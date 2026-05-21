import os
import time
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def make_one_hot_encoder():
    # support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X_train):
    # keras needs scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def encode_labels(y_train, y_test):
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("\nLabel encoding:")
    for class_name, encoded_value in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    ):
        print(f"{class_name} -> {encoded_value}")

    return y_train_encoded, y_test_encoded, label_encoder


def set_all_seeds(seed):
    # keep keras runs more stable
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_sampling_strategy_04(label_encoder):
    # smote targets from tuned xgboost experiment
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_smote_xgboost_04(X_train, label_encoder):
    # exp04 tuned smote xgboost base model
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy_04(label_encoder),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=2,
                objective="multi:softprob",
                num_class=5,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )


def make_selector_model():
    # selector model from exp14
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=2,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def make_final_xgboost_model():
    # final xgboost model from exp14
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=2,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def make_feature_selection_xgboost_14(X_train, label_encoder):
    # exp14 feature selection + smote + xgboost
    preprocessor = make_preprocessor(X_train)

    selector = SelectFromModel(
        estimator=make_selector_model(),
        threshold="0.5*mean",
        prefit=False
    )

    smote = SMOTE(
        sampling_strategy=make_sampling_strategy_04(label_encoder),
        k_neighbors=3,
        random_state=RANDOM_STATE
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("smote", smote),
            ("classifier", make_final_xgboost_model())
        ]
    )


def make_keras_config():
    # best setup from experiment 19
    return {
        "seed": 100,
        "hidden_1": 256,
        "hidden_2": 128,
        "hidden_3": 64,
        "dropout": 0.30,
        "learning_rate": 0.0005,
        "batch_size": 512,
        "epochs": 140,
        "patience": 12,
        "class_weights": {
            0: 1.0,
            1: 0.7,
            2: 2.0,
            3: 20.0,
            4: 60.0
        }
    }


def make_keras_model(input_dim, config):
    # clear old keras graph before making a new model
    tf.keras.backend.clear_session()
    set_all_seeds(config["seed"])

    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    # first dense block
    model.add(Dense(config["hidden_1"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # second dense block
    model.add(Dense(config["hidden_2"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # third dense block
    model.add(Dense(config["hidden_3"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # output layer for 5 classes
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, X_valid, y_valid, patience):
        super().__init__()

        # monitor macro f1 directly because this is our main metric
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.patience = patience

        self.best_score = -1
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.X_valid, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_valid,
            y_pred,
            average="macro",
            zero_division=0
        )

        if score > self.best_score:
            self.best_score = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        # restore best macro f1 weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_keras_and_predict_proba(X_train_raw, y_train, X_pred_raw):
    # keras uses its own scaled preprocessing
    config = make_keras_config()

    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    preprocessor = make_scaled_preprocessor(X_fit_raw)

    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_valid = preprocessor.transform(X_valid_raw)
    X_pred = preprocessor.transform(X_pred_raw)

    model = make_keras_model(
        input_dim=X_fit.shape[1],
        config=config
    )

    early_stopping = MacroF1EarlyStopping(
        X_valid=X_valid,
        y_valid=y_valid,
        patience=config["patience"]
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_valid, y_valid),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=config["class_weights"],
        callbacks=[early_stopping],
        verbose=0
    )

    probabilities = model.predict(X_pred, verbose=0)

    return probabilities, early_stopping.best_score, early_stopping.best_epoch


def predict_proba_with_safe_columns(model, X):
    # make sure predict_proba always returns 5 columns
    proba = model.predict_proba(X)

    if proba.shape[1] == 5:
        return proba

    fixed = np.zeros((proba.shape[0], 5), dtype=float)

    for index, class_id in enumerate(model.classes_):
        fixed[:, int(class_id)] = proba[:, index]

    return fixed


def make_stacking_features(proba_keras, proba_xgb04, proba_fs14):
    # meta model receives probabilities from all base models
    return np.hstack([
        proba_keras,
        proba_xgb04,
        proba_fs14
    ])


def evaluate_oof_base_model(name, oof_proba, y_train_encoded, label_encoder):
    y_pred_encoded = np.argmax(oof_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_true = label_encoder.inverse_transform(y_train_encoded)

    score = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"{name} OOF macro F1: {score:.4f}")

    return score


def build_oof_stacking_features(X_train, y_train_encoded, label_encoder):
    print("\n===== Building out-of-fold stacking features =====")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    n_samples = X_train.shape[0]

    oof_keras = np.zeros((n_samples, 5), dtype=float)
    oof_xgb04 = np.zeros((n_samples, 5), dtype=float)
    oof_fs14 = np.zeros((n_samples, 5), dtype=float)

    fold_scores = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n===== Stacking fold {fold_id} =====")

        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train_encoded[train_index]
        y_fold_valid = y_train_encoded[valid_index]

        fold_start_time = time.perf_counter()

        print("Training Keras exp19 base model...")
        keras_proba, keras_valid_f1, keras_best_epoch = train_keras_and_predict_proba(
            X_fold_train,
            y_fold_train,
            X_fold_valid
        )

        oof_keras[valid_index] = keras_proba

        print(f"Keras inner validation macro F1: {keras_valid_f1:.4f}")
        print(f"Keras best epoch: {keras_best_epoch}")

        print("Training exp04 SMOTE XGBoost base model...")
        xgb04_model = make_smote_xgboost_04(
            X_fold_train,
            label_encoder
        )

        xgb04_model.fit(X_fold_train, y_fold_train)
        oof_xgb04[valid_index] = predict_proba_with_safe_columns(
            xgb04_model,
            X_fold_valid
        )

        print("Training exp14 Feature Selection XGBoost base model...")
        fs14_model = make_feature_selection_xgboost_14(
            X_fold_train,
            label_encoder
        )

        fs14_model.fit(X_fold_train, y_fold_train)
        oof_fs14[valid_index] = predict_proba_with_safe_columns(
            fs14_model,
            X_fold_valid
        )

        fold_stack_features = make_stacking_features(
            oof_keras[valid_index],
            oof_xgb04[valid_index],
            oof_fs14[valid_index]
        )

        fold_meta = LogisticRegression(
            C=1.0,
            max_iter=3000,
            multi_class="auto",
            class_weight=None,
            random_state=RANDOM_STATE
        )

        fold_meta.fit(fold_stack_features, y_fold_valid)
        fold_pred = fold_meta.predict(fold_stack_features)

        fold_score = f1_score(
            y_fold_valid,
            fold_pred,
            average="macro",
            zero_division=0
        )

        fold_scores.append(fold_score)

        fold_elapsed = time.perf_counter() - fold_start_time

        print(f"Fold quick meta fit macro F1 on fold OOF features: {fold_score:.4f}")
        print(f"Fold time: {fold_elapsed:.2f} seconds")

    elapsed_time = time.perf_counter() - start_time

    print("\n===== OOF base model summary =====")

    evaluate_oof_base_model(
        "Keras exp19",
        oof_keras,
        y_train_encoded,
        label_encoder
    )

    evaluate_oof_base_model(
        "SMOTE XGBoost exp04",
        oof_xgb04,
        y_train_encoded,
        label_encoder
    )

    evaluate_oof_base_model(
        "Feature Selection XGBoost exp14",
        oof_fs14,
        y_train_encoded,
        label_encoder
    )

    stacking_features = make_stacking_features(
        oof_keras,
        oof_xgb04,
        oof_fs14
    )

    print(f"\nOOF stacking feature shape: {stacking_features.shape}")
    print(f"OOF build time: {elapsed_time:.2f} seconds")

    return stacking_features, elapsed_time


def make_meta_candidates():
    # keep meta models simple to avoid overfitting probabilities
    return [
        {
            "name": "meta_logreg_c05",
            "model": LogisticRegression(
                C=0.5,
                max_iter=3000,
                multi_class="auto",
                class_weight=None,
                random_state=RANDOM_STATE
            )
        },
        {
            "name": "meta_logreg_c1",
            "model": LogisticRegression(
                C=1.0,
                max_iter=3000,
                multi_class="auto",
                class_weight=None,
                random_state=RANDOM_STATE
            )
        },
        {
            "name": "meta_logreg_balanced_c05",
            "model": LogisticRegression(
                C=0.5,
                max_iter=3000,
                multi_class="auto",
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        },
        {
            "name": "meta_logreg_balanced_c1",
            "model": LogisticRegression(
                C=1.0,
                max_iter=3000,
                multi_class="auto",
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        }
    ]


def tune_meta_model(stacking_features, y_train_encoded):
    print("\n===== Meta-model tuning on OOF stacking features =====")

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    best_name = None
    best_model = None
    best_cv_mean = -1
    best_cv_std = 0

    for candidate in make_meta_candidates():
        name = candidate["name"]
        model = candidate["model"]

        scores = cross_val_score(
            model,
            stacking_features,
            y_train_encoded,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        print(f"{name}: CV macro F1 = {scores.mean():.4f} ± {scores.std():.4f}")

        if scores.mean() > best_cv_mean:
            best_cv_mean = scores.mean()
            best_cv_std = scores.std()
            best_name = name
            best_model = model

    print("\n===== Best Meta-model =====")
    print(f"Best meta model: {best_name}")
    print(f"Best meta CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")

    best_model.fit(stacking_features, y_train_encoded)

    return best_name, best_model, best_cv_mean, best_cv_std


def train_full_base_models_and_predict_test(X_train, y_train_encoded, X_test, label_encoder):
    print("\n===== Training full base models and predicting test probabilities =====")

    start_time = time.perf_counter()

    print("Training full Keras exp19 base model...")
    test_keras_proba, keras_valid_f1, keras_best_epoch = train_keras_and_predict_proba(
        X_train,
        y_train_encoded,
        X_test
    )

    print(f"Full Keras validation macro F1: {keras_valid_f1:.4f}")
    print(f"Full Keras best epoch: {keras_best_epoch}")

    print("Training full exp04 SMOTE XGBoost base model...")
    xgb04_model = make_smote_xgboost_04(
        X_train,
        label_encoder
    )

    xgb04_model.fit(X_train, y_train_encoded)
    test_xgb04_proba = predict_proba_with_safe_columns(
        xgb04_model,
        X_test
    )

    print("Training full exp14 Feature Selection XGBoost base model...")
    fs14_model = make_feature_selection_xgboost_14(
        X_train,
        label_encoder
    )

    fs14_model.fit(X_train, y_train_encoded)
    test_fs14_proba = predict_proba_with_safe_columns(
        fs14_model,
        X_test
    )

    test_stacking_features = make_stacking_features(
        test_keras_proba,
        test_xgb04_proba,
        test_fs14_proba
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"Test stacking feature shape: {test_stacking_features.shape}")
    print(f"Full base model train + test probability time: {elapsed_time:.2f} seconds")

    return test_stacking_features, elapsed_time


def evaluate_final_stacking(meta_model, test_stacking_features, y_test, label_encoder):
    print("\n===== Final Stacking Test Evaluation =====")

    y_pred_encoded = meta_model.predict(test_stacking_features)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"Stacking test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    stacking_features, oof_time = build_oof_stacking_features(
        X_train,
        y_train_encoded,
        label_encoder
    )

    meta_name, meta_model, meta_cv_mean, meta_cv_std = tune_meta_model(
        stacking_features,
        y_train_encoded
    )

    test_stacking_features, full_base_time = train_full_base_models_and_predict_test(
        X_train,
        y_train_encoded,
        X_test,
        label_encoder
    )

    y_pred, test_macro_f1 = evaluate_final_stacking(
        meta_model,
        test_stacking_features,
        y_test,
        label_encoder
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/23_stacking_keras_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 23_stacking_keras_xgboost")
    print("Base models: exp19 Keras + exp04 SMOTE XGBoost + exp14 Feature Selection XGBoost")
    print(f"Meta model: {meta_name}")
    print(f"Meta CV macro F1 on OOF features: {meta_cv_mean:.4f} ± {meta_cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"OOF stacking feature build time: {oof_time:.2f} seconds")
    print(f"Full base model train + test probability time: {full_base_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
