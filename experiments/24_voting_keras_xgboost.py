import os
import time
import warnings
import itertools
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def make_sampling_strategy(label_encoder):
    # same smote targets as tuned xgboost / feature selection xgboost
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_smote_xgboost_04(X_train, label_encoder):
    # experiment 04 tuned smote xgboost
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),
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
    # feature selector from experiment 14
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
    # final xgboost model from experiment 14
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
    # experiment 14 feature selection + smote + xgboost
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("selector", SelectFromModel(
                estimator=make_selector_model(),
                threshold="0.5*mean",
                prefit=False
            )),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
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
    # clear old keras graph before new training
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

    # 5 output classes
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

        # monitor macro f1 directly
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
        # restore best validation macro f1 weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_keras_and_predict_proba(X_train_raw, y_train, X_pred_raw):
    # keras has its own scaled preprocessing
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


def predict_proba_safe(model, X):
    # always return columns for all 5 classes
    proba = model.predict_proba(X)

    if proba.shape[1] == 5:
        return proba

    fixed = np.zeros((proba.shape[0], 5), dtype=float)

    for index, class_id in enumerate(model.classes_):
        fixed[:, int(class_id)] = proba[:, index]

    return fixed


def weighted_vote_proba(proba_keras, proba_xgb04, proba_fs14, weights):
    # order: keras, xgb04, fs14
    weighted = (
        weights[0] * proba_keras +
        weights[1] * proba_xgb04 +
        weights[2] * proba_fs14
    )

    return weighted / sum(weights)


def apply_multipliers(proba, label_encoder, multipliers):
    # small probability boosts for minority classes
    tuned = proba.copy()

    for class_name, multiplier in multipliers.items():
        class_index = int(label_encoder.transform([class_name])[0])
        tuned[:, class_index] *= multiplier

    # normalize back to probabilities
    tuned = tuned / tuned.sum(axis=1, keepdims=True)

    return tuned


def predict_labels_from_proba(proba, label_encoder):
    y_pred_encoded = np.argmax(proba, axis=1)
    return label_encoder.inverse_transform(y_pred_encoded)


def evaluate_proba(name, proba, y_true, label_encoder):
    y_pred = predict_labels_from_proba(proba, label_encoder)

    score = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"{name} macro F1: {score:.4f}")

    return score, y_pred


def make_weight_candidates():
    # order is [keras, smote xgb04, feature-selection xgb14]
    return [
        {"name": "weights_1_equal", "weights": [1, 1, 1]},
        {"name": "weights_2_keras_heavy", "weights": [3, 2, 2]},
        {"name": "weights_3_keras_heavier", "weights": [4, 2, 2]},
        {"name": "weights_4_keras_dominant", "weights": [5, 2, 2]},
        {"name": "weights_5_keras_xgb04", "weights": [3, 2, 1]},
        {"name": "weights_6_keras_fs14", "weights": [3, 1, 2]},
        {"name": "weights_7_xgb_balanced", "weights": [2, 2, 2]}
    ]


def train_base_models_predict_valid(X_train, y_train_encoded, X_valid, label_encoder):
    # train all base models once and return their probabilities
    print("Training Keras exp19 base model...")
    proba_keras, keras_valid_f1, keras_best_epoch = train_keras_and_predict_proba(
        X_train,
        y_train_encoded,
        X_valid
    )

    print(f"Keras inner validation macro F1: {keras_valid_f1:.4f}")
    print(f"Keras best epoch: {keras_best_epoch}")

    print("Training exp04 SMOTE XGBoost base model...")
    xgb04_model = make_smote_xgboost_04(
        X_train,
        label_encoder
    )

    xgb04_model.fit(X_train, y_train_encoded)
    proba_xgb04 = predict_proba_safe(
        xgb04_model,
        X_valid
    )

    print("Training exp14 Feature Selection XGBoost base model...")
    fs14_model = make_feature_selection_xgboost_14(
        X_train,
        label_encoder
    )

    fs14_model.fit(X_train, y_train_encoded)
    proba_fs14 = predict_proba_safe(
        fs14_model,
        X_valid
    )

    return proba_keras, proba_xgb04, proba_fs14


def tune_weights_by_cv(X_train, y_train_encoded, label_encoder):
    print("\n===== Voting weight tuning with 3-fold CV =====")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    candidates = make_weight_candidates()
    candidate_scores = {
        candidate["name"]: []
        for candidate in candidates
    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n===== CV fold {fold_id} =====")

        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train_encoded[train_index]
        y_fold_valid = y_train_encoded[valid_index]

        fold_start_time = time.perf_counter()

        proba_keras, proba_xgb04, proba_fs14 = train_base_models_predict_valid(
            X_fold_train,
            y_fold_train,
            X_fold_valid,
            label_encoder
        )

        for candidate in candidates:
            vote_proba = weighted_vote_proba(
                proba_keras,
                proba_xgb04,
                proba_fs14,
                candidate["weights"]
            )

            y_pred_encoded = np.argmax(vote_proba, axis=1)

            score = f1_score(
                y_fold_valid,
                y_pred_encoded,
                average="macro",
                zero_division=0
            )

            candidate_scores[candidate["name"]].append(score)

            print(
                f"{candidate['name']} "
                f"{candidate['weights']} fold macro F1: {score:.4f}"
            )

        fold_elapsed = time.perf_counter() - fold_start_time
        print(f"Fold time: {fold_elapsed:.2f} seconds")

    print("\n===== Voting CV Summary =====")

    best_candidate = None
    best_mean = -1
    best_std = 0

    for candidate in candidates:
        scores = np.array(candidate_scores[candidate["name"]])
        mean_score = scores.mean()
        std_score = scores.std()

        print(
            f"{candidate['name']} {candidate['weights']} | "
            f"CV macro F1: {mean_score:.4f} ± {std_score:.4f}"
        )

        if mean_score > best_mean:
            best_mean = mean_score
            best_std = std_score
            best_candidate = candidate

    elapsed_time = time.perf_counter() - start_time

    print("\n===== Best Voting Weights =====")
    print(f"Best configuration: {best_candidate['name']}")
    print(f"Best weights [keras, xgb04, fs14]: {best_candidate['weights']}")
    print(f"Best CV macro F1: {best_mean:.4f} ± {best_std:.4f}")
    print(f"Weight tuning CV time: {elapsed_time:.2f} seconds")

    return best_candidate, best_mean, best_std, elapsed_time


def tune_probability_multipliers(X_train, y_train_encoded, label_encoder, best_weights):
    print("\n===== Voting probability multiplier tuning =====")

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    proba_keras, proba_xgb04, proba_fs14 = train_base_models_predict_valid(
        X_fit,
        y_fit,
        X_valid,
        label_encoder
    )

    raw_proba = weighted_vote_proba(
        proba_keras,
        proba_xgb04,
        proba_fs14,
        best_weights
    )

    raw_score = f1_score(
        y_valid,
        np.argmax(raw_proba, axis=1),
        average="macro",
        zero_division=0
    )

    print(f"Validation macro F1 before multiplier tuning: {raw_score:.4f}")

    normal_values = [0.85, 0.90, 0.95, 1.0]
    r2l_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    u2r_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    best_multipliers = {
        "Normal": 1.0,
        "R2L": 1.0,
        "U2R": 1.0
    }

    best_score = raw_score

    for normal_mult, r2l_mult, u2r_mult in itertools.product(
        normal_values,
        r2l_values,
        u2r_values
    ):
        multipliers = {
            "Normal": normal_mult,
            "R2L": r2l_mult,
            "U2R": u2r_mult
        }

        tuned_proba = apply_multipliers(
            raw_proba,
            label_encoder,
            multipliers
        )

        y_pred_encoded = np.argmax(tuned_proba, axis=1)

        score = f1_score(
            y_valid,
            y_pred_encoded,
            average="macro",
            zero_division=0
        )

        if score > best_score:
            best_score = score
            best_multipliers = multipliers

    print(f"Best validation macro F1 after multiplier tuning: {best_score:.4f}")
    print(f"Best multipliers: {best_multipliers}")

    y_valid_raw = label_encoder.inverse_transform(np.argmax(raw_proba, axis=1))
    y_valid_tuned = label_encoder.inverse_transform(
        np.argmax(
            apply_multipliers(raw_proba, label_encoder, best_multipliers),
            axis=1
        )
    )

    print("\nValidation report before multiplier tuning:")
    print(classification_report(
        label_encoder.inverse_transform(y_valid),
        y_valid_raw,
        labels=LABELS,
        zero_division=0
    ))

    print("\nValidation report after multiplier tuning:")
    print(classification_report(
        label_encoder.inverse_transform(y_valid),
        y_valid_tuned,
        labels=LABELS,
        zero_division=0
    ))

    return best_multipliers, raw_score, best_score


def train_full_base_models_predict_test(X_train, y_train_encoded, X_test, label_encoder):
    print("\n===== Training full base models for final test evaluation =====")

    start_time = time.perf_counter()

    proba_keras, keras_valid_f1, keras_best_epoch = train_keras_and_predict_proba(
        X_train,
        y_train_encoded,
        X_test
    )

    print(f"Full Keras validation macro F1: {keras_valid_f1:.4f}")
    print(f"Full Keras best epoch: {keras_best_epoch}")

    print("Training full exp04 SMOTE XGBoost...")
    xgb04_model = make_smote_xgboost_04(
        X_train,
        label_encoder
    )

    xgb04_model.fit(X_train, y_train_encoded)
    proba_xgb04 = predict_proba_safe(
        xgb04_model,
        X_test
    )

    print("Training full exp14 Feature Selection XGBoost...")
    fs14_model = make_feature_selection_xgboost_14(
        X_train,
        label_encoder
    )

    fs14_model.fit(X_train, y_train_encoded)
    proba_fs14 = predict_proba_safe(
        fs14_model,
        X_test
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"Full base model train + prediction time: {elapsed_time:.2f} seconds")

    return proba_keras, proba_xgb04, proba_fs14, elapsed_time


def final_test_evaluation(
    proba_keras,
    proba_xgb04,
    proba_fs14,
    y_test,
    label_encoder,
    best_weights,
    best_multipliers
):
    print("\n===== Individual Base Model Test Evaluation =====")

    keras_score, _ = evaluate_proba(
        "Keras exp19",
        proba_keras,
        y_test,
        label_encoder
    )

    xgb04_score, _ = evaluate_proba(
        "SMOTE XGBoost exp04",
        proba_xgb04,
        y_test,
        label_encoder
    )

    fs14_score, _ = evaluate_proba(
        "Feature Selection XGBoost exp14",
        proba_fs14,
        y_test,
        label_encoder
    )

    print("\n===== Raw Voting Test Evaluation =====")

    raw_vote_proba = weighted_vote_proba(
        proba_keras,
        proba_xgb04,
        proba_fs14,
        best_weights
    )

    raw_score, raw_pred = evaluate_proba(
        "Raw voting",
        raw_vote_proba,
        y_test,
        label_encoder
    )

    print("\nRaw voting classification report:")
    print(classification_report(
        y_test,
        raw_pred,
        labels=LABELS,
        zero_division=0
    ))

    print("\n===== Tuned Voting Test Evaluation =====")

    tuned_vote_proba = apply_multipliers(
        raw_vote_proba,
        label_encoder,
        best_multipliers
    )

    tuned_score, tuned_pred = evaluate_proba(
        "Tuned voting",
        tuned_vote_proba,
        y_test,
        label_encoder
    )

    print("\nTuned voting classification report:")
    print(classification_report(
        y_test,
        tuned_pred,
        labels=LABELS,
        zero_division=0
    ))

    if tuned_score >= raw_score:
        best_name = "Tuned Voting"
        best_score = tuned_score
        best_pred = tuned_pred
    else:
        best_name = "Raw Voting"
        best_score = raw_score
        best_pred = raw_pred

    print("\n===== Final Voting Decision =====")
    print(f"Best final model: {best_name}")
    print(f"Best final test macro F1: {best_score:.4f}")

    return {
        "keras_score": keras_score,
        "xgb04_score": xgb04_score,
        "fs14_score": fs14_score,
        "raw_score": raw_score,
        "tuned_score": tuned_score,
        "best_name": best_name,
        "best_score": best_score,
        "best_pred": best_pred
    }


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    best_candidate, cv_mean, cv_std, cv_time = tune_weights_by_cv(
        X_train,
        y_train_encoded,
        label_encoder
    )

    best_multipliers, validation_raw_score, validation_tuned_score = tune_probability_multipliers(
        X_train,
        y_train_encoded,
        label_encoder,
        best_candidate["weights"]
    )

    proba_keras, proba_xgb04, proba_fs14, full_base_time = train_full_base_models_predict_test(
        X_train,
        y_train_encoded,
        X_test,
        label_encoder
    )

    final_results = final_test_evaluation(
        proba_keras,
        proba_xgb04,
        proba_fs14,
        y_test,
        label_encoder,
        best_candidate["weights"],
        best_multipliers
    )

    save_confusion_matrix(
        y_test,
        final_results["best_pred"],
        "outputs/confusion_matrices/24_voting_keras_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 24_voting_keras_xgboost")
    print("Base models: exp19 Keras + exp04 SMOTE XGBoost + exp14 Feature Selection XGBoost")
    print(f"Best weight configuration: {best_candidate['name']}")
    print(f"Best weights [keras, xgb04, fs14]: {best_candidate['weights']}")
    print(f"Best 3-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Validation macro F1 before multiplier tuning: {validation_raw_score:.4f}")
    print(f"Validation macro F1 after multiplier tuning: {validation_tuned_score:.4f}")
    print(f"Best multipliers: {best_multipliers}")
    print(f"Keras exp19 test macro F1 in this run: {final_results['keras_score']:.4f}")
    print(f"SMOTE XGBoost exp04 test macro F1 in this run: {final_results['xgb04_score']:.4f}")
    print(f"Feature Selection XGBoost exp14 test macro F1 in this run: {final_results['fs14_score']:.4f}")
    print(f"Raw voting test macro F1: {final_results['raw_score']:.4f}")
    print(f"Tuned voting test macro F1: {final_results['tuned_score']:.4f}")
    print(f"Best final model: {final_results['best_name']}")
    print(f"Best final test macro F1: {final_results['best_score']:.4f}")
    print(f"Voting CV time: {cv_time:.2f} seconds")
    print(f"Full base model train + prediction time: {full_base_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
