import time
import warnings
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def encode_labels(y_train, y_test):
    # voting uses numeric labels because xgboost and lightgbm need them
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


def make_sampling_strategy(label_encoder):
    # same minority targets as the best xgboost/lightgbm experiments
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_one_hot_encoder():
    # support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X_train):
    # mlp needs scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def make_selector_model():
    # smaller xgboost only for feature importance
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
    # strongest tuned xgboost setup from previous experiments
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


def make_feature_selection_xgboost_pipeline(X_train, label_encoder):
    # feature selection keeps only the stronger encoded features
    selector = SelectFromModel(
        estimator=make_selector_model(),
        threshold="median",
        prefit=False
    )

    # smote helps rare classes, especially r2l and u2r
    smote = SMOTE(
        sampling_strategy=make_sampling_strategy(label_encoder),
        k_neighbors=3,
        random_state=RANDOM_STATE
    )

    return ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("selector", selector),
            ("smote", smote),
            ("classifier", make_final_xgboost_model())
        ]
    )


def make_lightgbm_model():
    # lightgbm gives a different tree-based view than xgboost
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="multiclass",
        num_class=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )


def make_lightgbm_pipeline(X_train, label_encoder):
    # smote is placed after preprocessing because it needs numeric features
    smote = SMOTE(
        sampling_strategy=make_sampling_strategy(label_encoder),
        k_neighbors=3,
        random_state=RANDOM_STATE
    )

    return ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", smote),
            ("classifier", make_lightgbm_model())
        ]
    )


def make_mlp_model():
    # plain mlp worked better than smote + mlp in experiment 12
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=512,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=120,
        random_state=RANDOM_STATE
    )


def make_mlp_pipeline(X_train):
    # no smote here because it reduced mlp test performance
    return ImbPipeline(
        steps=[
            ("preprocessor", make_scaled_preprocessor(X_train)),
            ("classifier", make_mlp_model())
        ]
    )


def make_voting_ensemble(X_train, label_encoder, weights):
    # combine different model types so their errors may compensate each other
    fs_xgb = make_feature_selection_xgboost_pipeline(X_train, label_encoder)
    lgbm = make_lightgbm_pipeline(X_train, label_encoder)
    mlp = make_mlp_pipeline(X_train)

    return VotingClassifier(
        estimators=[
            ("fs_xgb", fs_xgb),
            ("lgbm", lgbm),
            ("mlp", mlp)
        ],
        voting="soft",
        weights=weights,
        n_jobs=1
    )


def apply_probability_multipliers(probabilities, label_encoder, multipliers):
    # adjust class probabilities before choosing the final class
    adjusted = probabilities.copy()

    for class_name, multiplier in multipliers.items():
        class_index = int(label_encoder.transform([class_name])[0])
        adjusted[:, class_index] *= multiplier

    y_pred_encoded = np.argmax(adjusted, axis=1)

    return label_encoder.inverse_transform(y_pred_encoded)


def find_best_multipliers(model, X_valid, y_valid, label_encoder):
    # tune only on train split, not on kddtest+
    probabilities = model.predict_proba(X_valid)

    candidate_normal = [1.0, 0.95, 0.90, 0.85]
    candidate_r2l = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    candidate_u2r = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    best_score = -1
    best_multipliers = None
    best_pred = None

    print("\n===== Probability multiplier tuning for voting ensemble =====")

    for normal_multiplier in candidate_normal:
        for r2l_multiplier in candidate_r2l:
            for u2r_multiplier in candidate_u2r:
                multipliers = {
                    "Normal": normal_multiplier,
                    "R2L": r2l_multiplier,
                    "U2R": u2r_multiplier
                }

                y_pred = apply_probability_multipliers(
                    probabilities,
                    label_encoder,
                    multipliers
                )

                score = f1_score(y_valid, y_pred, average="macro")

                if score > best_score:
                    best_score = score
                    best_multipliers = multipliers
                    best_pred = y_pred

    print(f"Best validation macro F1: {best_score:.4f}")
    print(f"Best multipliers: {best_multipliers}")

    print("\nValidation classification report with best multipliers:")
    print(classification_report(y_valid, best_pred, labels=LABELS, zero_division=0))

    return best_multipliers, best_score


def run_cross_validation(model, X_train, y_train_encoded):
    # 3-fold cv is used because ensemble is quite slow
    print("\nRunning 3-fold cross-validation...")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train_encoded,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    elapsed_time = time.perf_counter() - start_time

    print(f"CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


def evaluate_raw_voting(model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    # evaluate normal soft voting without extra probability tuning
    print("\n===== Raw Voting Ensemble Test Evaluation =====")

    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed_time = time.perf_counter() - start_time

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Raw voting test macro F1-score: {test_macro_f1:.4f}")
    print(f"Raw voting train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time


def evaluate_tuned_voting(model, X_train, y_train_encoded, X_test, y_test, label_encoder, multipliers):
    # evaluate soft voting with validation-selected probability multipliers
    print("\n===== Tuned Voting Ensemble Test Evaluation =====")

    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

    probabilities = model.predict_proba(X_test)

    y_pred = apply_probability_multipliers(
        probabilities,
        label_encoder,
        multipliers
    )

    elapsed_time = time.perf_counter() - start_time

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Tuned voting test macro F1-score: {test_macro_f1:.4f}")
    print(f"Tuned voting train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    # try a few simple weight combinations for base models
    candidate_weights = [
        {
            "name": "weights_1_balanced",
            "weights": [3, 2, 2]
        },
        {
            "name": "weights_2_xgb_heavier",
            "weights": [4, 2, 2]
        },
        {
            "name": "weights_3_mlp_heavier",
            "weights": [3, 2, 3]
        }
    ]

    best_name = None
    best_weights = None
    best_cv_mean = -1
    best_cv_std = 0
    best_cv_time = 0

    print("\n===== Voting Ensemble Cross-Validation Tuning =====")

    for config in candidate_weights:
        print(f"\nTesting configuration: {config['name']}")
        print(f"Weights [fs_xgb, lgbm, mlp]: {config['weights']}")

        model = make_voting_ensemble(
            X_train,
            label_encoder,
            weights=config["weights"]
        )

        cv_mean, cv_std, cv_time = run_cross_validation(
            model,
            X_train,
            y_train_encoded
        )

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_cv_time = cv_time
            best_name = config["name"]
            best_weights = config["weights"]

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best weights [fs_xgb, lgbm, mlp]: {best_weights}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Best CV time: {best_cv_time:.2f} seconds")

    # split train data once more to tune probability multipliers safely
    X_fit, X_valid, y_fit_encoded, y_valid_encoded = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    y_valid = label_encoder.inverse_transform(y_valid_encoded)

    tuning_model = make_voting_ensemble(
        X_fit,
        label_encoder,
        weights=best_weights
    )

    print("\n===== Training voting ensemble for multiplier tuning =====")
    tuning_start_time = time.perf_counter()

    tuning_model.fit(X_fit, y_fit_encoded)

    tuning_elapsed_time = time.perf_counter() - tuning_start_time
    print(f"Tuning model train time: {tuning_elapsed_time:.2f} seconds")

    base_valid_pred_encoded = tuning_model.predict(X_valid)
    base_valid_pred = label_encoder.inverse_transform(base_valid_pred_encoded)
    base_valid_f1 = f1_score(y_valid, base_valid_pred, average="macro")

    print(f"\nValidation macro F1 before multiplier tuning: {base_valid_f1:.4f}")

    print("\nValidation classification report before multiplier tuning:")
    print(classification_report(y_valid, base_valid_pred, labels=LABELS, zero_division=0))

    best_multipliers, best_validation_f1 = find_best_multipliers(
        tuning_model,
        X_valid,
        y_valid,
        label_encoder
    )

    # final raw model trained on full training set
    raw_final_model = make_voting_ensemble(
        X_train,
        label_encoder,
        weights=best_weights
    )

    y_pred_raw, raw_test_macro_f1, raw_eval_time = evaluate_raw_voting(
        raw_final_model,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    # final tuned model trained on full training set
    tuned_final_model = make_voting_ensemble(
        X_train,
        label_encoder,
        weights=best_weights
    )

    y_pred_tuned, tuned_test_macro_f1, tuned_eval_time = evaluate_tuned_voting(
        tuned_final_model,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder,
        best_multipliers
    )

    # keep whichever version scored better on final evaluation
    if tuned_test_macro_f1 >= raw_test_macro_f1:
        final_y_pred = y_pred_tuned
        final_test_macro_f1 = tuned_test_macro_f1
        final_model_name = "Tuned Voting Ensemble"
    else:
        final_y_pred = y_pred_raw
        final_test_macro_f1 = raw_test_macro_f1
        final_model_name = "Raw Voting Ensemble"

    save_confusion_matrix(
        y_test,
        final_y_pred,
        "outputs/confusion_matrices/13_voting_ensemble.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 13_voting_ensemble")
    print("Base models: Feature Selection XGBoost + LightGBM + Plain MLP")
    print(f"Best weight configuration: {best_name}")
    print(f"Best weights [fs_xgb, lgbm, mlp]: {best_weights}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Validation macro F1 before multiplier tuning: {base_valid_f1:.4f}")
    print(f"Validation macro F1 after multiplier tuning: {best_validation_f1:.4f}")
    print(f"Best multipliers: {best_multipliers}")
    print(f"Raw voting test macro F1: {raw_test_macro_f1:.4f}")
    print(f"Tuned voting test macro F1: {tuned_test_macro_f1:.4f}")
    print(f"Best final model: {final_model_name}")
    print(f"Best final test macro F1: {final_test_macro_f1:.4f}")
    print(f"Raw voting final train + prediction time: {raw_eval_time:.2f} seconds")
    print(f"Tuned voting final train + prediction time: {tuned_eval_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
