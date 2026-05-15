import time
import numpy as np

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
)


def encode_labels(y_train, y_test):
    # XGBoost works with numeric labels.
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
    # Same SMOTE targets as the current best tuned XGBoost.
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_xgboost_model():
    return XGBClassifier(
        # Current best configuration from 04_tuned_xgboost.
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


def make_pipeline(X_train, label_encoder):
    # SMOTE is applied after preprocessing because it needs numeric features.
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", make_xgboost_model())
        ]
    )


def apply_probability_multipliers(probabilities, label_encoder, multipliers):
    # copy probabilities so original output is not changed.
    adjusted = probabilities.copy()

    # increase or decrease selected class probabilities.
    for class_name, multiplier in multipliers.items():
        class_index = int(label_encoder.transform([class_name])[0])
        adjusted[:, class_index] *= multiplier

    # select class with the highest adjusted probability.
    y_pred_encoded = np.argmax(adjusted, axis=1)

    return label_encoder.inverse_transform(y_pred_encoded)


def find_best_multipliers(model, X_valid, y_valid, label_encoder):
    # we tune only class probability multipliers.
    # R2L and U2R are weak classes, so we try to boost them.
    probabilities = model.predict_proba(X_valid)

    candidate_normal = [1.0, 0.95, 0.90]
    candidate_r2l = [1.0, 1.2, 1.5, 2.0, 2.5]
    candidate_u2r = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    best_score = -1
    best_multipliers = None
    best_pred = None

    print("\n===== Threshold / probability multiplier tuning =====")

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


def run_standard_cross_validation(model, X_train, y_train_encoded):
    # standard 5-fold CV without probability multiplier tuning.
    # used for comparison with previous experiments.
    print("\nRunning standard 5-fold cross-validation...")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=5,
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

    print(f"Standard CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


def evaluate_on_test(model, X_train, y_train_encoded, X_test, y_test, label_encoder, multipliers):
    print("\n===== Final Test Evaluation =====")

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

    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    # split only training data for threshold tuning.
    # KDDTest+ is not used here.
    X_fit, X_valid, y_fit_encoded, y_valid_encoded = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    y_valid = label_encoder.inverse_transform(y_valid_encoded)

    tuning_model = make_pipeline(X_fit, label_encoder)

    print("\n===== Training model for threshold tuning =====")
    tuning_start_time = time.perf_counter()

    tuning_model.fit(X_fit, y_fit_encoded)

    tuning_elapsed_time = time.perf_counter() - tuning_start_time
    print(f"Tuning model train time: {tuning_elapsed_time:.2f} seconds")

    # Baseline validation score before threshold tuning.
    base_valid_pred_encoded = tuning_model.predict(X_valid)
    base_valid_pred = label_encoder.inverse_transform(base_valid_pred_encoded)
    base_valid_f1 = f1_score(y_valid, base_valid_pred, average="macro")

    print(f"\nValidation macro F1 before multiplier tuning: {base_valid_f1:.4f}")

    best_multipliers, best_validation_f1 = find_best_multipliers(
        tuning_model,
        X_valid,
        y_valid,
        label_encoder
    )

    final_model_for_cv = make_pipeline(X_train, label_encoder)

    cv_mean, cv_std, cv_time = run_standard_cross_validation(
        final_model_for_cv,
        X_train,
        y_train_encoded
    )

    final_model = make_pipeline(X_train, label_encoder)

    y_pred, test_macro_f1, final_eval_time = evaluate_on_test(
        final_model,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder,
        best_multipliers
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/06_threshold_tuning_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 06_threshold_tuning_xgboost")
    print(f"Validation macro F1 before tuning: {base_valid_f1:.4f}")
    print(f"Validation macro F1 after tuning: {best_validation_f1:.4f}")
    print(f"Best multipliers: {best_multipliers}")
    print(f"Standard 5-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Test macro F1 with threshold tuning: {test_macro_f1:.4f}")
    print(f"Tuning model train time: {tuning_elapsed_time:.2f} seconds")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
