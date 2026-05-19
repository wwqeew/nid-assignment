import time
import numpy as np

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
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


def make_tuned_xgboost_model():
    return XGBClassifier(
        # best xgboost parameters from 04_tuned_xgboost.py
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,

        # row and feature sampling to reduce overfitting
        subsample=0.9,
        colsample_bytree=0.9,

        # makes splits more conservative
        min_child_weight=2,

        # l2 regularization
        reg_lambda=2,

        # multiclass classification
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",

        # faster cpu training
        tree_method="hist",

        # reproducibility
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def make_model(X_train):
    # no smote here, only sample weights
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", make_tuned_xgboost_model())
        ]
    )


def make_sample_weights(y_encoded, label_encoder, class_weights):
    # convert class names into encoded class ids
    encoded_weights = {}

    for class_name, weight in class_weights.items():
        class_id = int(label_encoder.transform([class_name])[0])
        encoded_weights[class_id] = weight

    # create one weight per training row
    sample_weights = np.array(
        [encoded_weights[int(label)] for label in y_encoded],
        dtype=float
    )

    return sample_weights


def run_cv(X_train, y_train_encoded, label_encoder, class_weights, folds=3):
    # manual cv is used because sample_weight must be passed only to the classifier
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []
    start_time = time.perf_counter()

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train_encoded[train_index]
        y_fold_valid = y_train_encoded[valid_index]

        fold_weights = make_sample_weights(
            y_fold_train,
            label_encoder,
            class_weights
        )

        model = make_model(X_fold_train)

        model.fit(
            X_fold_train,
            y_fold_train,
            classifier__sample_weight=fold_weights
        )

        y_pred_encoded = model.predict(X_fold_valid)

        score = f1_score(
            y_fold_valid,
            y_pred_encoded,
            average="macro"
        )

        scores.append(score)

        print(f"Fold {fold_id} macro F1: {score:.4f}")

    elapsed_time = time.perf_counter() - start_time

    scores = np.array(scores)

    return scores.mean(), scores.std(), elapsed_time


def evaluate_on_test(model, X_train, y_train_encoded, X_test, y_test, label_encoder, class_weights):
    start_time = time.perf_counter()

    sample_weights = make_sample_weights(
        y_train_encoded,
        label_encoder,
        class_weights
    )

    model.fit(
        X_train,
        y_train_encoded,
        classifier__sample_weight=sample_weights
    )

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed_time = time.perf_counter() - start_time

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Final Test Evaluation =====")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    candidate_weights = [
        {
            "name": "weighted_1_no_extra_weights",
            # baseline inside this experiment
            "weights": {
                "DoS": 1,
                "Normal": 1,
                "Probe": 1,
                "R2L": 1,
                "U2R": 1
            }
        },
        {
            "name": "weighted_2_moderate_minority_focus",
            # moderate penalty for rare classes
            "weights": {
                "DoS": 1,
                "Normal": 1,
                "Probe": 2,
                "R2L": 8,
                "U2R": 30
            }
        },
        {
            "name": "weighted_3_strong_minority_focus",
            # stronger focus on r2l and u2r
            "weights": {
                "DoS": 1,
                "Normal": 1,
                "Probe": 2,
                "R2L": 12,
                "U2R": 60
            }
        },
        {
            "name": "weighted_4_very_strong_u2r_focus",
            # test if stronger u2r weight improves rare privilege escalation detection
            "weights": {
                "DoS": 1,
                "Normal": 1,
                "Probe": 2,
                "R2L": 10,
                "U2R": 100
            }
        },
        {
            "name": "weighted_5_reduce_normal_bias",
            # reduce normal weight so model is less comfortable predicting normal
            "weights": {
                "DoS": 1,
                "Normal": 0.7,
                "Probe": 2,
                "R2L": 12,
                "U2R": 80
            }
        }
    ]

    best_name = None
    best_weights = None
    best_cv_mean = -1
    best_cv_std = 0
    best_cv_time = 0

    print("\n===== Weighted XGBoost cross-validation tuning =====")

    for candidate in candidate_weights:
        print(f"\nTesting configuration: {candidate['name']}")
        print(f"Class weights: {candidate['weights']}")

        cv_mean, cv_std, cv_time = run_cv(
            X_train,
            y_train_encoded,
            label_encoder,
            candidate["weights"],
            folds=3
        )

        print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"CV time: {cv_time:.2f} seconds")

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_cv_time = cv_time
            best_name = candidate["name"]
            best_weights = candidate["weights"]

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best class weights: {best_weights}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Best configuration CV time: {best_cv_time:.2f} seconds")

    best_model = make_model(X_train)

    y_pred, test_macro_f1, final_eval_time = evaluate_on_test(
        best_model,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder,
        best_weights
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/10_weighted_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 10_weighted_xgboost")
    print(f"Best configuration: {best_name}")
    print(f"Best class weights: {best_weights}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
