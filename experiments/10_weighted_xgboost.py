import time

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
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


def make_sampling_strategy(label_encoder, r2l_target, u2r_target):
    # same idea as in the best tuned xgboost experiment
    # oversample only the weakest minority classes
    return {
        int(label_encoder.transform(["R2L"])[0]): r2l_target,
        int(label_encoder.transform(["U2R"])[0]): u2r_target
    }


def make_tuned_xgboost_model(params):
    return XGBClassifier(
        # best xgboost parameters from 04_tuned_xgboost.py
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],

        # row and feature sampling to reduce overfitting
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        # makes splits more conservative
        min_child_weight=params["min_child_weight"],

        # l2 regularization
        reg_lambda=params["reg_lambda"],

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


def make_smoteenn(label_encoder, params):
    # smote creates synthetic r2l and u2r samples first
    smote = SMOTE(
        sampling_strategy=make_sampling_strategy(
            label_encoder,
            params["r2l_target"],
            params["u2r_target"]
        ),
        k_neighbors=3,
        random_state=RANDOM_STATE
    )

    # enn removes noisy samples after smote
    # kind_sel mode is less aggressive than all
    enn = EditedNearestNeighbours(
        n_neighbors=params["enn_neighbors"],
        kind_sel=params["enn_kind_sel"],
        sampling_strategy="all"
    )

    return SMOTEENN(
        smote=smote,
        enn=enn,
        random_state=RANDOM_STATE
    )


def make_model(label_encoder, params):
    # smoteenn is placed after preprocessing because it needs numeric features
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(params["X_train"])),
            ("smoteenn", make_smoteenn(label_encoder, params)),
            ("classifier", make_tuned_xgboost_model(params))
        ]
    )


def run_cv(model, X_train, y_train_encoded, folds=3):
    # 3-fold cv is used to compare configurations faster
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    start_time = time.perf_counter()

    scores = cross_val_score(
        model,
        X_train,
        y_train_encoded,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    elapsed_time = time.perf_counter() - start_time

    return scores.mean(), scores.std(), elapsed_time


def evaluate_on_test(model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

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

    candidate_params = [
        {
            "name": "smoteenn_1_best_xgb_conservative_cleaning",
            "X_train": X_train,

            # best xgboost settings from experiment 04
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2,

            # same smote targets as the current best model
            "r2l_target": 12000,
            "u2r_target": 3000,

            # conservative enn cleaning
            "enn_neighbors": 3,
            "enn_kind_sel": "mode"
        },
        {
            "name": "smoteenn_2_stronger_minority_sampling",
            "X_train": X_train,

            # same xgboost settings with stronger minority sampling
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2,

            "r2l_target": 16000,
            "u2r_target": 5000,

            "enn_neighbors": 3,
            "enn_kind_sel": "mode"
        },
        {
            "name": "smoteenn_3_more_regularized_xgb",
            "X_train": X_train,

            # more conservative xgboost settings from experiment 04
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_lambda": 3,

            "r2l_target": 12000,
            "u2r_target": 3000,

            "enn_neighbors": 3,
            "enn_kind_sel": "mode"
        },
        {
            "name": "smoteenn_4_aggressive_cleaning",
            "X_train": X_train,

            # best xgboost settings with stronger enn cleaning
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2,

            "r2l_target": 12000,
            "u2r_target": 3000,

            # kind_sel all removes more borderline samples
            "enn_neighbors": 3,
            "enn_kind_sel": "all"
        }
    ]

    best_name = None
    best_model = None
    best_cv_mean = -1
    best_cv_std = 0
    best_cv_time = 0

    print("\n===== SMOTEENN + XGBoost cross-validation tuning =====")

    for params in candidate_params:
        print(f"\nTesting configuration: {params['name']}")
        print(
            f"SMOTE targets: R2L={params['r2l_target']}, "
            f"U2R={params['u2r_target']}; "
            f"ENN kind_sel={params['enn_kind_sel']}"
        )

        model = make_model(label_encoder, params)

        cv_mean, cv_std, cv_time = run_cv(
            model,
            X_train,
            y_train_encoded,
            folds=3
        )

        print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"CV time: {cv_time:.2f} seconds")

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_cv_time = cv_time
            best_name = params["name"]
            best_model = model

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Best configuration CV time: {best_cv_time:.2f} seconds")

    y_pred, test_macro_f1, final_eval_time = evaluate_on_test(
        best_model,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/09_smoteenn_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 09_smoteenn_xgboost")
    print(f"Best configuration: {best_name}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
