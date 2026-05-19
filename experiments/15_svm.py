import time
import warnings
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    save_confusion_matrix
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_one_hot_encoder():
    # support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X_train):
    # svm needs scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def make_sampling_strategy():
    # same minority targets as the best smote based experiments
    return {
        "R2L": 12000,
        "U2R": 3000
    }


def make_svm_model(class_weight=None, C=1.0):
    return LinearSVC(
        # linear svm is much faster than rbf svm on this dataset
        C=C,

        # class_weight is used for cost-sensitive learning
        class_weight=class_weight,

        # dual false is usually better when samples are more than features
        dual=False,

        # more iterations because svm may need time to converge
        max_iter=20000,

        # fixed seed for reproducibility
        random_state=RANDOM_STATE
    )


def make_pipeline(X_train, config):
    preprocessor = make_scaled_preprocessor(X_train)

    if config["use_smote"]:
        # smote is placed after preprocessing because it needs numeric features
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(
                    sampling_strategy=make_sampling_strategy(),
                    k_neighbors=3,
                    random_state=RANDOM_STATE
                )),
                ("classifier", make_svm_model(
                    class_weight=config["class_weight"],
                    C=config["C"]
                ))
            ]
        )

    # plain sklearn pipeline without oversampling
    return SklearnPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", make_svm_model(
                class_weight=config["class_weight"],
                C=config["C"]
            ))
        ]
    )


def run_cv(model, X_train, y_train, folds=3):
    # 3-fold cv is used because svm can be slower than tree models
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    start_time = time.perf_counter()

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    elapsed_time = time.perf_counter() - start_time

    return scores.mean(), scores.std(), elapsed_time


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    start_time = time.perf_counter()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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

    candidate_configs = [
        {
            "name": "svm_1_plain",
            "C": 1.0,
            "class_weight": None,
            "use_smote": False
        },
        {
            "name": "svm_2_balanced",
            "C": 1.0,
            "class_weight": "balanced",
            "use_smote": False
        },
        {
            "name": "svm_3_custom_weights",
            "C": 1.0,
            "class_weight": {
                "DoS": 1,
                "Normal": 0.7,
                "Probe": 2,
                "R2L": 12,
                "U2R": 80
            },
            "use_smote": False
        },
        {
            "name": "svm_4_custom_weights_lower_c",
            "C": 0.5,
            "class_weight": {
                "DoS": 1,
                "Normal": 0.7,
                "Probe": 2,
                "R2L": 12,
                "U2R": 80
            },
            "use_smote": False
        },
        {
            "name": "svm_5_smote_plain",
            "C": 1.0,
            "class_weight": None,
            "use_smote": True
        },
        {
            "name": "svm_6_smote_custom_weights",
            "C": 0.5,
            "class_weight": {
                "DoS": 1,
                "Normal": 0.8,
                "Probe": 1.5,
                "R2L": 5,
                "U2R": 20
            },
            "use_smote": True
        }
    ]

    best_name = None
    best_config = None
    best_model = None
    best_cv_mean = -1
    best_cv_std = 0
    best_cv_time = 0

    print("\n===== SVM cross-validation tuning =====")

    for config in candidate_configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"C: {config['C']}")
        print(f"class_weight: {config['class_weight']}")
        print(f"use_smote: {config['use_smote']}")

        model = make_pipeline(X_train, config)

        cv_mean, cv_std, cv_time = run_cv(
            model,
            X_train,
            y_train,
            folds=3
        )

        print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"CV time: {cv_time:.2f} seconds")

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_cv_time = cv_time
            best_name = config["name"]
            best_config = config
            best_model = model

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best config details: {best_config}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Best configuration CV time: {best_cv_time:.2f} seconds")

    y_pred, test_macro_f1, final_eval_time = evaluate_on_test(
        best_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/15_svm.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 15_svm")
    print(f"Best configuration: {best_name}")
    print(f"Best config details: {best_config}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
