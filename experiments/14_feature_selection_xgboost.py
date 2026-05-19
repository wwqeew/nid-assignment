import time

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    # same smote targets as the best tuned xgboost experiment
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_selector_model():
    return XGBClassifier(
        # smaller model only for feature importance estimation
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,

        # keep it stable and not too aggressive
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


def make_model(X_train, label_encoder, config):
    # preprocessing converts categorical columns into numeric one-hot columns
    preprocessor = make_preprocessor(X_train)

    # selector removes weak features based on xgboost feature importance
    selector = SelectFromModel(
        estimator=make_selector_model(),
        threshold=config["threshold"],
        prefit=False
    )

    # smote is placed after feature selection because it needs numeric features
    smote = SMOTE(
        sampling_strategy=make_sampling_strategy(label_encoder),
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


def run_cv(model, X_train, y_train_encoded, folds=3):
    # 3-fold cv is used for faster comparison between feature selection thresholds
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


def get_selected_feature_count(model, X_train):
    # count features before and after selection
    preprocessor = model.named_steps["preprocessor"]
    selector = model.named_steps["selector"]

    transformed_sample = preprocessor.transform(X_train.iloc[:1])
    total_features = transformed_sample.shape[1]
    selected_features = selector.get_support().sum()

    return total_features, selected_features


def evaluate_on_test(model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed_time = time.perf_counter()

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    total_features, selected_features = get_selected_feature_count(model, X_train)

    print("\n===== Final Test Evaluation =====")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Original encoded feature count: {total_features}")
    print(f"Selected feature count: {selected_features}")
    print(f"Final train + prediction time: {elapsed_time - start_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time - start_time, total_features, selected_features


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    candidate_configs = [
        {
            "name": "fs_1_keep_more_features",
            # keeps features with importance above half of mean importance
            "threshold": "0.5*mean"
        },
        {
            "name": "fs_2_medium_selection",
            # balanced option, removes lower half by importance
            "threshold": "median"
        },
        {
            "name": "fs_3_aggressive_selection",
            # keeps only features above mean importance
            "threshold": "mean"
        },
        {
            "name": "fs_4_very_aggressive_selection",
            # keeps only clearly strong features
            "threshold": "1.25*mean"
        }
    ]

    best_name = None
    best_config = None
    best_model = None
    best_cv_mean = -1
    best_cv_std = 0
    best_cv_time = 0

    print("\n===== Feature Selection + XGBoost cross-validation tuning =====")

    for config in candidate_configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"Selection threshold: {config['threshold']}")

        model = make_model(
            X_train,
            label_encoder,
            config
        )

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
            best_name = config["name"]
            best_config = config
            best_model = model

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best config details: {best_config}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Best configuration CV time: {best_cv_time:.2f} seconds")

    y_pred, test_macro_f1, final_eval_time, total_features, selected_features = evaluate_on_test(
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
        "outputs/confusion_matrices/14_feature_selection_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 14_feature_selection_xgboost")
    print(f"Best configuration: {best_name}")
    print(f"Best config details: {best_config}")
    print(f"Best 3-fold CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Original encoded feature count: {total_features}")
    print(f"Selected feature count: {selected_features}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
