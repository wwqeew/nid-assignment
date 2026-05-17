import time
import numpy as np

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
)


def add_engineered_features(X):
    # Create a copy to avoid changing the original dataset.
    X = X.copy()

    # Basic traffic volume features.
    X["total_bytes"] = X["src_bytes"] + X["dst_bytes"]
    X["byte_ratio"] = X["src_bytes"] / (X["dst_bytes"] + 1)
    X["byte_diff"] = X["src_bytes"] - X["dst_bytes"]

    # Error-rate summary features.
    X["error_rate_sum"] = X["serror_rate"] + X["rerror_rate"]
    X["srv_error_rate_sum"] = X["srv_serror_rate"] + X["srv_rerror_rate"]

    X["dst_host_error_rate_sum"] = (
        X["dst_host_serror_rate"] + X["dst_host_rerror_rate"]
    )

    X["dst_host_srv_error_rate_sum"] = (
        X["dst_host_srv_serror_rate"] + X["dst_host_srv_rerror_rate"]
    )

    # Login-related suspicious activity.
    X["login_risk"] = (
        X["hot"] +
        X["num_failed_logins"] +
        X["is_guest_login"]
    )

    # Root / privilege-related activity.
    X["root_activity"] = (
        X["root_shell"] +
        X["su_attempted"] +
        X["num_root"] +
        X["num_file_creations"] +
        X["num_shells"]
    )

    # Destination host/service relationship.
    X["host_srv_ratio"] = X["dst_host_srv_count"] / (X["dst_host_count"] + 1)

    # Clean possible numerical issues.
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)

    return X


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
    # Same SMOTE targets as the best tuned SMOTE + XGBoost experiment.
    # R2L and U2R are oversampled, but not up to the Normal class size.
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_tuned_xgboost_model():
    return XGBClassifier(
        # Best parameters from 04_tuned_xgboost.py.
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


def evaluate_model(model_name, model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    print(f"\n===== {model_name} =====")

    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed_time = time.perf_counter() - start_time
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test macro F1-score: {macro_f1:.4f}")
    print(f"Training + test prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, macro_f1, elapsed_time


def run_cross_validation(model, X_train, y_train_encoded):
    print("\nRunning 5-fold cross-validation...")

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

    print(f"CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"Cross-validation time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()

    print("\nAdding engineered features...")
    X_train_fe = add_engineered_features(X_train)
    X_test_fe = add_engineered_features(X_test)

    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Feature count after engineering: {X_train_fe.shape[1]}")

    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    # Tuned XGBoost with engineered features, without SMOTE.
    # This checks whether feature engineering alone helps.
    tuned_xgboost_fe = SklearnPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train_fe)),
            ("classifier", make_tuned_xgboost_model())
        ]
    )

    # Tuned SMOTE + XGBoost with engineered features.
    # This is the direct comparison with the best current model.
    tuned_smote_xgboost_fe = ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train_fe)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),

                # k=3 is safer for U2R because this class has very few real samples.
                k_neighbors=3,

                random_state=RANDOM_STATE
            )),
            ("classifier", make_tuned_xgboost_model())
        ]
    )

    y_pred_plain, test_macro_f1_plain, plain_time = evaluate_model(
        "Tuned XGBoost + Feature Engineering",
        tuned_xgboost_fe,
        X_train_fe,
        y_train_encoded,
        X_test_fe,
        y_test,
        label_encoder
    )

    y_pred_smote, test_macro_f1_smote, smote_time = evaluate_model(
        "Tuned SMOTE + XGBoost + Feature Engineering",
        tuned_smote_xgboost_fe,
        X_train_fe,
        y_train_encoded,
        X_test_fe,
        y_test,
        label_encoder
    )

    cv_mean, cv_std, cv_time = run_cross_validation(
        tuned_smote_xgboost_fe,
        X_train_fe,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_pred_smote,
        "outputs/confusion_matrices/08_tuned_feature_engineering_xgboost.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 08_feature_engineering_xgboost")
    print("Parameters: same as 04_tuned_xgboost.py")
    print("SMOTE targets: R2L=12000, U2R=3000")
    print(f"Tuned XGBoost + FE test macro F1: {test_macro_f1_plain:.4f}")
    print(f"Tuned XGBoost + FE train + prediction time: {plain_time:.2f} seconds")
    print(f"Tuned SMOTE + XGBoost + FE CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Tuned SMOTE + XGBoost + FE test macro F1: {test_macro_f1_smote:.4f}")
    print(f"Tuned SMOTE + XGBoost + FE train + prediction time: {smote_time:.2f} seconds")
    print(f"Tuned SMOTE + XGBoost + FE CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")

    if test_macro_f1_smote > test_macro_f1_plain:
        print("Best model in this experiment: Tuned SMOTE + XGBoost + Feature Engineering")
    else:
        print("Best model in this experiment: Tuned XGBoost + Feature Engineering")


if __name__ == "__main__":
    main()
