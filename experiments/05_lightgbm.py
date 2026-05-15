import time

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from lightgbm import LGBMClassifier

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    make_preprocessor,
    save_confusion_matrix
)


def encode_labels(y_train, y_test):
    # lightGBM works with numeric class labels: 0, 1, 2, ...
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
    # oversample only weak minority classes.
    # same targets as tuned XGBoost, so the comparison is fair.
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_lightgbm_model():
    return LGBMClassifier(
        # more trees with lower learning rate usually improves generalization.
        n_estimators=500,
        # 0.05 is safer than 0.1 and reduces overfitting risk.
        learning_rate=0.05,
        # LightGBM controls tree complexity mostly through num_leaves.
        # 31 is standard and balanced
        num_leaves=31,
        # -1 means no fixed depth limit; num_leaves still controls complexity.
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="multiclass",
        num_class=5,
        random_state=RANDOM_STATE,
        # use all available CPU cores.
        n_jobs=-1,
        # hide LightGBM training warnings/noise.
        verbose=-1
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

    # StratifiedKFold keeps class proportions in each fold.
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

    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    # plain LightGBM baseline without resampling.
    # checks whether LightGBM alone can beat XGBoost / Random Forest.
    lightgbm_plain = SklearnPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", make_lightgbm_model())
        ]
    )

    # SMOTE + LightGBM focuses on minority classes.
    smote_lightgbm = ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", make_lightgbm_model())
        ]
    )

    y_pred_plain, test_macro_f1_plain, plain_time = evaluate_model(
        "LightGBM",
        lightgbm_plain,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    y_pred_smote, test_macro_f1_smote, smote_time = evaluate_model(
        "SMOTE + LightGBM",
        smote_lightgbm,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    cv_mean, cv_std, cv_time = run_cross_validation(
        smote_lightgbm,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_pred_smote,
        "outputs/confusion_matrices/05_smote_lightgbm.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 05_lightgbm")
    print(f"LightGBM test macro F1: {test_macro_f1_plain:.4f}")
    print(f"LightGBM train + prediction time: {plain_time:.2f} seconds")
    print(f"SMOTE + LightGBM CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"SMOTE + LightGBM test macro F1: {test_macro_f1_smote:.4f}")
    print(f"SMOTE + LightGBM train + prediction time: {smote_time:.2f} seconds")
    print(f"SMOTE + LightGBM CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")

    if test_macro_f1_smote > test_macro_f1_plain:
        print("Best model in this experiment: SMOTE + LightGBM")
    else:
        print("Best model in this experiment: LightGBM")


if __name__ == "__main__":
    main()
