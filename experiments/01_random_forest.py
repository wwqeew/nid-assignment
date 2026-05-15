from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from common import (
    RANDOM_STATE,
    load_data,
    make_preprocessor,
    evaluate_model,
    run_cross_validation,
    save_confusion_matrix
)


def main():
    X_train, y_train, X_test, y_test = load_data()
    preprocessor = make_preprocessor(X_train)

    rf_default = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    rf_balanced = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    evaluate_model(
        "Random Forest Default",
        rf_default,
        X_train,
        y_train,
        X_test,
        y_test
    )

    y_pred_balanced, test_macro_f1 = evaluate_model(
        "Random Forest Balanced",
        rf_balanced,
        X_train,
        y_train,
        X_test,
        y_test
    )

    cv_mean, cv_std = run_cross_validation(
        rf_balanced,
        X_train,
        y_train
    )

    save_confusion_matrix(
        y_test,
        y_pred_balanced,
        "outputs/confusion_matrices/01_random_forest_balanced.png"
    )

    print("\n===== Experiment Summary =====")
    print("Experiment: 01_random_forest")
    print("Best model: Random Forest Balanced")
    print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
