from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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

    smote_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(
                sampling_strategy={
                    "R2L": 8000,
                    "U2R": 2000
                },
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )

    y_pred, test_macro_f1 = evaluate_model(
        "SMOTE + Random Forest",
        smote_rf,
        X_train,
        y_train,
        X_test,
        y_test
    )

    cv_mean, cv_std = run_cross_validation(
        smote_rf,
        X_train,
        y_train
    )

    save_confusion_matrix(
        y_test,
        y_pred,
        "outputs/confusion_matrices/02_smote_random_forest.png"
    )

    print("\n===== Experiment Summary =====")
    print("Experiment: 02_smote_random_forest")
    print("Model: SMOTE + Random Forest")
    print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
