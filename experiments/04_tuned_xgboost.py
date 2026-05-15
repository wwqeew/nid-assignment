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


def make_sampling_strategy(label_encoder, r2l_target, u2r_target):
    # We oversample only the rarest classes. R2L and U2R are the main problem for macro F1.
    return {
        int(label_encoder.transform(["R2L"])[0]): r2l_target,
        int(label_encoder.transform(["U2R"])[0]): u2r_target
    }


def make_model(label_encoder, params):
    # SMOTE is placed after preprocessing because it needs numeric features.
    # XGBoost then learns on the balanced training set.
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(params["X_train"])),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(
                    label_encoder,
                    params["r2l_target"],
                    params["u2r_target"]
                ),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", XGBClassifier(
                # more trees + smaller learning rate usually generalizes better
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],

                # smaller learning rate for more stable boosting
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],

                # min_child_weight makes splits more conservative
                min_child_weight=params["min_child_weight"],
                # L2 regularization to avoid overly complex trees
                reg_lambda=params["reg_lambda"],
                objective="multi:softprob",
                num_class=5,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]
    )


def run_cv(model, X_train, y_train_encoded, folds=3):
    # 3-fold CV is used for faster parameter comparison
    cv = StratifiedKFold(
        n_splits=folds,
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

    return scores.mean(), scores.std()


def evaluate_on_test(model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Final Test Evaluation =====")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def main():
    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    candidate_params = [
        {
            "name": "tuned_1_more_trees_lower_lr",
            "X_train": X_train,
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2,
            "r2l_target": 12000,
            "u2r_target": 3000
        },
        {
            "name": "tuned_2_more_regularized",
            "X_train": X_train,
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_lambda": 3,
            "r2l_target": 12000,
            "u2r_target": 3000
        },
        {
            "name": "tuned_3_stronger_minority_sampling",
            "X_train": X_train,
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2,
            "r2l_target": 16000,
            "u2r_target": 5000
        }
    ]

    best_name = None
    best_model = None
    best_cv_mean = -1
    best_cv_std = 0

    print("\n===== Cross-validation tuning =====")

    for params in candidate_params:
        print(f"\nTesting configuration: {params['name']}")

        model = make_model(label_encoder, params)
        cv_mean, cv_std = run_cv(
            model,
            X_train,
            y_train_encoded,
            folds=3
        )

        print(f"CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")

        if cv_mean > best_cv_mean:
            best_cv_mean = cv_mean
            best_cv_std = cv_std
            best_name = params["name"]
            best_model = model

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_name}")
    print(f"Best CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")

    y_pred, test_macro_f1 = evaluate_on_test(
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
        "outputs/confusion_matrices/04_tuned_xgboost.png"
    )

    print("\n===== Experiment Summary =====")
    print("Experiment: 04_tuned_xgboost")
    print(f"Best configuration: {best_name}")
    print(f"Best CV macro F1: {best_cv_mean:.4f} ± {best_cv_std:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
