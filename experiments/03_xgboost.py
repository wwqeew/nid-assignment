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


def encode_labels(y_train, y_test):
    # XGBoost works best with numeric class labels: 0, 1, 2, ...
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
    # We oversample only the rarest classes. R2L and U2R are the main problem for macro F1.
    return {
        int(label_encoder.transform(["R2L"])[0]): 8000,
        int(label_encoder.transform(["U2R"])[0]): 2000
    }


def make_xgboost_model():
    return XGBClassifier(
        n_estimators=300, #enough capacity without being too slow.
        max_depth=6, #balanced starting point
        learning_rate=0.1,
        subsample=0.9, #90% of rows per tree to reduce overfitting.
        colsample_bytree=0.9, 
        objective="multi:softprob", #multiclass classification with probability output.
        num_class=5,
        eval_metric="mlogloss", #standard metric for XGBoost
        tree_method="hist",
        random_state=RANDOM_STATE, #our fixed seed
        n_jobs=-1
    )


def evaluate_model(model_name, model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    print(f"\n===== {model_name} =====")

    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, macro_f1


def run_cross_validation(model, X_train, y_train_encoded):
    print("\nRunning 5-fold cross-validation...")
   
    cv = StratifiedKFold( #StratifiedKFold keeps class proportions in each fold, important because R2L and U2R are rare.
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

    print(f"CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean(), scores.std()


def main():
    X_train, y_train, X_test, y_test = load_data()

    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    xgboost_plain = SklearnPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", make_xgboost_model())
        ]
    )

    # SMOTE + XGBoost focuses on minority classes.
    # SMOTE is placed after preprocessing because it needs numeric features.
    smote_xgboost = ImbPipeline(
        steps=[
            ("preprocessor", make_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),

                # k=3 is safer for very small classes like U2R.
                # default k=5 may be less stable when there are few real samples.
                k_neighbors=3,

                random_state=RANDOM_STATE
            )),
            ("classifier", make_xgboost_model())
        ]
    )

    y_pred_plain, test_macro_f1_plain = evaluate_model(
        "XGBoost",
        xgboost_plain,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    y_pred_smote, test_macro_f1_smote = evaluate_model(
        "SMOTE + XGBoost",
        smote_xgboost,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    cv_mean, cv_std = run_cross_validation(
        smote_xgboost,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_pred_smote,
        "outputs/confusion_matrices/03_smote_xgboost.png"
    )

    print("\n===== Experiment Summary =====")
    print("Experiment: 03_xgboost")
    print(f"XGBoost test macro F1: {test_macro_f1_plain:.4f}")
    print(f"SMOTE + XGBoost CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"SMOTE + XGBoost test macro F1: {test_macro_f1_smote:.4f}")

    if test_macro_f1_smote > test_macro_f1_plain:
        print("Best model in this experiment: SMOTE + XGBoost")
    else:
        print("Best model in this experiment: XGBoost")


if __name__ == "__main__":
    main()
