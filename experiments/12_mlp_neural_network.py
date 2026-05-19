import time
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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
    # mlp needs scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def encode_labels(y_train, y_test):
    # mlp works with numeric class labels
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
    # same minority targets as the best smote + xgboost experiments
    return {
        int(label_encoder.transform(["R2L"])[0]): 12000,
        int(label_encoder.transform(["U2R"])[0]): 3000
    }


def make_mlp_model():
    return MLPClassifier(
        # two hidden layers make this a real feed-forward neural network
        hidden_layer_sizes=(128, 64),

        # relu is the standard activation for hidden layers
        activation="relu",

        # adam works well as a default optimizer for mlp
        solver="adam",

        # small regularization to reduce overfitting
        alpha=0.0005,

        # not too small, so training stays stable and reasonably fast
        batch_size=512,

        # standard starting learning rate
        learning_rate_init=0.001,

        # stop when validation score stops improving
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,

        # enough iterations, early stopping usually stops earlier
        max_iter=120,

        random_state=RANDOM_STATE
    )


def evaluate_model(model_name, model, X_train, y_train_encoded, X_test, y_test, label_encoder):
    print(f"\n===== {model_name} =====")

    start_time = time.perf_counter()

    model.fit(X_train, y_train_encoded)

    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    elapsed_time = time.perf_counter() - start_time

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Training + test prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1, elapsed_time


def run_cross_validation(model, X_train, y_train_encoded):
    print("\nRunning 3-fold cross-validation...")

    start_time = time.perf_counter()

    # 3-fold cv is used because mlp is slower than tree models
    cv = StratifiedKFold(
        n_splits=3,
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

    # plain mlp without smote
    # this checks whether neural network alone can handle imbalance
    mlp_plain = SklearnPipeline(
        steps=[
            ("preprocessor", make_scaled_preprocessor(X_train)),
            ("classifier", make_mlp_model())
        ]
    )

    # smote + mlp
    # smote is placed after preprocessing because it needs numeric features
    smote_mlp = ImbPipeline(
        steps=[
            ("preprocessor", make_scaled_preprocessor(X_train)),
            ("smote", SMOTE(
                sampling_strategy=make_sampling_strategy(label_encoder),
                k_neighbors=3,
                random_state=RANDOM_STATE
            )),
            ("classifier", make_mlp_model())
        ]
    )

    y_pred_plain, test_macro_f1_plain, plain_time = evaluate_model(
        "MLP Neural Network",
        mlp_plain,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    y_pred_smote, test_macro_f1_smote, smote_time = evaluate_model(
        "SMOTE + MLP Neural Network",
        smote_mlp,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    cv_mean, cv_std, cv_time = run_cross_validation(
        smote_mlp,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_pred_smote,
        "outputs/confusion_matrices/12_smote_mlp_neural_network.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 12_mlp_neural_network")
    print(f"MLP test macro F1: {test_macro_f1_plain:.4f}")
    print(f"MLP train + prediction time: {plain_time:.2f} seconds")
    print(f"SMOTE + MLP CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"SMOTE + MLP test macro F1: {test_macro_f1_smote:.4f}")
    print(f"SMOTE + MLP train + prediction time: {smote_time:.2f} seconds")
    print(f"SMOTE + MLP CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")

    if test_macro_f1_smote > test_macro_f1_plain:
        print("Best model in this experiment: SMOTE + MLP Neural Network")
    else:
        print("Best model in this experiment: MLP Neural Network")


if __name__ == "__main__":
    main()
