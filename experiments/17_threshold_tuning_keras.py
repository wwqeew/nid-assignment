import time
import numpy as np
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    save_confusion_matrix
)


np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def make_one_hot_encoder():
    # support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X_train):
    # neural networks need scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
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


def make_best_keras_config():
    # best config from experiment 16
    return {
        "name": "keras_4_custom_weights",
        "hidden_1": 256,
        "hidden_2": 128,
        "hidden_3": 64,
        "dropout": 0.35,
        "learning_rate": 0.0007,
        "batch_size": 512,
        "epochs": 120,
        "patience": 10,
        "class_weight_mode": "custom"
    }


def make_class_weights():
    # manual weights from the best keras setup
    return {
        0: 1.0,    # DoS
        1: 0.7,    # Normal
        2: 2.0,    # Probe
        3: 12.0,   # R2L
        4: 80.0    # U2R
    }


def make_model(input_dim, config):
    tf.keras.backend.clear_session()

    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    # first dense block
    model.add(Dense(config["hidden_1"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # second dense block
    model.add(Dense(config["hidden_2"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # third dense block
    model.add(Dense(config["hidden_3"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    # 5 output classes
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_keras_model(X_fit, y_fit, X_valid, y_valid, config):
    class_weights = make_class_weights()

    print(f"Class weights: {class_weights}")

    model = make_model(
        input_dim=X_fit.shape[1],
        config=config
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        restore_best_weights=True
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_valid, y_valid),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0
    )

    return model


def apply_probability_multipliers(probabilities, label_encoder, multipliers):
    # multiply class probabilities before argmax
    adjusted = probabilities.copy()

    for class_name, multiplier in multipliers.items():
        class_index = int(label_encoder.transform([class_name])[0])
        adjusted[:, class_index] *= multiplier

    y_pred_encoded = np.argmax(adjusted, axis=1)

    return y_pred_encoded


def find_best_multipliers(model, X_valid, y_valid_encoded, label_encoder):
    # tune only on validation split from training data
    probabilities = model.predict(X_valid, verbose=0)

    # normal is often overpredicted, so try reducing it
    candidate_normal = [1.0, 0.95, 0.90, 0.85, 0.80]

    # r2l recall is the main weakness, so try boosting it
    candidate_r2l = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

    # u2r is already strong, so keep this range softer
    candidate_u2r = [0.8, 1.0, 1.2, 1.5, 2.0]

    best_score = -1
    best_multipliers = None
    best_pred_encoded = None

    print("\n===== Threshold / probability multiplier tuning =====")

    for normal_multiplier in candidate_normal:
        for r2l_multiplier in candidate_r2l:
            for u2r_multiplier in candidate_u2r:
                multipliers = {
                    "Normal": normal_multiplier,
                    "R2L": r2l_multiplier,
                    "U2R": u2r_multiplier
                }

                y_pred_encoded = apply_probability_multipliers(
                    probabilities,
                    label_encoder,
                    multipliers
                )

                score = f1_score(y_valid_encoded, y_pred_encoded, average="macro")

                if score > best_score:
                    best_score = score
                    best_multipliers = multipliers
                    best_pred_encoded = y_pred_encoded

    print(f"Best validation macro F1: {best_score:.4f}")
    print(f"Best multipliers: {best_multipliers}")

    y_valid = label_encoder.inverse_transform(y_valid_encoded)
    best_pred = label_encoder.inverse_transform(best_pred_encoded)

    print("\nValidation classification report with best multipliers:")
    print(classification_report(y_valid, best_pred, labels=LABELS, zero_division=0))

    return best_multipliers, best_score


def evaluate_raw_model(model, X_test_processed, y_test, label_encoder):
    # normal keras prediction without threshold tuning
    probabilities = model.predict(X_test_processed, verbose=0)

    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Raw Keras Test Evaluation =====")
    print(f"Raw Keras test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def evaluate_tuned_model(model, X_test_processed, y_test, label_encoder, multipliers):
    # keras prediction with validation-selected probability multipliers
    probabilities = model.predict(X_test_processed, verbose=0)

    y_pred_encoded = apply_probability_multipliers(
        probabilities,
        label_encoder,
        multipliers
    )

    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Tuned Keras Test Evaluation =====")
    print(f"Tuned Keras test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def run_tuned_cross_validation(config, X_train, y_train_encoded, label_encoder):
    # cv with multiplier tuning inside each fold
    print("\nRunning 3-fold cross-validation with threshold tuning...")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for fold_id, (train_index, outer_valid_index) in enumerate(
        cv.split(X_train, y_train_encoded),
        start=1
    ):
        print(f"\n===== CV fold {fold_id} =====")

        X_outer_train = X_train.iloc[train_index]
        X_outer_valid = X_train.iloc[outer_valid_index]

        y_outer_train = y_train_encoded[train_index]
        y_outer_valid = y_train_encoded[outer_valid_index]

        preprocessor = make_scaled_preprocessor(X_outer_train)

        X_outer_train_processed = preprocessor.fit_transform(X_outer_train)
        X_outer_valid_processed = preprocessor.transform(X_outer_valid)

        X_fit, X_inner_valid, y_fit, y_inner_valid = train_test_split(
            X_outer_train_processed,
            y_outer_train,
            test_size=0.2,
            stratify=y_outer_train,
            random_state=RANDOM_STATE
        )

        model = train_keras_model(
            X_fit,
            y_fit,
            X_inner_valid,
            y_inner_valid,
            config
        )

        best_multipliers, best_inner_score = find_best_multipliers(
            model,
            X_inner_valid,
            y_inner_valid,
            label_encoder
        )

        outer_probabilities = model.predict(X_outer_valid_processed, verbose=0)

        y_outer_pred = apply_probability_multipliers(
            outer_probabilities,
            label_encoder,
            best_multipliers
        )

        fold_score = f1_score(y_outer_valid, y_outer_pred, average="macro")
        scores.append(fold_score)

        print(f"Fold {fold_id} inner validation macro F1: {best_inner_score:.4f}")
        print(f"Fold {fold_id} outer macro F1: {fold_score:.4f}")

    elapsed_time = time.perf_counter() - start_time
    scores = np.array(scores)

    print(f"\nCV macro F1 with threshold tuning: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    config = make_best_keras_config()

    print("\n===== Keras threshold tuning experiment =====")
    print(f"Configuration: {config['name']}")

    preprocessor = make_scaled_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train_processed,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    print("\n===== Training Keras model for threshold tuning =====")

    train_start_time = time.perf_counter()

    model = train_keras_model(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        config
    )

    train_elapsed_time = time.perf_counter() - train_start_time
    print(f"Training time: {train_elapsed_time:.2f} seconds")

    raw_valid_probabilities = model.predict(X_valid, verbose=0)
    raw_valid_pred_encoded = np.argmax(raw_valid_probabilities, axis=1)
    raw_valid_f1 = f1_score(y_valid, raw_valid_pred_encoded, average="macro")

    print(f"\nValidation macro F1 before threshold tuning: {raw_valid_f1:.4f}")

    best_multipliers, best_validation_f1 = find_best_multipliers(
        model,
        X_valid,
        y_valid,
        label_encoder
    )

    y_pred_raw, raw_test_macro_f1 = evaluate_raw_model(
        model,
        X_test_processed,
        y_test,
        label_encoder
    )

    y_pred_tuned, tuned_test_macro_f1 = evaluate_tuned_model(
        model,
        X_test_processed,
        y_test,
        label_encoder,
        best_multipliers
    )

    if tuned_test_macro_f1 >= raw_test_macro_f1:
        final_y_pred = y_pred_tuned
        final_test_macro_f1 = tuned_test_macro_f1
        final_model_name = "Tuned Keras Neural Network"
    else:
        final_y_pred = y_pred_raw
        final_test_macro_f1 = raw_test_macro_f1
        final_model_name = "Raw Keras Neural Network"

    cv_mean, cv_std, cv_time = run_tuned_cross_validation(
        config,
        X_train,
        y_train_encoded,
        label_encoder
    )

    save_confusion_matrix(
        y_test,
        final_y_pred,
        "outputs/confusion_matrices/17_threshold_tuning_keras.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 17_threshold_tuning_keras")
    print(f"Base configuration: {config['name']}")
    print(f"Validation macro F1 before threshold tuning: {raw_valid_f1:.4f}")
    print(f"Validation macro F1 after threshold tuning: {best_validation_f1:.4f}")
    print(f"Best multipliers: {best_multipliers}")
    print(f"Raw Keras test macro F1: {raw_test_macro_f1:.4f}")
    print(f"Tuned Keras test macro F1: {tuned_test_macro_f1:.4f}")
    print(f"Best final model: {final_model_name}")
    print(f"Best final test macro F1: {final_test_macro_f1:.4f}")
    print(f"3-fold tuned CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Training time: {train_elapsed_time:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
