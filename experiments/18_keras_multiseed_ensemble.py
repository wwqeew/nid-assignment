import os
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
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
    # best setup from experiment 16
    return {
        "name": "keras_4_custom_weights_multiseed",
        "hidden_1": 256,
        "hidden_2": 128,
        "hidden_3": 64,
        "dropout": 0.35,
        "learning_rate": 0.0007,
        "batch_size": 512,
        "epochs": 120,
        "patience": 10
    }


def make_class_weights():
    # custom weights from the best keras experiment
    return {
        0: 1.0,    # DoS
        1: 0.7,    # Normal
        2: 2.0,    # Probe
        3: 12.0,   # R2L
        4: 80.0    # U2R
    }


def set_all_seeds(seed):
    # reduce randomness between repeated runs
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_model(input_dim, config, seed):
    # clear previous keras graph before creating a new model
    tf.keras.backend.clear_session()
    set_all_seeds(seed)

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


def train_single_model(X_train_processed, y_train_encoded, config, seed):
    # train one keras model with one random seed
    model = make_model(
        input_dim=X_train_processed.shape[1],
        config=config,
        seed=seed
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        restore_best_weights=True
    )

    model.fit(
        X_train_processed,
        y_train_encoded,
        validation_split=0.1,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=make_class_weights(),
        callbacks=[early_stopping],
        verbose=0
    )

    return model


def train_seed_ensemble(X_train_processed, y_train_encoded, config, seeds):
    # train several keras models and keep them as one ensemble
    models = []

    for seed in seeds:
        print(f"\nTraining seed model: {seed}")

        start_time = time.perf_counter()

        model = train_single_model(
            X_train_processed,
            y_train_encoded,
            config,
            seed
        )

        elapsed_time = time.perf_counter() - start_time
        print(f"Seed {seed} train time: {elapsed_time:.2f} seconds")

        models.append(model)

    return models


def predict_average_proba(models, X_processed):
    # average probabilities from all seed models
    probabilities = []

    for model in models:
        proba = model.predict(X_processed, verbose=0)
        probabilities.append(proba)

    return np.mean(probabilities, axis=0)


def evaluate_single_seed_model(model, X_test_processed, y_test, label_encoder, seed):
    # check how each individual seed performs
    probabilities = model.predict(X_test_processed, verbose=0)
    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n===== Seed {seed} Test Evaluation =====")
    print(f"Seed {seed} test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return test_macro_f1


def evaluate_ensemble(models, X_test_processed, y_test, label_encoder):
    # final ensemble prediction by averaged softmax probabilities
    probabilities = predict_average_proba(models, X_test_processed)

    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== Multi-seed Keras Ensemble Test Evaluation =====")
    print(f"Ensemble test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def run_multiseed_cv(config, X_train, y_train_encoded, seeds):
    # 3-fold cv with the same multi-seed averaging idea
    print("\nRunning 3-fold cross-validation for multi-seed ensemble...")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n===== CV fold {fold_id} =====")

        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train_encoded[train_index]
        y_fold_valid = y_train_encoded[valid_index]

        preprocessor = make_scaled_preprocessor(X_fold_train)

        X_fold_train_processed = preprocessor.fit_transform(X_fold_train)
        X_fold_valid_processed = preprocessor.transform(X_fold_valid)

        fold_models = train_seed_ensemble(
            X_fold_train_processed,
            y_fold_train,
            config,
            seeds
        )

        fold_probabilities = predict_average_proba(
            fold_models,
            X_fold_valid_processed
        )

        y_fold_pred = np.argmax(fold_probabilities, axis=1)

        fold_score = f1_score(y_fold_valid, y_fold_pred, average="macro")
        scores.append(fold_score)

        print(f"Fold {fold_id} ensemble macro F1: {fold_score:.4f}")

    elapsed_time = time.perf_counter() - start_time
    scores = np.array(scores)

    print(f"\nCV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


def main():
    total_start_time = time.perf_counter()

    # use 3 seeds first; more seeds may improve stability but will be slower
    seeds = [42, 7, 21]

    config = make_best_keras_config()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    print("\n===== Keras multi-seed ensemble experiment =====")
    print(f"Configuration: {config['name']}")
    print(f"Seeds: {seeds}")
    print(f"Class weights: {make_class_weights()}")

    preprocessor = make_scaled_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    train_start_time = time.perf_counter()

    models = train_seed_ensemble(
        X_train_processed,
        y_train_encoded,
        config,
        seeds
    )

    train_elapsed_time = time.perf_counter() - train_start_time

    print(f"\nTotal final ensemble training time: {train_elapsed_time:.2f} seconds")

    seed_scores = []

    for model, seed in zip(models, seeds):
        seed_score = evaluate_single_seed_model(
            model,
            X_test_processed,
            y_test,
            label_encoder,
            seed
        )

        seed_scores.append(seed_score)

    y_pred_ensemble, ensemble_test_macro_f1 = evaluate_ensemble(
        models,
        X_test_processed,
        y_test,
        label_encoder
    )

    cv_mean, cv_std, cv_time = run_multiseed_cv(
        config,
        X_train,
        y_train_encoded,
        seeds
    )

    save_confusion_matrix(
        y_test,
        y_pred_ensemble,
        "outputs/confusion_matrices/18_keras_multiseed_ensemble.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 18_keras_multiseed_ensemble")
    print(f"Base configuration: {config['name']}")
    print(f"Seeds: {seeds}")
    print(f"Class weights: {make_class_weights()}")
    print(f"Individual seed test macro F1 scores: {[round(score, 4) for score in seed_scores]}")
    print(f"Individual seed mean test macro F1: {np.mean(seed_scores):.4f}")
    print(f"Individual seed std test macro F1: {np.std(seed_scores):.4f}")
    print(f"Ensemble test macro F1: {ensemble_test_macro_f1:.4f}")
    print(f"3-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Final ensemble training time: {train_elapsed_time:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
