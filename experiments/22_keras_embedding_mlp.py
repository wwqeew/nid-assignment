import os
import time
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Embedding,
    Flatten,
    Concatenate
)
from tensorflow.keras.optimizers import Adam

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    save_confusion_matrix
)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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


def set_all_seeds(seed):
    # keep keras runs more stable
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_candidate_configs():
    # embedding mlp configs around the best experiment 19 setup
    return [
        {
            "name": "emb_1_service8_best_weights",
            "seed": 100,
            "protocol_emb_dim": 2,
            "service_emb_dim": 8,
            "flag_emb_dim": 4,
            "numeric_dense": 64,
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.30,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "max_epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.7,
                2: 2.0,
                3: 20.0,
                4: 60.0
            }
        },
        {
            "name": "emb_2_service12_best_weights",
            "seed": 100,
            "protocol_emb_dim": 2,
            "service_emb_dim": 12,
            "flag_emb_dim": 4,
            "numeric_dense": 64,
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.30,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "max_epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.7,
                2: 2.0,
                3: 20.0,
                4: 60.0
            }
        },
        {
            "name": "emb_3_service16_best_weights",
            "seed": 100,
            "protocol_emb_dim": 3,
            "service_emb_dim": 16,
            "flag_emb_dim": 4,
            "numeric_dense": 64,
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.30,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "max_epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.7,
                2: 2.0,
                3: 20.0,
                4: 60.0
            }
        },
        {
            "name": "emb_4_service12_more_dropout",
            "seed": 100,
            "protocol_emb_dim": 2,
            "service_emb_dim": 12,
            "flag_emb_dim": 4,
            "numeric_dense": 64,
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.35,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "max_epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.7,
                2: 2.0,
                3: 20.0,
                4: 60.0
            }
        },
        {
            "name": "emb_5_service12_more_r2l",
            "seed": 100,
            "protocol_emb_dim": 2,
            "service_emb_dim": 12,
            "flag_emb_dim": 4,
            "numeric_dense": 64,
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.30,
            "learning_rate": 0.0005,
            "batch_size": 512,
            "max_epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.7,
                2: 2.0,
                3: 22.0,
                4: 60.0
            }
        }
    ]


class TabularEmbeddingPreprocessor:
    def __init__(self):
        self.categorical_features = ["protocol_type", "service", "flag"]
        self.scaler = StandardScaler()
        self.category_maps = {}
        self.numeric_features = None

    def fit(self, X):
        # numeric columns are everything except categorical columns
        self.numeric_features = [
            col for col in X.columns
            if col not in self.categorical_features
        ]

        self.scaler.fit(X[self.numeric_features])

        # reserve 0 for unknown categories
        for col in self.categorical_features:
            values = sorted(X[col].astype(str).unique())
            self.category_maps[col] = {
                value: index + 1
                for index, value in enumerate(values)
            }

        return self

    def transform(self, X):
        data = {}

        for col in self.categorical_features:
            mapping = self.category_maps[col]

            encoded = X[col].astype(str).map(mapping).fillna(0).astype("int32").values
            data[col] = encoded

        data["numeric"] = self.scaler.transform(X[self.numeric_features]).astype("float32")

        return data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def get_cardinality(self, col):
        # +1 because index 0 is reserved for unknown
        return len(self.category_maps[col]) + 1


def make_model(preprocessor, config):
    tf.keras.backend.clear_session()
    set_all_seeds(config["seed"])

    protocol_input = Input(shape=(1,), name="protocol_type")
    service_input = Input(shape=(1,), name="service")
    flag_input = Input(shape=(1,), name="flag")
    numeric_input = Input(
        shape=(len(preprocessor.numeric_features),),
        name="numeric"
    )

    # categorical embeddings
    protocol_emb = Embedding(
        input_dim=preprocessor.get_cardinality("protocol_type"),
        output_dim=config["protocol_emb_dim"],
        name="protocol_embedding"
    )(protocol_input)
    protocol_emb = Flatten()(protocol_emb)

    service_emb = Embedding(
        input_dim=preprocessor.get_cardinality("service"),
        output_dim=config["service_emb_dim"],
        name="service_embedding"
    )(service_input)
    service_emb = Flatten()(service_emb)

    flag_emb = Embedding(
        input_dim=preprocessor.get_cardinality("flag"),
        output_dim=config["flag_emb_dim"],
        name="flag_embedding"
    )(flag_input)
    flag_emb = Flatten()(flag_emb)

    # small numeric branch
    numeric_branch = Dense(config["numeric_dense"], activation="relu")(numeric_input)
    numeric_branch = BatchNormalization()(numeric_branch)
    numeric_branch = Dropout(config["dropout"])(numeric_branch)

    # combine categorical embeddings with numeric features
    x = Concatenate()([
        protocol_emb,
        service_emb,
        flag_emb,
        numeric_branch
    ])

    # first dense block
    x = Dense(config["hidden_1"], activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(config["dropout"])(x)

    # second dense block
    x = Dense(config["hidden_2"], activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(config["dropout"])(x)

    # third dense block
    x = Dense(config["hidden_3"], activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(config["dropout"])(x)

    output = Dense(5, activation="softmax", name="output")(x)

    model = Model(
        inputs={
            "protocol_type": protocol_input,
            "service": service_input,
            "flag": flag_input,
            "numeric": numeric_input
        },
        outputs=output
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, X_valid, y_valid, patience):
        super().__init__()

        # monitor macro f1 directly, because this is our main metric
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.patience = patience

        self.best_score = -1
        self.best_weights = None
        self.best_epoch = 0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.X_valid, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)

        score = f1_score(
            self.y_valid,
            y_pred,
            average="macro",
            zero_division=0
        )

        if score > self.best_score:
            self.best_score = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        # restore best validation macro f1 weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_model(X_fit, y_fit, X_valid, y_valid, preprocessor, config):
    model = make_model(preprocessor, config)

    macro_f1_stopping = MacroF1EarlyStopping(
        X_valid=X_valid,
        y_valid=y_valid,
        patience=config["patience"]
    )

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_valid, y_valid),
        epochs=config["max_epochs"],
        batch_size=config["batch_size"],
        class_weight=config["class_weights"],
        callbacks=[macro_f1_stopping],
        verbose=0
    )

    return model, macro_f1_stopping.best_score, macro_f1_stopping.best_epoch


def evaluate_encoded(model, X_processed, y_encoded):
    probabilities = model.predict(X_processed, verbose=0)
    y_pred_encoded = np.argmax(probabilities, axis=1)

    macro_f1 = f1_score(
        y_encoded,
        y_pred_encoded,
        average="macro",
        zero_division=0
    )

    return y_pred_encoded, macro_f1


def tune_configs_on_validation(X_train, y_train_encoded):
    # keep kddtest+ untouched during tuning
    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    preprocessor = TabularEmbeddingPreprocessor()
    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_valid = preprocessor.transform(X_valid_raw)

    candidate_configs = make_candidate_configs()

    best_result = None
    all_results = []

    print("\n===== Keras categorical embeddings MLP tuning =====")

    for config in candidate_configs:
        print(f"\nTesting configuration: {config['name']}")
        print(f"Seed: {config['seed']}")
        print(f"Protocol emb dim: {config['protocol_emb_dim']}")
        print(f"Service emb dim: {config['service_emb_dim']}")
        print(f"Flag emb dim: {config['flag_emb_dim']}")
        print(f"Dropout: {config['dropout']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Class weights: {config['class_weights']}")

        start_time = time.perf_counter()

        model, callback_best_f1, best_epoch = train_model(
            X_fit,
            y_fit,
            X_valid,
            y_valid,
            preprocessor,
            config
        )

        y_valid_pred_encoded, valid_macro_f1 = evaluate_encoded(
            model,
            X_valid,
            y_valid
        )

        elapsed_time = time.perf_counter() - start_time

        result = {
            "config": config,
            "valid_macro_f1": valid_macro_f1,
            "callback_best_f1": callback_best_f1,
            "best_epoch": best_epoch,
            "elapsed_time": elapsed_time
        }

        all_results.append(result)

        print(f"Validation macro F1: {valid_macro_f1:.4f}")
        print(f"Callback best validation macro F1: {callback_best_f1:.4f}")
        print(f"Best epoch: {best_epoch}")
        print(f"Train + validation time: {elapsed_time:.2f} seconds")

        if best_result is None or valid_macro_f1 > best_result["valid_macro_f1"]:
            best_result = result

    print("\n===== Validation Tuning Summary =====")

    sorted_results = sorted(
        all_results,
        key=lambda item: item["valid_macro_f1"],
        reverse=True
    )

    for index, result in enumerate(sorted_results, start=1):
        config = result["config"]

        print(
            f"{index}. {config['name']} | "
            f"valid_macro_f1={result['valid_macro_f1']:.4f} | "
            f"best_epoch={result['best_epoch']} | "
            f"service_emb_dim={config['service_emb_dim']} | "
            f"weights={config['class_weights']}"
        )

    print("\n===== Best Validation Configuration =====")
    print(f"Best configuration: {best_result['config']['name']}")
    print(f"Best validation macro F1: {best_result['valid_macro_f1']:.4f}")
    print(f"Best epoch: {best_result['best_epoch']}")
    print(f"Best class weights: {best_result['config']['class_weights']}")

    return best_result


def train_final_and_evaluate(best_result, X_train, y_train_encoded, X_test, y_test, label_encoder):
    config = best_result["config"]

    print("\n===== Final Embedding MLP Test Evaluation =====")
    print(f"Final configuration: {config['name']}")
    print(f"Final seed: {config['seed']}")
    print(f"Final class weights: {config['class_weights']}")

    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    preprocessor = TabularEmbeddingPreprocessor()

    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_valid = preprocessor.transform(X_valid_raw)
    X_test_processed = preprocessor.transform(X_test)

    start_time = time.perf_counter()

    model, final_valid_f1, final_best_epoch = train_model(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        preprocessor,
        config
    )

    probabilities = model.predict(X_test_processed, verbose=0)
    y_test_pred_encoded = np.argmax(probabilities, axis=1)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    elapsed_time = time.perf_counter() - start_time

    test_macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0
    )

    print(f"Final validation macro F1 during training: {final_valid_f1:.4f}")
    print(f"Final best epoch: {final_best_epoch}")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, labels=LABELS, zero_division=0))

    return y_test_pred, test_macro_f1, elapsed_time, final_valid_f1, final_best_epoch


def run_cv(best_result, X_train, y_train_encoded):
    print("\nRunning 3-fold cross-validation for best embedding setup...")

    config = best_result["config"]

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []
    best_epochs = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n===== CV fold {fold_id} =====")

        X_outer_train = X_train.iloc[train_index]
        X_outer_valid = X_train.iloc[valid_index]

        y_outer_train = y_train_encoded[train_index]
        y_outer_valid = y_train_encoded[valid_index]

        X_fit_raw, X_inner_valid_raw, y_fit, y_inner_valid = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=0.2,
            stratify=y_outer_train,
            random_state=RANDOM_STATE
        )

        preprocessor = TabularEmbeddingPreprocessor()

        X_fit = preprocessor.fit_transform(X_fit_raw)
        X_inner_valid = preprocessor.transform(X_inner_valid_raw)
        X_outer_valid_processed = preprocessor.transform(X_outer_valid)

        model, inner_best_f1, best_epoch = train_model(
            X_fit,
            y_fit,
            X_inner_valid,
            y_inner_valid,
            preprocessor,
            config
        )

        y_outer_pred_encoded, fold_score = evaluate_encoded(
            model,
            X_outer_valid_processed,
            y_outer_valid
        )

        scores.append(fold_score)
        best_epochs.append(best_epoch)

        print(f"Fold {fold_id} inner validation macro F1: {inner_best_f1:.4f}")
        print(f"Fold {fold_id} best epoch: {best_epoch}")
        print(f"Fold {fold_id} outer macro F1: {fold_score:.4f}")

    elapsed_time = time.perf_counter() - start_time
    scores = np.array(scores)

    print(f"\nCV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV best epochs: {best_epochs}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time, best_epochs


def main():
    total_start_time = time.perf_counter()

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    best_result = tune_configs_on_validation(
        X_train,
        y_train_encoded
    )

    y_test_pred, test_macro_f1, final_eval_time, final_valid_f1, final_best_epoch = train_final_and_evaluate(
        best_result,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    cv_mean, cv_std, cv_time, cv_best_epochs = run_cv(
        best_result,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_test_pred,
        "outputs/confusion_matrices/22_keras_embedding_mlp.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 22_keras_embedding_mlp")
    print(f"Best configuration: {best_result['config']['name']}")
    print(f"Best seed: {best_result['config']['seed']}")
    print(f"Best embedding dims: protocol={best_result['config']['protocol_emb_dim']}, service={best_result['config']['service_emb_dim']}, flag={best_result['config']['flag_emb_dim']}")
    print(f"Best class weights: {best_result['config']['class_weights']}")
    print(f"Best validation macro F1 during tuning: {best_result['valid_macro_f1']:.4f}")
    print(f"Final validation macro F1 during training: {final_valid_f1:.4f}")
    print(f"Final best epoch: {final_best_epoch}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"3-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"CV best epochs: {cv_best_epochs}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
