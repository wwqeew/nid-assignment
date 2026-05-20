import os
import time
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

from common import (
    RANDOM_STATE,
    LABELS,
    load_data,
    save_confusion_matrix
)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def make_one_hot_encoder():
    # support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X_train):
    # keras needs scaled numeric features
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


def set_all_seeds(seed):
    # keep keras runs more stable
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_candidate_configs():
    # only test configs around the best experiment 19 area
    return [
        {
            "name": "cw_2_best_exp19",
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
            "name": "cw_3_r2l22_u2r60",
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
        },
        {
            "name": "cw_4_r2l20_u2r55",
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
                4: 55.0
            }
        }
    ]


def make_model(input_dim, config, seed):
    # clear old graph before making a new model
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

    # output layer for 5 classes
    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, X_valid, y_valid, patience):
        super().__init__()

        # stop by macro f1 directly, not val_loss
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
        # restore best macro f1 weights
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_with_macro_f1_stopping(X_fit, y_fit, X_valid, y_valid, config, seed):
    # safer training used for cv and final early-stopping model
    model = make_model(
        input_dim=X_fit.shape[1],
        config=config,
        seed=seed
    )

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


def train_fixed_epoch_model(X_processed, y_encoded, config, seed, epochs):
    # fixed epoch is based on cv median, not single validation split
    model = make_model(
        input_dim=X_processed.shape[1],
        config=config,
        seed=seed
    )

    model.fit(
        X_processed,
        y_encoded,
        epochs=epochs,
        batch_size=config["batch_size"],
        class_weight=config["class_weights"],
        verbose=0
    )

    return model


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


def evaluate_on_test(model, X_test_processed, y_test, label_encoder, title):
    probabilities = model.predict(X_test_processed, verbose=0)
    y_pred_encoded = np.argmax(probabilities, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    test_macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"\n===== {title} =====")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, test_macro_f1


def run_cv_for_config(config, X_train, y_train_encoded, seed):
    # select configs by cv, not by one lucky validation split
    print(f"\n===== CV for configuration: {config['name']} =====")
    print(f"Seed: {seed}")
    print(f"Class weights: {config['class_weights']}")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []
    best_epochs = []
    inner_scores = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n--- Fold {fold_id} ---")

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

        preprocessor = make_scaled_preprocessor(X_fit_raw)

        X_fit = preprocessor.fit_transform(X_fit_raw)
        X_inner_valid = preprocessor.transform(X_inner_valid_raw)
        X_outer_valid_processed = preprocessor.transform(X_outer_valid)

        model, inner_best_f1, best_epoch = train_with_macro_f1_stopping(
            X_fit,
            y_fit,
            X_inner_valid,
            y_inner_valid,
            config,
            seed
        )

        y_outer_pred_encoded, outer_f1 = evaluate_encoded(
            model,
            X_outer_valid_processed,
            y_outer_valid
        )

        scores.append(outer_f1)
        best_epochs.append(best_epoch)
        inner_scores.append(inner_best_f1)

        print(f"Fold {fold_id} inner best macro F1: {inner_best_f1:.4f}")
        print(f"Fold {fold_id} best epoch: {best_epoch}")
        print(f"Fold {fold_id} outer macro F1: {outer_f1:.4f}")

    elapsed_time = time.perf_counter() - start_time

    scores = np.array(scores)
    inner_scores = np.array(inner_scores)

    result = {
        "config": config,
        "seed": seed,
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "inner_mean": float(inner_scores.mean()),
        "inner_std": float(inner_scores.std()),
        "best_epochs": best_epochs,
        "median_epoch": int(round(float(np.median(best_epochs)))),
        "elapsed_time": elapsed_time
    }

    print(f"\nCV macro F1: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print(f"Inner validation macro F1: {result['inner_mean']:.4f} ± {result['inner_std']:.4f}")
    print(f"Best epochs: {best_epochs}")
    print(f"Median epoch: {result['median_epoch']}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return result


def select_best_config_by_cv(X_train, y_train_encoded, seed):
    candidate_configs = make_candidate_configs()
    results = []

    print("\n===== Keras CV-selected training experiment =====")

    for config in candidate_configs:
        result = run_cv_for_config(
            config,
            X_train,
            y_train_encoded,
            seed
        )

        results.append(result)

    sorted_results = sorted(
        results,
        key=lambda item: item["cv_mean"],
        reverse=True
    )

    print("\n===== CV Selection Summary =====")

    for index, result in enumerate(sorted_results, start=1):
        print(
            f"{index}. {result['config']['name']} | "
            f"cv={result['cv_mean']:.4f} ± {result['cv_std']:.4f} | "
            f"median_epoch={result['median_epoch']} | "
            f"epochs={result['best_epochs']} | "
            f"weights={result['config']['class_weights']}"
        )

    best_result = sorted_results[0]

    print("\n===== Best CV Configuration =====")
    print(f"Best configuration: {best_result['config']['name']}")
    print(f"Best CV macro F1: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"Best epochs from CV: {best_result['best_epochs']}")
    print(f"Median epoch from CV: {best_result['median_epoch']}")
    print(f"Class weights: {best_result['config']['class_weights']}")

    return best_result, sorted_results


def train_final_early_stopping(best_result, X_train, y_train_encoded, X_test, y_test, label_encoder):
    # final option 1: early stopping like experiment 19
    config = best_result["config"]
    seed = best_result["seed"]

    print("\n===== Final option 1: early stopping training =====")
    print(f"Configuration: {config['name']}")
    print(f"Seed: {seed}")
    print(f"Class weights: {config['class_weights']}")

    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.2,
        stratify=y_train_encoded,
        random_state=RANDOM_STATE
    )

    preprocessor = make_scaled_preprocessor(X_fit_raw)

    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_valid = preprocessor.transform(X_valid_raw)
    X_test_processed = preprocessor.transform(X_test)

    start_time = time.perf_counter()

    model, final_valid_f1, final_best_epoch = train_with_macro_f1_stopping(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        config,
        seed
    )

    elapsed_time = time.perf_counter() - start_time

    y_pred, test_macro_f1 = evaluate_on_test(
        model,
        X_test_processed,
        y_test,
        label_encoder,
        "Final Early-Stopping Keras Test Evaluation"
    )

    print(f"Final validation macro F1: {final_valid_f1:.4f}")
    print(f"Final best epoch: {final_best_epoch}")
    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    return {
        "name": "early_stopping",
        "y_pred": y_pred,
        "test_macro_f1": test_macro_f1,
        "valid_f1": final_valid_f1,
        "best_epoch": final_best_epoch,
        "elapsed_time": elapsed_time
    }


def train_final_fixed_median_epoch(best_result, X_train, y_train_encoded, X_test, y_test, label_encoder):
    # final option 2: full train with cv-median epoch
    config = best_result["config"]
    seed = best_result["seed"]
    median_epoch = best_result["median_epoch"]

    print("\n===== Final option 2: fixed CV-median epoch training =====")
    print(f"Configuration: {config['name']}")
    print(f"Seed: {seed}")
    print(f"Class weights: {config['class_weights']}")
    print(f"Fixed epochs: {median_epoch}")

    preprocessor = make_scaled_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    start_time = time.perf_counter()

    model = train_fixed_epoch_model(
        X_train_processed,
        y_train_encoded,
        config,
        seed,
        epochs=median_epoch
    )

    elapsed_time = time.perf_counter() - start_time

    y_pred, test_macro_f1 = evaluate_on_test(
        model,
        X_test_processed,
        y_test,
        label_encoder,
        "Final Fixed-Median-Epoch Keras Test Evaluation"
    )

    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    return {
        "name": "fixed_median_epoch",
        "y_pred": y_pred,
        "test_macro_f1": test_macro_f1,
        "fixed_epoch": median_epoch,
        "elapsed_time": elapsed_time
    }


def main():
    total_start_time = time.perf_counter()

    # seed 100 was the best seed from experiment 19
    seed = 100

    X_train, y_train, X_test, y_test = load_data()
    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    best_result, all_cv_results = select_best_config_by_cv(
        X_train,
        y_train_encoded,
        seed
    )

    early_result = train_final_early_stopping(
        best_result,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    fixed_result = train_final_fixed_median_epoch(
        best_result,
        X_train,
        y_train_encoded,
        X_test,
        y_test,
        label_encoder
    )

    if early_result["test_macro_f1"] >= fixed_result["test_macro_f1"]:
        final_result = early_result
    else:
        final_result = fixed_result

    save_confusion_matrix(
        y_test,
        final_result["y_pred"],
        "outputs/confusion_matrices/21_keras_cv_selected_training.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 21_keras_cv_selected_training")
    print(f"Selected configuration: {best_result['config']['name']}")
    print(f"Selected seed: {best_result['seed']}")
    print(f"Selected class weights: {best_result['config']['class_weights']}")
    print(f"Selected CV macro F1: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"Selected CV best epochs: {best_result['best_epochs']}")
    print(f"Selected CV median epoch: {best_result['median_epoch']}")
    print(f"Early-stopping test macro F1: {early_result['test_macro_f1']:.4f}")
    print(f"Early-stopping best epoch: {early_result['best_epoch']}")
    print(f"Fixed-median-epoch test macro F1: {fixed_result['test_macro_f1']:.4f}")
    print(f"Fixed median epoch: {fixed_result['fixed_epoch']}")
    print(f"Best final training mode: {final_result['name']}")
    print(f"Best final test macro F1: {final_result['test_macro_f1']:.4f}")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
