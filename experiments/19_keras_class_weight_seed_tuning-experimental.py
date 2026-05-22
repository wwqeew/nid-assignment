import os
import time
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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


try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


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
    # keep keras runs as stable as possible
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def make_candidate_configs():
    # configs focus on r2l recall while trying not to destroy u2r
    return [
        {
            "name": "cw_1_current_best",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.20,
            "learning_rate": 0.00001,
            "batch_size": 512,
            "epochs": 180,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.67,
                2: 2.22,
                3: 19.95,
                4: 60.0
            }
        },
                {
            "name": "cw_2_current_best",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.20,
            "learning_rate": 0.000001,
            "batch_size": 512,
            "epochs": 140,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.67,
                2: 2.22,
                3: 19.95,
                4: 60.0
            }
        },
        {
            "name": "cw_3_weight_change_U2R_dropout_increase",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.17,
            "learning_rate": 0.00001,
            "batch_size": 512,
            "epochs": 180,
            "patience": 12,
            "class_weights": {
                0: 1.0,
                1: 0.67,
                2: 2.23,
                3: 19.96,
                4: 60.0
            }
        },
                {
            "name": "cw_4_best_result_replica",
            "hidden_1": 128,
            "hidden_2": 64,
            "hidden_3": 32,
            "dropout": 0.2,
            "learning_rate": 0.00001,
            "batch_size": 512,
            "epochs": 180,
            "patience": 12,
            "class_weights": {
                0: 0.54858,
                1: 0.37412,
                2: 2.16145,
                3: 25.3211,
                4: 479.8952
            }
        },
        # {
        #     "name": "cw_3_even_more_r2l",
        #     "hidden_1": 256,
        #     "hidden_2": 128,
        #     "hidden_3": 64,
        #     "dropout": 0.30,
        #     "learning_rate": 0.0005,
        #     "batch_size": 512,
        #     "epochs": 140,
        #     "patience": 12,
        #     "class_weights": {
        #         0: 1.0,
        #         1: 0.7,
        #         2: 2.0,
        #         3: 20.5,
        #         4: 60.5
        #     }
        # },
        # {
        #     "name": "cw_4_less_u2r_more_r2l",
        #     "hidden_1": 256,
        #     "hidden_2": 128,
        #     "hidden_3": 64,
        #     "dropout": 0.30,
        #     "learning_rate": 0.0005,
        #     "batch_size": 512,
        #     "epochs": 140,
        #     "patience": 12,
        #     "class_weights": {
        #         0: 1.0,
        #         1: 0.7,
        #         2: 2.0,
        #         3: 20.0,
        #         4: 60.0
        #     }
        # },
        # {
        #     "name": "cw_5_probe_r2l_boost",
        #     "hidden_1": 256,
        #     "hidden_2": 128,
        #     "hidden_3": 64,
        #     "dropout": 0.30,
        #     "learning_rate": 0.0005,
        #     "batch_size": 512,
        #     "epochs": 140,
        #     "patience": 12,
        #     "class_weights": {
        #         0: 1.0,
        #         1: 0.7,
        #         2: 2.0,
        #         3: 21.0,
        #         4: 61.0
        #     }
        # }
    ]


def make_model(input_dim, config, seed):
    # clear previous graph before creating a new model
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


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, X_valid, y_valid, patience):
        super().__init__()

        # monitor macro f1 directly instead of val_loss
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


def train_model(X_fit, y_fit, X_valid, y_valid, config, seed):
    # train one keras model using macro f1 early stopping
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
        epochs=config["epochs"],
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
    # split only training data, kddtest+ stays untouched
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

    candidate_configs = make_candidate_configs()

    # seed 21 was strong in the previous experiment, so keep it here, 
    # Also no need to change seed, results are very similar accross results, therefor model is robust enough at any seed
    seeds = [7, 100, 777]

    best_result = None
    all_results = []

    print("\n===== Keras class weight + seed tuning =====")

    for config in candidate_configs:
        for seed in seeds:
            print(f"\nTesting configuration: {config['name']}")
            print(f"Seed: {seed}")
            print(f"Dropout: {config['dropout']}")
            print(f"Learning rate: {config['learning_rate']}")
            print(f"Class weights: {config['class_weights']}")

            start_time = time.perf_counter()

            model, best_train_valid_f1, best_epoch = train_model(
                X_fit,
                y_fit,
                X_valid,
                y_valid,
                config,
                seed
            )

            y_valid_pred_encoded, valid_macro_f1 = evaluate_encoded(
                model,
                X_valid,
                y_valid
            )

            elapsed_time = time.perf_counter() - start_time

            result = {
                "config": config,
                "seed": seed,
                "valid_macro_f1": valid_macro_f1,
                "callback_best_f1": best_train_valid_f1,
                "best_epoch": best_epoch,
                "elapsed_time": elapsed_time
            }

            all_results.append(result)

            print(f"Validation macro F1: {valid_macro_f1:.4f}")
            print(f"Callback best validation macro F1: {best_train_valid_f1:.4f}")
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

    for index, result in enumerate(sorted_results[:10], start=1):
        print(
            f"{index}. {result['config']['name']} | "
            f"seed={result['seed']} | "
            f"valid_macro_f1={result['valid_macro_f1']:.4f} | "
            f"best_epoch={result['best_epoch']}"
        )

    print("\n===== Best Validation Configuration =====")
    print(f"Best configuration: {best_result['config']['name']}")
    print(f"Best seed: {best_result['seed']}")
    print(f"Best validation macro F1: {best_result['valid_macro_f1']:.4f}")
    print(f"Best epoch: {best_result['best_epoch']}")
    print(f"Best class weights: {best_result['config']['class_weights']}")

    return best_result


def train_final_and_evaluate(best_result, X_train, y_train_encoded, X_test, y_test, label_encoder):
    # final model uses best config and seed from validation tuning
    config = best_result["config"]
    seed = best_result["seed"]

    print("\n===== Final Keras Test Evaluation =====")
    print(f"Final configuration: {config['name']}")
    print(f"Final seed: {seed}")
    print(f"Final class weights: {config['class_weights']}")

    X_fit_raw, X_valid_raw, y_fit, y_valid = train_test_split(
        X_train,
        y_train_encoded,
        test_size=0.1,
        stratify=y_train_encoded,
        random_state=seed
    )

    preprocessor = make_scaled_preprocessor(X_fit_raw)

    X_fit = preprocessor.fit_transform(X_fit_raw)
    X_valid = preprocessor.transform(X_valid_raw)
    X_test_processed = preprocessor.transform(X_test)

    start_time = time.perf_counter()

    model, best_train_valid_f1, best_epoch = train_model(
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        config,
        seed
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

    print(f"Final validation macro F1 during training: {best_train_valid_f1:.4f}")
    print(f"Final best epoch: {best_epoch}")
    print(f"Test macro F1-score: {test_macro_f1:.4f}")
    print(f"Final train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, labels=LABELS, zero_division=0))

    return y_test_pred, test_macro_f1, elapsed_time, best_train_valid_f1, best_epoch


def run_cv(best_result, X_train, y_train_encoded):
    # cv only for the selected best setup
    print("\nRunning 3-fold cross-validation for best Keras setup...")

    config = best_result["config"]
    seed = best_result["seed"]

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train_encoded), start=1):
        print(f"\n===== CV fold {fold_id} =====")

        X_outer_train = X_train.iloc[train_index]
        X_outer_valid = X_train.iloc[valid_index]

        y_outer_train = y_train_encoded[train_index]
        y_outer_valid = y_train_encoded[valid_index]

        X_fit_raw, X_inner_valid_raw, y_fit, y_inner_valid = train_test_split(
            X_outer_train,
            y_outer_train,
            test_size=0.1,
            stratify=y_outer_train,
            random_state=seed
        )

        preprocessor = make_scaled_preprocessor(X_fit_raw)

        X_fit = preprocessor.fit_transform(X_fit_raw)
        X_inner_valid = preprocessor.transform(X_inner_valid_raw)
        X_outer_valid_processed = preprocessor.transform(X_outer_valid)

        model, best_inner_f1, best_epoch = train_model(
            X_fit,
            y_fit,
            X_inner_valid,
            y_inner_valid,
            config,
            seed
        )

        y_outer_pred_encoded, fold_score = evaluate_encoded(
            model,
            X_outer_valid_processed,
            y_outer_valid
        )

        scores.append(fold_score)

        print(f"Fold {fold_id} inner validation macro F1: {best_inner_f1:.4f}")
        print(f"Fold {fold_id} best epoch: {best_epoch}")
        print(f"Fold {fold_id} outer macro F1: {fold_score:.4f}")

    elapsed_time = time.perf_counter() - start_time
    scores = np.array(scores)

    print(f"\nCV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


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

    cv_mean, cv_std, cv_time = run_cv(
        best_result,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        y_test_pred,
        "outputs/confusion_matrices/19_keras_class_weight_seed_tuning.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Experiment Summary =====")
    print("Experiment: 19_keras_class_weight_seed_tuning")
    print(f"Best configuration: {best_result['config']['name']}")
    print(f"Best seed: {best_result['seed']}")
    print(f"Best class weights: {best_result['config']['class_weights']}")
    print(f"Best validation macro F1 during tuning: {best_result['valid_macro_f1']:.4f}")
    print(f"Final validation macro F1 during training: {final_valid_f1:.4f}")
    print(f"Final best epoch: {final_best_epoch}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"3-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Final train + prediction time: {final_eval_time:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total experiment time: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
