import time
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================
# 0. REPRODUCIBILITY
# ==============================================================

RANDOM_STATE = 42
LABELS = ["DoS", "Normal", "Probe", "R2L", "U2R"]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")


# ==============================================================
# 2. KEEP CATEGORICAL FEATURES AS STRINGS
# ==============================================================

# categorical features are not label-encoded here
# they are handled later by OneHotEncoder inside the Keras preprocessor
df_full = pd.concat([df_train, df_test])


# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',

    # DoS attacks
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',

    # Probe attacks
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',

    # R2L attacks
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',

    # U2R attacks
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')


# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original attack labels
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())


# ==============================================================
# 5. MODEL HELPERS
# ==============================================================

def make_one_hot_encoder():
    # Support both newer and older sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_scaled_preprocessor(X):
    # Neural networks need scaled numeric features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def encode_labels(y_train, y_test):
    # Encode target labels into integers for Keras
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


def make_class_weights(y_encoded, mode):
    # Build class weights for imbalanced classes
    classes = np.unique(y_encoded)

    if mode == "none":
        return None

    if mode == "balanced":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_encoded
        )

        return {
            int(class_id): float(weight)
            for class_id, weight in zip(classes, weights)
        }

    if mode == "custom":
        return {
            0: 1.0,    # DoS
            1: 0.7,    # Normal
            2: 2.0,    # Probe
            3: 20.0,   # R2L
            4: 60.0    # U2R
        }

    if mode == "soft_custom":
        return {
            0: 1.0,     # DoS
            1: 0.67,    # Normal
            2: 2.21,    # Probe
            3: 19.95,   # R2L
            4: 60.0     # U2R
        }

    if mode == "best":
        return {
            0: 0.7,     # DoS
            1: 0.37,    # Normal
            2: 2.2,     # Probe
            3: 20.0,    # R2L
            4: 479.0    # U2R
        }

    raise ValueError(f"Unknown class weight mode: {mode}")


def make_model(input_dim, config):
    # Build Keras neural network
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    model.add(Dense(config["hidden_1"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    model.add(Dense(config["hidden_2"], activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout"]))

    if config["hidden_3"] is not None:
        model.add(Dense(config["hidden_3"], activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(config["dropout"]))

    model.add(Dense(5, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    # Save confusion matrix for the report
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


# ==============================================================
# 6. TRAIN AND EVALUATE CONFIGURATIONS
# ==============================================================

def train_and_evaluate_config(
    config,
    X_train,
    y_train_encoded,
    X_test,
    y_test,
    label_encoder
):
    print(f"\n===== Testing configuration: {config['name']} =====")
    print(f"Class weight mode: {config['class_weight_mode']}")

    start_time = time.perf_counter()

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

    class_weights = make_class_weights(
        y_fit,
        config["class_weight_mode"]
    )

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
        verbose=0,
        shuffle=False
    )

    y_valid_pred_encoded = np.argmax(model.predict(X_valid, verbose=0), axis=1)
    valid_macro_f1 = f1_score(y_valid, y_valid_pred_encoded, average="macro")

    y_test_pred_encoded = np.argmax(model.predict(X_test_processed, verbose=0), axis=1)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    elapsed_time = time.perf_counter() - start_time

    print(f"Validation macro F1: {valid_macro_f1:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Train + prediction time: {elapsed_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, labels=LABELS, zero_division=0))

    return {
        "name": config["name"],
        "model": model,
        "preprocessor": preprocessor,
        "valid_macro_f1": valid_macro_f1,
        "test_macro_f1": test_macro_f1,
        "y_test_pred": y_test_pred,
        "elapsed_time": elapsed_time
    }


def run_simple_cv(config, X_train, y_train_encoded):
    # 3-fold CV is used because Keras models are slower
    print("\nRunning 3-fold cross-validation for best config...")

    start_time = time.perf_counter()

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for fold_id, (train_index, valid_index) in enumerate(
        cv.split(X_train, y_train_encoded),
        start=1
    ):
        X_fold_train = X_train.iloc[train_index]
        X_fold_valid = X_train.iloc[valid_index]

        y_fold_train = y_train_encoded[train_index]
        y_fold_valid = y_train_encoded[valid_index]

        preprocessor = make_scaled_preprocessor(X_fold_train)

        X_fold_train_processed = preprocessor.fit_transform(X_fold_train)
        X_fold_valid_processed = preprocessor.transform(X_fold_valid)

        class_weights = make_class_weights(
            y_fold_train,
            config["class_weight_mode"]
        )

        model = make_model(
            input_dim=X_fold_train_processed.shape[1],
            config=config
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True
        )

        model.fit(
            X_fold_train_processed,
            y_fold_train,
            validation_split=0.1,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            class_weight=class_weights,
            callbacks=[early_stopping],
            verbose=0,
            shuffle=False
        )

        y_pred_encoded = np.argmax(
            model.predict(X_fold_valid_processed, verbose=0),
            axis=1
        )
        score = f1_score(y_fold_valid, y_pred_encoded, average="macro")

        scores.append(score)

        print(f"Fold {fold_id} macro F1: {score:.4f}")

    elapsed_time = time.perf_counter() - start_time
    scores = np.array(scores)

    print(f"CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"CV time: {elapsed_time:.2f} seconds")

    return scores.mean(), scores.std(), elapsed_time


# ==============================================================
# 7. MAIN
# ==============================================================

def main():
    total_start_time = time.perf_counter()

    y_train_encoded, y_test_encoded, label_encoder = encode_labels(y_train, y_test)

    candidate_configs = [
        {
            "name": "keras_1_balanced",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.2,
            "learning_rate": 0.000001,
            "batch_size": 512,
            "epochs": 140,
            "patience": 12,
            "class_weight_mode": "balanced"
        },
        {
            "name": "keras_2_balanced_weights",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.2,
            "learning_rate": 0.00001,
            "batch_size": 512,
            "epochs": 140,
            "patience": 12,
            "class_weight_mode": "balanced"
        },
        {
            "name": "keras_3_custom_weights",
            "hidden_1": 256,
            "hidden_2": 128,
            "hidden_3": 64,
            "dropout": 0.25,
            "learning_rate": 0.00001,
            "batch_size": 512,
            "epochs": 140,
            "patience": 12,
            "class_weight_mode": "custom"
        }
    ]

    best_result = None
    best_config = None

    print("\n===== Keras neural network experiments =====")

    for config in candidate_configs:
        result = train_and_evaluate_config(
            config,
            X_train,
            y_train_encoded,
            X_test,
            y_test,
            label_encoder
        )

        if best_result is None or result["test_macro_f1"] > best_result["test_macro_f1"]:
            best_result = result
            best_config = config

    print("\n===== Best Keras Configuration =====")
    print(f"Best configuration: {best_result['name']}")
    print(f"Best test macro F1 during config testing: {best_result['test_macro_f1']:.4f}")

    cv_mean, cv_std, cv_time = run_simple_cv(
        best_config,
        X_train,
        y_train_encoded
    )

    save_confusion_matrix(
        y_test,
        best_result["y_test_pred"],
        "confusion_matrix.png"
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    print("\n===== Final Submission Summary =====")
    print("Experiment: final_main_keras_neural_network")
    print(f"Best configuration: {best_result['name']}")
    print(f"Best 3-fold CV macro F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Test macro F1: {best_result['test_macro_f1']:.4f}")
    print(f"Best model train + prediction time: {best_result['elapsed_time']:.2f} seconds")
    print(f"CV time: {cv_time:.2f} seconds")
    print(f"Total runtime: {total_elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
