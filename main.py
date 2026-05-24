import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================
# 0. REPRODUCIBILITY
# ==============================================================

RANDOM_STATE = 42

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
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge train and test for consistent encoding
df_full = pd.concat([df_train, df_test])

cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le


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
# 5. FINAL MODEL CONFIGURATION
# ==============================================================

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

config = {
    "hidden_1": 256,
    "hidden_2": 128,
    "hidden_3": 64,
    "dropout": 0.25,
    "learning_rate": 0.00001,
    "batch_size": 512,
    "epochs": 140,
    "patience": 12
}

# Custom weights for rare classes
class_weights = {
    0: 1.0,     # DoS
    1: 0.7,     # Normal
    2: 2.0,     # Probe
    3: 20.0,    # R2L
    4: 60.0     # U2R
}


def make_one_hot_encoder():
    # Support both new and old sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X):
    # Scale numeric features and one-hot encode categorical features
    categorical_features = ["protocol_type", "service", "flag"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    return ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )


def build_model(input_dim):
    # Final Keras neural network architecture
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(config["hidden_1"], activation="relu"),
        BatchNormalization(),
        Dropout(config["dropout"]),

        Dense(config["hidden_2"], activation="relu"),
        BatchNormalization(),
        Dropout(config["dropout"]),

        Dense(config["hidden_3"], activation="relu"),
        BatchNormalization(),
        Dropout(config["dropout"]),

        Dense(5, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_confusion_matrix(y_true, y_pred, output_path="confusion_matrix.png"):
    # Save confusion matrix image for the report
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {output_path}")


def main():
    start_time = time.perf_counter()

    # Encode target categories
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)

    print("\nTarget label encoding:")
    for class_name, encoded_value in zip(
        target_encoder.classes_,
        target_encoder.transform(target_encoder.classes_)
    ):
        print(f"{class_name} -> {encoded_value}")

    # Preprocess train and test data
    preprocessor = make_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Build final model using all training data
    model = build_model(input_dim=X_train_processed.shape[1])

    # Stop training if validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        restore_best_weights=True
    )

    print("\nTraining final Keras model...")
    print(f"Class weights: {class_weights}")

    model.fit(
        X_train_processed,
        y_train_encoded,
        validation_split=0.1,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False
    )

    # Evaluate on KDDTest+
    y_test_pred_encoded = np.argmax(model.predict(X_test_processed, verbose=0), axis=1)
    y_test_pred = target_encoder.inverse_transform(y_test_pred_encoded)

    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    print(f"\nTest macro F1: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_test_pred, labels=labels, zero_division=0))

    save_confusion_matrix(y_test, y_test_pred)

    elapsed_time = time.perf_counter() - start_time

    print("\n===== Final Model Summary =====")
    print("Algorithm: Keras Neural Network")
    print(f"Hidden layers: {config['hidden_1']}/{config['hidden_2']}/{config['hidden_3']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Patience: {config['patience']}")
    print(f"Class weights: {class_weights}")
    print(f"Test macro F1: {test_macro_f1:.4f}")
    print(f"Total runtime: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
