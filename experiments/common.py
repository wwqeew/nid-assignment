import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


# fixed seed for reproducible results
RANDOM_STATE = 42

TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "class", "level"
]

CATEGORY_MAP = {
    "normal": "Normal",

    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",

    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",

    "warezclient": "R2L", "guess_passwd": "R2L", "ftp_write": "R2L",
    "imap": "R2L", "phf": "R2L", "multihop": "R2L", "warezmaster": "R2L",
    "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L",
    "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",

    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R"
}

LABELS = ["DoS", "Normal", "Probe", "R2L", "U2R"]


def make_one_hot_encoder():
    # supports both newer and older scikit-learn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data():
    # load train and test datasets
    print("Loading NSL-KDD data...")

    df_train = pd.read_csv(TRAIN_URL, names=COLUMNS)
    df_test = pd.read_csv(TEST_URL, names=COLUMNS)

    # convert detailed attack names into 5 categories
    df_train["category"] = df_train["class"].map(CATEGORY_MAP)
    df_test["category"] = df_test["class"].map(CATEGORY_MAP)

    # stop if some attack name was not mapped
    if df_train["category"].isna().any() or df_test["category"].isna().any():
        raise ValueError("Some attack classes were not mapped.")

    # eemove target-related and useless columns
    drop_columns = ["class", "level", "num_outbound_cmds"]

    X_train = df_train.drop(columns=drop_columns + ["category"])
    y_train = df_train["category"]

    X_test = df_test.drop(columns=drop_columns + ["category"])
    y_test = df_test["category"]

    # print basic dataset information
    print(f"Training records: {X_train.shape[0]}")
    print(f"Test records: {X_test.shape[0]}")
    print(f"Features before encoding: {X_train.shape[1]}")

    print("\nTraining distribution:")
    print(y_train.value_counts())

    print("\nTest distribution:")
    print(y_test.value_counts())

    return X_train, y_train, X_test, y_test


def make_preprocessor(X_train):
    # categorical features need encoding
    categorical_features = ["protocol_type", "service", "flag"]

    # all other features are numerical
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    # apply one-hot encoding only to categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), categorical_features),
            ("num", "passthrough", numeric_features)
        ]
    )

    return preprocessor


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    # train model and evaluate it on the test set
    print(f"\n===== {model_name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Test macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    return y_pred, macro_f1


def run_cross_validation(model, X_train, y_train):
    # estimate model performance using 5-fold cross-validation
    print("\nRunning 5-fold cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1
    )

    print(f"CV macro F1: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores.mean(), scores.std()


def save_confusion_matrix(y_test, y_pred, filename):
    # create output folder if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # build confusion matrix with fixed label order
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    # save confusion matrix as PNG
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
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved to: {filename}")
