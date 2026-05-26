import os
import re
import glob
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, mutual_info_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# -------------------- SETUP --------------------
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

tf.get_logger().setLevel('ERROR')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-1", "RNN")
window_size = 100
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
BATCH_SIZE = 32
EPOCHS = 20
os.makedirs(results_root, exist_ok=True)


# -------------------- UTILS --------------------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)],
                  key=lambda x: int(re.findall(r"\d+", x)[0]))


def segment_sequences(df, seq_len=100):
    segments = []
    for i in range(0, len(df) - seq_len + 1, seq_len):
        window = df[['dec_tree_out_1']].iloc[i:i + seq_len].values
        if window.shape[0] == seq_len:
            segments.append(window)
    return np.array(segments)


def compute_leakage_metrics(y_true, y_pred):
    labels_in_data = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_in_data)
    mi = float(mutual_info_score(y_true, y_pred))

    p_true = np.bincount(y_true, minlength=len(labels_in_data)) / len(y_true)
    p_pred = np.bincount(y_pred, minlength=len(labels_in_data)) / len(y_pred)

    p_true += 1e-12
    p_pred += 1e-12
    kl = float(np.sum(rel_entr(p_true, p_pred)))

    return mi, kl, cm, labels_in_data


def compute_full_metrics(y_true, y_pred, label_encoder):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    mi, kl, cm, labels_in_data = compute_leakage_metrics(y_true, y_pred)
    metrics = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mutual_information": round(mi, 4),
        "kl_divergence": round(kl, 4),
        "confusion_matrix": [[int(x) for x in row] for row in cm],
        "labels": [str(label_encoder.classes_[l]) for l in labels_in_data]
    }
    return metrics


def save_metrics_and_reports(base_path, y_true, y_pred, label_encoder):
    os.makedirs(base_path, exist_ok=True)
    metrics = compute_full_metrics(y_true, y_pred, label_encoder)
    # Save metrics.json
    with open(os.path.join(base_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    # Save report.txt
    with open(os.path.join(base_path, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {metrics['precision']}\n")
        f.write(f"Macro Recall: {metrics['recall']}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']}\n")
    # Save confusion_matrix.csv
    pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(
        os.path.join(base_path, "confusion_matrix.csv")
    )


def build_rnn(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# -------------------- LOAD DATA --------------------
user_data = {}
all_labels = []
for user in get_user_dirs(data_root):
    user_path = os.path.join(data_root, user)
    X_user, y_user = [], []
    for file_path in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True):
        rel_path = os.path.relpath(file_path, start=user_path).split(os.sep)
        if len(rel_path) >= 3 and rel_path[0].lower() == "processed":
            activity = rel_path[1].strip().lower()
            if activity not in VALID_ACTIVITIES:
                continue
            df = pd.read_csv(file_path)
            if 'dec_tree_out_1' not in df.columns or df['dec_tree_out_1'].isnull().any() or len(df) < window_size:
                continue
            segs = segment_sequences(df, window_size)
            if segs.size > 0:
                X_user.extend(segs)
                y_user.extend([activity] * len(segs))
    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))
        all_labels.extend(y_user)

le = LabelEncoder().fit(all_labels)


# -------------------- CROSS-USER --------------------
for ratio in train_test_ratios:
    split_name = f"train{int(ratio[0] * 100)}_test{int(ratio[1] * 100)}"
    split_path = os.path.join(results_root, "cross-user", split_name)
    os.makedirs(split_path, exist_ok=True)
    keys = list(user_data.keys())
    n_train = int(ratio[0] * len(keys))
    train_users, test_users = keys[:n_train], keys[n_train:]
    if not train_users or not test_users:
        continue
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = le.transform(np.hstack([user_data[u][1] for u in test_users]))
    model = build_rnn(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    save_metrics_and_reports(split_path, y_test, y_pred, le)
    with open(os.path.join(split_path, "summary.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=le.classes_, digits=4, zero_division=0))


# -------------------- INTRA-USER --------------------
intra_path = os.path.join(results_root, "intra-user")
os.makedirs(intra_path, exist_ok=True)
all_preds, all_trues = [], []
for user in user_data:
    X, y = user_data[user]
    y_enc = le.transform(y)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
        model = build_rnn(X.shape[1:], len(le.classes_))
        model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        all_preds.extend(y_pred)
        all_trues.extend(y_test)
    except Exception as e:
        print(f"⚠️ Intra-user error for {user}: {e}")

if all_preds:
    save_metrics_and_reports(intra_path, all_trues, all_preds, le)
    with open(os.path.join(intra_path, "summary.txt"), "w") as f:
        f.write(classification_report(all_trues, all_preds, target_names=le.classes_, digits=4, zero_division=0))


# -------------------- LEAVE-ONE-USER-OUT --------------------
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
all_preds, all_trues = [], []
for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    if not train_users:
        continue
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test, y_test = user_data[test_user]
    y_test = le.transform(y_test)
    model = build_rnn(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_preds.extend(y_pred)
    all_trues.extend(y_test)

if all_preds:
    save_metrics_and_reports(loo_path, all_trues, all_preds, le)
    with open(os.path.join(loo_path, "summary.txt"), "w") as f:
        f.write(classification_report(all_trues, all_preds, target_names=le.classes_, digits=4, zero_division=0))