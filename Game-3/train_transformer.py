import os
import re
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mutual_info_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-3", "Transformer")
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
WINDOW_SIZE = 100
EPOCHS = 20
BATCH_SIZE = 32
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]

ALL_FEATURES = [
    'acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
    'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]',
    'dec_tree_out_1'
]

os.makedirs(results_root, exist_ok=True)

# ---------- MODEL ----------
def build_transformer(input_shape, num_classes, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)  # shape: (100, 7)
    x = LayerNormalization(epsilon=1e-6)(inputs)

    # Project input to match attention/feed-forward dimensions
    x_proj = Dense(ff_dim)(x)

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x_proj, x_proj)
    x = Add()([x_proj, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_output = Dense(ff_dim, activation='relu')(x)
    ff_output = Dropout(dropout_rate)(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- HELPERS ----------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def segment_sequences(df, seq_len=100):
    segments = []
    for i in range(0, len(df) - seq_len + 1, seq_len):
        window = df[ALL_FEATURES].iloc[i:i+seq_len].values
        if window.shape[0] == seq_len:
            segments.append(window)
    return np.array(segments)

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = mutual_info_score(y_true, y_pred)
    y_true_idx = [labels.index(y) for y in y_true]
    y_pred_idx = [labels.index(y) for y in y_pred]
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = np.sum(rel_entr(p_true, p_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "mutual_information": float(mi),
        "kl_divergence": float(kl),
        "confusion_matrix": cm.tolist(),
        "labels": [str(l) for l in labels]
    }

def save_results(y_true, y_pred, le, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = list(range(len(le.classes_)))
    class_names = list(le.classes_)
    metrics = compute_metrics(y_true, y_pred, labels)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_precision = metrics["precision"]
    weighted_recall = metrics["recall"]
    weighted_f1 = metrics["f1_score"]
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Average KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Average Mutual Information: {metrics['mutual_information']:.4f}\n")
        f.write(f"Average Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n")

    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names)\
        .to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

# ---------- LOAD DATA ----------
user_data = {}
all_labels = []
for user in get_user_dirs(data_root):
    path = os.path.join(data_root, user)
    X_user, y_user = [], []
    for file_path in glob.glob(os.path.join(path, '**', '*.csv'), recursive=True):
        rel_path = os.path.relpath(file_path, start=path).split(os.sep)
        if len(rel_path) >= 3 and rel_path[0].lower() == "processed":
            activity = rel_path[1].strip().lower()
            if activity not in VALID_ACTIVITIES:
                continue
            df = pd.read_csv(file_path)
            if not all(f in df.columns for f in ALL_FEATURES) or df[ALL_FEATURES].isnull().any().any() or len(df) < WINDOW_SIZE:
                continue
            segs = segment_sequences(df, WINDOW_SIZE)
            if segs.size > 0:
                X_user.extend(segs)
                y_user.extend([activity]*len(segs))
    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))
        all_labels.extend(y_user)

le = LabelEncoder().fit(all_labels)

# ---------- CROSS-USER ----------
for ratio in train_test_ratios:
    split_name = f"train{int(ratio[0]*100)}_test{int(ratio[1]*100)}"
    out_dir = os.path.join(results_root, "cross-user", split_name)
    keys = list(user_data.keys())
    n_train = int(ratio[0] * len(keys))
    train_users = keys[:n_train]
    test_users = keys[n_train:]
    if not train_users or not test_users:
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = le.transform(np.hstack([user_data[u][1] for u in test_users]))

    model = build_transformer(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    save_results(list(y_test), list(y_pred), le, out_dir)

# ---------- INTRA-USER ----------
all_true, all_pred = [], []
for user in user_data:
    X, y = user_data[user]
    y_enc = le.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
    model = build_transformer(X.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

save_results(all_true, all_pred, le, os.path.join(results_root, "intra-user"))

# ---------- LOUO ----------
all_true, all_pred = [], []
for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test, y_test = user_data[test_user]
    y_test = le.transform(y_test)
    model = build_transformer(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

save_results(all_true, all_pred, le, os.path.join(results_root, "cross-user", "leave-one-out"))

