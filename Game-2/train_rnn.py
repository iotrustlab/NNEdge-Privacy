import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    mutual_info_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------- CONFIG --------
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root_base = os.path.join(data_root, "Results", "Game-2", "RNN")
sequence_length = 100
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
BATCH_SIZE = 32
EPOCHS = 20

# -------- HELPERS --------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def extract_sequences(df, modality):
    if modality == "accel":
        cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1']
    elif modality == "gyro":
        cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
    else:
        raise ValueError("Unknown modality")
    segments = []
    for i in range(0, len(df) - sequence_length + 1, sequence_length):
        seg = df[cols].iloc[i:i+sequence_length].values
        if seg.shape[0] == sequence_length:
            segments.append(seg)
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
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']:.4f}\n")

    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names)\
        .to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

def build_rnn(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# -------- MAIN --------
for modality in ["accel", "gyro"]:
    print(f"\n🔧 Running Game-2 RNN ({modality.upper()})")
    start_time = time.time()
    results_root = os.path.join(results_root_base, modality)
    os.makedirs(results_root, exist_ok=True)

    user_data, all_labels = {}, []
    for user in get_user_dirs(data_root):
        X_user, y_user = [], []
        user_path = os.path.join(data_root, user)
        for file in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True):
            parts = os.path.relpath(file, start=user_path).split(os.sep)
            if "processed" not in [p.lower() for p in parts]:
                continue
            activity = None
            for idx, part in enumerate(parts):
                if part.lower() == "processed" and idx + 1 < len(parts):
                    activity = parts[idx + 1].lower()
                    break
            if activity not in VALID_ACTIVITIES:
                continue
            df = pd.read_csv(file)
            if df.isnull().values.any():
                continue
            if modality == "accel" and not all(c in df.columns for c in ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1']):
                continue
            if modality == "gyro" and not all(c in df.columns for c in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']):
                continue
            if len(df) < sequence_length:
                continue
            segs = extract_sequences(df, modality)
            if segs.size > 0:
                X_user.extend(segs)
                y_user.extend([activity] * len(segs))
        if X_user:
            user_data[user] = (np.array(X_user), np.array(y_user))
            all_labels.extend(y_user)

    le = LabelEncoder().fit(all_labels)

    # --- CROSS-USER ---
    for train_ratio, test_ratio in train_test_ratios:
        split_name = f"train{int(train_ratio*100)}_test{int(test_ratio*100)}"
        split_path = os.path.join(results_root, "cross-user", split_name)
        users = list(user_data.keys())
        train_users = users[:int(train_ratio * len(users))]
        test_users = users[int(train_ratio * len(users)):]
        if not train_users or not test_users:
            continue
        X_train = np.vstack([user_data[u][0] for u in train_users])
        y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
        X_test = np.vstack([user_data[u][0] for u in test_users])
        y_test = le.transform(np.hstack([user_data[u][1] for u in test_users]))
        model = build_rnn(X_train.shape[1:], len(le.classes_))
        model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        save_results(list(y_test), list(y_pred), le, split_path)

    # --- INTRA-USER ---
    intra_path = os.path.join(results_root, "intra-user")
    all_preds, all_trues = [], []
    for user in user_data:
        X, y = user_data[user]
        y_enc = le.transform(y)
        if len(X) < 2: continue
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
            model = build_rnn(X.shape[1:], len(le.classes_))
            model.fit(X_tr, to_categorical(y_tr), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            y_pred = np.argmax(model.predict(X_te), axis=1)
            all_preds.extend(y_pred)
            all_trues.extend(y_te)
        except Exception as e:
            print(f"⚠️ Error in intra-user {user}: {e}")
    if all_preds:
        save_results(all_trues, all_preds, le, intra_path)

    # --- LOUO ---
    loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
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
        save_results(all_trues, all_preds, le, loo_path)

    print(f"✅ Completed {modality.upper()} in {time.time() - start_time:.2f} sec")
