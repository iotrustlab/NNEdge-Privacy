import os
import re
import glob
import json
import numpy as np
import pandas as pd
import pickle
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mutual_info_score, precision_score, recall_score, f1_score
from scipy.special import rel_entr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import random
from _paths import (
    get_data_root,
    get_max_combinations,
    get_results_root,
    get_split_configs,
    get_window_sizes,
    load_experiment_csv,
)
from experiment_metrics import compute_vulnerability



# -------- HELPERS --------
def compute_metrics(y_true, y_pred, labels, probabilities=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    mi = mutual_info_score(y_true, y_pred)
    p_true = np.bincount(y_true, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = float(np.sum(rel_entr(p_true, p_pred)))
    h_true = -np.sum(p_true * np.log2(p_true))
    h_pred = -np.sum(p_pred * np.log2(p_pred))
    nmi_percentage = float((mi / np.sqrt(h_true * h_pred + 1e-12)) * 100)
    reverse_kl_percentage = float((1 - min(1.0, np.sum(rel_entr(p_pred, p_true)) / np.log(len(labels)))) * 100)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": np.mean([v["f1-score"] for k, v in report.items() if isinstance(v, dict)]),
        "precision": np.mean([v["precision"] for k, v in report.items() if isinstance(v, dict)]),
        "recall": np.mean([v["recall"] for k, v in report.items() if isinstance(v, dict)]),
        "mutual_information": mi,
        "mutual_information_nmi_percentage": nmi_percentage,
        "kl_divergence": kl,
        "kl_divergence_reverse_percentage": reverse_kl_percentage,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels)
    }
    if probabilities is not None:
        metrics["vulnerability"] = compute_vulnerability(probabilities)
    return metrics, report, cm
# Updated script with debug prints for file depths and improved RNN


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------- CONFIG ----------
data_root = get_data_root()
results_root = get_results_root("Game-3", "RNN")
window_sizes = get_window_sizes([20, 50, 70, 100])
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
EPOCHS = 25
BATCH_SIZE = 32
FEATURES = [
    'acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
    'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]',
    'dec_tree_out_1'
]
os.makedirs(results_root, exist_ok=True)


# -------- HELPERS --------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def extract_sequences(df, window_size):
    return np.array([
        df[FEATURES].iloc[i:i + window_size].values
        for i in range(0, len(df) - window_size + 1, window_size)
        if df[FEATURES].iloc[i:i + window_size].shape[0] == window_size
    ])

def compute_leakage_metrics(y_true, y_pred):
    """Compute metrics including class-independent information-theoretic measures."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = float(mutual_info_score(y_true, y_pred))
    label_to_index = {label: i for i, label in enumerate(labels)}
    y_true_idx = [label_to_index[y] for y in y_true]
    y_pred_idx = [label_to_index[y] for y in y_pred]
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = float(np.sum(rel_entr(p_true, p_pred)))
    
    # Class-independent metrics
    # Calculate entropies for NMI normalization
    def entropy(labels_arr):
        _, counts = np.unique(labels_arr, return_counts=True)
        probs = counts / len(labels_arr)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    h_true = entropy(y_true_idx)
    h_pred = entropy(y_pred_idx)
    
    # Normalized Mutual Information (class-independent)
    nmi = mi / np.sqrt(h_true * h_pred + 1e-12)
    nmi_percentage = nmi * 100
    
    # Reverse KL divergence (pred->true instead of true->pred)
    reverse_kl = float(np.sum(rel_entr(p_pred, p_true)))
    
    # Normalized reverse KL (class-independent)
    kl_max = np.log(len(labels))
    reverse_kl_normalized = min(1.0, reverse_kl / kl_max)
    reverse_kl_percentage = (1 - reverse_kl_normalized) * 100
    
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(np.mean([v['f1-score'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "precision": float(np.mean([v['precision'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "recall": float(np.mean([v['recall'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "mutual_information": mi,
        "mutual_information_nmi_percentage": float(nmi_percentage),
        "kl_divergence": kl,
        "kl_divergence_reverse_percentage": float(reverse_kl_percentage),
        "confusion_matrix": [[int(x) for x in row] for row in cm],
        "labels": [str(label) for label in labels]
    }

def build_rnn(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def save_outputs(y_true, y_pred, le, path, probabilities, report_header=""):
    os.makedirs(path, exist_ok=True)
    metrics, _, _ = compute_metrics(y_true, y_pred, list(le.classes_), probabilities)

    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4, zero_division=0)
    with open(os.path.join(path, "summary.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(path, "report.txt"), "w") as f:
        f.write(f"{report_header}\n")
        f.write(f"Macro Precision: {metrics['precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall']:.4f}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']:.4f}\n")
        f.write(f"NMI Percentage: {metrics['mutual_information_nmi_percentage']:.4f}%\n")
        f.write(f"Reverse KL Percentage: {metrics['kl_divergence_reverse_percentage']:.4f}%\n")
        f.write(f"Vulnerability: {metrics['vulnerability']:.4f}\n")

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    cm_df.to_csv(os.path.join(path, "confusion_matrix.csv"))


# ---------- MULTI-WINDOW EXPERIMENT ----------
user_cache_path = os.path.join(get_results_root("Game-3", "RNN"), "user_data_all.pkl")
user_cache_dir = os.path.dirname(user_cache_path)
os.makedirs(user_cache_dir, exist_ok=True)
if os.path.exists(user_cache_path):
    print(f"� Loading all user data from {user_cache_path}")
    with open(user_cache_path, "rb") as f:
        all_user_data = pickle.load(f)
    user_dirs = list(all_user_data.keys())
else:
    print("🔍 Loading user directories...")
    user_dirs = get_user_dirs(data_root)
    all_user_data = {}
    for user in tqdm(user_dirs, desc="Users", ncols=80):
        print(f"👤 Inspecting {user}")
        X_user, y_user = [], []
        user_path = os.path.join(data_root, user)
        csv_files = [f for f in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)
                     if "Raw" not in f]
        for file_path in tqdm(csv_files, desc=f"{user} files", leave=False, ncols=80):
            rel_path = os.path.relpath(file_path, start=user_path)
            parts = rel_path.split(os.sep)
            try:
                if len(parts) >= 3 and parts[0].lower() == "processed":
                    activity = parts[1].strip().lower()
                else:
                    continue
                if activity not in VALID_ACTIVITIES:
                    continue
                df = load_experiment_csv(file_path)
                if any([col not in df.columns or df[col].isnull().any() for col in FEATURES]):
                    continue
                X_user.append((df, activity))
                y_user.append(activity)
            except Exception as e:
                print(f"  ❌ Error: {file_path} -> {e}")
        if X_user:
            all_user_data[user] = X_user
            print(f"  ✅ Loaded {len(X_user)} files for {user}")
        else:
            print(f"  ⚠️ No usable data for {user}")
    with open(user_cache_path, "wb") as f:
        pickle.dump(all_user_data, f)

window_sizes = get_window_sizes([20, 50, 70, 100])
split_configs = get_split_configs([(8,3), (6,5), (4,7), (1,10)])
for window_size in window_sizes:
    print(f"\n===== WINDOW SIZE: {window_size} =====")
    results_root_win = os.path.join(results_root, f"Window_{window_size}")
    os.makedirs(results_root_win, exist_ok=True)
    for n_train, n_test in split_configs:
        split_name = f"split{n_train}_{n_test}"
        split_path = os.path.join(results_root_win, split_name)
        os.makedirs(split_path, exist_ok=True)
        user_list = list(all_user_data.keys())
        combos = list(itertools.combinations(user_list, n_train))
        max_combinations = get_max_combinations(2)
        combo_files = [f for f in os.listdir(split_path) if f.startswith("combo_") and f.endswith(".json")]
        if len(combo_files) >= max_combinations:
            print(f"✅ {max_combinations} or more combos already done for {split_name} (window {window_size}), skipping.")
            continue
        if len(combos) > max_combinations:
            random.seed(42)  # For reproducibility
            combos = random.sample(combos, max_combinations)
        metrics_list = []
        confusion_matrix_saved = False
        np.random.seed(42)
        random_idx = np.random.randint(0, len(combos)) if combos else 0
        for idx, train_users in enumerate(combos):
            test_users = [u for u in user_list if u not in train_users]
            if len(test_users) != n_test:
                continue
            train_str = "_".join(sorted(train_users))
            test_str = "_".join(sorted(test_users))
            combo_file = os.path.join(split_path, f"combo_{train_str}__{test_str}.json")
            if os.path.exists(combo_file):
                continue
            user_data = {}
            for user in tqdm(all_user_data.keys(), desc="Users", ncols=80):
                X_user, y_user = [], []
                for df, activity in all_user_data[user]:
                    if len(df) < window_size:
                        continue
                    feats = extract_sequences(df, window_size)
                    if feats.size == 0:
                        continue
                    X_user.extend(feats)
                    y_user.extend([activity] * len(feats))
                if X_user:
                    user_data[user] = (np.array(X_user), np.array(y_user))
            X_train = np.vstack([user_data[u][0] for u in train_users])
            y_train = np.hstack([user_data[u][1] for u in train_users])
            X_test = np.vstack([user_data[u][0] for u in test_users])
            y_test = np.hstack([user_data[u][1] for u in test_users])
            scaler = StandardScaler().fit(X_train.reshape(-1, 1))
            X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
            le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)
            # Game-3 RNN model architecture
            X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], len(FEATURES)))
            X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], len(FEATURES)))
            model = build_rnn((window_size, len(FEATURES)), len(le.classes_))
            model.fit(X_train_rnn, to_categorical(y_train_enc), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            y_proba = model.predict(X_test_rnn, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            metrics, report_dict, cm = compute_metrics(y_test_enc, y_pred, le.classes_, y_proba)
            metrics_list.append({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "mutual_information_nmi_percentage": metrics["mutual_information_nmi_percentage"],
                "kl_divergence_reverse_percentage": metrics["kl_divergence_reverse_percentage"],
                "vulnerability": metrics["vulnerability"],
            })
            with open(combo_file, "w") as f:
                json.dump({
                    "train_users": train_users,
                    "test_users": test_users,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "mutual_information_nmi_percentage": metrics["mutual_information_nmi_percentage"],
                    "kl_divergence_reverse_percentage": metrics["kl_divergence_reverse_percentage"],
                    "vulnerability": metrics["vulnerability"],
                }, f, indent=4)
            if idx == random_idx and not confusion_matrix_saved:
                save_outputs(y_test_enc, y_pred, le, split_path, y_proba)
                confusion_matrix_saved = True
        if metrics_list:
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            with open(os.path.join(split_path, "avg_metrics.json"), "w") as f:
                json.dump(avg_metrics, f, indent=4)
    print(f"✅ RNN training for window size {window_size} completed.")
