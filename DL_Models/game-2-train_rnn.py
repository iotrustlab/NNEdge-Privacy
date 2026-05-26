import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import re
import glob
import json
import pickle
import itertools
import random
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, mutual_info_score
from scipy.special import rel_entr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from _paths import (
    get_data_root,
    get_max_combinations,
    get_results_root,
    get_split_configs,
    get_window_sizes,
    load_experiment_csv,
)
from experiment_metrics import compute_vulnerability

warnings.filterwarnings("ignore")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow GPU devices found: {gpus}")
else:
    print("⚠️ No GPU found for TensorFlow. Training will use CPU.")

data_root = get_data_root()
results_root = get_results_root("Game-2", "RNN")
window_sizes = get_window_sizes([20, 50, 70])
split_configs = get_split_configs([(8,3), (6,5), (4,7), (1,10)])
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
BATCH_SIZE = 32
EPOCHS = 20
MODES = {
    "accel": ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1'],
    "gyro": ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
}
os.makedirs(results_root, exist_ok=True)

# -------- HELPERS --------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def segment_sequences(df, features, window_size):
    segments = []
    for i in range(0, len(df) - window_size + 1, window_size):
        window = df[features].iloc[i:i + window_size].values
        if window.shape[0] == window_size:
            segments.append(window)
    return np.array(segments)

def compute_metrics(y_true, y_pred, le, probabilities=None):
    """Compute metrics including class-independent information-theoretic measures."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    mi = mutual_info_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(le.classes_)
    p_true = np.bincount(y_true, minlength=num_classes) / len(y_true)
    p_pred = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
    kl = float(np.sum(rel_entr(p_true + 1e-12, p_pred + 1e-12)))
    
    # Class-independent metrics
    # Calculate entropies for NMI normalization
    def entropy(labels_arr):
        _, counts = np.unique(labels_arr, return_counts=True)
        probs = counts / len(labels_arr)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    h_true = entropy(y_true)
    h_pred = entropy(y_pred)
    
    # Normalized Mutual Information (class-independent)
    nmi = mi / np.sqrt(h_true * h_pred + 1e-12)
    nmi_percentage = nmi * 100
    
    # Reverse KL divergence (pred->true instead of true->pred)
    reverse_kl = float(np.sum(rel_entr(p_pred + 1e-12, p_true + 1e-12)))
    
    # Normalized reverse KL (class-independent)
    kl_max = np.log(num_classes)
    reverse_kl_normalized = min(1.0, reverse_kl / kl_max)
    reverse_kl_percentage = (1 - reverse_kl_normalized) * 100
    
    labels = list(map(str, le.classes_))
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mutual_information": round(mi, 4),
        "mutual_information_nmi_percentage": round(nmi_percentage, 4),
        "kl_divergence": round(kl, 4),
        "kl_divergence_reverse_percentage": round(reverse_kl_percentage, 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }
    if probabilities is not None:
        metrics["vulnerability"] = round(compute_vulnerability(probabilities), 4)
    return metrics

def save_results(base_path, y_true, y_pred, le, probabilities):
    os.makedirs(base_path, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, le, probabilities)
    with open(os.path.join(base_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(base_path, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {metrics['precision']}\n")
        f.write(f"Macro Recall: {metrics['recall']}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']}\n")
        f.write(f"NMI Percentage: {metrics['mutual_information_nmi_percentage']}%\n")
        f.write(f"Reverse KL Percentage: {metrics['kl_divergence_reverse_percentage']}%\n")
        f.write(f"Vulnerability: {metrics['vulnerability']}\n")
    pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(
        os.path.join(base_path, "confusion_matrix.csv")
    )
    summary_report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4, zero_division=0)
    with open(os.path.join(base_path, "summary.txt"), "w") as f:
        f.write(summary_report)
    return metrics

def build_rnn(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for mode, features in MODES.items():
    print(f"\n🔧 Mode: {mode} — using features {features}\n")
    for window_size in window_sizes:
        print(f"===== WINDOW SIZE: {window_size} =====")
        # Caching
        user_cache_path = os.path.join(results_root, mode, f"user_data_all_window{window_size}.pkl")
        user_cache_dir = os.path.dirname(user_cache_path)
        os.makedirs(user_cache_dir, exist_ok=True)
        if os.path.exists(user_cache_path):
            print(f"🔄 Loading all user data from {user_cache_path}")
            with open(user_cache_path, "rb") as f:
                all_user_data = pickle.load(f)
            user_dirs = list(all_user_data.keys())
        else:
            print("🔍 Loading user directories...")
            user_dirs = get_user_dirs(data_root)
            all_user_data = {}
            all_labels = []
            for user in user_dirs:
                X_user, y_user = [], []
                user_path = os.path.join(data_root, user)
                file_paths = [f for f in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)
                             if "Raw" not in f]
                for file_path in file_paths:
                    rel = os.path.relpath(file_path, start=user_path).split(os.sep)
                    # Find activity name
                    for idx, folder in enumerate(rel):
                        if folder.lower() == "processed" and idx + 1 < len(rel):
                            activity = rel[idx + 1].strip().lower()
                            break
                    else:
                        continue
                    if activity not in VALID_ACTIVITIES:
                        continue
                    df = load_experiment_csv(file_path)
                    if df[features].isnull().values.any() or len(df) < window_size:
                        continue
                    segments = segment_sequences(df, features, window_size)
                    if segments.size > 0:
                        X_user.extend(segments)
                        y_user.extend([activity] * len(segments))
                if X_user:
                    all_user_data[user] = (np.array(X_user), np.array(y_user))
                    all_labels.extend(y_user)
            with open(user_cache_path, "wb") as f:
                pickle.dump(all_user_data, f)
        print(f"Total Users Processed: {len(all_user_data)}")
        if not all_user_data:
            print(f"⚠️ No data for mode: {mode}")
            continue
        le = LabelEncoder().fit([lbl for user in all_user_data for lbl in all_user_data[user][1]])
        results_root_mode = os.path.join(results_root, mode, f"Window_{window_size}")
        os.makedirs(results_root_mode, exist_ok=True)
        user_list = list(all_user_data.keys())
        for n_train, n_test in split_configs:
            split_name = f"split{n_train}_{n_test}"
            split_path = os.path.join(results_root_mode, split_name)
            os.makedirs(split_path, exist_ok=True)
            combos = list(itertools.combinations(user_list, n_train))
            max_combinations = get_max_combinations(2)
            combo_files = [f for f in os.listdir(split_path) if f.startswith("combo_") and f.endswith(".json")]
            if len(combo_files) >= max_combinations:
                print(f"✅ {max_combinations} or more combos already done for {split_name} (window {window_size}), skipping training and calculating avg_metrics.json.")
                # Calculate avg_metrics.json from existing combo files
                all_metrics = []
                for combo_file in combo_files:
                    with open(os.path.join(split_path, combo_file), "r") as f:
                        combo_data = json.load(f)
                        if "metrics" in combo_data:
                            all_metrics.append(combo_data["metrics"])
                if all_metrics:
                    avg_metrics = {}
                    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
                    for k in keys:
                        avg_metrics[k] = round(np.mean([m[k] for m in all_metrics]), 4)
                    with open(os.path.join(split_path, "avg_metrics.json"), "w") as f:
                        json.dump(avg_metrics, f, indent=4)
                continue
            if len(combos) > max_combinations:
                random.seed(42)  # For reproducibility
                combos = random.sample(combos, max_combinations)
            all_metrics = []
            combo_ids = []
            for idx, train_users in enumerate(combos):
                test_users = [u for u in user_list if u not in train_users]
                if len(test_users) != n_test:
                    continue
                train_str = "_".join(sorted(train_users))
                test_str = "_".join(sorted(test_users))
                combo_id = f"combo_{idx+1}_{train_str}_vs_{test_str}"
                combo_json = os.path.join(split_path, f"{combo_id}.json")
                if os.path.exists(combo_json):
                    print(f"  Combo {combo_id} already exists, skipping.")
                    continue
                X_train = np.vstack([all_user_data[u][0] for u in train_users])
                y_train = le.transform(np.hstack([all_user_data[u][1] for u in train_users]))
                X_test = np.vstack([all_user_data[u][0] for u in test_users])
                y_test = le.transform(np.hstack([all_user_data[u][1] for u in test_users]))
                print(f"\n➡️ Combo {combo_id}")
                print(f"   Train Users: {train_users}")
                print(f"   Test Users: {test_users}")
                print(f"   Train Samples: {X_train.shape[0]} | Test Samples: {X_test.shape[0]}")
                model = build_rnn((X_train.shape[1], X_train.shape[2]), len(le.classes_))
                model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
                y_proba = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_proba, axis=1)
                # Ensure split_path is a string, not a numpy array
                out_path = split_path if isinstance(split_path, str) else str(split_path)
                metrics = save_results(out_path, y_test, y_pred, le, y_proba)
                combo_metadata = {"train_users": train_users, "test_users": test_users, **metrics, "metrics": metrics}
                with open(combo_json, "w") as f:
                    json.dump(combo_metadata, f, indent=2)
                all_metrics.append(metrics)
                combo_ids.append(combo_id)
            # Save avg_metrics.json after all combos
            if all_metrics:
                avg_metrics = {}
                keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
                for k in keys:
                    avg_metrics[k] = round(np.mean([m[k] for m in all_metrics]), 4)
                with open(os.path.join(split_path, "avg_metrics.json"), "w") as f:
                    json.dump(avg_metrics, f, indent=4)
                # Save confusion matrix and plots for a random combo
                random.seed(42)
                chosen_idx = random.randint(0, len(combo_ids)-1)
                chosen_combo = combo_ids[chosen_idx]
                chosen_json = os.path.join(split_path, f"{chosen_combo}.json")
                with open(chosen_json, "r") as f:
                    chosen_data = json.load(f)
                chosen_metrics = chosen_data["metrics"]
                pd.DataFrame(chosen_metrics["confusion_matrix"], index=chosen_metrics["labels"], columns=chosen_metrics["labels"]).to_csv(
                    os.path.join(split_path, "confusion_matrix.csv")
                )
