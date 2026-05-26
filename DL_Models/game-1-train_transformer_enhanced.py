# --- GAME-1 TRANSFORMER WITH CLASS-INDEPENDENT METRICS & 2 RANDOM COMBINATIONS ---
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mutual_info_score, precision_score, recall_score, f1_score
from scipy.special import rel_entr
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Flatten
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

def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ TensorFlow GPU devices found: {gpus}")
else:
    print("⚠️ No GPU found for TensorFlow. Training will use CPU.")

# ---------- CONFIG ----------
data_root = get_data_root()
window_sizes = get_window_sizes([20, 50, 70, 100])
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
EPOCHS = 20
BATCH_SIZE = 32

def extract_sequences(df, window_size):
    return np.array([
        df[['dec_tree_out_1']].iloc[i:i + window_size].values
        for i in range(0, len(df) - window_size + 1, window_size)
        if df[['dec_tree_out_1']].iloc[i:i + window_size].shape[0] == window_size
    ])

def compute_metrics(y_true, y_pred, labels):
    """Compute metrics including class-independent information-theoretic measures."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    mi = mutual_info_score(y_true, y_pred)
    p_true = np.bincount(y_true, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = float(np.sum(rel_entr(p_true, p_pred)))
    
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
    reverse_kl = float(np.sum(rel_entr(p_pred, p_true)))
    
    # Normalized reverse KL (class-independent)
    kl_max = np.log(len(labels))
    reverse_kl_normalized = min(1.0, reverse_kl / kl_max)
    reverse_kl_percentage = (1 - reverse_kl_normalized) * 100
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": np.mean([v["f1-score"] for k, v in report.items() if isinstance(v, dict)]),
        "precision": np.mean([v["precision"] for k, v in report.items() if isinstance(v, dict)]),
        "recall": np.mean([v["recall"] for k, v in report.items() if isinstance(v, dict)]),
        "mutual_information": mi,
        "mutual_information_nmi_percentage": float(nmi_percentage),
        "kl_divergence": kl,
        "kl_divergence_reverse_percentage": float(reverse_kl_percentage),
        "confusion_matrix": cm.tolist(),
        "labels": list(labels)
    }, report, cm

def save_outputs(path, metrics, report_dict, cm, le):
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(os.path.join(path, "confusion_matrix.csv"))
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix.png"))
    plt.close()
    with open(os.path.join(path, "report.txt"), "w") as f:
        f.write(f"Average KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Average Mutual Information: {metrics['mutual_information']:.4f}\n")
        f.write(f"NMI Percentage: {metrics['mutual_information_nmi_percentage']:.4f}%\n")
        f.write(f"Reverse KL Percentage: {metrics['kl_divergence_reverse_percentage']:.4f}%\n")
        f.write(f"Average Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall']:.4f}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"Weighted Precision: {report_dict['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted Recall: {report_dict['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted F1-score: {report_dict['weighted avg']['f1-score']:.4f}\n")

def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    attention = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size, dropout=dropout
    )(inputs, inputs)
    attention = Dropout(dropout)(attention)
    
    # Add & Norm
    attention = Add()([inputs, attention])
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Feed-forward network
    ff = Dense(ff_dim, activation="relu")(attention)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    
    # Add & Norm
    output = Add()([attention, ff])
    output = LayerNormalization(epsilon=1e-6)(output)
    
    return output

def build_transformer(input_shape, num_classes, head_size=32, num_heads=4, ff_dim=64, num_transformer_blocks=2, dropout=0.1):
    inputs = Input(shape=input_shape)
    
    # Apply transformer blocks
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)
    
    # Global average pooling and classification
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# ---------- LOAD DATA WITH CACHING ----------
# Load all user data once and cache
user_cache_path = os.path.join(get_results_root("Game-1", "Transformer"), "user_data_all.pkl")
# Ensure cache directory exists before saving
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
                if 'dec_tree_out_1' not in df.columns or df['dec_tree_out_1'].isnull().any():
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

# ---------- MULTI-WINDOW EXPERIMENT ----------
for window_size in window_sizes:
    print(f"\n===== WINDOW SIZE: {window_size} =====")
    results_root = get_results_root("Game-1", "Transformer", f"Window_{window_size}")
    os.makedirs(results_root, exist_ok=True)
    split_configs = get_split_configs([(8,3), (6,5), (4,7), (1,10)])
    for n_train, n_test in split_configs:
        split_name = f"split{n_train}_{n_test}"
        split_path = os.path.join(results_root, split_name)
        os.makedirs(split_path, exist_ok=True)
        user_list = list(all_user_data.keys())
        combos = list(itertools.combinations(user_list, n_train))
        max_combinations = get_max_combinations(2)
        combo_files = [f for f in os.listdir(split_path) if f.startswith("combo_") and f.endswith(".json")]
        if len(combo_files) >= max_combinations:
            print(f"✅ {max_combinations} combos already done for {split_name} (window {window_size}), skipping.")
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
            # Build user data for this window size from cache
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
            model = build_transformer((window_size, 1), len(le.classes_))
            model.fit(X_train, to_categorical(y_train_enc), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_enc, y_pred, average='weighted', zero_division=0)
            metrics, report_dict, cm = compute_metrics(y_test_enc, y_pred, le.classes_)
            metrics_list.append({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "mutual_information_nmi_percentage": metrics["mutual_information_nmi_percentage"],
                "kl_divergence_reverse_percentage": metrics["kl_divergence_reverse_percentage"]
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
                    "kl_divergence_reverse_percentage": metrics["kl_divergence_reverse_percentage"]
                }, f, indent=4)
            if idx == random_idx and not confusion_matrix_saved:
                save_outputs(split_path, metrics, report_dict, cm, le)
                confusion_matrix_saved = True
        if metrics_list:
            avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}
            with open(os.path.join(split_path, "avg_metrics.json"), "w") as f:
                json.dump(avg_metrics, f, indent=4)
    print(f"✅ Transformer training for window size {window_size} completed.")

print(f"🎉 Game-1 Transformer training completed with class-independent metrics!")
