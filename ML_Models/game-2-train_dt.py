"""
Game-2 Decision Tree Training Script for Cross-User HAR Privacy Analysis

Modified Version Features:
1. Randomly selects only 5 combinations per window size and split configuration
2. Includes class-independent information-theoretic metrics:
   - NMI (Normalized Mutual Information) Percentage: Class-independent mutual information
   - Reverse KL Percentage: Class-independent calibration metric (higher = better)

These metrics enable fair comparison across datasets with different numbers of activity classes.
"""

# ---------- IMPORTS ----------
import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mutual_info_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.special import rel_entr
from _paths import (
    get_data_root,
    get_max_combinations,
    get_results_root,
    get_split_configs,
    get_window_sizes,
    load_experiment_csv,
)
from experiment_metrics import compute_vulnerability

# ---------- CONFIG ----------
data_root = get_data_root()
window_sizes = get_window_sizes([20, 50, 70, 100])
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# ---------- HELPERS ----------
def extract_features(df, window_size, modality):
    if modality == "accel":
        cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1']
    elif modality == "gyro":
        cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
    else:
        raise ValueError("Unknown modality")
    feats = []
    for i in range(0, len(df) - window_size + 1, window_size):
        win = df[cols].iloc[i:i + window_size]
        stats = win.agg(['mean', 'std', 'min', 'max']).values.flatten()
        feats.append(stats)
    return np.array(feats)

def get_user_dirs(base):
    dirs = [d for d in os.listdir(base) if re.match(r"User \d+", d)]
    return sorted(dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

def compute_leakage_metrics(y_true, y_pred, labels):
    """Compute class-independent information-theoretic metrics for privacy analysis."""
    from sklearn.metrics import confusion_matrix, mutual_info_score
    from scipy.special import rel_entr
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Raw mutual information (class-dependent)
    mi_raw = mutual_info_score(y_true, y_pred)
    
    # Calculate entropies for normalization
    def entropy(labels):
        """Calculate entropy of a label distribution."""
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log2(probs + 1e-12))
    
    h_true = entropy(y_true)
    h_pred = entropy(y_pred)
    
    # Normalized Mutual Information (NMI) - class-independent
    nmi = mi_raw / np.sqrt(h_true * h_pred + 1e-12)
    nmi_percentage = nmi * 100
    
    # For KL divergence calculation, we need to handle predicted probabilities
    # Since we only have hard predictions, we'll use the empirical distributions
    label_to_index = {label: i for i, label in enumerate(labels)}
    y_true_idx = [label_to_index[y] for y in y_true]
    y_pred_idx = [label_to_index[y] for y in y_pred]
    
    # Calculate empirical distributions
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    
    # Add small epsilon to avoid log(0)
    p_true += 1e-12
    p_pred += 1e-12
    
    # Raw KL divergence (class-dependent)
    kl_raw = np.sum(rel_entr(p_true, p_pred))
    
    # Normalized KL divergence (class-independent)
    # Theoretical maximum is log(C) where C is number of classes
    kl_max = np.log(len(labels))
    kl_normalized = min(1.0, kl_raw / kl_max)
    
    # Reverse KL percentage (higher is better, like accuracy)
    reverse_kl_percentage = (1 - kl_normalized) * 100
    
    return {
        "mutual_information_raw": float(mi_raw),
        "mutual_information_nmi_percentage": float(nmi_percentage),
        "kl_divergence_raw": float(kl_raw),
        "kl_divergence_reverse_percentage": float(reverse_kl_percentage),
        "confusion_matrix": cm.tolist()
    }

import pickle
import itertools
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- MULTI-WINDOW & MODALITY EXPERIMENT ----------
for modality in ["accel", "gyro"]:
    for window_size in window_sizes:
        print(f"\n===== MODALITY: {modality.upper()} | WINDOW SIZE: {window_size} =====")
        results_root = get_results_root("Game-2", "DT", modality, f"Window_{window_size}")
        os.makedirs(results_root, exist_ok=True)
        cache_path = os.path.join(results_root, "user_data_decision_tree.pkl")
        split_configs = get_split_configs([(8,3), (6,5), (4,7), (1,10)])
        split_complete = True
        for n_train, n_test in split_configs:
            split_name = f"split{n_train}_{n_test}"
            split_path = os.path.join(results_root, split_name)
            avg_metrics_path = os.path.join(split_path, "avg_metrics.json")
            if not os.path.exists(avg_metrics_path):
                split_complete = False
                break
        if split_complete and os.path.exists(cache_path):
            print(f"✅ Modality {modality} Window {window_size} already complete. Skipping.")
            continue
        start_time = time.time()
        # Load or build user data cache
        if os.path.exists(cache_path):
            print(f"🔄 Loading cached user data from {cache_path}")
            with open(cache_path, "rb") as f:
                user_data = pickle.load(f)
            user_dirs = list(user_data.keys())
        else:
            print("🔍 Loading user directories...")
            user_dirs = get_user_dirs(data_root)
            user_data = {}
            for user in user_dirs:
                print(f"👤 Inspecting {user}")
                X_user, y_user = [], []
                user_path = os.path.join(data_root, user)
                csv_files = [f for f in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)
                             if "Raw" not in f]
                for file_path in csv_files:
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
                        if modality == "accel" and not all(col in df.columns for col in ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']):
                            continue
                        if modality == "gyro" and not all(col in df.columns for col in ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']):
                            continue
                        if len(df) < window_size:
                            continue
                        feats = extract_features(df, window_size, modality)
                        if feats.size == 0:
                            continue
                        X_user.extend(feats)
                        y_user.extend([activity] * len(feats))
                    except Exception as e:
                        print(f"  ❌ Error: {file_path} -> {e}")
                if X_user:
                    user_data[user] = (np.array(X_user), np.array(y_user))
                    print(f"  ✅ Loaded {len(X_user)} samples")
                else:
                    print(f"  ⚠️ No usable data for {user}")
            # Save cache
            with open(cache_path, "wb") as f:
                pickle.dump(user_data, f)

        # ---------- COMBINATORIAL SPLITS ----------
        user_list = list(user_data.keys())
        for n_train, n_test in split_configs:
            split_name = f"split{n_train}_{n_test}"
            split_path = os.path.join(results_root, split_name)
            os.makedirs(split_path, exist_ok=True)
            
            # Generate all possible combinations
            all_combos = list(itertools.combinations(user_list, n_train))
            
            # Randomly select up to the requested number of combinations
            random.seed(42)  # For reproducibility
            max_combos = min(get_max_combinations(2), len(all_combos))
            selected_combos = random.sample(all_combos, max_combos)
            
            print(f"  📊 Split {split_name}: Testing {max_combos} random combinations out of {len(all_combos)} possible")
            
            metrics_list = []
            confusion_matrix_saved = False
            
            for idx, train_users in enumerate(selected_combos):
                test_users = [u for u in user_list if u not in train_users]
                if len(test_users) != n_test:
                    continue
                
                train_str = "_".join(sorted(train_users))
                test_str = "_".join(sorted(test_users))
                combo_file = os.path.join(split_path, f"combo_{train_str}__{test_str}.json")
                
                if os.path.exists(combo_file):
                    # Load existing results
                    with open(combo_file, "r") as f:
                        existing_metrics = json.load(f)
                        # Extract the metrics we need for averaging
                        metrics_list.append({
                            "accuracy": existing_metrics["accuracy"],
                            "precision": existing_metrics["precision"],
                            "recall": existing_metrics["recall"],
                            "f1_score": existing_metrics["f1_score"],
                            "mutual_information_nmi_percentage": existing_metrics.get("mutual_information_nmi_percentage", 0),
                            "kl_divergence_reverse_percentage": existing_metrics.get("kl_divergence_reverse_percentage", 0),
                            "vulnerability": existing_metrics.get("vulnerability", 0),
                        })
                    continue
                
                X_train = np.vstack([user_data[u][0] for u in train_users])
                y_train = np.hstack([user_data[u][1] for u in train_users])
                X_test = np.vstack([user_data[u][0] for u in test_users])
                y_test = np.hstack([user_data[u][1] for u in test_users])
                scaler = StandardScaler().fit(X_train)
                model = DecisionTreeClassifier(random_state=42)
                model.fit(scaler.transform(X_train), y_train)
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                vulnerability = compute_vulnerability(model.predict_proba(X_test_scaled))
                
                # Calculate traditional metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Calculate class-independent information-theoretic metrics
                labels_sorted = sorted(list(set(y_train) | set(y_test)))
                leakage_metrics = compute_leakage_metrics(y_test, y_pred, labels_sorted)
                
                # Store metrics for averaging
                metrics_list.append({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "mutual_information_nmi_percentage": leakage_metrics["mutual_information_nmi_percentage"],
                    "kl_divergence_reverse_percentage": leakage_metrics["kl_divergence_reverse_percentage"],
                    "vulnerability": vulnerability,
                })
                
                # Save detailed results for this combo
                combo_results = {
                    "train_users": list(train_users),
                    "test_users": test_users,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "mutual_information_raw": leakage_metrics["mutual_information_raw"],
                    "mutual_information_nmi_percentage": leakage_metrics["mutual_information_nmi_percentage"],
                    "kl_divergence_raw": leakage_metrics["kl_divergence_raw"],
                    "kl_divergence_reverse_percentage": leakage_metrics["kl_divergence_reverse_percentage"],
                    "vulnerability": vulnerability,
                    "num_classes": len(labels_sorted),
                    "classes": labels_sorted
                }
                
                with open(combo_file, "w") as f:
                    json.dump(combo_results, f, indent=4)
                
                if idx == 0 and not confusion_matrix_saved:
                    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
                    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
                        os.path.join(split_path, "confusion_matrix.csv"))
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_sorted, yticklabels=labels_sorted)
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix ({split_name})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(split_path, "confusion_matrix.png"))
                    plt.close()
                    confusion_matrix_saved = True
            
            # Save average metrics including new class-independent metrics
            if metrics_list:
                avg_metrics = {
                    "accuracy": float(np.mean([m["accuracy"] for m in metrics_list])),
                    "precision": float(np.mean([m["precision"] for m in metrics_list])),
                    "recall": float(np.mean([m["recall"] for m in metrics_list])),
                    "f1_score": float(np.mean([m["f1_score"] for m in metrics_list])),
                    "mutual_information_nmi_percentage": float(np.mean([m["mutual_information_nmi_percentage"] for m in metrics_list])),
                    "kl_divergence_reverse_percentage": float(np.mean([m["kl_divergence_reverse_percentage"] for m in metrics_list])),
                    "vulnerability": float(np.mean([m["vulnerability"] for m in metrics_list])),
                    "num_combinations_tested": len(metrics_list),
                    "total_combinations_possible": len(all_combos)
                }
                with open(os.path.join(split_path, "avg_metrics.json"), "w") as f:
                    json.dump(avg_metrics, f, indent=4)
        print(f"✅ Decision Tree training for modality {modality} window size {window_size} completed in {time.time() - start_time:.2f} seconds.")
