import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, mutual_info_score
)
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-1", "DT")  # DT = Decision Tree
window_size = 20
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
os.makedirs(results_root, exist_ok=True)

# ---------- HELPERS ----------
def extract_features(df):
    feats = []
    for i in range(0, len(df) - window_size + 1, window_size):
        win = df[['dec_tree_out_1']].iloc[i:i + window_size]
        stats = win.agg(['mean', 'std', 'min', 'max']).values.flatten()
        feats.append(stats)
    return np.array(feats)

def get_user_dirs(base):
    dirs = [d for d in os.listdir(base) if re.match(r"User \d+", d)]
    return sorted(dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

def compute_leakage_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = mutual_info_score(y_true, y_pred)
    label_to_index = {label: i for i, label in enumerate(labels)}
    y_true_idx = [label_to_index[y] for y in y_true]
    y_pred_idx = [label_to_index[y] for y in y_pred]
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = np.sum(rel_entr(p_true, p_pred))
    return {
        "mutual_information": float(mi),
        "kl_divergence": float(kl),
        "confusion_matrix": cm.tolist()
    }

# ---------- LOAD DATA ----------
start_time = time.time()
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

            df = pd.read_csv(file_path)
            if 'dec_tree_out_1' not in df.columns or df['dec_tree_out_1'].isnull().any():
                continue
            if len(df) < window_size:
                continue

            feats = extract_features(df)
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

# ---------- CROSS-USER ----------
for train_ratio, test_ratio in train_test_ratios:
    split_name = f"train{int(train_ratio*100)}_test{int(test_ratio*100)}"
    split_path = os.path.join(results_root, "cross-user", split_name)
    os.makedirs(split_path, exist_ok=True)

    n_train = int(train_ratio * len(user_data))
    train_users = list(user_data.keys())[:n_train]
    test_users = list(user_data.keys())[n_train:]

    if not train_users or not test_users:
        print(f"⚠️ Skipping {split_name}: insufficient users")
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = np.hstack([user_data[u][1] for u in test_users])

    scaler = StandardScaler().fit(X_train)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    with open(os.path.join(split_path, "report.txt"), "w") as f:
        f.write(f"=== Experiment Summary ===\n")
        f.write(f"Train Users: {train_users}\nTest Users: {test_users}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    joblib.dump(model, os.path.join(split_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(split_path, "scaler.pkl"))

    labels_sorted = sorted(list(set(y_train) | set(y_test)))
    metrics = compute_leakage_metrics(y_test, y_pred, labels_sorted)
    with open(os.path.join(split_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(metrics["confusion_matrix"], index=labels_sorted, columns=labels_sorted).to_csv(
        os.path.join(split_path, "confusion_matrix.csv"))

# ---------- INTRA-USER ----------
intra_path = os.path.join(results_root, "intra-user")
os.makedirs(intra_path, exist_ok=True)
all_preds, all_trues = [], []

for user in user_data:
    X, y = user_data[user]
    if len(X) < 2:
        continue
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
        scaler = StandardScaler().fit(X_train)
        model = DecisionTreeClassifier(random_state=42)
        model.fit(scaler.transform(X_train), y_train)
        y_pred = model.predict(scaler.transform(X_test))

        all_preds.extend(y_pred)
        all_trues.extend(y_test)

        report_user = classification_report(y_test, y_pred, digits=4, zero_division=0)
        with open(os.path.join(intra_path, f"report_{user}.txt"), "w") as f:
            f.write(f"User: {user}\n\n{report_user}")
    except Exception as e:
        print(f"  ⚠️ Intra-user error for {user}: {e}")

if all_preds and all_trues:
    acc_total = accuracy_score(all_trues, all_preds)
    report_all = classification_report(all_trues, all_preds, digits=4, zero_division=0)
    with open(os.path.join(intra_path, "summary.txt"), "w") as f:
        f.write(f"Intra-user Combined Report\n")
        f.write(f"Accuracy: {acc_total:.4f}\n\n{report_all}")
    labels_sorted = sorted(list(set(all_trues) | set(all_preds)))
    metrics = compute_leakage_metrics(all_trues, all_preds, labels_sorted)
    with open(os.path.join(intra_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(metrics["confusion_matrix"], index=labels_sorted, columns=labels_sorted).to_csv(
        os.path.join(intra_path, "confusion_matrix.csv"))

# ---------- LEAVE-ONE-USER-OUT ----------
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
all_preds, all_trues = [], []

for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    if not train_users:
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test, y_test = user_data[test_user]

    scaler = StandardScaler().fit(X_train)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    with open(os.path.join(loo_path, f"report_{test_user}.txt"), "w") as f:
        f.write(f"Leave-One-Out Evaluation\nTest User: {test_user}\nTrain Users: {train_users}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

    all_preds.extend(y_pred)
    all_trues.extend(y_test)

if all_preds and all_trues:
    acc_loo = accuracy_score(all_trues, all_preds)
    report = classification_report(all_trues, all_preds, digits=4, zero_division=0)
    with open(os.path.join(loo_path, "summary.txt"), "w") as f:
        f.write("Leave-One-Out Combined Summary\n")
        f.write(f"Accuracy: {acc_loo:.4f}\n\n{report}")
    labels_sorted = sorted(list(set(all_trues) | set(all_preds)))
    metrics = compute_leakage_metrics(all_trues, all_preds, labels_sorted)
    with open(os.path.join(loo_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(metrics["confusion_matrix"], index=labels_sorted, columns=labels_sorted).to_csv(
        os.path.join(loo_path, "confusion_matrix.csv"))

print(f"✅ Decision Tree training for Game-1 completed in {time.time() - start_time:.2f} seconds.")
