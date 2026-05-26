import os
import re
import glob
import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mutual_info_score
from sklearn.model_selection import train_test_split
from scipy.special import rel_entr
import joblib
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-1", "NB")
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
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = [label_to_index[y] for y in y_true]
    y_pred_idx = [label_to_index[y] for y in y_pred]
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = np.sum(rel_entr(p_true, p_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
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
    csv_files = [f for f in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True) if "Raw" not in f]

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
            print(f"     ❌ Error loading {file_path}: {e}")

    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))

# ---------- TRAINING FUNCTION ----------
def run_and_save(model, scaler, X_test, y_test, split_path, train_users, test_users, y_pred):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    labels_sorted = sorted(list(set(y_test) | set(y_pred)))
    metrics = compute_leakage_metrics(y_test, y_pred, labels_sorted)

    with open(os.path.join(split_path, "report.txt"), "w") as f:
        f.write("=== Experiment Summary ===\\n")
        f.write(f"Train Users: {train_users}\\n")
        f.write(f"Test Users: {test_users}\\n")
        f.write(f"Accuracy: {acc:.4f}\\n\\n")
        f.write(report)

    with open(os.path.join(split_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(metrics["confusion_matrix"], index=labels_sorted, columns=labels_sorted)\
        .to_csv(os.path.join(split_path, "confusion_matrix.csv"))

    joblib.dump(model, os.path.join(split_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(split_path, "scaler.pkl"))

# ---------- CROSS-USER ----------
print("\\n🚀 Running Cross-User Experiments")
for train_ratio, test_ratio in train_test_ratios:
    split_name = f"train{int(train_ratio*100)}_test{int(test_ratio*100)}"
    split_path = os.path.join(results_root, "cross-user", split_name)
    os.makedirs(split_path, exist_ok=True)

    n_train = int(train_ratio * len(user_data))
    train_users = list(user_data.keys())[:n_train]
    test_users = list(user_data.keys())[n_train:]

    if not train_users or not test_users:
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = np.hstack([user_data[u][1] for u in test_users])

    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    run_and_save(model, scaler, X_test, y_test, split_path, train_users, test_users, y_pred)

# ---------- INTRA-USER ----------
print("\\n🚀 Running Intra-User Experiments")
intra_path = os.path.join(results_root, "intra-user")
os.makedirs(intra_path, exist_ok=True)
for user in user_data:
    X, y = user_data[user]
    if len(np.unique(y)) < 2 or len(X) < 2:
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    user_path = os.path.join(intra_path, f"user_{user.replace(' ', '_')}")
    os.makedirs(user_path, exist_ok=True)
    run_and_save(model, scaler, X_test, y_test, user_path, [user], [user], y_pred)

# ---------- LOUO ----------
print("\\n🚀 Running Leave-One-User-Out Experiments")
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test, y_test = user_data[test_user]

    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    user_path = os.path.join(loo_path, f"test_{test_user.replace(' ', '_')}")
    os.makedirs(user_path, exist_ok=True)
    run_and_save(model, scaler, X_test, y_test, user_path, train_users, [test_user], y_pred)

print(f"✅ All experiments completed in {time.time() - start_time:.2f} seconds.")