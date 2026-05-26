# Naive Bayes Script with Fixed File Handling and Complete Result Saving
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-3", "NB")
window_size = 20
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
os.makedirs(results_root, exist_ok=True)

# ---------- HELPERS ----------
def extract_features_full(df):
    required_cols = [
        'acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
        'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]',
        'dec_tree_out_1'
    ]
    if not all(c in df.columns for c in required_cols):
        return np.empty((0, 28))
    feats = []
    for i in range(0, len(df) - window_size + 1, window_size):
        window = df[required_cols].iloc[i:i + window_size]
        stats = window.agg(['mean', 'std', 'min', 'max']).values.flatten()
        feats.append(stats)
    return np.array(feats)

def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)],
                  key=lambda x: int(re.findall(r"\d+", x)[0]))

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
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "mutual_information": round(float(mi), 4),
        "kl_divergence": round(float(kl), 4),
        "confusion_matrix": cm.tolist()
    }

def save_results(base_path, y_true, y_pred, labels):
    os.makedirs(base_path, exist_ok=True)
    metrics = compute_leakage_metrics(y_true, y_pred, labels)

    # Save metrics.json
    with open(os.path.join(base_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix CSV
    pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels).to_csv(
        os.path.join(base_path, "confusion_matrix.csv")
    )

    # Save summary.txt (full classification report)
    summary_report = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    with open(os.path.join(base_path, "summary.txt"), "w") as f:
        f.write(summary_report)

    # Save report.txt (Macro metrics & leakage)
    report_dict = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
    macro = report_dict.get('macro avg', {})
    with open(os.path.join(base_path, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {macro.get('precision', 0):.4f}\n")
        f.write(f"Macro Recall: {macro.get('recall', 0):.4f}\n")
        f.write(f"Macro F1-score: {macro.get('f1-score', 0):.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']}\n")

# ---------- LOAD DATA ----------
print("🔍 Loading user directories...")
user_dirs = get_user_dirs(data_root)
user_data = {}

for user in user_dirs:
    print(f"\n👤 Processing {user}")
    X_user, y_user = [], []
    user_path = os.path.join(data_root, user)
    file_paths = glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)

    for file_path in file_paths:
        rel = os.path.relpath(file_path, start=user_path).split(os.sep)
        if "processed" not in [r.lower() for r in rel]:
            print(f"   ⚠️ Skipping file (no 'processed' folder): {file_path}")
            continue

        for idx, folder in enumerate(rel):
            if folder.lower() == "processed" and idx + 1 < len(rel):
                activity = rel[idx + 1].strip().lower()
                break
        else:
            print(f"   ⚠️ Skipping file (no activity folder after 'processed'): {file_path}")
            continue

        if activity not in VALID_ACTIVITIES:
            print(f"   ⚠️ Skipping file (invalid activity '{activity}'): {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"   ⚠️ Skipping file (read error): {file_path} | Error: {e}")
            continue

        if df.isnull().any().any():
            print(f"   ⚠️ Skipping file (NaN values found): {file_path}")
            continue

        if len(df) < window_size:
            print(f"   ⚠️ Skipping file (not enough samples, found {len(df)}): {file_path}")
            continue

        feats = extract_features_full(df)
        if feats.size == 0:
            print(f"   ⚠️ Skipping file (feature extraction failed): {file_path}")
            continue

        X_user.extend(feats)
        y_user.extend([activity] * len(feats))

    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))
        print(f"   ✔ {user}: {len(X_user)} samples")
    else:
        print(f"   ⚠️ {user}: no usable data")

# ---------- CROSS-USER ----------
for train_ratio, test_ratio in train_test_ratios:
    split_name = f"train{int(train_ratio*100)}_test{int(test_ratio*100)}"
    split_path = os.path.join(results_root, "cross-user", split_name)
    os.makedirs(split_path, exist_ok=True)

    n_train = int(train_ratio * len(user_data))
    train_users = list(user_data.keys())[:n_train]
    test_users = list(user_data.keys())[n_train:]

    if not train_users or not test_users:
        print(f"⚠️ Skipping {split_name}: not enough users to split")
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = np.hstack([user_data[u][1] for u in test_users])

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print(f"⚠️ Skipping {split_name}: label diversity too low")
        continue

    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    labels_sorted = sorted(list(set(y_train) | set(y_test)))
    save_results(split_path, y_test, y_pred, labels_sorted)
    joblib.dump(model, os.path.join(split_path, "model.pkl"))
    joblib.dump(scaler, os.path.join(split_path, "scaler.pkl"))

# ---------- INTRA-USER ----------
intra_path = os.path.join(results_root, "intra-user")
os.makedirs(intra_path, exist_ok=True)
all_preds, all_trues = [], []

for user in user_data:
    X, y = user_data[user]
    if len(X) < 2:
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    all_preds.extend(y_pred)
    all_trues.extend(y_test)

if all_preds and all_trues:
    labels_sorted = sorted(list(set(all_trues) | set(all_preds)))
    save_results(intra_path, all_trues, all_preds, labels_sorted)

# ---------- LEAVE-ONE-USER-OUT ----------
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
all_preds, all_trues = [], []

for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test, y_test = user_data[test_user]

    scaler = StandardScaler().fit(X_train)
    model = GaussianNB().fit(scaler.transform(X_train), y_train)
    y_pred = model.predict(scaler.transform(X_test))

    all_preds.extend(y_pred)
    all_trues.extend(y_test)

if all_preds and all_trues:
    labels_sorted = sorted(list(set(all_trues) | set(all_preds)))
    save_results(loo_path, all_trues, all_preds, labels_sorted)

print(f"\n✅ Training completed")
