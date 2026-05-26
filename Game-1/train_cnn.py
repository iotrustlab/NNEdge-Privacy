import os
import re
import glob
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mutual_info_score
from scipy.special import rel_entr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-1", "CNN")
window_size = 100
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
EPOCHS = 20
BATCH_SIZE = 32
os.makedirs(results_root, exist_ok=True)

# ---------- HELPERS ----------
def extract_sequences(df):
    return np.array([
        df[['dec_tree_out_1']].iloc[i:i + window_size].values
        for i in range(0, len(df) - window_size + 1, window_size)
        if df[['dec_tree_out_1']].iloc[i:i + window_size].shape[0] == window_size
    ])

def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    mi = mutual_info_score(y_true, y_pred)
    p_true = np.bincount(y_true, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = float(np.sum(rel_entr(p_true, p_pred)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": np.mean([v["f1-score"] for k, v in report.items() if isinstance(v, dict)]),
        "precision": np.mean([v["precision"] for k, v in report.items() if isinstance(v, dict)]),
        "recall": np.mean([v["recall"] for k, v in report.items() if isinstance(v, dict)]),
        "mutual_information": mi,
        "kl_divergence": kl,
        "confusion_matrix": cm.tolist(),
        "labels": list(labels)
    }, report, cm

def save_outputs(path, metrics, report_dict, cm, le):
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(os.path.join(path, "confusion_matrix.csv"))
    with open(os.path.join(path, "report.txt"), "w") as f:
        f.write(f"Average KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Average Mutual Information: {metrics['mutual_information']:.4f}\n")
        f.write(f"Average Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall']:.4f}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"Weighted Precision: {report_dict['weighted avg']['precision']:.4f}\n")
        f.write(f"Weighted Recall: {report_dict['weighted avg']['recall']:.4f}\n")
        f.write(f"Weighted F1-score: {report_dict['weighted avg']['f1-score']:.4f}\n")
    with open(os.path.join(path, "summary.txt"), "w") as f:
        f.write(classification_report(
            list(range(len(le.classes_))),
            list(range(len(le.classes_))),
            target_names=le.classes_,
            digits=4
        ))



def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ---------- LOAD DATA ----------
user_data = {}
for user in get_user_dirs(data_root):
    X_user, y_user = [], []
    user_path = os.path.join(data_root, user)
    csv_files = [f for f in glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True) if "Raw" not in f]
    for file_path in csv_files:
        rel_path = os.path.relpath(file_path, start=user_path)
        parts = rel_path.split(os.sep)
        if len(parts) >= 3 and parts[0].lower() == "processed":
            activity = parts[1].strip().lower()
            if activity not in VALID_ACTIVITIES:
                continue
            df = pd.read_csv(file_path)
            if 'dec_tree_out_1' not in df.columns or df['dec_tree_out_1'].isnull().any() or len(df) < window_size:
                continue
            feats = extract_sequences(df)
            X_user.extend(feats)
            y_user.extend([activity] * len(feats))
    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))

# ---------- EVALUATION ----------
for ratio in train_test_ratios:
    trn, tst = int(ratio[0] * 100), int(ratio[1] * 100)
    split_path = os.path.join(results_root, "cross-user", f"train{trn}_test{tst}")
    os.makedirs(split_path, exist_ok=True)
    users = list(user_data.keys())
    train_users, test_users = users[:int(ratio[0] * len(users))], users[int(ratio[0] * len(users)):]
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

    model = build_cnn((window_size, 1), len(le.classes_))
    model.fit(X_train, to_categorical(y_train_enc), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    metrics, report_dict, cm = compute_metrics(y_test_enc, y_pred, le.classes_)
    save_outputs(split_path, metrics, report_dict, cm, le)

# ---------- LOUO ----------
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
all_preds, all_trues = [], []

for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = np.hstack([user_data[u][1] for u in train_users])
    X_test, y_test = user_data[test_user]

    scaler = StandardScaler().fit(X_train.reshape(-1, 1))
    X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    model = build_cnn((window_size, 1), len(le.classes_))
    model.fit(X_train, to_categorical(y_train_enc), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_preds.extend(y_pred)
    all_trues.extend(y_test_enc)

metrics, report_dict, cm = compute_metrics(np.array(all_trues), np.array(all_preds), le.classes_)
save_outputs(loo_path, metrics, report_dict, cm, le)

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
        scaler = StandardScaler().fit(X_train.reshape(-1, 1))
        X_train = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)

        model = build_cnn((window_size, 1), len(le.classes_))
        model.fit(X_train, to_categorical(y_train_enc), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        all_preds.extend(y_pred)
        all_trues.extend(y_test_enc)
    except Exception as e:
        print(f"⚠️ Error on user {user}: {e}")

metrics, report_dict, cm = compute_metrics(np.array(all_trues), np.array(all_preds), le.classes_)
save_outputs(intra_path, metrics, report_dict, cm, le)
