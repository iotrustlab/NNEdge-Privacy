# Updated script with debug prints for file depths and improved RNN
import os, re, glob, json, time, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mutual_info_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------- CONFIG --------
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-3", "RNN")
sequence_length = 100
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
BATCH_SIZE = 32
EPOCHS = 25
os.makedirs(results_root, exist_ok=True)

# -------- HELPERS --------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def extract_full_imu_sequences(df):
    cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
    if not all(c in df.columns for c in cols): return np.array([])
    segments = []
    for i in range(0, len(df) - sequence_length + 1, sequence_length):
        seg = df[cols].iloc[i:i+sequence_length].values
        if seg.shape[0] == sequence_length:
            segments.append(seg)
    return np.array(segments)

def compute_leakage_metrics(y_true, y_pred):
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
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(np.mean([v['f1-score'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "precision": float(np.mean([v['precision'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "recall": float(np.mean([v['recall'] for k, v in report_dict.items() if isinstance(v, dict)])),
        "mutual_information": mi,
        "kl_divergence": kl,
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

def save_outputs(y_true, y_pred, le, path, report_header=""):
    os.makedirs(path, exist_ok=True)
    metrics = compute_leakage_metrics(y_true, y_pred)

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

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    cm_df.to_csv(os.path.join(path, "confusion_matrix.csv"))

# -------- LOAD DATA --------
user_data = {}
all_labels = []

for user in get_user_dirs(data_root):
    print(f"📦 Loading {user}")
    user_path = os.path.join(data_root, user)
    X_user, y_user = [], []
    file_paths = glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)
    print(f"   Found {len(file_paths)} CSV files")

    for file_path in file_paths:
        rel = os.path.relpath(file_path, start=user_path).split(os.sep)
        if "processed" not in [part.lower() for part in rel]:
            continue
        for idx, part in enumerate(rel):
            if part.lower() == "processed" and idx + 1 < len(rel):
                activity = rel[idx + 1].strip().lower()
                break
        else:
            continue

        if activity not in VALID_ACTIVITIES:
            continue

        df = pd.read_csv(file_path)
        if df.isnull().values.any() or len(df) < sequence_length:
            continue

        segs = extract_full_imu_sequences(df)
        if segs.size > 0:
            X_user.extend(segs)
            y_user.extend([activity] * len(segs))
            print(f"   ✔ {file_path} | Segments: {len(segs)}")

    if X_user:
        user_data[user] = (np.array(X_user), np.array(y_user))
        all_labels.extend(y_user)
        print(f"   ➡ Total segments for {user}: {len(X_user)}")
    else:
        print(f"   ⚠️ No valid data found for {user}")

le = LabelEncoder().fit(all_labels)

# -------- CROSS-USER --------
for tr, ts in train_test_ratios:
    split_name = f"train{int(tr*100)}_test{int(ts*100)}"
    split_path = os.path.join(results_root, "cross-user", split_name)
    users = list(user_data.keys())
    n_train = int(tr * len(users))
    train_users, test_users = users[:n_train], users[n_train:]
    if not train_users or not test_users:
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = le.transform(np.hstack([user_data[u][1] for u in test_users]))

    print(f"\n➡️ Cross-User Split: {split_name}")
    print(f"   Train Users: {train_users}")
    print(f"   Test Users: {test_users}")
    print(f"   Train Samples: {X_train.shape[0]} | Test Samples: {X_test.shape[0]}")

    model = build_rnn(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    save_outputs(y_test, y_pred, le, split_path)

# -------- INTRA-USER --------
intra_path = os.path.join(results_root, "intra-user")
os.makedirs(intra_path, exist_ok=True)
all_preds, all_trues = [], []

print("\n➡️ Intra-User Evaluation")
for user in user_data:
    X, y = user_data[user]
    y_enc = le.transform(y)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
        model = build_rnn(X.shape[1:], len(le.classes_))
        model.fit(X_tr, to_categorical(y_tr), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_te), axis=1)
        all_preds.extend(y_pred)
        all_trues.extend(y_te)
        print(f"   {user}: {len(X_tr)} train / {len(X_te)} test samples")
    except Exception as e:
        print(f"⚠️ Intra-user error for {user}: {e}")

if all_preds:
    save_outputs(all_trues, all_preds, le, intra_path)

# -------- LEAVE-ONE-USER-OUT --------
loo_path = os.path.join(results_root, "cross-user", "leave-one-out")
os.makedirs(loo_path, exist_ok=True)
all_preds, all_trues = [], []

print("\n➡️ Leave-One-User-Out Evaluation")
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
    print(f"   Test User: {test_user} | Train Samples: {X_train.shape[0]} | Test Samples: {X_test.shape[0]}")

if all_preds:
    save_outputs(all_trues, all_preds, le, loo_path)