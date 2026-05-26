# This is your original script with debug statements added for data loading,
# fixed path depth handling, and an improved CNN architecture.
# Now includes saving summary.txt with detailed classification reports.

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, mutual_info_score
from scipy.special import rel_entr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.exceptions import UndefinedMetricWarning

# -------------------- SETUP --------------------
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------- CONFIG --------------------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-2", "CNN")
sequence_length = 100
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
EPOCHS = 30
BATCH_SIZE = 32
MODES = {
    "accel": ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]', 'dec_tree_out_1'],
    "gyro": ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]', 'dec_tree_out_1']
}
os.makedirs(results_root, exist_ok=True)

def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)],
                  key=lambda x: int(re.findall(r"\d+", x)[0]))

def segment_sequences(df, features):
    segments = []
    for i in range(0, len(df) - sequence_length + 1, sequence_length):
        window = df[features].iloc[i:i + sequence_length].values
        if window.shape[0] == sequence_length:
            segments.append(window)
    return np.array(segments)

def compute_metrics(y_true, y_pred, le):
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

    labels = list(map(str, le.classes_))
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mutual_information": round(mi, 4),
        "kl_divergence": round(kl, 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def save_results(base_path, y_true, y_pred, le):
    os.makedirs(base_path, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, le)

    with open(os.path.join(base_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(base_path, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {metrics['precision']}\n")
        f.write(f"Macro Recall: {metrics['recall']}\n")
        f.write(f"Macro F1-score: {metrics['f1_score']}\n")
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']}\n")

    pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]).to_csv(
        os.path.join(base_path, "confusion_matrix.csv")
    )

    # Save full classification report
    summary_report = classification_report(y_true, y_pred, target_names=le.classes_, digits=4, zero_division=0)
    with open(os.path.join(base_path, "summary.txt"), "w") as f:
        f.write(summary_report)

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(256, 3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

for mode, features in MODES.items():
    print(f"\n🔧 Mode: {mode} — using features {features}\n")
    user_data = {}
    all_labels = []

    for user in get_user_dirs(data_root):
        X_user, y_user = [], []
        user_path = os.path.join(data_root, user)
        print(f"📂 Processing {user}")

        file_paths = glob.glob(os.path.join(user_path, '**', '*.csv'), recursive=True)
        file_count = 0
        for file_path in file_paths:
            rel = os.path.relpath(file_path, start=user_path).split(os.sep)
            if "processed" not in [r.lower() for r in rel]:
                continue

            # Find activity name irrespective of depth
            for idx, folder in enumerate(rel):
                if folder.lower() == "processed" and idx + 1 < len(rel):
                    activity = rel[idx + 1].strip().lower()
                    break
            else:
                continue

            if activity not in VALID_ACTIVITIES:
                continue

            df = pd.read_csv(file_path)
            if df[features].isnull().values.any() or len(df) < sequence_length:
                continue

            segments = segment_sequences(df, features)
            if segments.size > 0:
                X_user.extend(segments)
                y_user.extend([activity] * len(segments))
                file_count += 1

        print(f"   ✔ {file_count} valid files | {len(X_user)} segments")

        if X_user:
            user_data[user] = (np.array(X_user), np.array(y_user))
            all_labels.extend(y_user)

    print(f"Total Users Processed: {len(user_data)}")
    if not user_data:
        print(f"⚠️ No data for mode: {mode}")
        continue

    le = LabelEncoder().fit(all_labels)
    mode_root = os.path.join(results_root, mode)
    os.makedirs(mode_root, exist_ok=True)

    # ---- CROSS-USER SPLITS ----
    for tr, ts in train_test_ratios:
        split_name = f"train{int(tr * 100)}_test{int(ts * 100)}"
        split_path = os.path.join(mode_root, "cross-user", split_name)
        os.makedirs(split_path, exist_ok=True)

        keys = list(user_data.keys())
        n_train = int(tr * len(keys))
        train_users, test_users = keys[:n_train], keys[n_train:]
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

        model = build_cnn((X_train.shape[1], X_train.shape[2]), len(le.classes_))
        model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)

        save_results(split_path, y_test, y_pred, le)

    # ---- INTRA-USER ----
    intra_path = os.path.join(mode_root, "intra-user")
    os.makedirs(intra_path, exist_ok=True)
    all_preds, all_trues = [], []

    print("\n➡️ Intra-User Evaluation")
    for user in user_data:
        X, y = user_data[user]
        y_enc = le.transform(y)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
            model = build_cnn((X.shape[1], X.shape[2]), len(le.classes_))
            model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            y_pred = np.argmax(model.predict(X_test), axis=1)
            all_preds.extend(y_pred)
            all_trues.extend(y_test)
            print(f"   {user}: {len(X_train)} train / {len(X_test)} test samples")
        except Exception as e:
            print(f"⚠️ Intra-user error for {user}: {e}")

    if all_preds:
        save_results(intra_path, all_trues, all_preds, le)

    # ---- LEAVE-ONE-USER-OUT ----
    loo_path = os.path.join(mode_root, "cross-user", "leave-one-out")
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

        model = build_cnn((X_train.shape[1], X_train.shape[2]), len(le.classes_))
        model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(model.predict(X_test), axis=1)
        all_preds.extend(y_pred)
        all_trues.extend(y_test)

        print(f"   Test User: {test_user} | Train Samples: {X_train.shape[0]} | Test Samples: {X_test.shape[0]}")

    if all_preds:
        save_results(loo_path, all_trues, all_preds, le)
