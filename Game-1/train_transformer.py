import os
import re
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mutual_info_score, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras.utils import to_categorical
from scipy.special import rel_entr
import tensorflow as tf
import warnings
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Layer


warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- CONFIG ----------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
results_root = os.path.join(data_root, "Results", "Game-1", "Transformer")
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}
WINDOW_SIZE = 100
EPOCHS = 20
BATCH_SIZE = 32
train_test_ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]

os.makedirs(results_root, exist_ok=True)



class PositionalEncoding(Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, pos, i, d_model):
        angles = pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return angles

    def positional_encoding(self, sequence_length, d_model):
        angle_rads = self.get_angles(
            np.arange(sequence_length)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        # Apply sin to even indices; cos to odd indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-Head Self Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward Network
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

def build_transformer(input_shape, num_classes, num_layers=3, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = PositionalEncoding(input_shape[0], input_shape[1])(inputs)

    for _ in range(num_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- HELPERS ----------
def get_user_dirs(base):
    return sorted([d for d in os.listdir(base) if re.match(r"User \d+", d)], key=lambda x: int(re.findall(r"\d+", x)[0]))

def segment_sequences(df, seq_len=100):
    segments = []
    for i in range(0, len(df) - seq_len + 1, seq_len):
        window = df[['dec_tree_out_1']].iloc[i:i+seq_len].values
        if window.shape[0] == seq_len:
            segments.append(window)
    return np.array(segments)

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = mutual_info_score(y_true, y_pred)
    y_true_idx = [labels.index(y) for y in y_true]
    y_pred_idx = [labels.index(y) for y in y_pred]
    p_true = np.bincount(y_true_idx, minlength=len(labels)) / len(y_true)
    p_pred = np.bincount(y_pred_idx, minlength=len(labels)) / len(y_pred)
    p_true += 1e-12
    p_pred += 1e-12
    kl = np.sum(rel_entr(p_true, p_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "mutual_information": float(mi),
        "kl_divergence": float(kl),
        "confusion_matrix": cm.tolist(),
        "labels": [str(l) for l in labels]
    }

def save_results(y_true, y_pred, le, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = list(range(len(le.classes_)))
    class_names = list(le.classes_)
    metrics = compute_metrics(y_true, y_pred, labels)

    # metrics.json
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # summary.txt
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report + "\n")

    # report.txt
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_precision = metrics["precision"]
    weighted_recall = metrics["recall"]
    weighted_f1 = metrics["f1_score"]
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Average KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Average Mutual Information: {metrics['mutual_information']:.4f}\n")
        f.write(f"Average Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n")
        f.write(f"Weighted Precision: {weighted_precision:.4f}\n")
        f.write(f"Weighted Recall: {weighted_recall:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n")

    # confusion_matrix.csv
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names)\
        .to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

# ---------- LOAD DATA ----------
user_data = {}
all_labels = []

for user in get_user_dirs(data_root):
    path = os.path.join(data_root, user)
    print(f"\n📂 Processing {user}")
    X_user, y_user = [], []
    for file_path in glob.glob(os.path.join(path, '**', '*.csv'), recursive=True):
        rel_path = os.path.relpath(file_path, start=path).split(os.sep)

        # Dynamically find "processed" and extract activity
        activity = None
        for idx, part in enumerate(rel_path):
            if part.lower() == "processed" and idx + 1 < len(rel_path):
                activity = rel_path[idx + 1].lower()
                break
        if activity is None or activity not in VALID_ACTIVITIES:
            continue

        print(f"📄 Loading file: {file_path} | Activity: {activity}")

        df = pd.read_csv(file_path)
        if 'dec_tree_out_1' not in df.columns:
            print(f"⚠️ Skipping file (missing 'dec_tree_out_1'): {file_path}")
            continue
        if df['dec_tree_out_1'].isnull().any():
            print(f"⚠️ Skipping file (NaN values found): {file_path}")
            continue
        if len(df) < WINDOW_SIZE:
            print(f"⚠️ Skipping file (too few samples): {file_path}")
            continue

        segs = segment_sequences(df, WINDOW_SIZE)
        if segs.size > 0:
            X_user.extend(segs)
            y_user.extend([activity]*len(segs))

    if X_user:
        print(f"✅ {user}: {len(X_user)} sequences loaded.")
        user_data[user] = (np.array(X_user), np.array(y_user))
        all_labels.extend(y_user)
    else:
        print(f"⚠️ No valid data found for {user}.")

le = LabelEncoder().fit(all_labels)
print(f"\n🏷️ Label classes: {list(le.classes_)}")

# ---------- CROSS-USER ----------
print("\n🚀 Starting CROSS-USER Experiments...")
for ratio in train_test_ratios:
    split_name = f"train{int(ratio[0]*100)}_test{int(ratio[1]*100)}"
    out_dir = os.path.join(results_root, "cross-user", split_name)
    keys = list(user_data.keys())
    n_train = int(ratio[0] * len(keys))
    train_users = keys[:n_train]
    test_users = keys[n_train:]

    print(f"\n🔧 Split: {split_name} | Train Users: {train_users} | Test Users: {test_users}")

    if not train_users or not test_users:
        print(f"⚠️ Skipping split {split_name}: Insufficient users.")
        continue

    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test = np.vstack([user_data[u][0] for u in test_users])
    y_test = le.transform(np.hstack([user_data[u][1] for u in test_users]))

    print(f"🧠 Training Transformer model on {X_train.shape[0]} samples...")
    model = build_transformer(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    print(f"✅ Model trained. Evaluating on {X_test.shape[0]} samples...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    save_results(list(y_test), list(y_pred), le, out_dir)
    print(f"📊 Results saved to {out_dir}")

# ---------- INTRA-USER ----------
print("\n🚀 Starting INTRA-USER Experiments...")
all_true, all_pred = [], []
for user in user_data:
    print(f"\n🔍 Intra-user split for {user}")
    X, y = user_data[user]
    y_enc = le.transform(y)
    if len(X) < 2:
        print(f"⚠️ Skipping {user}: Not enough data.")
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.5, stratify=y_enc, random_state=42)
    print(f"🧠 Training on {X_train.shape[0]} samples...")
    model = build_transformer(X.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    print(f"✅ Model trained. Evaluating on {X_test.shape[0]} samples...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

save_results(all_true, all_pred, le, os.path.join(results_root, "intra-user"))
print(f"📊 Intra-user results saved to {os.path.join(results_root, 'intra-user')}")

# ---------- LOUO ----------
print("\n🚀 Starting LEAVE-ONE-USER-OUT (LOUO) Experiments...")
all_true, all_pred = [], []
for test_user in user_data:
    train_users = [u for u in user_data if u != test_user]
    if not train_users:
        continue
    print(f"\n🔍 LOUO | Test User: {test_user} | Train Users: {train_users}")
    X_train = np.vstack([user_data[u][0] for u in train_users])
    y_train = le.transform(np.hstack([user_data[u][1] for u in train_users]))
    X_test, y_test = user_data[test_user]
    y_test = le.transform(y_test)
    print(f"🧠 Training on {X_train.shape[0]} samples...")
    model = build_transformer(X_train.shape[1:], len(le.classes_))
    model.fit(X_train, to_categorical(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    print(f"✅ Model trained. Evaluating on {X_test.shape[0]} samples...")
    y_pred = np.argmax(model.predict(X_test), axis=1)
    all_true.extend(y_test)
    all_pred.extend(y_pred)

louo_dir = os.path.join(results_root, "cross-user", "leave-one-out")
save_results(all_true, all_pred, le, louo_dir)
print(f"📊 LOUO results saved to {louo_dir}")