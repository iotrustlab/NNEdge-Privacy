def coral_transform(Xs, Xt):
    # CORAL: Align second-order statistics (covariance)
    # Xs: source (train), Xt: target (test)
    cov_src = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6
    cov_tar = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-6
    U_src, S_src, _ = np.linalg.svd(cov_src)
    U_tar, S_tar, _ = np.linalg.svd(cov_tar)
    cov_src_inv_sqrt = U_src @ np.diag(1.0 / np.sqrt(S_src)) @ U_src.T
    cov_tar_sqrt = U_tar @ np.diag(np.sqrt(S_tar)) @ U_tar.T
    Xs_coral = (Xs - Xs.mean(axis=0)) @ cov_src_inv_sqrt @ cov_tar_sqrt + Xt.mean(axis=0)
    return Xs_coral
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "Results"
CACHE_PAMAP = "Cache/pamap2_cache.pkl"
CACHE_UCI = "Cache/uci_har_cache.pkl"
CACHE_STM = "Cache/stm_modality_pred_cache.pkl"
ACTIVITY_CLASSES = [
    "Upstairs", "Downstairs", "Walking", "Sitting", "Standing", "Laying", "Jogging"
]

def activity_map(label):
    mapping = {
        "ascending_stairs": "Upstairs", "WALKING_UPSTAIRS": "Upstairs", "Upstairs": "Upstairs",
        "descending_stairs": "Downstairs", "WALKING_DOWNSTAIRS": "Downstairs", "Downstairs": "Downstairs",
        "walking": "Walking", "WALKING": "Walking", "Walking": "Walking", "nordic_walking": "Walking", "vacuum_cleaning": "Walking",
        "sitting": "Sitting", "SITTING": "Sitting", "Sitting": "Sitting",
        "standing": "Standing", "STANDING": "Standing", "Standing": "Standing",
        "lying": "Laying", "LAYING": "Laying", "Laying": "Laying",
        "running": "Jogging", "Jogging": "Jogging", "rope_jumping": "Jogging", "cycling": "Jogging",
    }
    return mapping.get(label, label)

def get_raw_window(meta, force_modality=None):
    df = pd.read_csv(meta["file"])
    modality = force_modality if force_modality is not None else meta["modality"]
    if modality == "accel":
        accel_cols = [c for c in df.columns if 'accel' in c.lower() or 'acc' in c.lower()]
        data = df[accel_cols[:3]].values
        if "STMDATASET" in meta["file"]:
            data = data * 0.00981
    else:
        gyro_cols = [c for c in df.columns if 'gyro' in c.lower()]
        data = df[gyro_cols[:3]].values
        if "STMDATASET" in meta["file"]:
            data = data * 0.001 * (np.pi/180)
    return data[meta["start"]:meta["end"]]

def load_raw_windows(cache_path, n_max=None):
    print(f"Loading raw windows from {cache_path} ...")
    with open(cache_path, "rb") as f:
        items = pickle.load(f)
    features, modality_labels, placement_labels, activity_labels, window_metadata = items
    X_raw, y_act = [], []
    max_len_accel = 0
    max_len_gyro = 0
    temp_windows = []
    for i, meta in enumerate(window_metadata):
        if n_max is not None and i >= n_max:
            break
        act = activity_map(activity_labels[i])
        if (act in ACTIVITY_CLASSES) and (act != 'ironing'):
            raw_win_accel = get_raw_window(meta, force_modality="accel")
            raw_win_gyro = get_raw_window(meta, force_modality="gyro")
            flat_accel = raw_win_accel.flatten()
            flat_gyro = raw_win_gyro.flatten()
            temp_windows.append((flat_accel, flat_gyro))
            y_act.append(act)
            if len(flat_accel) > max_len_accel:
                max_len_accel = len(flat_accel)
            if len(flat_gyro) > max_len_gyro:
                max_len_gyro = len(flat_gyro)
        if i % 5000 == 0 and i > 0:
            print(f"Processed {i} windows...")
    print(f"Loaded {len(temp_windows)} valid windows. Padding to max lengths accel={max_len_accel}, gyro={max_len_gyro}.")
    # Pad all windows to max_len for both modalities and concatenate
    for flat_accel, flat_gyro in temp_windows:
        if len(flat_accel) < max_len_accel:
            pad_width = max_len_accel - len(flat_accel)
            flat_accel = np.pad(flat_accel, (0, pad_width), 'constant')
        if len(flat_gyro) < max_len_gyro:
            pad_width = max_len_gyro - len(flat_gyro)
            flat_gyro = np.pad(flat_gyro, (0, pad_width), 'constant')
        X_raw.append(np.concatenate([flat_accel, flat_gyro]))
    return np.array(X_raw), np.array(y_act), max_len_accel + max_len_gyro

if __name__ == "__main__":
    # Save raw data cache after loading (after X_pamap_raw and X_uci_raw are defined)
    print("Checking for available GPUs...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {gpus}")
    else:
        print("No GPU found, running on CPU.")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading training data...")
    # Try to load raw cache files if they exist
    pamap_raw_cache = "Cache/pamap2_raw_cache.pkl"
    uci_raw_cache = "Cache/uci_raw_cache.pkl"
    if os.path.exists(pamap_raw_cache):
        print(f"Loading PAMAP2 raw cache from {pamap_raw_cache}")
        with open(pamap_raw_cache, "rb") as f:
            X_pamap_raw, y_pamap_act = pickle.load(f)
        pamap_len = X_pamap_raw.shape[1]
    else:
        X_pamap_raw, y_pamap_act, pamap_len = load_raw_windows(CACHE_PAMAP)
        with open(pamap_raw_cache, "wb") as f:
            pickle.dump((X_pamap_raw, y_pamap_act), f)
    if os.path.exists(uci_raw_cache):
        print(f"Loading UCI raw cache from {uci_raw_cache}")
        with open(uci_raw_cache, "rb") as f:
            X_uci_raw, y_uci_act = pickle.load(f)
        uci_len = X_uci_raw.shape[1]
    else:
        X_uci_raw, y_uci_act, uci_len = load_raw_windows(CACHE_UCI)
        with open(uci_raw_cache, "wb") as f:
            pickle.dump((X_uci_raw, y_uci_act), f)
    global_max_len = max(pamap_len, uci_len)
    print(f"Global max window length for train: {global_max_len}")
    # Pad all train windows to global_max_len
    def pad_to_global(X, max_len):
        X_pad = []
        for win in X:
            if len(win) < max_len:
                win = np.pad(win, (0, max_len - len(win)), 'constant')
            X_pad.append(win)
        return np.array(X_pad)
    X_pamap_raw = pad_to_global(X_pamap_raw, global_max_len)
    X_uci_raw = pad_to_global(X_uci_raw, global_max_len)
    print("Concatenating training data...")
    X_train_raw = np.concatenate([X_pamap_raw, X_uci_raw], axis=0)
    y_train_act = np.concatenate([y_pamap_act, y_uci_act], axis=0)
    act_label_list = ACTIVITY_CLASSES
    y_train_enc = np.array([act_label_list.index(a) for a in y_train_act])
    print(f"Training set shape: {X_train_raw.shape}")

    # Select only top 10 generalizable accel features
    accel_indices = [293, 296, 290, 668, 524, 413, 416, 299, 527, 254]
    n_features = X_train_raw.shape[1] // 2
    X_train_accel = X_train_raw[:, :n_features][:, accel_indices]
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_accel)
    print(f"Training set shape (accel only): {X_train_std.shape}")

    print("Building TensorFlow MLP model...")
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train_std.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(act_label_list), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Starting training...")
    history = model.fit(X_train_std, y_train_enc, epochs=30, batch_size=256, verbose=2)
    print("Training complete.")
    print("Loading STM test set...")
    with open(CACHE_STM, "rb") as f:
        stm_data = pickle.load(f)
    stm_activity_labels = stm_data[3]
    stm_pred_modality = stm_data[1]
    window_metadata = stm_data[-1]
    stm_raw_cache = "Cache/stm_raw_cache.pkl"
    def pad_to_global_test(X, max_len):
        X_pad = []
        for win in X:
            if len(win) < max_len:
                win = np.pad(win, (0, max_len - len(win)), 'constant')
            X_pad.append(win)
        return np.array(X_pad)
    if os.path.exists(stm_raw_cache):
        print(f"Loading STM raw cache from {stm_raw_cache}")
        with open(stm_raw_cache, "rb") as f:
            X_stm_raw, y_stm_act = pickle.load(f)
    else:
        print("Processing STM windows and saving cache...")
        with open(CACHE_STM, "rb") as f:
            stm_data = pickle.load(f)
        stm_activity_labels = stm_data[3]
        stm_pred_modality = stm_data[1]
        window_metadata = stm_data[-1]
        X_stm_raw, y_stm_act = [], []
        max_len_accel = 0
        max_len_gyro = 0
        temp_windows = []
        for i, meta in enumerate(window_metadata):
            act = activity_map(stm_activity_labels[i])
            if (act in ACTIVITY_CLASSES) and (act != 'ironing'):
                raw_win_accel = get_raw_window(meta, force_modality="accel")
                raw_win_gyro = get_raw_window(meta, force_modality="gyro")
                flat_accel = raw_win_accel.flatten()
                flat_gyro = raw_win_gyro.flatten()
                temp_windows.append((flat_accel, flat_gyro, stm_pred_modality[i]))
                y_stm_act.append(act)
                if len(flat_accel) > max_len_accel:
                    max_len_accel = len(flat_accel)
                if len(flat_gyro) > max_len_gyro:
                    max_len_gyro = len(flat_gyro)
            if i % 5000 == 0 and i > 0:
                print(f"Processed {i} STM windows...")
        print(f"Loaded {len(temp_windows)} STM windows. Padding to max lengths accel={max_len_accel}, gyro={max_len_gyro}.")
        # Pad and concatenate, tag with predicted modality
        for flat_accel, flat_gyro, pred_mod in temp_windows:
            if len(flat_accel) < max_len_accel:
                pad_width = max_len_accel - len(flat_accel)
                flat_accel = np.pad(flat_accel, (0, pad_width), 'constant')
            if len(flat_gyro) < max_len_gyro:
                pad_width = max_len_gyro - len(flat_gyro)
                flat_gyro = np.pad(flat_gyro, (0, pad_width), 'constant')
            X_stm_raw.append(np.concatenate([flat_accel, flat_gyro]))
        X_stm_raw = np.array(X_stm_raw)
        with open(stm_raw_cache, "wb") as f:
            pickle.dump((X_stm_raw, y_stm_act), f)
    y_stm_enc = np.array([act_label_list.index(a) for a in y_stm_act])
    X_stm_raw = pad_to_global_test(X_stm_raw, global_max_len)
    # Select only top 10 generalizable accel features for test set
    X_stm_accel = X_stm_raw[:, :n_features][:, accel_indices]
    X_stm_std = scaler.transform(X_stm_accel)
    print(f"Test set shape (accel only): {X_stm_std.shape}")
    print("Applying CORAL domain adaptation...")
    X_train_coral = coral_transform(X_train_std, X_stm_std)
    print("Training MLP on CORAL-transformed train data...")
    model_coral = keras.Sequential([
        keras.layers.Input(shape=(X_train_coral.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(len(act_label_list), activation='softmax')
    ])
    model_coral.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_coral = model_coral.fit(X_train_coral, y_train_enc, epochs=30, batch_size=256, verbose=2)
    print("Predicting on STM test set with CORAL model...")
    y_stm_pred_coral = np.argmax(model_coral.predict(X_stm_std, verbose=2), axis=1)
    report = classification_report(y_stm_enc, np.argmax(model.predict(X_stm_std, verbose=2), axis=1), target_names=ACTIVITY_CLASSES)
    report_coral = classification_report(y_stm_enc, y_stm_pred_coral, target_names=ACTIVITY_CLASSES)
    print("--- Standard ---")
    print(report)
    print("--- CORAL ---")
    print(report_coral)
    with open(f"{RESULTS_DIR}/stm_activity_classification_report_mlp_raw.txt", "w") as f:
        f.write("--- Standard ---\n")
        f.write(report)
        f.write("\n--- CORAL ---\n")
        f.write(report_coral)
    cm = confusion_matrix(y_stm_enc, np.argmax(model.predict(X_stm_std, verbose=0), axis=1))
    cm_coral = confusion_matrix(y_stm_enc, y_stm_pred_coral)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('STM Test Confusion Matrix (MLP Raw, Accel Only)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/stm_confusion_matrix_mlp_raw_accel.png")
    plt.close()
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_coral, annot=True, fmt='d', cmap='Greens', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('STM Test Confusion Matrix (MLP Raw, Accel Only, CORAL)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/stm_confusion_matrix_mlp_raw_accel_coral.png")
    plt.close()
    print(f"Test confusion matrices saved to {RESULTS_DIR}/stm_confusion_matrix_mlp_raw_accel.png and ...accel_coral.png")
