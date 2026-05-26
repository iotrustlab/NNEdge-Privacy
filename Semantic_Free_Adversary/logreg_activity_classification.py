from tqdm import tqdm

def extract_features_for_dataset(accel_windows, gyro_windows, desc="Extracting features"):
    print(f"{desc} for {len(accel_windows)} samples...")
    features = []
    for i in tqdm(range(len(accel_windows)), desc=desc):
        feats = extract_window_features(accel_windows[i], gyro_windows[i])
        features.append(feats)
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(accel_windows)} samples")
    print(f"Done {desc.lower()}.")
    return np.array(features)

def extract_stat_features(window):
    """
    Given a window of shape (window_size, n_channels), return a feature vector of 32 features per channel.
    Features: mean, std, min, max, median, energy, entropy, skewness, kurtosis, percentiles, etc.
    """
    feats = []
    for i in range(window.shape[1]):
        channel = window[:, i]
        feats.append(np.mean(channel))
        feats.append(np.std(channel))
        feats.append(np.min(channel))
        feats.append(np.max(channel))
        feats.append(np.median(channel))
        feats.append(np.percentile(channel, 25))
        feats.append(np.percentile(channel, 75))
        feats.append(np.percentile(channel, 10))
        feats.append(np.percentile(channel, 90))
        feats.append(np.sum(channel ** 2))  # energy
        # Entropy (discretize)
        hist, _ = np.histogram(channel, bins=10, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        feats.append(entropy)
        # Skewness
        if hasattr(channel, 'skew'):
            feats.append(channel.skew())
        else:
            feats.append(np.mean((channel - np.mean(channel)) ** 3) / (np.std(channel) ** 3 + 1e-8))
        # Kurtosis
        if hasattr(channel, 'kurtosis'):
            feats.append(channel.kurtosis())
        else:
            feats.append(np.mean((channel - np.mean(channel)) ** 4) / (np.std(channel) ** 4 + 1e-8))
        # Absolute mean
        feats.append(np.mean(np.abs(channel)))
        # Absolute max
        feats.append(np.max(np.abs(channel)))
        # Range
        feats.append(np.max(channel) - np.min(channel))
        # Zero crossings
        feats.append(np.sum(np.diff(np.sign(channel)) != 0))
        # Mean crossings
        mean_crossings = np.sum(np.diff(np.sign(channel - np.mean(channel))) != 0)
        feats.append(mean_crossings)
        # Slope (linear fit)
        x = np.arange(len(channel))
        slope = np.polyfit(x, channel, 1)[0]
        feats.append(slope)
        # RMS
        feats.append(np.sqrt(np.mean(channel ** 2)))
        # Variance
        feats.append(np.var(channel))
        # Standard error
        feats.append(np.std(channel) / np.sqrt(len(channel)))
        # Coefficient of variation
        feats.append(np.std(channel) / (np.mean(channel) + 1e-8))
        # Interquartile range
        feats.append(np.percentile(channel, 75) - np.percentile(channel, 25))
        # 5th and 95th percentiles
        feats.append(np.percentile(channel, 5))
        feats.append(np.percentile(channel, 95))
        # Max-min ratio
        feats.append((np.max(channel) + 1e-8) / (np.min(channel) + 1e-8))
        # Min-max ratio
        feats.append((np.min(channel) + 1e-8) / (np.max(channel) + 1e-8))
        # Mean-min ratio
        feats.append((np.mean(channel) + 1e-8) / (np.min(channel) + 1e-8))
        # Mean-max ratio
        feats.append((np.mean(channel) + 1e-8) / (np.max(channel) + 1e-8))
    print(f"Extracted {len(feats) // window.shape[1]} features per channel (total {len(feats)})")
    return np.array(feats)

def extract_window_features(accel_window, gyro_window):
    """
    Compute statistical features for both accel and gyro windows and concatenate.
    accel_window, gyro_window: shape (window_size, n_channels)
    Returns: feature vector for the window
    """
    accel_feats = extract_stat_features(accel_window)
    gyro_feats = extract_stat_features(gyro_window)
    print(f"Accel features shape: {accel_feats.shape}, Gyro features shape: {gyro_feats.shape}")
    return np.concatenate([accel_feats, gyro_feats])


USE_CORAL = False # Set to True to enable CORAL

# Simple CORAL implementation for feature alignment
def coral_transform(Xs, Xt):
    # Xs: source (train), Xt: target (test)
    # Center
    Xs_c = Xs - np.mean(Xs, axis=0)
    Xt_c = Xt - np.mean(Xt, axis=0)
    # Covariance
    Cs = np.cov(Xs_c, rowvar=False) + np.eye(Xs.shape[1])
    Ct = np.cov(Xt_c, rowvar=False) + np.eye(Xt.shape[1])
    # Whitening source
    U_s, S_s, _ = np.linalg.svd(Cs)
    Cs_inv_sqrt = U_s @ np.diag(1.0 / np.sqrt(S_s)) @ U_s.T
    Xs_whitened = Xs_c @ Cs_inv_sqrt
    # Coloring with target
    U_t, S_t, _ = np.linalg.svd(Ct)
    Ct_sqrt = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T
    Xs_coral = Xs_whitened @ Ct_sqrt + np.mean(Xt, axis=0)
    return Xs_coral, Xt
"""
Multinomial Logistic Regression for Activity Classification.
Uses modality-specific discriminative features for each window.
Unified activity mapping.
Uses predicted modality for STM test set.
Saves report in Results/
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

RESULTS_DIR = "Results"
CACHE_PAMAP = "Cache/pamap2_cache.pkl"
CACHE_STM = "Cache/stm_cache_with_pred_modalities.pkl"
CACHE_UCI = "Cache/uci_har_cache.pkl"

# Feature indices for each modality (from analysis)

# Automatic feature selection function
def auto_select_features(X, y, n_features=10):
    # X: (n_samples, n_features), y: (n_samples,)
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    return selected_indices

# --- STM test set feature extraction using predicted modality ---
def extract_stm_modality_features(cache_path, modality_preds, activity_classes):
    # Diagnostic: print structure of STM cache
    with open(cache_path, "rb") as f:
        stm_data = pickle.load(f)
    print(f"STM cache type: {type(stm_data)}, length: {len(stm_data)}")
    for idx in range(min(6, len(stm_data))):
        entry = stm_data[idx]
        print(f"Index {idx}: type {type(entry)}")
        if isinstance(entry, list) and len(entry) > 0:
            print(f"  First element type: {type(entry[0])}, shape: {getattr(entry[0], 'shape', 'N/A')}")
        elif hasattr(entry, 'shape'):
            print(f"  Shape: {entry.shape}")
        else:
            print(f"  Value: {entry}")
    # ...existing code...
    """
    Loads STM cache, filters activities, and for each sample uses predicted modality to select the modality-wise feature vector (accel or gyro).
    Returns: X_stm_modality (n_samples, 32), y_stm_final (encoded activity labels)
    """
    with open(cache_path, "rb") as f:
        stm_data = pickle.load(f)
        # Assume STM cache structure: [accel_windows, gyro_windows, modality_labels, activity_labels, ...]
        accel_windows = np.array(stm_data[0])  # shape (n_samples, window_size, n_channels)
        gyro_windows = np.array(stm_data[1])   # shape (n_samples, window_size, n_channels)
        modality_labels = np.array(stm_data[2])  # shape (n_samples,)
        activity_stm = stm_data[3]
    # Map activities
    y_stm_act = np.array([activity_map(a) for a in activity_stm])
    # Filter to only the 7 classes
    mask = np.array([(a in activity_classes) and (a != 'ironing') for a in y_stm_act])
    accel_windows = accel_windows[mask]
    gyro_windows = gyro_windows[mask]
    modality_labels = modality_labels[mask]
    y_stm_act = y_stm_act[mask]
    # For each sample, use modality label to assign accel/gyro, compute features, select indices, and concatenate
    X_stm_modality = []
    for i in range(len(y_stm_act)):
        print(f"Sample {i}: accel_windows[i] shape: {accel_windows[i].shape}")
        print(f"Sample {i}: gyro_windows[i] shape: {gyro_windows[i].shape}")
        if accel_windows[i].ndim != 2:
            raise ValueError(f"accel_windows[{i}] is not 2D, shape: {accel_windows[i].shape}")
        if gyro_windows[i].ndim != 2:
            raise ValueError(f"gyro_windows[{i}] is not 2D, shape: {gyro_windows[i].shape}")
        if modality_labels[i] == 0:
            # modality 0: first window is accel, second is gyro
            accel_feats = extract_stat_features(accel_windows[i])
            gyro_feats = extract_stat_features(gyro_windows[i])
        else:
            # modality 1: first window is gyro, second is accel
            gyro_feats = extract_stat_features(accel_windows[i])
            accel_feats = extract_stat_features(gyro_windows[i])
        print(f"Sample {i}: accel_feats shape {accel_feats.shape}, values: {accel_feats}")
        print(f"Sample {i}: gyro_feats shape {gyro_feats.shape}, values: {gyro_feats}")
        # Select discriminative indices
    sel_feats = np.array([accel_feats[j] for j in SELECTED_FEATURES])
    # Append modality as a feature
    sel_feats = np.append(sel_feats, modality_labels[i])
    print(f"Sample {i}: selected features shape {sel_feats.shape}, values: {sel_feats}")
    X_stm_modality.append(sel_feats)
    X_stm_modality = np.array(X_stm_modality)
    # Encode activity labels
    y_stm_final = np.array([activity_classes.index(a) for a in y_stm_act])
    return X_stm_modality, y_stm_final

def select_discriminative_features_stm(X, modality_preds):
    # X: (n_samples, 32+32) modality features (already concatenated)
    # Select discriminative indices from both accel and gyro features
    # This function is now redundant, as selection is done in extract_stm_modality_features
    return X

def extract_selected_features_for_window(accel_window, gyro_window):
    # Always use extract_stat_features for each window
    feats = extract_stat_features(accel_window)  # shape (32,)
    sel_feats = np.array([feats[j] for j in SELECTED_FEATURES])
    return sel_feats


def process_dataset_windows(accel_windows, gyro_windows, modality_labels=None):
    features = []
    for i in range(len(accel_windows)):
        window = accel_windows[i] if modality_labels is None or modality_labels[i] == 0 else gyro_windows[i]
        feats = extract_stat_features(window)
        sel_feats = np.array([feats[j] for j in SELECTED_FEATURES])
        # Append modality as a feature if modality_labels is not None
        if modality_labels is not None:
            sel_feats = np.append(sel_feats, modality_labels[i])
        features.append(sel_feats)
    return np.array(features)

def extract_and_save_selected_features():
    """
    Loads train/test cache files, extracts features for each window, selects discriminative features,
    and saves the processed arrays for downstream activity classification.
    For test set, uses predicted modality from logreg_modality_classifier.py.
    """
    # Load train caches
    with open(CACHE_PAMAP, "rb") as f:
        pamap_data = pickle.load(f)
    print("pamap_data length:", len(pamap_data))
    print("pamap_data types:", [type(x) for x in pamap_data])
    for idx, item in enumerate(pamap_data):
        if hasattr(item, 'shape'):
            print(f"pamap_data[{idx}] shape: {item.shape}")
        else:
            print(f"pamap_data[{idx}] type: {type(item)}")
    # Only use pamap_data[3] for window data
    pamap_windows = np.array(pamap_data[3])  # shape: (n_samples, window_size, n_channels)
    with open(CACHE_UCI, "rb") as f:
        uci_data = pickle.load(f)
    print("uci_data length:", len(uci_data))
    print("uci_data types:", [type(x) for x in uci_data])
    for idx, item in enumerate(uci_data):
        if hasattr(item, 'shape'):
            print(f"uci_data[{idx}] shape: {item.shape}")
        else:
            print(f"uci_data[{idx}] type: {type(item)}")
    try:
        uci_accel_windows = np.array(uci_data[3])
        uci_gyro_windows = np.array(uci_data[4])
        print(f"uci_accel_windows shape: {uci_accel_windows.shape}")
        print(f"uci_gyro_windows shape: {uci_gyro_windows.shape}")
    except Exception as e:
        print(f"Error accessing uci_data[3] or [4]: {e}")
    # Debug: Verify 32 features are extracted for all samples (modality-wise)
    print("--- Debug: Verifying 32 features per sample (modality-wise, PAMAP) ---")
    all_ok = True
    # Load PAMAP and UCI feature arrays as in modality classifier
    with open(CACHE_PAMAP, "rb") as f:
        pamap_cache = pickle.load(f)
    with open(CACHE_UCI, "rb") as f:
        uci_cache = pickle.load(f)
    X_pamap = np.array(pamap_cache[0])
    X_uci = np.array(uci_cache[0])
    # Combine train sets
    X_train = np.concatenate([X_pamap, X_uci], axis=0)
    print(f"Combined train set shape: {X_train.shape}")
    # Verify extraction of 32 modality-wise activity features for each sample
    all_ok = True
    for i, sample in enumerate(X_train):
        if sample.shape[0] < 32:
            print(f"ERROR: Sample {i} has {sample.shape[0]} features, expected at least 32.")
            all_ok = False
        if i < 3:
            print(f"Sample {i} features: {sample[:32]}")
    if all_ok:
        print("OK: 32 features extracted for each sample in combined train set.")
    
    


# Unified activity mapping
def activity_map(label):
    mapping = {
        # stairs
        "ascending_stairs": "Upstairs",
        "WALKING_UPSTAIRS": "Upstairs",
        "Upstairs": "Upstairs",
        "descending_stairs": "Downstairs",
        "WALKING_DOWNSTAIRS": "Downstairs",
        "Downstairs": "Downstairs",
        # walking
        "walking": "Walking",
        "WALKING": "Walking",
        "Walking": "Walking",
        "nordic_walking": "Walking",
        "vacuum_cleaning": "Walking",
        # sitting
        "sitting": "Sitting",
        "SITTING": "Sitting",
        "Sitting": "Sitting",
        # standing
        "standing": "Standing",
        "STANDING": "Standing",
        "Standing": "Standing",
        # laying
        "lying": "Laying",
        "LAYING": "Laying",
        "Laying": "Laying",
        # jogging/running/cycling/rope_jumping
        "running": "Jogging",
        "Jogging": "Jogging",
        "rope_jumping": "Jogging",
        "cycling": "Jogging",
    }
    return mapping.get(label, label)

# Fixed list of 7 activity classes
ACTIVITY_CLASSES = [
    "Upstairs",
    "Downstairs",
    "Walking",
    "Sitting",
    "Standing",
    "Laying",
    "Jogging"
]

def filter_activities(X, y, acts):
    mask = np.array([(a in acts) and (a != 'ironing') for a in y])
    return X[mask], y[mask]

def load_cache(path, features=None):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if features is not None:
        return np.array(data[0])[:, features], np.array(data[1]), np.array(data[2])
    return data


if __name__ == "__main__":
   
    # Load features and activity labels
    with open(CACHE_PAMAP, "rb") as f:
        X_pamap_full, _, _, activity_pamap = pickle.load(f)
        X_pamap_full = np.array(X_pamap_full)
    with open(CACHE_UCI, "rb") as f:
        X_uci_full, _, _, activity_uci = pickle.load(f)
        X_uci_full = np.array(X_uci_full)
    with open(CACHE_STM, "rb") as f:
        X_stm_full, _, _, activity_stm = pickle.load(f)
        X_stm_full = np.array(X_stm_full)
    # Print all unique activity labels in STM test set before mapping
    # Map activities
    y_pamap_act = np.array([activity_map(a) for a in activity_pamap])
    y_uci_act = np.array([activity_map(a) for a in activity_uci])
    y_stm_act = np.array([activity_map(a) for a in activity_stm])
    # Filter to only the 7 classes (ignore 'Other')
    # Assume features are [accel..., gyro...]
    X_pamap_act, y_pamap_act = filter_activities(X_pamap_full, y_pamap_act, ACTIVITY_CLASSES)
    X_uci_act, y_uci_act = filter_activities(X_uci_full, y_uci_act, ACTIVITY_CLASSES)
    X_stm_act, y_stm_act = filter_activities(X_stm_full, y_stm_act, ACTIVITY_CLASSES)
    # Load modality labels (for train: true, for test: predicted)
    y_pamap_mod = load_cache(CACHE_PAMAP)[1]
    y_uci_mod = load_cache(CACHE_UCI)[1]
    # Prepare train set features from precomputed feature vectors
    def select_discriminative_features_from_cache(features_list, modality_list):
        # features_list: list of np.ndarray, each shape (33,)
        # modality_list: list or np.ndarray, each 0 (accel) or 1 (gyro)
        selected = []
        for i in range(len(features_list)):
            feats = features_list[i]
            sel_feats = np.array([feats[j] for j in SELECTED_FEATURES])
            # Append modality as a feature
            sel_feats = np.append(sel_feats, modality_list[i])
            selected.append(sel_feats)
        return np.array(selected)

    # Load train features and labels
    with open(CACHE_PAMAP, "rb") as f:
        pamap_data = pickle.load(f)
    pamap_features = pamap_data[0]
    pamap_modality = pamap_data[1]
    pamap_location = pamap_data[2]
    pamap_activity = pamap_data[3]

    # Filter to only the 7 classes
    pamap_activity_mapped = np.array([activity_map(a) for a in pamap_activity])
    pamap_mask = np.array([(a in ACTIVITY_CLASSES) and (a != 'ironing') for a in pamap_activity_mapped])
    pamap_features = [pamap_features[i] for i in range(len(pamap_features)) if pamap_mask[i]]
    pamap_modality = [pamap_modality[i] for i in range(len(pamap_modality)) if pamap_mask[i]]
    pamap_activity_mapped = pamap_activity_mapped[pamap_mask]

    # Repeat for UCI
    with open(CACHE_UCI, "rb") as f:
        uci_data = pickle.load(f)
    uci_features = uci_data[0]
    uci_modality = uci_data[1]
    uci_location = uci_data[2]
    uci_activity = uci_data[3]
    uci_activity_mapped = np.array([activity_map(a) for a in uci_activity])
    uci_mask = np.array([(a in ACTIVITY_CLASSES) and (a != 'ironing') for a in uci_activity_mapped])
    uci_features = [uci_features[i] for i in range(len(uci_features)) if uci_mask[i]]
    uci_modality = [uci_modality[i] for i in range(len(uci_modality)) if uci_mask[i]]
    uci_activity_mapped = uci_activity_mapped[uci_mask]

    # Combine train sets
    X_train_features = pamap_features + uci_features
    y_train_modality = pamap_modality + uci_modality
    y_train_activity = np.concatenate([pamap_activity_mapped, uci_activity_mapped], axis=0)


    # Split accel and gyro features (assume 32 accel + 32 gyro + modality)
    X_train_features_arr = np.array(X_train_features)
    accel_feats = X_train_features_arr[:, :32]
    gyro_feats = X_train_features_arr[:, 32:64]
    # Automatic feature selection
    n_select = 5  # Decreased number of features for accel and gyro
    accel_indices = auto_select_features(accel_feats, y_train_activity, n_features=n_select)
    gyro_indices = auto_select_features(gyro_feats, y_train_activity, n_features=n_select)
    SELECTED_FEATURES = list(accel_indices) + [32 + i for i in gyro_indices]
    print(f"Automatically selected accel feature indices: {accel_indices}")
    print(f"Automatically selected gyro feature indices: {gyro_indices}")
    print(f"SELECTED_FEATURES (combined): {SELECTED_FEATURES}")

    # Select discriminative features
    def select_discriminative_features_from_cache(features_list, modality_list):
        selected = []
        for i in range(len(features_list)):
            feats = features_list[i]
            sel_feats = np.array([feats[j] for j in SELECTED_FEATURES])
            sel_feats = np.append(sel_feats, modality_list[i])
            selected.append(sel_feats)
        return np.array(selected)

    X_train_sel = select_discriminative_features_from_cache(X_train_features, y_train_modality)
    print(f"Selected train features shape: {X_train_sel.shape}")

    # Encode activity labels using fixed 7 classes
    act_label_list = ACTIVITY_CLASSES
    y_train_act_enc = np.array([act_label_list.index(a) for a in y_train_activity])
    # Print class balance in train dataset
    print("Class balance in TRAIN set:")
    for act in act_label_list:
        print(f"{act}: {(y_train_activity == act).sum()}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Domain adaptation (CORAL) and normalization

    # --- STM test set preparation and classification (after training) ---
    with open(CACHE_STM, "rb") as f:
        stm_data = pickle.load(f)
    stm_features = stm_data[0]
    stm_modality = stm_data[1]
    stm_location = stm_data[2]
    stm_activity = stm_data[3]
    stm_activity_mapped = np.array([activity_map(a) for a in stm_activity])
    stm_mask = np.array([(a in ACTIVITY_CLASSES) and (a != 'ironing') for a in stm_activity_mapped])
    stm_features = [stm_features[i] for i in range(len(stm_features)) if stm_mask[i]]
    stm_modality = [stm_modality[i] for i in range(len(stm_modality)) if stm_mask[i]]
    stm_activity_mapped = stm_activity_mapped[stm_mask]

    # Select discriminative features for STM test set
    X_stm_sel = select_discriminative_features_from_cache(stm_features, stm_modality)
    print(f"STM test set (discriminative features) shape: {X_stm_sel.shape}")

    scaler = StandardScaler()
    if USE_CORAL:
        print("Applying CORAL to train and test features...")
        X_train_coral, X_stm_coral = coral_transform(X_train_sel, X_stm_sel)
        X_train_final_std = scaler.fit_transform(X_train_coral)
        X_stm_final_std = scaler.transform(X_stm_coral)
    else:
        X_train_final_std = scaler.fit_transform(X_train_sel)
        X_stm_final_std = scaler.transform(X_stm_sel)

    # Train multinomial logistic regression

    clf_act = LogisticRegression(penalty='l2', class_weight='balanced', C=1.0, max_iter=500,
                                multi_class='multinomial', solver='lbfgs')
    clf_act.fit(X_train_final_std, y_train_act_enc)
    print("Training complete: Multinomial logistic regression fitted on selected train features.")

    # Predict activities on TRAIN set
    y_train_pred = clf_act.predict(X_train_final_std)
    print("Train set predictions complete.")
    train_report = classification_report(y_train_act_enc, y_train_pred, target_names=ACTIVITY_CLASSES)
    print(train_report)
    with open(f"{RESULTS_DIR}/train_activity_classification_report.txt", "w") as f:
        f.write(train_report)
    print(f"Train classification report saved to {RESULTS_DIR}/train_activity_classification_report.txt")

    # Confusion matrix for train
    cm_train = confusion_matrix(y_train_act_enc, y_train_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Train Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/train_confusion_matrix.png")
    plt.close()
    print(f"Train confusion matrix saved to {RESULTS_DIR}/train_confusion_matrix.png")

    # Predict activities on STM test set
    y_stm_final = np.array([act_label_list.index(a) for a in stm_activity_mapped])
    y_stm_pred = clf_act.predict(X_stm_final_std)
    print("STM test set predictions complete.")
    # Print classification report
    report = classification_report(y_stm_final, y_stm_pred, target_names=ACTIVITY_CLASSES)
    print(report)
    # Save report
    with open(f"{RESULTS_DIR}/stm_activity_classification_report.txt", "w") as f:
        f.write(report)
    print(f"Classification report saved to {RESULTS_DIR}/stm_activity_classification_report.txt")

    # Confusion matrix for test
    cm_test = confusion_matrix(y_stm_final, y_stm_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_confusion_matrix.png")
    plt.close()
    print(f"Test confusion matrix saved to {RESULTS_DIR}/test_confusion_matrix.png")