"""
Logistic Regression Modality and Placement Classifier
- Uses selected features: feat_0, feat_4, feat_22, feat_26, feat_30
- Outlier clipping, log transform (where applicable), standardization
- Placement mapping unified
- Stratified K-Fold CV, F1-score and balanced accuracy
"""
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

RESULTS_DIR = "Results"
SELECTED_FEATURES = [0, 4, 22, 26, 30]  # For modality only
PLACEMENT_FEATURES = [1, 5, 9, 12, 23, 30, 31]  # For placement classifier
CACHE_PAMAP = "Cache/pamap2_cache.pkl"
CACHE_STM = "Cache/stm_cache.pkl"
CACHE_UCI = "Cache/uci_har_cache.pkl"
PLACEMENT_MAP = {
    'right_ankle': 'right-ankle',
    'left_ankle': 'left-ankle',
    'right_wrist': 'right-wrist',
    'left_wrist': 'left-wrist',
    'chest': 'right-pocket',
    'right_pocket': 'right-pocket',
}
PLACEMENT_LABELS = ["left-ankle", "right-ankle", "left-wrist", "right-wrist", "right-pocket"]

# Utility functions
def load_cache(path, feature_idxs):
    with open(path, "rb") as f:
        tup = pickle.load(f)
    X = np.array(tup[0])[:, feature_idxs]
    y = np.array(tup[1])
    placement = tup[2] if len(tup) > 2 else None
    return X, y, placement

def map_placement_labels(labels):
    return [PLACEMENT_MAP.get(l, "Other") for l in labels]

def encode_labels(labels, label_list):
    return np.array([label_list.index(l) if l in label_list else -1 for l in labels])

def clip_outliers(X, lower=1, upper=99):
    X_clip = X.copy()
    for i in range(X.shape[1]):
        l, u = np.percentile(X[:, i], lower), np.percentile(X[:, i], upper)
        X_clip[:, i] = np.clip(X[:, i], l, u)
    return X_clip

def log_transform(X, feature_indices):
    X_log = X.copy()
    for i in feature_indices:
        # Only apply log1p to positive values
        pos_mask = X_log[:, i] > 0
        X_log[pos_mask, i] = np.log1p(X_log[pos_mask, i])
    return X_log

def filter_valid(X, y):
    mask = y != -1
    return X[mask], y[mask]










def balance_classes(X, y, label_names):
    # Downsample each class to the count of the least frequent class
    from collections import Counter
    idxs = []
    counts = Counter(y)
    min_count = min([counts[i] for i in range(len(label_names)) if counts[i] > 0])
    for i, label in enumerate(label_names):
        class_idxs = np.where(y == i)[0]
        if len(class_idxs) > min_count:
            class_idxs = np.random.choice(class_idxs, min_count, replace=False)
        idxs.extend(class_idxs)
    idxs = np.array(idxs)
    return X[idxs], y[idxs]

def run_logreg(X_train, y_train, X_test, y_test, num_classes, label_names, task_name,
              multinomial=False, features_for_log=None):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    if multinomial:
        clf = LogisticRegression(penalty='l2', class_weight='balanced', C=1.0, max_iter=500,
                                multi_class='multinomial', solver='lbfgs')
    else:
        clf = LogisticRegression(penalty='l2', class_weight='balanced', C=1.0, max_iter=500)
    clf.fit(X_train_std, y_train)
    preds_train = clf.predict(X_train_std)
    preds_test = clf.predict(X_test_std)
    prob_train = clf.predict_proba(X_train_std)
    prob_test = clf.predict_proba(X_test_std)
    f1 = f1_score(y_test, preds_test, average='weighted')
    bal_acc = balanced_accuracy_score(y_test, preds_test)
    report = classification_report(y_test, preds_test, target_names=label_names)
    print(f"{task_name} F1-score: {f1:.4f}, Balanced Acc: {bal_acc:.4f}")
    return preds_test, f1, bal_acc, report, prob_test

def cross_validate(X, y, num_classes, label_names, task_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, bals = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        _, f1, bal_acc, _, _ = run_logreg(X_tr, y_tr, X_te, y_te, num_classes, label_names, task_name)
        f1s.append(f1)
        bals.append(bal_acc)
    print(f"{task_name} CV F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"{task_name} CV Balanced Acc: {np.mean(bals):.4f} ± {np.std(bals):.4f}")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Modality classification (unchanged)
    X_pamap, y_pamap_mod, _ = load_cache(CACHE_PAMAP, SELECTED_FEATURES)
    X_stm, y_stm_mod, _ = load_cache(CACHE_STM, SELECTED_FEATURES)
    X_uci, y_uci_mod, _ = load_cache(CACHE_UCI, SELECTED_FEATURES)
    # Outlier clipping
    X_pamap_clip = clip_outliers(X_pamap)
    X_stm_clip = clip_outliers(X_stm)
    X_uci_clip = clip_outliers(X_uci)
    # Log transform (feat_22, feat_30: indices 2, 4)
    X_pamap_log = log_transform(X_pamap_clip, [2, 4])
    X_stm_log = log_transform(X_stm_clip, [2, 4])
    X_uci_log = log_transform(X_uci_clip, [2, 4])
    # Combine all for training
    X_train_mod = np.concatenate([X_pamap_log, X_uci_log], axis=0)
    y_train_mod = np.concatenate([y_pamap_mod, y_uci_mod], axis=0)
    # Standardization and model
    # Predict modality for both accel and gyro feature vectors in STM test set
    mod_preds, mod_f1, mod_bal_acc, mod_report, mod_probs = run_logreg(
        X_train_mod, y_train_mod, X_stm_log, y_stm_mod, 2, ["Accel", "Gyro"], "Modality")
    cross_validate(X_train_mod, y_train_mod, 2, ["Accel", "Gyro"], "Modality")

    # Save both predictions and probabilities for all STM feature vectors
    stm_modality_results = {
        "modality_preds": mod_preds,  # shape (n_stm_feature_vectors,)
        "modality_probs": mod_probs,  # shape (n_stm_feature_vectors, 2)
        "stm_true_modality": y_stm_mod,  # shape (n_stm_feature_vectors,)
        "modality_report": mod_report,
        "modality_f1": mod_f1,
        "modality_bal_acc": mod_bal_acc,
    }
    with open(f"{RESULTS_DIR}/stm_logreg_modality_predictions.pkl", "wb") as f:
        pickle.dump(stm_modality_results, f)
    with open(f"{RESULTS_DIR}/logreg_modality_report.txt", "w") as f:
        f.write("--- Modality Classification ---\n")
        f.write(mod_report + "\n")
        f.write(f"F1-score: {mod_f1:.4f}, Balanced Acc: {mod_bal_acc:.4f}\n")

    print("Modality classification results saved to Results/stm_logreg_modality_predictions.pkl")

    # Save STM cache with predicted modality labels (after mod_preds is defined)
    with open(CACHE_STM, "rb") as f:
        stm_cache = pickle.load(f)
    stm_pred_cache = list(stm_cache)
    stm_pred_cache[1] = mod_preds  # override modality labels with predictions
    with open("Cache/stm_modality_pred_cache.pkl", "wb") as f:
        pickle.dump(tuple(stm_pred_cache), f)
    print("STM cache with predicted modality saved to Cache/stm_modality_pred_cache.pkl")
