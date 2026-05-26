import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import f_classif
from scipy.stats import ks_2samp

RESULTS_DIR = "Results"
CACHE_PAMAP = "Cache/pamap2_cache.pkl"
CACHE_STM = "Cache/stm_cache_with_pred_modalities.pkl"
CACHE_UCI = "Cache/uci_har_cache.pkl"
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

def domain_adapt_feature_selection(X_train, y_train, X_test, n_features=3):
    # Compute discriminative power (ANOVA F-score)
    f_scores, _ = f_classif(X_train, y_train)
    # Compute stability (KS statistic between train and test)
    ks_stats = np.array([ks_2samp(X_train[:, i], X_test[:, i]).statistic for i in range(X_train.shape[1])])
    # Normalize scores
    f_scores_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-8)
    ks_stats_norm = ks_stats / (np.max(ks_stats) + 1e-8)
    # Combine: high discriminative, low domain shift
    combined_score = f_scores_norm - ks_stats_norm
    selected_indices = np.argsort(combined_score)[-n_features:][::-1]
    print("Selected feature indices (domain adaptation-aware):", selected_indices)
    print("Combined scores for selected features:", combined_score[selected_indices])
    return selected_indices

def select_discriminative_features(features_list, modality_list, selected_indices):
    selected = []
    for i in range(len(features_list)):
        feats = features_list[i]
        sel_feats = np.array([feats[j] for j in selected_indices])
        sel_feats = np.append(sel_feats, modality_list[i])
        selected.append(sel_feats)
    return np.array(selected)

if __name__ == "__main__":
    # Load train features and labels
    with open(CACHE_PAMAP, "rb") as f:
        pamap_data = pickle.load(f)
    pamap_features = pamap_data[0]
    pamap_modality = pamap_data[1]
    pamap_activity = pamap_data[3]
    pamap_activity_mapped = np.array([activity_map(a) for a in pamap_activity])
    pamap_mask = np.array([(a in ACTIVITY_CLASSES) and (a != 'ironing') for a in pamap_activity_mapped])
    pamap_features = [pamap_features[i] for i in range(len(pamap_features)) if pamap_mask[i]]
    pamap_modality = [pamap_modality[i] for i in range(len(pamap_modality)) if pamap_mask[i]]
    pamap_activity_mapped = pamap_activity_mapped[pamap_mask]

    with open(CACHE_UCI, "rb") as f:
        uci_data = pickle.load(f)
    uci_features = uci_data[0]
    uci_modality = uci_data[1]
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

    # STM test set
    with open(CACHE_STM, "rb") as f:
        stm_data = pickle.load(f)
    stm_features = stm_data[0]
    stm_modality = stm_data[1]
    stm_activity = stm_data[3]
    stm_activity_mapped = np.array([activity_map(a) for a in stm_activity])
    stm_mask = np.array([(a in ACTIVITY_CLASSES) and (a != 'ironing') for a in stm_activity_mapped])
    stm_features = [stm_features[i] for i in range(len(stm_features)) if stm_mask[i]]
    stm_modality = [stm_modality[i] for i in range(len(stm_modality)) if stm_mask[i]]
    stm_activity_mapped = stm_activity_mapped[stm_mask]

    # Convert to arrays
    X_train_features_arr = np.array(X_train_features)
    X_test_features_arr = np.array(stm_features)
    n_select = 3
    selected_indices = domain_adapt_feature_selection(X_train_features_arr, y_train_activity, X_test_features_arr, n_features=n_select)

    X_train_sel = select_discriminative_features(X_train_features, y_train_modality, selected_indices)
    print(f"Selected train features shape: {X_train_sel.shape}")
    act_label_list = ACTIVITY_CLASSES
    y_train_act_enc = np.array([act_label_list.index(a) for a in y_train_activity])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_train_final_std = scaler.fit_transform(X_train_sel)

    # Train Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf_rf.fit(X_train_final_std, y_train_act_enc)
    print("Training complete: Random Forest fitted on selected train features.")

    # Train set predictions
    y_train_pred = clf_rf.predict(X_train_final_std)
    train_report = classification_report(y_train_act_enc, y_train_pred, target_names=ACTIVITY_CLASSES)
    print(train_report)
    with open(f"{RESULTS_DIR}/train_activity_classification_report_rf_domainadapt.txt", "w") as f:
        f.write(train_report)
    cm_train = confusion_matrix(y_train_act_enc, y_train_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Train Confusion Matrix (RF DomainAdapt)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/train_confusion_matrix_rf_domainadapt.png")
    plt.close()
    print(f"Train confusion matrix saved to {RESULTS_DIR}/train_confusion_matrix_rf_domainadapt.png")

    # Test set
    X_test_sel = select_discriminative_features(stm_features, stm_modality, selected_indices)
    print(f"STM test set (domainadapt features) shape: {X_test_sel.shape}")
    X_test_final_std = scaler.transform(X_test_sel)
    y_stm_final = np.array([act_label_list.index(a) for a in stm_activity_mapped])
    y_stm_pred = clf_rf.predict(X_test_final_std)
    report = classification_report(y_stm_final, y_stm_pred, target_names=ACTIVITY_CLASSES)
    print(report)
    with open(f"{RESULTS_DIR}/stm_activity_classification_report_rf_domainadapt.txt", "w") as f:
        f.write(report)
    cm_test = confusion_matrix(y_stm_final, y_stm_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Test Confusion Matrix (RF DomainAdapt)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_confusion_matrix_rf_domainadapt.png")
    plt.close()
    print(f"Test confusion matrix saved to {RESULTS_DIR}/test_confusion_matrix_rf_domainadapt.png")
