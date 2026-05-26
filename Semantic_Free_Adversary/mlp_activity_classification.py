import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

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

def auto_select_features(X, y, n_features=10):
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
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

    # Feature selection
    X_train_features_arr = np.array(X_train_features)
    accel_feats = X_train_features_arr[:, :32]
    gyro_feats = X_train_features_arr[:, 32:64]
    n_select = 5
    accel_indices = auto_select_features(accel_feats, y_train_activity, n_features=n_select)
    gyro_indices = auto_select_features(gyro_feats, y_train_activity, n_features=n_select)
    SELECTED_FEATURES = list(accel_indices) + [32 + i for i in gyro_indices]
    print(f"Automatically selected accel feature indices: {accel_indices}")
    print(f"Automatically selected gyro feature indices: {gyro_indices}")
    print(f"SELECTED_FEATURES (combined): {SELECTED_FEATURES}")

    X_train_sel = select_discriminative_features(X_train_features, y_train_modality, SELECTED_FEATURES)
    print(f"Selected train features shape: {X_train_sel.shape}")
    act_label_list = ACTIVITY_CLASSES
    y_train_act_enc = np.array([act_label_list.index(a) for a in y_train_activity])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_train_final_std = scaler.fit_transform(X_train_sel)

    # Train MLP
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
    clf_mlp.fit(X_train_final_std, y_train_act_enc)
    print("Training complete: MLP fitted on selected train features.")

    # Train set predictions (MLP)
    y_train_pred_mlp = clf_mlp.predict(X_train_final_std)
    train_report_mlp = classification_report(y_train_act_enc, y_train_pred_mlp, target_names=ACTIVITY_CLASSES)
    print(train_report_mlp)
    with open(f"{RESULTS_DIR}/train_activity_classification_report_mlp.txt", "w") as f:
        f.write(train_report_mlp)
    cm_train_mlp = confusion_matrix(y_train_act_enc, y_train_pred_mlp)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_train_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Train Confusion Matrix (MLP)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/train_confusion_matrix_mlp.png")
    plt.close()
    print(f"Train confusion matrix saved to {RESULTS_DIR}/train_confusion_matrix_mlp.png")

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
    X_stm_sel = select_discriminative_features(stm_features, stm_modality, SELECTED_FEATURES)
    print(f"STM test set (discriminative features) shape: {X_stm_sel.shape}")
    X_stm_final_std = scaler.transform(X_stm_sel)

    y_stm_final = np.array([act_label_list.index(a) for a in stm_activity_mapped])
    y_stm_pred_mlp = clf_mlp.predict(X_stm_final_std)
    report_mlp = classification_report(y_stm_final, y_stm_pred_mlp, target_names=ACTIVITY_CLASSES)
    print(report_mlp)
    with open(f"{RESULTS_DIR}/stm_activity_classification_report_mlp.txt", "w") as f:
        f.write(report_mlp)
    cm_test_mlp = confusion_matrix(y_stm_final, y_stm_pred_mlp)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_test_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIVITY_CLASSES, yticklabels=ACTIVITY_CLASSES)
    plt.title('Test Confusion Matrix (MLP)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/test_confusion_matrix_mlp.png")
    plt.close()
    print(f"Test confusion matrix saved to {RESULTS_DIR}/test_confusion_matrix_mlp.png")
