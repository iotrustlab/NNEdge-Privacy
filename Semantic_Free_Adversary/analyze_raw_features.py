import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

pamap_raw_cache = "Cache/pamap2_raw_cache.pkl"
uci_raw_cache = "Cache/uci_raw_cache.pkl"
stm_raw_cache = "Cache/stm_raw_cache.pkl"
ACTIVITY_CLASSES = [
    "Upstairs", "Downstairs", "Walking", "Sitting", "Standing", "Laying", "Jogging"
]

def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def describe_features(X, y, title, modality):
    summary = []
    for i, activity in enumerate(ACTIVITY_CLASSES):
        idx = np.where(y == i)[0]
        if len(idx) == 0:
            continue
        means = np.mean(X[idx], axis=0)
        stds = np.std(X[idx], axis=0)
        summary.append(
            f"{title} - {activity} ({modality}):\n"
            f"  Mean (first 5 features): {means[:5]}\n"
            f"  Std (first 5 features): {stds[:5]}\n"
            f"  Sample count: {len(idx)}\n"
        )
    return "\n".join(summary)

def describe_pca(X, y, title, modality):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    summary = [f"{title} ({modality}) PCA explained variance: {explained}"]
    for i, activity in enumerate(ACTIVITY_CLASSES):
        idx = np.where(y == i)[0]
        if len(idx) == 0:
            continue
        mean_pca = np.mean(X_pca[idx], axis=0)
        summary.append(
            f"  {activity}: mean PC1={mean_pca[0]:.2f}, mean PC2={mean_pca[1]:.2f}, count={len(idx)}"
        )
    return "\n".join(summary)

def main():
    X_pamap_raw, y_pamap_act = load_cache(pamap_raw_cache)
    X_uci_raw, y_uci_act = load_cache(uci_raw_cache)
    X_stm_raw, y_stm_act = load_cache(stm_raw_cache)

    act_label_list = ACTIVITY_CLASSES
    y_pamap_enc = np.array([act_label_list.index(a) for a in y_pamap_act])
    y_uci_enc = np.array([act_label_list.index(a) for a in y_uci_act])
    y_stm_enc = np.array([act_label_list.index(a) for a in y_stm_act])

    # Pad train arrays to same number of columns
    def pad_to_max(X, max_len):
        if X.shape[1] < max_len:
            return np.pad(X, ((0, 0), (0, max_len - X.shape[1])), 'constant')
        return X
    max_len = max(X_pamap_raw.shape[1], X_uci_raw.shape[1])
    X_pamap_raw = pad_to_max(X_pamap_raw, max_len)
    X_uci_raw = pad_to_max(X_uci_raw, max_len)

    X_train = np.concatenate([X_pamap_raw, X_uci_raw], axis=0)
    y_train = np.concatenate([y_pamap_enc, y_uci_enc], axis=0)

    n_features = X_train.shape[1] // 2
    modalities = {"accel": slice(0, n_features), "gyro": slice(n_features, None)}

    analysis = []
    for modality, sl in modalities.items():
        # Train
        analysis.append(describe_features(X_train[:, sl], y_train, "Train", modality))
        analysis.append(describe_pca(X_train[:, sl], y_train, "Train", modality))
        # Test
        analysis.append(describe_features(X_stm_raw[:, sl], y_stm_enc, "Test", modality))
        analysis.append(describe_pca(X_stm_raw[:, sl], y_stm_enc, "Test", modality))

    with open("Results/feature_analysis_summary.txt", "w") as f:
        f.write("\n\n".join(analysis))

    # --- Feature Generalizability Analysis ---
    def feature_scores(X_train, y_train, X_test, y_test, modality_name):
        n_features = X_train.shape[1]
        scores = []
        for feat in range(n_features):
            # For each activity, get mean in train and test
            train_means = []
            test_means = []
            for act in range(len(ACTIVITY_CLASSES)):
                train_idx = np.where(y_train == act)[0]
                test_idx = np.where(y_test == act)[0]
                if len(train_idx) > 0:
                    train_means.append(np.mean(X_train[train_idx, feat]))
                else:
                    train_means.append(np.nan)
                if len(test_idx) > 0:
                    test_means.append(np.mean(X_test[test_idx, feat]))
                else:
                    test_means.append(np.nan)
            # Domain sensitivity: mean absolute difference between train and test means
            domain_sensitivity = np.nanmean(np.abs(np.array(train_means) - np.array(test_means)))
            # Discriminability: variance of means across activities (average of train and test)
            discriminability = np.nanmean([
                np.var(train_means), np.var(test_means)
            ])
            scores.append((feat, domain_sensitivity, discriminability))
        # Rank by low domain sensitivity and high discriminability
        scores.sort(key=lambda x: (x[1], -x[2]))
        return scores

    generalizability_report = []
    for modality, sl in modalities.items():
        scores = feature_scores(X_train[:, sl], y_train, X_stm_raw[:, sl], y_stm_enc, modality)
        generalizability_report.append(f"Top 10 generalizable features for {modality} (index, domain_sensitivity, discriminability):")
        for idx, ds, disc in scores[:10]:
            generalizability_report.append(f"  Feature {idx}: domain_sensitivity={ds:.4f}, discriminability={disc:.4f}")
        generalizability_report.append("")

    with open("Results/generalizable_features.txt", "w") as f:
        f.write("\n".join(generalizability_report))

if __name__ == "__main__":
    main()