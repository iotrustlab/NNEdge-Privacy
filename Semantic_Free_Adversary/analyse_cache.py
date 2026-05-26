import pickle
import numpy as np

def inspect_cache(path):
    with open(path, 'rb') as f:
        X, modality, placement, activity = pickle.load(f)
    print(f"File: {path}")
    print("  X shape:", np.shape(X))
    print("  modality shape:", np.shape(modality), "unique:", np.unique(modality))
    print("  placement shape:", np.shape(placement), "unique:", np.unique(placement))
    print("  activity shape:", np.shape(activity), "unique:", np.unique(activity))
    print("  X sample:", X[:5])
    print("  X NaN count:", np.isnan(X).sum())
    print()

inspect_cache('Cache/pamap2_cache.pkl')
inspect_cache('Cache/stm_cache.pkl')
inspect_cache('Cache/uci_har_cache.pkl')