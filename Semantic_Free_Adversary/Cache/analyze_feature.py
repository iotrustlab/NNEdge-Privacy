# Global activity_map function for unified activity mapping
def activity_map(label):
    mapping = {
        "ascending_stairs": "Upstairs",
        "WALKING_UPSTAIRS": "Upstairs",
        "Upstairs": "Upstairs",
        "descending_stairs": "Downstairs",
        "WALKING_DOWNSTAIRS": "Downstairs",
        "Downstairs": "Downstairs",
        "walking": "Walking",
        "WALKING": "Walking",
        "Walking": "Walking",
        "nordic_walking": "Walking",
        "sitting": "Sitting",
        "SITTING": "Sitting",
        "Sitting": "Sitting",
        "standing": "Standing",
        "STANDING": "Standing",
        "Standing": "Standing",
        "lying": "Laying",
        "LAYING": "Laying",
        "Laying": "Laying",
        "running": "Jogging/Running",
        "Jogging": "Jogging/Running",
        "rope_jumping": "Jogging/Running",
    }
    return mapping.get(label, "Other")
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load features and labels
def unify_label(label):
    mapping = {
        'right_ankle': 'right-ankle',
        'left_ankle': 'left-ankle',
        'right_wrist': 'right-wrist',
        'left_wrist': 'left-wrist',
        'chest': 'right-pocket',
        'right_pocket': 'right-pocket',
    }
    return mapping.get(label, label)

with open('Cache/pamap2_cache.pkl', 'rb') as f:
    X_pamap, modality_pamap, placement_pamap, activity_pamap = pickle.load(f)
    placement_pamap = np.array([unify_label(lbl) for lbl in placement_pamap])
with open('Cache/stm_cache.pkl', 'rb') as f:
    X_stm, modality_stm, placement_stm, activity_stm = pickle.load(f)
    placement_stm = np.array([unify_label(lbl) for lbl in placement_stm])

# Debug: print modality and activity counts in STM test set
import collections
print("Test set modality counts:", collections.Counter(modality_stm))
print("Test set activity counts:", collections.Counter(activity_stm))
for mod in [0, 1]:
    for act in set(activity_stm):
        mask = (np.array(modality_stm) == mod) & (np.array(activity_stm) == act)
        print(f"Modality {mod}, Activity {act}: {np.sum(mask)} samples")
with open('Cache/uci_har_cache.pkl', 'rb') as f:
    X_uci, modality_uci, placement_uci, activity_uci = pickle.load(f)

selected_indices = [1, 5, 6, 7, 9]
feature_names = [f'feat_{i}' for i in range(len(X_pamap[0]))]

with open('Cache/analyze_feature.txt', 'w') as f:
    # Prepare activity labels for train and test (include UCI in train)
    # Only filter 'ironing' from PAMAP2 once, after loading
    # (Already done above, do not repeat here)
    # Now concatenate and map
    y_train_raw = np.concatenate([
        np.array(activity_pamap),
        np.array(activity_uci)
    ])
    y_test_raw = np.array(activity_stm)
    # Unified activity mapping for all datasets
    def activity_map_unified(label):
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
            # UCI HAR
            "WALKING": "Walking",
            "WALKING_UPSTAIRS": "Upstairs",
            "WALKING_DOWNSTAIRS": "Downstairs",
            "SITTING": "Sitting",
            "STANDING": "Standing",
            "LAYING": "Laying",
        }
        return mapping.get(label, label)
    # Map all activities in train and test sets
    y_train_act = np.array([activity_map_unified(a) for a in y_train_raw])
    y_test_act = np.array([activity_map_unified(a) for a in y_test_raw])
    # Print unique mapped activities for verification
    print("Unique mapped train activities:", np.unique(y_train_act, return_counts=True))
    print("Unique mapped test activities:", np.unique(y_test_act, return_counts=True))
    # Print unique mapped activities and their counts for both sets
    print("Unique mapped train activities:", np.unique(y_train_act, return_counts=True))
    print("Unique mapped test activities:", np.unique(y_test_act, return_counts=True))
    modalities = [0, 1]  # 0: Accel, 1: Gyro
    modality_names = {0: 'Accel', 1: 'Gyro'}
    # Debug: print sample counts and unique activities for each modality
    modality_train_all = np.concatenate([modality_pamap, modality_uci], axis=0)
    X_train_all = np.concatenate([X_pamap, X_uci], axis=0)
    modality_stm = np.array(modality_stm)
    X_stm = np.array(X_stm)
    for mod in modalities:
        print(f"\n--- DEBUG: Modality {mod} ({modality_names[mod]}) ---")
        print(f"modality_train_all shape: {modality_train_all.shape}, unique: {np.unique(modality_train_all)}")
        print(f"modality_stm shape: {modality_stm.shape}, unique: {np.unique(modality_stm)}")
        train_mod_mask = modality_train_all == mod
        test_mod_mask = modality_stm == mod
        print(f"train_mod_mask sum: {np.sum(train_mod_mask)}, test_mod_mask sum: {np.sum(test_mod_mask)}")
        print(f"First 10 train_mod_mask: {train_mod_mask[:10]}")
        print(f"First 10 test_mod_mask: {test_mod_mask[:10]}")
        print(f"X_train_all shape: {X_train_all.shape}, X_stm shape: {X_stm.shape}")
        print(f"X_train_mod shape: {X_train_all[train_mod_mask].shape}, X_test_mod shape: {X_stm[test_mod_mask].shape}")
        print(f"y_train_act shape: {y_train_act.shape}, y_test_act shape: {y_test_act.shape}")
        print(f"y_train_mod shape: {y_train_act[train_mod_mask].shape}, y_test_mod shape: {y_test_act[test_mod_mask].shape}")
        print(f"First 10 y_train_mod: {y_train_act[train_mod_mask][:10]}")
        print(f"First 10 y_test_mod: {y_test_act[test_mod_mask][:10]}")
        train_mask = modality_pamap == mod
        test_mask = modality_stm == mod
        X_train_mod = X_pamap[train_mask]
        y_train_mod = y_train_act[train_mask]
        X_test_mod = X_stm[test_mask]
        y_test_mod = y_test_act[test_mask]
        print(f"Modality {modality_names[mod]}: train samples =", X_train_mod.shape[0], "test samples =", X_test_mod.shape[0])
        print(f"Modality {modality_names[mod]}: unique train activities =", np.unique(y_train_mod))
        print(f"Modality {modality_names[mod]}: unique test activities =", np.unique(y_test_mod))
        print("Unique modality_pamap:", np.unique(modality_pamap))
        print("Unique modality_stm:", np.unique(modality_stm))
        # Modality analysis (Accel vs Gyro)
        f.write('--- Modality Classification Feature Analysis ---\n')
    
    
    # Combine train sets
    X_train = np.concatenate([X_pamap, X_uci], axis=0)
    y_train = np.concatenate([modality_pamap, modality_uci], axis=0)
    X_test = np.array(X_stm)
    y_test = np.array(modality_stm)
    for idx in range(X_train.shape[1]):
        train_accel = X_train[np.array(y_train) == 0, idx]
        train_gyro = X_train[np.array(y_train) == 1, idx]
        test_accel = X_test[np.array(y_test) == 0, idx]
        test_gyro = X_test[np.array(y_test) == 1, idx]
        t_stat_train, p_train = ttest_ind(train_accel, train_gyro, equal_var=False)
        t_stat_test, p_test = ttest_ind(test_accel, test_gyro, equal_var=False)
        # Calculate std, mean diff, domain shift
        std_accel_train = np.std(train_accel)
        std_gyro_train = np.std(train_gyro)
        std_accel_test = np.std(test_accel)
        std_gyro_test = np.std(test_gyro)
        mean_diff_train = np.mean(train_accel) - np.mean(train_gyro)
        mean_diff_test = np.mean(test_accel) - np.mean(test_gyro)
        domain_shift_accel = np.mean(test_accel) - np.mean(train_accel)
        domain_shift_gyro = np.mean(test_gyro) - np.mean(train_gyro)
        f.write(f'Feature {idx} ({feature_names[idx]}):\n')
        f.write(f'  Train: mean_accel={np.mean(train_accel):.3f}, std_accel={std_accel_train:.3f}, mean_gyro={np.mean(train_gyro):.3f}, std_gyro={std_gyro_train:.3f}, mean_diff={mean_diff_train:.3f}, t={t_stat_train:.2f}, p={p_train:.2e}\n')
        f.write(f'  Test:  mean_accel={np.mean(test_accel):.3f}, std_accel={std_accel_test:.3f}, mean_gyro={np.mean(test_gyro):.3f}, std_gyro={std_gyro_test:.3f}, mean_diff={mean_diff_test:.3f}, t={t_stat_test:.2f}, p={p_test:.2e}\n')
        f.write(f'  Domain shift: accel={domain_shift_accel:.3f}, gyro={domain_shift_gyro:.3f}\n')
        
    f.write('\n--- Placement Classification Feature Analysis (no UCI) ---\n')
    # Placement analysis (PAMAP2 + STM only)
    X_train = np.array(X_pamap)
    y_train = np.array(placement_pamap)
    X_test = np.array(X_stm)
    y_test = np.array(placement_stm)
    unique_places = list(set(list(y_train) + list(y_test)))
    for idx in range(X_train.shape[1]):
        f.write(f'Feature {idx} ({feature_names[idx]}):\n')
        for place in unique_places:
            train_vals = X_train[y_train == place, idx]
            test_vals = X_test[y_test == place, idx]
            train_mean = np.mean(train_vals) if len(train_vals) > 0 else float('nan')
            test_mean = np.mean(test_vals) if len(test_vals) > 0 else float('nan')
            train_std = np.std(train_vals) if len(train_vals) > 0 else float('nan')
            test_std = np.std(test_vals) if len(test_vals) > 0 else float('nan')
            domain_shift = test_mean - train_mean if not np.isnan(test_mean) and not np.isnan(train_mean) else float('nan')
            f.write(f'  Place {place}: train_mean={train_mean:.3f}, train_std={train_std:.3f}, test_mean={test_mean:.3f}, test_std={test_std:.3f}, domain_shift={domain_shift:.3f}\n')
        
    f.write('\nAnalysis complete.\n')

    f.write('\n--- Activity Classification Feature Analysis (PAMAP2 + STM) ---\n')
    X_train = np.array(X_pamap)
    y_train = np.array([activity_map_unified(a) for a in activity_pamap])
    X_test = np.array(X_stm)
    y_test = np.array([activity_map_unified(a) for a in activity_stm])
    unique_acts = sorted(set(list(y_train) + list(y_test)))
    for idx in range(X_train.shape[1]):
        f.write(f'Feature {idx} ({feature_names[idx]}):\n')
        for act in unique_acts:
            train_mask = (y_train == act)
            test_mask = (y_test == act)
            # Debug: print shapes
            print(f"      Debug: X_train shape={X_train.shape}, train_mask sum={np.sum(train_mask)}")
            print(f"      Debug: X_test shape={X_test.shape}, test_mask sum={np.sum(test_mask)}")
            # Only index if array is 2D and mask is not empty
            if X_train.ndim == 2 and np.sum(train_mask) > 0:
                train_vals = X_train[train_mask, idx]
            else:
                train_vals = np.array([])
            if X_test.ndim == 2 and np.sum(test_mask) > 0:
                test_vals = X_test[test_mask, idx]
            else:
                test_vals = np.array([])
            train_mean = np.mean(train_vals) if len(train_vals) > 0 else float('nan')
            test_mean = np.mean(test_vals) if len(test_vals) > 0 else float('nan')
            train_std = np.std(train_vals) if len(train_vals) > 0 else float('nan')
            test_std = np.std(test_vals) if len(test_vals) > 0 else float('nan')
            domain_shift = test_mean - train_mean if not np.isnan(test_mean) and not np.isnan(train_mean) else float('nan')
            f.write(f'  Activity {act}: train_mean={train_mean if not np.isnan(train_mean) else "NA"}, train_std={train_std if not np.isnan(train_std) else "NA"}, test_mean={test_mean if not np.isnan(test_mean) else "NA"}, test_std={test_std if not np.isnan(test_std) else "NA"}, domain_shift={domain_shift if not np.isnan(domain_shift) else "NA"}\n')
        # 
    f.write('\nActivity analysis complete.\n')

    f.write('\n--- Modality-wise Activity Classification Feature Analysis (PAMAP2 + STM) ---\n')
    # For train (PAMAP2), test (STM)
    modalities = [0, 1]  # 0: Accel, 1: Gyro
    modality_names = {0: 'Accel', 1: 'Gyro'}
    # Map activities for unified analysis
    y_train_act = np.concatenate([
        np.array([activity_map_unified(a) for a in activity_pamap]),
        np.array([activity_map_unified(a) for a in activity_uci])
    ])
    y_test_act = np.array([activity_map_unified(a) for a in activity_stm])
    # Use only common activities
    common_acts = sorted(set(y_train_act) & set(y_test_act))
    # Concatenate train features and modality labels
    X_train_all = np.concatenate([X_pamap, X_uci], axis=0)
    modality_train_all = np.concatenate([modality_pamap, modality_uci], axis=0)
    for mod in modalities:
        f.write(f'\nModality: {modality_names[mod]}\n')
        train_mod_mask = modality_train_all == mod
        test_mod_mask = modality_stm == mod
        X_train_mod = X_train_all[train_mod_mask]
        y_train_mod = y_train_act[train_mod_mask]
        X_test_mod = X_stm[test_mod_mask]
        y_test_mod = y_test_act[test_mod_mask]
        print(f"Modality {modality_names[mod]}: train samples =", X_train_mod.shape[0], "test samples =", X_test_mod.shape[0])
        print(f"Modality {modality_names[mod]}: unique train activities =", np.unique(y_train_mod))
        print(f"Modality {modality_names[mod]}: unique test activities =", np.unique(y_test_mod))
        if X_train_mod.shape[0] == 0 or X_test_mod.shape[0] == 0 or len(X_train_mod.shape) < 2:
            f.write('  No samples or features for this modality in train or test.\n')
            continue
        print(f"Unique test activities for modality {mod}: {np.unique(y_test_mod)}")
        print(f"Common acts: {common_acts}")
        for idx in range(X_train_mod.shape[1]):
            f.write(f'Feature {idx} ({feature_names[idx]}):\n')
            for act in common_acts:
                train_act_mask = np.array([str(a).strip().lower() == str(act).strip().lower() for a in y_train_mod])
                test_act_mask = np.array([str(a).strip().lower() == str(act).strip().lower() for a in y_test_mod])
                if X_train_mod.ndim == 2 and np.sum(train_act_mask) > 0:
                    train_vals = X_train_mod[train_act_mask, idx]
                else:
                    train_vals = np.array([])
                if X_test_mod.ndim == 2 and np.sum(test_act_mask) > 0:
                    test_vals = X_test_mod[test_act_mask, idx]
                else:
                    test_vals = np.array([])
                train_mean = np.mean(train_vals) if len(train_vals) > 0 else float('nan')
                test_mean = np.mean(test_vals) if len(test_vals) > 0 else float('nan')
                train_std = np.std(train_vals) if len(train_vals) > 0 else float('nan')
                test_std = np.std(test_vals) if len(test_vals) > 0 else float('nan')
                domain_shift = test_mean - train_mean if not np.isnan(test_mean) and not np.isnan(train_mean) else float('nan')
                f.write(f'  Activity {act}: train_mean={train_mean if not np.isnan(train_mean) else "NA"}, train_std={train_std if not np.isnan(train_std) else "NA"}, test_mean={test_mean if not np.isnan(test_mean) else "NA"}, test_std={test_std if not np.isnan(test_std) else "NA"}, domain_shift={domain_shift if not np.isnan(domain_shift) else "NA"}\n')
        
    f.write('\nModality-wise activity analysis complete.\n')
