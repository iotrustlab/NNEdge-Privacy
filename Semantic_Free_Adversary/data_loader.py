import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

PAMAP2_ROOT = "../Processed_PAMAP"
UCI_ROOT = "../Processed_UCI"
STM_ROOT = "../STMDATASET/Data"
CACHE_DIR = "Cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_features(window):
    feats = []
    for i in range(window.shape[1]):
        x = window[:, i]
        feats.extend([
            np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
            np.ptp(x), np.sum(x**2), np.sum(np.abs(np.fft.rfft(x))),
            np.argmax(np.abs(np.fft.rfft(x))), np.mean(np.abs(np.fft.rfft(x))),
            ((np.diff(np.sign(x)) != 0).sum())
        ])
    return np.array(feats)

def process_pamap2(axis_perm=None):
    features, modality_labels, placement_labels, activity_labels, window_metadata = [], [], [], [], []
    for subj in os.listdir(PAMAP2_ROOT):
        subj_path = os.path.join(PAMAP2_ROOT, subj)
        if not os.path.isdir(subj_path): continue
        for act in os.listdir(subj_path):
            act_path = os.path.join(subj_path, act)
            if not os.path.isdir(act_path): continue
            for place_file in os.listdir(act_path):
                raw_place = place_file.replace('.csv','')
                place_path = os.path.join(act_path, place_file)
                df = pd.read_csv(place_path)
                feat_cols = [c for c in df.columns if c != 'Timestamp']
                accel_cols = [c for c in feat_cols if 'accel' in c.lower()]
                gyro_cols = [c for c in feat_cols if 'gyro' in c.lower()]
                if len(accel_cols) >= 3 and len(gyro_cols) >= 3:
                    data_accel = df[accel_cols[:3]].values
                    data_gyro = df[gyro_cols[:3]].values
                    if axis_perm is not None:
                        data_accel = data_accel[:, axis_perm[:3]]
                        data_gyro = data_gyro[:, axis_perm[:3]]
                    win_size = 250
                    step = 125
                    for start in range(0, len(data_accel)-win_size+1, step):
                        window_accel = data_accel[start:start+win_size]
                        window_gyro = data_gyro[start:start+win_size]
                        feats_accel = extract_features(window_accel)
                        feats_gyro = extract_features(window_gyro)
                        if np.isnan(feats_accel).any() or np.isnan(feats_gyro).any():
                            continue
                        features.append(feats_accel)
                        modality_labels.append(0)
                        placement_labels.append(raw_place)
                        activity_labels.append(act)
                        window_metadata.append({"file": place_path, "start": start, "end": start+win_size, "modality": "accel"})
                        features.append(feats_gyro)
                        modality_labels.append(1)
                        placement_labels.append(raw_place)
                        activity_labels.append(act)
                        window_metadata.append({"file": place_path, "start": start, "end": start+win_size, "modality": "gyro"})
    with open(os.path.join(CACHE_DIR, "pamap2_cache.pkl"), "wb") as f:
        pickle.dump((features, modality_labels, placement_labels, activity_labels, window_metadata), f)
    print("✅ PAMAP2 cache saved.")

def process_uci(axis_perm=None):
    features, modality_labels, placement_labels, activity_labels, window_metadata = [], [], [], [], []
    for subj in os.listdir(UCI_ROOT):
        subj_path = os.path.join(UCI_ROOT, subj)
        if not os.path.isdir(subj_path): continue
        for act in os.listdir(subj_path):
            act_path = os.path.join(subj_path, act)
            if not os.path.isdir(act_path): continue
            for win_file in os.listdir(act_path):
                win_path = os.path.join(act_path, win_file)
                df = pd.read_csv(win_path)
                accel_cols = [c for c in df.columns if 'acc' in c.lower()]
                gyro_cols = [c for c in df.columns if 'gyro' in c.lower()]
                if len(accel_cols) >= 3 and len(gyro_cols) >= 3:
                    data_accel = df[accel_cols[:3]].values
                    data_gyro = df[gyro_cols[:3]].values
                    if axis_perm is not None:
                        data_accel = data_accel[:, axis_perm[:3]]
                        data_gyro = data_gyro[:, axis_perm[:3]]
                    win_size = 250
                    step = 125
                    if len(data_accel) < win_size or len(data_gyro) < win_size:
                        # Process entire file as one window
                        feats_accel = extract_features(data_accel)
                        feats_gyro = extract_features(data_gyro)
                        if np.isnan(feats_accel).any() or np.isnan(feats_gyro).any():
                            continue
                        features.append(feats_accel)
                        modality_labels.append(0)
                        placement_labels.append('UCI')
                        activity_labels.append(act)
                        window_metadata.append({"file": win_path, "start": 0, "end": len(data_accel), "modality": "accel"})
                        features.append(feats_gyro)
                        modality_labels.append(1)
                        placement_labels.append('UCI')
                        activity_labels.append(act)
                        window_metadata.append({"file": win_path, "start": 0, "end": len(data_gyro), "modality": "gyro"})
                    else:
                        for start in range(0, len(data_accel)-win_size+1, step):
                            window_accel = data_accel[start:start+win_size]
                            window_gyro = data_gyro[start:start+win_size]
                            feats_accel = extract_features(window_accel)
                            feats_gyro = extract_features(window_gyro)
                            if np.isnan(feats_accel).any() or np.isnan(feats_gyro).any():
                                continue
                            features.append(feats_accel)
                            modality_labels.append(0)
                            placement_labels.append('UCI')
                            activity_labels.append(act)
                            window_metadata.append({"file": win_path, "start": start, "end": start+win_size, "modality": "accel"})
                            features.append(feats_gyro)
                            modality_labels.append(1)
                            placement_labels.append('UCI')
                            activity_labels.append(act)
                            window_metadata.append({"file": win_path, "start": start, "end": start+win_size, "modality": "gyro"})
    with open(os.path.join(CACHE_DIR, "uci_har_cache.pkl"), "wb") as f:
        pickle.dump((features, modality_labels, placement_labels, activity_labels, window_metadata), f)
    print("✅ UCI-HAR cache saved.")

def process_stm(axis_perm=None):
    features, modality_labels, placement_labels, activity_labels, window_metadata = [], [], [], [], []
    for user in os.listdir(STM_ROOT):
        user_path = os.path.join(STM_ROOT, user, "Processed")
        if not os.path.isdir(user_path): continue
        for act in os.listdir(user_path):
            act_path = os.path.join(user_path, act)
            if not os.path.isdir(act_path): continue
            for item in os.listdir(act_path):
                item_path = os.path.join(act_path, item)
                if os.path.isdir(item_path):
                    # Placement folder: iterate CSVs inside
                    for csv_file in os.listdir(item_path):
                        if not csv_file.endswith('.csv'): continue
                        raw_place = csv_file.split('_')[0]
                        csv_path = os.path.join(item_path, csv_file)
                        df = pd.read_csv(csv_path)
                        feat_cols = [c for c in df.columns if c not in ['time[us]', 'dec_tree_out_1']]
                        accel_cols = [c for c in feat_cols if 'acc' in c]
                        gyro_cols = [c for c in feat_cols if 'gyro' in c]
                        if len(accel_cols) == 3 and len(gyro_cols) == 3:
                            data_accel = df[accel_cols].values * 0.00981
                            data_gyro = df[gyro_cols].values * 0.001 * (3.141592653589793/180)
                            if axis_perm is not None:
                                data_accel = data_accel[:, axis_perm[:3]]
                                data_gyro = data_gyro[:, axis_perm[:3]]
                            win_size = 250
                            step = 125
                            for start in range(0, len(data_accel)-win_size+1, step):
                                window_accel = data_accel[start:start+win_size]
                                window_gyro = data_gyro[start:start+win_size]
                                feats_accel = extract_features(window_accel)
                                feats_gyro = extract_features(window_gyro)
                                if np.isnan(feats_accel).any() or np.isnan(feats_gyro).any():
                                    continue
                                features.append(feats_accel)
                                modality_labels.append(0)
                                placement_labels.append(raw_place)
                                activity_labels.append(act)
                                window_metadata.append({"file": csv_path, "start": start, "end": start+win_size, "modality": "accel"})
                                features.append(feats_gyro)
                                modality_labels.append(1)
                                placement_labels.append(raw_place)
                                activity_labels.append(act)
                                window_metadata.append({"file": csv_path, "start": start, "end": start+win_size, "modality": "gyro"})
                elif item.endswith('.csv'):
                    # Direct CSV file in activity folder
                    raw_place = item.replace('.csv','')
                    df = pd.read_csv(item_path)
                    feat_cols = [c for c in df.columns if c not in ['time[us]', 'dec_tree_out_1']]
                    accel_cols = [c for c in feat_cols if 'acc' in c]
                    gyro_cols = [c for c in feat_cols if 'gyro' in c]
                    if len(accel_cols) == 3 and len(gyro_cols) == 3:
                        data_accel = df[accel_cols].values * 0.00981
                        data_gyro = df[gyro_cols].values * 0.001 * (3.141592653589793/180)
                        if axis_perm is not None:
                            data_accel = data_accel[:, axis_perm[:3]]
                            data_gyro = data_gyro[:, axis_perm[:3]]
                        win_size = 250
                        step = 125
                        for start in range(0, len(data_accel)-win_size+1, step):
                            window_accel = data_accel[start:start+win_size]
                            window_gyro = data_gyro[start:start+win_size]
                            feats_accel = extract_features(window_accel)
                            feats_gyro = extract_features(window_gyro)
                            if np.isnan(feats_accel).any() or np.isnan(feats_gyro).any():
                                continue
                            features.append(feats_accel)
                            modality_labels.append(0)
                            placement_labels.append(raw_place)
                            activity_labels.append(act)
                            window_metadata.append({"file": item_path, "start": start, "end": start+win_size, "modality": "accel"})
                            features.append(feats_gyro)
                            modality_labels.append(1)
                            placement_labels.append(raw_place)
                            activity_labels.append(act)
                            window_metadata.append({"file": item_path, "start": start, "end": start+win_size, "modality": "gyro"})
    with open(os.path.join(CACHE_DIR, "stm_cache.pkl"), "wb") as f:
        pickle.dump((features, modality_labels, placement_labels, activity_labels, window_metadata), f)
    print("✅ STM cache saved.")

if __name__ == "__main__":
    axis_perm = [2,0,1,5,3,4]
    process_pamap2(axis_perm)
    process_uci(axis_perm)
    process_stm(axis_perm)
