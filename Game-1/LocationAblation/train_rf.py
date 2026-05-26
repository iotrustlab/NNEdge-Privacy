"""
Game-1 Random Forest Script with Sensor Location Ablation Study
- Only uses 'dec_tree_out_1' (Game-1 setting)
- Cross-User (multiple ratios), Intra-User, LOUO, and LOLO evaluations
- Trains on one location, tests on the other four or vice versa
"""

import os
import re
import glob
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, mutual_info_score
)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from scipy.special import rel_entr
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------- CONFIG --------
data_root = os.environ.get("NNEDGE_PRIVACY_STM_ROOT", os.path.join(REPO_ROOT, "STMDATASET", "Data"))
game = "Game-1"
model_name = "RF"
locations = ["left-ankle", "left-wrist", "right-ankle", "right-pocket", "right-wrist"]
window_size = 20
ratios = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]
VALID_ACTIVITIES = {"jogging", "laying", "sitting", "standing", "upstairs", "downstairs", "walking"}

# -------- HELPERS --------
def create_optimized_rf(X_train, y_train):
    # Define the Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [200, 300],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }

    # Use GridSearchCV to find best parameters
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted'
    )
    
    # Fit and return the best model
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def extract_features(df):
    feats = []
    for i in range(0, len(df) - window_size + 1, window_size):
        win = df[['dec_tree_out_1']].iloc[i:i + window_size]
        
        # Statistical features
        basic_stats = win.agg(['mean', 'std', 'min', 'max', 'median', 'skew']).values.flatten()
        
        # Time-domain features
        values = win['dec_tree_out_1'].values
        diff = np.diff(values)
        
        # Peak features
        peaks = np.where(diff[:-1] * diff[1:] < 0)[0]
        peak_count = len(peaks)
        mean_peak_distance = np.mean(np.diff(peaks)) if peak_count > 1 else 0
        
        # Shape features
        rms = np.sqrt(np.mean(np.square(values)))
        crest_factor = np.max(np.abs(values)) / rms if rms != 0 else 0
        
        # Percentile features
        percentiles = np.percentile(values, [10, 25, 75, 90])
        
        # Combine all features
        window_features = np.concatenate([
            basic_stats,
            [np.mean(np.abs(diff)), np.std(diff)],  # Diff statistics
            [peak_count, mean_peak_distance],  # Peak features
            [rms, crest_factor],  # Shape features
            percentiles,  # Percentile features
        ])
        
        feats.append(window_features)
    return np.array(feats)

def get_user_dirs(base):
    user_dirs = [d for d in os.listdir(base) if re.match(r"User \d+$", d)]
    return sorted(user_dirs, key=lambda x: int(re.findall(r"\d+", x)[0]))

def compute_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mi = mutual_info_score(y_true, y_pred)
    p_true = np.bincount([labels.index(y) for y in y_true], minlength=len(labels)) / len(y_true)
    p_pred = np.bincount([labels.index(y) for y in y_pred], minlength=len(labels)) / len(y_pred)
    kl = np.sum(rel_entr(p_true + 1e-12, p_pred + 1e-12))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "mutual_information": float(mi),
        "kl_divergence": float(kl),
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def save_results(y_true, y_pred, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Macro Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Macro Recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Macro F1-score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"KL Divergence: {metrics['kl_divergence']:.4f}\n")
        f.write(f"Mutual Information: {metrics['mutual_information']:.4f}\n")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))

def load_data_by_location():
    data = {loc: {} for loc in locations}
    print("🗂️ Starting data loading...")

    users = get_user_dirs(data_root)
    print(f"🔍 Found {len(users)} users: {users}")

    for user in users:
        user_path = os.path.join(data_root, user, "Processed")
        print(f"\n📁 Processing user: {user}")
        if not os.path.isdir(user_path):
            print(f"❌ Missing user path: {user_path}")
            continue

        try:
            activities = os.listdir(user_path)
        except Exception as e:
            print(f"❌ Could not list activities in {user_path}: {e}")
            continue

        for activity in activities:
            activity_path = os.path.join(user_path, activity)
            if not os.path.isdir(activity_path):
                print(f"🚫 Skipping non-dir: {activity_path}")
                continue

            activity_name = activity.lower()
            if activity_name not in VALID_ACTIVITIES:
                print(f"⚠️ Skipping unknown activity '{activity}'")
                continue

            print(f"✅ Found activity: {activity_name}")

            for loc in locations:
                loc_path = os.path.join(activity_path, loc)
                feats, labels = [], []

                # --- Case 1: Folder named after location ---
                if os.path.isdir(loc_path):
                    print(f"📂 Reading files from folder: {loc_path}")
                    files = glob.glob(os.path.join(loc_path, "*.csv"))
                else:
                    # --- Case 2: Look for matching single files directly in activity folder ---
                    pattern = os.path.join(activity_path, f"*{loc}*.csv")
                    files = glob.glob(pattern)
                    if files:
                        print(f"📄 Found direct files matching {loc} in {activity_path}: {len(files)}")

                for file in files:
                    print(f"   📄 Reading file: {file}")
                    try:
                        df = pd.read_csv(file)
                        if 'dec_tree_out_1' not in df.columns:
                            print(f"   ⚠️ 'dec_tree_out_1' missing in {file}")
                            continue
                        if df['dec_tree_out_1'].isnull().any():
                            print(f"   ⚠️ Null values in 'dec_tree_out_1' in {file}")
                            continue

                        f = extract_features(df)
                        if f.size == 0:
                            print(f"   ⚠️ No features extracted from {file}")
                            continue

                        feats.append(f)
                        labels.extend([activity_name] * f.shape[0])
                        print(f"   ✅ {f.shape[0]} windows extracted")

                    except Exception as e:
                        print(f"   ❌ Error reading file {file}: {e}")
                        continue

                if feats:
                    if user not in data[loc]:
                        data[loc][user] = (np.vstack(feats), np.array(labels))
                    else:
                        X_prev, y_prev = data[loc][user]
                        X_new = np.vstack(feats)
                        y_new = np.array(labels)
                        data[loc][user] = (
                            np.vstack([X_prev, X_new]),
                            np.concatenate([y_prev, y_new])
                        )
                    print(f"   📊 Loaded {len(labels)} samples for user {user} at location {loc}")

    # Final per-location summary
    for loc in data:
        print(f"\n📦 Summary for location '{loc}': {len(data[loc])} users loaded.")

    return data


# -------- LEAVE-ONE-LOCATION-OUT (ALL CASES) --------
def run_lo_location():
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "leave-one-location-out")
    os.makedirs(output_root, exist_ok=True)

    for test_loc in locations:
        print(f"\n🚨 LOLO: Testing on {test_loc}")

        # --- Cross-User ---
        for train_ratio, test_ratio in ratios:
            subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
            out_path = os.path.join(output_root, subdir, f"test_on_{test_loc}")
            train_users, test_users = [], []
            for loc in locations:
                users = list(data_by_loc[loc].keys())
                n_train = int(train_ratio * len(users))
                train_users += [(loc, u) for u in users[:n_train] if loc != test_loc]
                if loc == test_loc:
                    test_users += [(loc, u) for u in users[n_train:]]

            X_train, y_train, X_test, y_test = [], [], [], []
            for loc, user in train_users:
                X, y = data_by_loc[loc][user]
                X_train.append(X)
                y_train.extend(y)
            for loc, user in test_users:
                X, y = data_by_loc[loc][user]
                X_test.append(X)
                y_test.extend(y)

            if X_train and X_test:
                X_train = np.vstack(X_train)
                X_test = np.vstack(X_test)
                scaler = StandardScaler().fit(X_train)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_test))
                labels_sorted = sorted(list(set(y_test) | set(y_pred)))
                save_results(y_test, y_pred.tolist(), labels_sorted, out_path)

        # --- Intra-User ---
        intra_preds, intra_trues = [], []
        for user in data_by_loc.get(test_loc, {}):
            X, y = data_by_loc[test_loc][user]
            if len(X) < 2 or len(np.unique(y)) < 2:
                continue
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
            scaler = StandardScaler().fit(X_tr)
            model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_tr), y_tr)
            y_pred = model.predict(scaler.transform(X_te))
            intra_preds.extend(y_pred)
            intra_trues.extend(y_te)

        if intra_preds:
            intra_path = os.path.join(output_root, "intra-user", f"test_on_{test_loc}")
            save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

        # --- Leave-One-User-Out ---
        all_users = list(data_by_loc.get(test_loc, {}).keys())
        louo_preds, louo_trues = [], []
        for test_user in all_users:
            train_users = [u for u in all_users if u != test_user]
            X_train, y_train = [], []
            for u in train_users:
                X, y = data_by_loc[test_loc][u]
                X_train.append(X)
                y_train.extend(y)
            X_test, y_test = data_by_loc[test_loc][test_user]
            if X_train and len(X_test) > 0:
                X_train = np.vstack(X_train)
                scaler = StandardScaler().fit(X_train)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_test))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"test_on_{test_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)



def run_train_one_test_others():
    data_by_loc = load_data_by_location()
    output_root = os.path.join(data_root, "Results", game, model_name, "train-one-test-others")
    os.makedirs(output_root, exist_ok=True)

    for train_loc in locations:
        test_locs = [loc for loc in locations if loc != train_loc]
        print(f"\n🚨 Training on {train_loc}, testing on: {test_locs}")

        # --- Cross-User ---
        for train_ratio, test_ratio in ratios:
            subdir = f"cross-user/train{int(train_ratio*100)}_test{int(test_ratio*100)}"
            out_path = os.path.join(output_root, subdir, f"train_on_{train_loc}")
            train_users = list(data_by_loc[train_loc].keys())
            n_train = int(train_ratio * len(train_users))
            train_users = train_users[:n_train]

            test_users = []
            for loc in test_locs:
                test_users += [(loc, u) for u in data_by_loc[loc].keys()]

            X_train, y_train, X_test, y_test = [], [], [], []
            for u in train_users:
                X, y = data_by_loc[train_loc][u]
                X_train.append(X)
                y_train.extend(y)

            for loc, u in test_users:
                X, y = data_by_loc[loc][u]
                X_test.append(X)
                y_test.extend(y)

            if X_train and X_test:
                X_train = np.vstack(X_train)
                X_test = np.vstack(X_test)
                scaler = StandardScaler().fit(X_train)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_test))
                labels_sorted = sorted(list(set(y_test) | set(y_pred)))
                save_results(y_test, y_pred.tolist(), labels_sorted, out_path)

        # --- Intra-User ---
        intra_preds, intra_trues = [], []
        for loc in test_locs:
            for user in data_by_loc[loc]:
                X, y = data_by_loc[loc][user]
                if len(X) < 2 or len(np.unique(y)) < 2:
                    continue
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
                X_train, y_train = [], []
                for u in data_by_loc[train_loc]:
                    X_tmp, y_tmp = data_by_loc[train_loc][u]
                    X_train.append(X_tmp)
                    y_train.extend(y_tmp)

                if not X_train:
                    continue
                X_train = np.vstack(X_train)
                scaler = StandardScaler().fit(X_train)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_te))
                intra_preds.extend(y_pred)
                intra_trues.extend(y_te)

        if intra_preds:
            intra_path = os.path.join(output_root, "intra-user", f"train_on_{train_loc}")
            save_results(intra_trues, intra_preds, sorted(set(intra_trues)), intra_path)

        # --- Leave-One-User-Out ---
        louo_preds, louo_trues = [], []
        for loc in test_locs:
            test_users = list(data_by_loc[loc].keys())
            for test_user in test_users:
                X_test, y_test = data_by_loc[loc][test_user]
                X_train, y_train = [], []
                for u in data_by_loc[train_loc]:
                    X_tmp, y_tmp = data_by_loc[train_loc][u]
                    X_train.append(X_tmp)
                    y_train.extend(y_tmp)

                if not X_train or len(X_test) == 0:
                    continue
                X_train = np.vstack(X_train)
                scaler = StandardScaler().fit(X_train)
                model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train), y_train)
                y_pred = model.predict(scaler.transform(X_test))
                louo_preds.extend(y_pred)
                louo_trues.extend(y_test)

        if louo_preds:
            louo_path = os.path.join(output_root, "leave-one-user-out", f"train_on_{train_loc}")
            save_results(louo_trues, louo_preds, sorted(set(louo_trues)), louo_path)



# To run LOLO across all 3 use cases:
run_lo_location()
run_train_one_test_others()

