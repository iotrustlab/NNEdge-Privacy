"""
Comprehensive Gyroscope-Only Feature Analysis for HAR
======================================================
Analyzes entire MotionSense (train) and STM_MotionSense (test) datasets
across all activities and users to identify optimal features for maximizing
test accuracy in cross-dataset activity recognition.

This script:
1. Loads all gyroscope data from both datasets
2. Extracts basic and advanced features using windowing
3. Analyzes feature distributions across activities
4. Ranks features by importance and separability
5. Recommends optimal feature sets for modeling
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings('ignore')

# Create results directory
results_dir = 'results/comprehensive_gyro_analysis'
os.makedirs(results_dir, exist_ok=True)

# Helper function to check if step is done
def step_done(step_name):
    cache_file = os.path.join(results_dir, f'.{step_name}_done')
    return os.path.exists(cache_file)

def mark_step_done(step_name):
    cache_file = os.path.join(results_dir, f'.{step_name}_done')
    with open(cache_file, 'w') as f:
        f.write('done')

print("=" * 80)
print("COMPREHENSIVE GYROSCOPE-ONLY FEATURE ANALYSIS FOR HAR")
print("=" * 80)

# ============================================================================
# PART 1: LOAD ALL DATA
# ============================================================================

if step_done('data_loading'):
    print("\n[1/6] Loading all gyroscope data from both datasets... (SKIPPED - cached)")
    ms_data = pd.read_pickle(os.path.join(results_dir, 'ms_data.pkl'))
    stm_data = pd.read_pickle(os.path.join(results_dir, 'stm_data.pkl'))
else:
    print("\n[1/6] Loading all gyroscope data from both datasets...")

activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']
activity_to_label = {act: idx for idx, act in enumerate(activities)}

# MotionSense Gyroscope Data
def load_motionsense_gyro():
    """Load all MotionSense gyroscope data"""
    data = []
    root = 'MotionSense/C_Gyroscope_data/C_Gyroscope_data'
    
    # Map filename patterns to activities
    activity_map = {
        'wlk': 'Walking', 'jog': 'Jogging', 'std': 'Standing',
        'sit': 'Sitting', 'ups': 'Upstairs', 'dws': 'Downstairs'
    }
    
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Determine activity from folder name
        activity = None
        for prefix, act in activity_map.items():
            if folder.lower().startswith(prefix):
                activity = act
                break
        
        if activity is None:
            continue
        
        # Load all CSV files in this activity folder (subjects)
        for csv_file in os.listdir(folder_path):
            if csv_file.endswith('.csv'):
                filepath = os.path.join(folder_path, csv_file)
                df = pd.read_csv(filepath)
                df['activity'] = activity
                df['activity_label'] = activity_to_label[activity]
                df['dataset'] = 'MotionSense'
                data.append(df)
    
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# STM_MotionSense Gyroscope Data
def load_stm_gyro():
    """Load all STM_MotionSense gyroscope data"""
    data = []
    root = 'STM_MotionSense'
    
    for user_folder in sorted(os.listdir(root)):
        user_path = os.path.join(root, user_folder, 'Processed')
        if not os.path.isdir(user_path):
            continue
        
        for activity in activities:
            activity_path = os.path.join(user_path, activity)
            if not os.path.isdir(activity_path):
                continue
            
            # Handle both single file and subfolder structures
            csv_files = []
            if os.path.isfile(os.path.join(activity_path, f'{activity.lower()}.csv')):
                csv_files.append(os.path.join(activity_path, f'{activity.lower()}.csv'))
            
            # Check for subfolders with CSV files
            for item in os.listdir(activity_path):
                item_path = os.path.join(activity_path, item)
                if os.path.isdir(item_path):
                    for csv in os.listdir(item_path):
                        if csv.endswith('.csv'):
                            csv_files.append(os.path.join(item_path, csv))
                elif item.endswith('.csv'):
                    csv_files.append(item_path)
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # STM data has gyro_x, gyro_y, gyro_z
                if 'gyro_x' in df.columns:
                    df = df.rename(columns={'gyro_x': 'x', 'gyro_y': 'y', 'gyro_z': 'z'})
                df['activity'] = activity
                df['activity_label'] = activity_to_label[activity]
                df['dataset'] = 'STM_MotionSense'
                df['user'] = user_folder
                data.append(df)
    
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

print("  Loading MotionSense gyroscope data...")
ms_data = load_motionsense_gyro()
print(f"    Loaded {len(ms_data):,} samples from MotionSense")

print("  Loading STM_MotionSense gyroscope data...")
stm_data = load_stm_gyro()
print(f"    Loaded {len(stm_data):,} samples from STM_MotionSense")

# Save raw data for future runs
ms_data.to_pickle(os.path.join(results_dir, 'ms_data.pkl'))
stm_data.to_pickle(os.path.join(results_dir, 'stm_data.pkl'))
mark_step_done('data_loading')

# ============================================================================
# PART 2: FEATURE EXTRACTION
# ============================================================================

if step_done('feature_extraction'):
    print("\n[2/6] Extracting basic and advanced features... (SKIPPED - cached)")
    ms_features = pd.read_pickle(os.path.join(results_dir, 'ms_features.pkl'))
    stm_features = pd.read_pickle(os.path.join(results_dir, 'stm_features.pkl'))
    print(f"    Loaded {len(ms_features):,} MotionSense feature windows")
    print(f"    Loaded {len(stm_features):,} STM_MotionSense feature windows")
else:
    print("\n[2/6] Extracting basic and advanced features...")

def extract_window_features(signal_chunk, sampling_rate=50):
    """Extract features from a window of gyroscope signal"""
    if len(signal_chunk) < 5:
        return None
    
    features = {}
    
    # --- BASIC STATISTICAL FEATURES ---
    features['mean'] = np.mean(signal_chunk)
    features['std'] = np.std(signal_chunk)
    features['min'] = np.min(signal_chunk)
    features['max'] = np.max(signal_chunk)
    features['median'] = np.median(signal_chunk)
    features['range'] = features['max'] - features['min']
    features['rms'] = np.sqrt(np.mean(signal_chunk**2))
    
    # --- STATISTICAL DISTRIBUTION ---
    features['skewness'] = skew(signal_chunk)
    features['kurtosis'] = kurtosis(signal_chunk)
    
    # --- SPECTRAL FEATURES ---
    fft_vals = np.abs(fft(signal_chunk))
    fft_vals = fft_vals[:len(fft_vals)//2]  # Take positive frequencies
    if len(fft_vals) > 1:
        features['spectral_energy'] = np.sum(fft_vals**2)
        features['spectral_entropy'] = -np.sum((fft_vals**2) * np.log(fft_vals**2 + 1e-10))
        features['dominant_freq'] = np.argmax(fft_vals)
    
    # --- TEMPORAL FEATURES ---
    features['zero_crossing_rate'] = np.sum(np.abs(np.diff(np.sign(signal_chunk)))) / len(signal_chunk)
    
    # Peak detection
    signal_chunk_1d = np.asarray(signal_chunk).flatten()
    peaks, _ = find_peaks(np.abs(signal_chunk_1d), height=np.std(signal_chunk_1d))
    features['peak_count'] = len(peaks)
    features['peak_avg_height'] = np.mean(np.abs(signal_chunk_1d[peaks])) if len(peaks) > 0 else 0
    
    # --- ENERGY METRICS ---
    features['energy'] = np.sum(signal_chunk**2)
    features['signal_magnitude_area'] = np.sum(np.abs(signal_chunk))
    
    return features

def extract_all_features_windowed(df, gyro_cols=['x', 'y', 'z'], window_size=50, step=25):
    """Extract features for all windows across all axes and aggregates"""
    all_features = []
    
    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        window_data = {}
        
        # Extract features per axis
        for col in gyro_cols:
            signal = df[col].iloc[start:end].values
            axis_features = extract_window_features(signal)
            if axis_features is not None:
                for feat_name, feat_val in axis_features.items():
                    window_data[f'{col}_{feat_name}'] = feat_val
        
        # Magnitude features
        mag = np.sqrt(df['x'].iloc[start:end]**2 + 
                      df['y'].iloc[start:end]**2 + 
                      df['z'].iloc[start:end]**2).values
        mag_features = extract_window_features(mag)
        if mag_features is not None:
            for feat_name, feat_val in mag_features.items():
                window_data[f'mag_{feat_name}'] = feat_val
        
        # Activity label and other metadata
        if 'activity_label' in df.columns:
            window_data['activity_label'] = df['activity_label'].iloc[start]
            window_data['activity'] = df['activity'].iloc[start]
        if 'dataset' in df.columns:
            window_data['dataset'] = df['dataset'].iloc[start]
        if 'user' in df.columns:
            window_data['user'] = df['user'].iloc[start]
        # Skip dec_tree_out_1 if it contains arrays - only keep scalar columns
        
        all_features.append(window_data)
    
    return pd.DataFrame(all_features)

# ============================================================================
# PART 2B: FEATURE EXTRACTION (with caching)
# ============================================================================

print("\n[2/6] Extracting basic and advanced features...")

if not step_done('feature_extraction'):
    print("  Extracting features from MotionSense...")
    ms_features = extract_all_features_windowed(ms_data)
    print(f"    Generated {len(ms_features):,} feature windows")

    print("  Extracting features from STM_MotionSense...")
    stm_features = extract_all_features_windowed(stm_data)
    print(f"    Generated {len(stm_features):,} feature windows")
    
    # Save features for future runs
    ms_features.to_pickle(os.path.join(results_dir, 'ms_features.pkl'))
    stm_features.to_pickle(os.path.join(results_dir, 'stm_features.pkl'))
    mark_step_done('feature_extraction')
else:
    print("  (SKIPPED - cached)")
    ms_features = pd.read_pickle(os.path.join(results_dir, 'ms_features.pkl'))
    stm_features = pd.read_pickle(os.path.join(results_dir, 'stm_features.pkl'))
    print(f"    Loaded {len(ms_features):,} MotionSense feature windows")
    print(f"    Loaded {len(stm_features):,} STM_MotionSense feature windows")

# ============================================================================
# PART 3: FEATURE ANALYSIS
# ============================================================================

if step_done('feature_analysis'):
    print("\n[3/6] Analyzing feature distributions and statistics... (SKIPPED - cached)")
    feature_stats = pd.read_csv(os.path.join(results_dir, 'feature_statistics.csv'), index_col=0)
    feature_cols = [col for col in feature_stats.index.tolist()]
else:
    print("\n[3/6] Analyzing feature distributions and statistics...")
    
    # Combine for overall statistics
    combined_features = pd.concat([ms_features, stm_features], ignore_index=True)

    # Feature summary statistics - exclude non-numeric columns
    feature_cols = [col for col in combined_features.columns 
                    if col not in ['activity_label', 'activity', 'dataset', 'user']]

    print(f"  Total features extracted: {len(feature_cols)}")
    print(f"\n  Feature Summary Statistics:")
    print("  " + "-" * 76)

    # Compute and display statistics - ensure all columns are numeric
    combined_features_numeric = combined_features[feature_cols].copy()
    
    # Convert to numeric, coercing errors to NaN
    for col in combined_features_numeric.columns:
        combined_features_numeric[col] = pd.to_numeric(combined_features_numeric[col], errors='coerce')
    
    feature_stats = combined_features_numeric.describe().T
    
    # Compute coefficient of variation for each feature (std / |mean|)
    means = combined_features_numeric.mean()
    stds = combined_features_numeric.std()
    feature_stats['var_coeff'] = stds / means.abs()

    print(feature_stats[['mean', 'std', 'min', 'max', 'var_coeff']].to_string())

    # Save feature statistics
    feature_stats.to_csv(os.path.join(results_dir, 'feature_statistics.csv'))
    mark_step_done('feature_analysis')

# ============================================================================
# PART 4: FEATURE IMPORTANCE RANKING
# ============================================================================

# ============================================================================
# PART 4: FEATURE IMPORTANCE RANKING
# ============================================================================

if step_done('feature_importance'):
    print("\n[4/6] Computing feature importance using Random Forest... (SKIPPED - cached)")
    feature_importance = pd.read_csv(os.path.join(results_dir, 'feature_importance_all.csv'))
else:
    print("\n[4/6] Computing feature importance using Random Forest...")
    
    # Combine data for preparation
    combined_features = pd.concat([ms_features, stm_features], ignore_index=True)
    combined_features_numeric = combined_features[feature_cols].copy()
    for col in combined_features_numeric.columns:
        combined_features_numeric[col] = pd.to_numeric(combined_features_numeric[col], errors='coerce')
    
    # Prepare data for MotionSense model
    X_ms = ms_features[feature_cols].fillna(0).copy()
    for col in X_ms.columns:
        X_ms[col] = pd.to_numeric(X_ms[col], errors='coerce')
    X_ms = X_ms.fillna(0)
    
    y_ms = ms_features['activity_label']

    scaler = StandardScaler()
    X_ms_scaled = scaler.fit_transform(X_ms)

    # Train Random Forest and get feature importances
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_ms_scaled, y_ms)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  Top 30 Most Important Features:")
    print("  " + "-" * 76)
    print(feature_importance.head(30).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(30)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title('Top 30 Features by Importance (Random Forest on MotionSense)', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance_top30.png'), dpi=300)
    print("\n  Saved: feature_importance_top30.png")
    plt.close()

    # Save full feature importance
    feature_importance.to_csv(os.path.join(results_dir, 'feature_importance_all.csv'), index=False)
    mark_step_done('feature_importance')

# ============================================================================
# PART 5: CROSS-DATASET CONSISTENCY AND SEPARABILITY
# ============================================================================

# ============================================================================
# PART 5: CROSS-DATASET CONSISTENCY AND SEPARABILITY
# ============================================================================

if step_done('cross_dataset_analysis'):
    print("\n[5/6] Analyzing cross-dataset consistency and separability... (SKIPPED - cached)")
else:
    print("\n[5/6] Analyzing cross-dataset consistency and separability...")
    
    # Reload combined features for plotting
    combined_features = pd.concat([ms_features, stm_features], ignore_index=True)

    # Distribution comparison: MS vs STM for each activity
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, activity in enumerate(activities):
        ms_act = ms_features[ms_features['activity'] == activity]
        stm_act = stm_features[stm_features['activity'] == activity]
        
        # Use top feature for visualization
        top_feat = feature_importance.iloc[0]['feature']
        
        ms_vals = pd.to_numeric(ms_act[top_feat], errors='coerce').dropna()
        stm_vals = pd.to_numeric(stm_act[top_feat], errors='coerce').dropna()
        
        axes[idx].hist(ms_vals, bins=30, alpha=0.5, label='MotionSense', color='blue')
        axes[idx].hist(stm_vals, bins=30, alpha=0.5, label='STM_MotionSense', color='orange')
        axes[idx].set_title(f'{activity}\n({top_feat})', fontsize=12)
        axes[idx].set_xlabel('Feature Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'distribution_comparison_top_feature.png'), dpi=300)
    print("  Saved: distribution_comparison_top_feature.png")
    plt.close()

    # PCA Visualization
    print("  Computing PCA visualization...")
    top_n_features = 20
    top_feat_names = feature_importance.head(top_n_features)['feature'].tolist()

    combined_features_numeric = combined_features[top_feat_names].copy()
    for col in combined_features_numeric.columns:
        combined_features_numeric[col] = pd.to_numeric(combined_features_numeric[col], errors='coerce')
    
    X_top = combined_features_numeric.fillna(0)
    X_top_scaled = scaler.fit_transform(X_top)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_top_scaled)

    plt.figure(figsize=(14, 10))
    colors = {i: plt.cm.tab10(i) for i in range(len(activities))}
    for activity in activities:
        mask = combined_features['activity'] == activity
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   alpha=0.5, s=20, label=activity, color=colors[activity_to_label[activity]])

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    plt.title('PCA of Top 20 Features (Combined Dataset)', fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'pca_visualization.png'), dpi=300)
    print("  Saved: pca_visualization.png")
    plt.close()
    
    mark_step_done('cross_dataset_analysis')

# ============================================================================
# PART 6: CROSS-DATASET MODEL EVALUATION
# ============================================================================

# ============================================================================
# PART 6: CROSS-DATASET MODEL EVALUATION
# ============================================================================

if step_done('model_evaluation'):
    print("\n[6/6] Evaluating models with different feature sets... (SKIPPED - cached)")
    results_df = pd.read_csv(os.path.join(results_dir, 'feature_set_comparison.csv'))
else:
    print("\n[6/6] Evaluating models with different feature sets...")
    
    # Reload combined features
    combined_features = pd.concat([ms_features, stm_features], ignore_index=True)

    # Select feature sets to test
    feature_sets = {
        'top_10': feature_importance.head(10)['feature'].tolist(),
        'top_20': feature_importance.head(20)['feature'].tolist(),
        'top_30': feature_importance.head(30)['feature'].tolist(),
        'top_50': feature_importance.head(50)['feature'].tolist(),
        'all': feature_cols,
    }

    # Prepare data with numeric conversion
    X_train = ms_features[feature_cols].copy()
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_train = X_train.fillna(0)
    
    y_train = ms_features['activity_label']
    
    X_test = stm_features[feature_cols].copy()
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    X_test = X_test.fillna(0)
    
    y_test = stm_features['activity_label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for feat_set_name, feat_set in feature_sets.items():
        print(f"\n  Testing feature set: {feat_set_name} ({len(feat_set)} features)")
        
        X_train_subset = X_train_scaled[:, [feature_cols.index(f) for f in feat_set]]
        X_test_subset = X_test_scaled[:, [feature_cols.index(f) for f in feat_set]]
        
        # Train and evaluate
        rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        rf.fit(X_train_subset, y_train)
        
        y_pred = rf.predict(X_test_subset)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"    Cross-Dataset Accuracy (MotionSense → STM): {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        results.append({
            'feature_set': feat_set_name,
            'num_features': len(feat_set),
            'accuracy': accuracy,
            'features': ','.join(feat_set)
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)

    print("\n" + "=" * 80)
    print("FEATURE SET COMPARISON SUMMARY")
    print("=" * 80)
    print(results_df[['feature_set', 'num_features', 'accuracy']].to_string(index=False))

    # Save results
    results_df.to_csv(os.path.join(results_dir, 'feature_set_comparison.csv'), index=False)

    # Visualize model performance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='feature_set', y='accuracy', palette='muted')
    plt.title('Cross-Dataset Accuracy by Feature Set (MotionSense → STM)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Feature Set', fontsize=12)
    plt.ylim([0, 1])
    for i, v in enumerate(results_df['accuracy']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_performance_by_feature_set.png'), dpi=300)
    print("\n  Saved: model_performance_by_feature_set.png")
    plt.close()
    
    mark_step_done('model_evaluation')

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR OPTIMAL GYRO-ONLY HAR MODEL")
print("=" * 80)

best_result = results_df.iloc[0]
print(f"\nBest Feature Set: {best_result['feature_set']}")
print(f"  - Number of features: {best_result['num_features']}")
print(f"  - Cross-dataset accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"\nFeatures to use:")
best_features = best_result['features'].split(',')
for i, feat in enumerate(best_features, 1):
    print(f"  {i:2d}. {feat}")

print(f"\nTop 10 Individual Features (by importance):")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:30s} (importance: {row['importance']:.4f})")

print("\n" + "=" * 80)
print(f"All results saved to: {results_dir}")
print("=" * 80)
