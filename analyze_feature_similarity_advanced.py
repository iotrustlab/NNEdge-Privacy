import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Create results directory
results_dir = 'results/feature_analysis'
os.makedirs(results_dir, exist_ok=True)

# Paths
motionsense_accel_root = 'MotionSense/B_Accelerometer_data/B_Accelerometer_data'
motionsense_gyro_root = 'MotionSense/C_Gyroscope_data/C_Gyroscope_data'
stm_root = 'STM_MotionSense'

# Activities to analyze (folder names)
activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']

# Helper to collect all pocket files for a dataset
def collect_files(root, dataset='motionsense'):
    files = []
    if dataset == 'motionsense':
        for act in os.listdir(root):
            if act.lower().startswith('wlk'):
                act_name = 'Walking'
            elif act.lower().startswith('jog'):
                act_name = 'Jogging'
            elif act.lower().startswith('std'):
                act_name = 'Standing'
            elif act.lower().startswith('sit'):
                act_name = 'Sitting'
            elif act.lower().startswith('ups'):
                act_name = 'Upstairs'
            elif act.lower().startswith('dws'):
                act_name = 'Downstairs'
            else:
                continue
            act_path = os.path.join(root, act)
            for f in os.listdir(act_path):
                if f.endswith('.csv'):
                    files.append((os.path.join(act_path, f), act_name))
    else:  # STM
        for user in os.listdir(root):
            user_path = os.path.join(root, user, 'Processed')
            if not os.path.isdir(user_path):
                continue
            for act in os.listdir(user_path):
                if act not in activities:
                    continue
                act_path = os.path.join(user_path, act)
                # Flat structure
                for f in os.listdir(act_path):
                    fpath = os.path.join(act_path, f)
                    if os.path.isfile(fpath) and f.endswith('.csv') and 'pocket' in f.lower():
                        files.append((fpath, act))
                # Subfolder structure
                pocket_dir = os.path.join(act_path, 'right-pocket')
                if os.path.isdir(pocket_dir):
                    for f in os.listdir(pocket_dir):
                        fpath = os.path.join(pocket_dir, f)
                        if os.path.isfile(fpath) and f.endswith('.csv'):
                            files.append((fpath, act))
    return files

# Load and concatenate all data for each activity
def load_data(files, dataset, gyro_files=None):
    dfs = {act: [] for act in activities}
    gyro_dfs = {act: [] for act in activities} if gyro_files is not None else None
    for idx, (f, act) in enumerate(files):
        try:
            df = pd.read_csv(f)
            if df is None or df.empty:
                continue
            if dataset == 'motionsense':
                if all(col in df.columns for col in ['x', 'y', 'z']):
                    dfs[act].append(df[['x', 'y', 'z']])
                if gyro_files is not None:
                    gyro_f, _ = gyro_files[idx]
                    gyro_df = pd.read_csv(gyro_f)
                    if gyro_df is not None and not gyro_df.empty and all(col in gyro_df.columns for col in ['x', 'y', 'z']):
                        gyro_dfs[act].append(gyro_df[['x', 'y', 'z']])
            else:
                accel_cols = ['x', 'y', 'z']
                gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
                if all(col in df.columns for col in accel_cols):
                    dfs[act].append(df[accel_cols])
                if gyro_dfs is not None and all(col in df.columns for col in gyro_cols):
                    gyro_dfs[act].append(df[gyro_cols])
        except Exception as e:
            pass
    for act in activities:
        if dfs[act]:
            dfs[act] = pd.concat(dfs[act], ignore_index=True)
        else:
            dfs[act] = pd.DataFrame()
        if gyro_dfs is not None:
            if gyro_dfs[act]:
                gyro_dfs[act] = pd.concat(gyro_dfs[act], ignore_index=True)
            else:
                gyro_dfs[act] = pd.DataFrame()
    return dfs, gyro_dfs

# Advanced feature extraction functions
def extract_advanced_features(data, window_size=50):
    """Extract advanced HAR features from sensor data"""
    if data.empty or len(data) < window_size:
        return None
    
    features = {}
    cols = data.columns.tolist()
    
    for col in cols:
        signal = data[col].values
        
        # Statistical features
        features[f'{col}_mean'] = np.mean(signal)
        features[f'{col}_std'] = np.std(signal)
        features[f'{col}_min'] = np.min(signal)
        features[f'{col}_max'] = np.max(signal)
        features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
        features[f'{col}_median'] = np.median(signal)
        features[f'{col}_skew'] = stats.skew(signal)
        features[f'{col}_kurtosis'] = stats.kurtosis(signal)
        features[f'{col}_q25'] = np.percentile(signal, 25)
        features[f'{col}_q75'] = np.percentile(signal, 75)
        
        # Energy and power features
        features[f'{col}_energy'] = np.sum(signal ** 2)
        features[f'{col}_rms'] = np.sqrt(np.mean(signal ** 2))
        
        # Frequency domain features (using FFT)
        fft_vals = np.abs(fft(signal))[:len(signal)//2]
        features[f'{col}_fft_max'] = np.max(fft_vals)
        features[f'{col}_fft_mean'] = np.mean(fft_vals)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features[f'{col}_zero_crossings'] = len(zero_crossings)
        
        # Peak count
        peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0]
        features[f'{col}_peak_count'] = len(peaks)
    
    # Magnitude features (for accel: sqrt(x^2 + y^2 + z^2))
    if set(['x', 'y', 'z']).issubset(set(cols)):
        magnitude = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_energy'] = np.sum(magnitude ** 2)
    
    return features

# Collect files
ms_accel_files = collect_files(motionsense_accel_root, 'motionsense')
ms_gyro_files = collect_files(motionsense_gyro_root, 'motionsense')
stm_files = collect_files(stm_root, 'stm')

# Load data
ms_data, ms_gyro_data = load_data(ms_accel_files, 'motionsense', ms_gyro_files)
stm_data, stm_gyro_data = load_data(stm_files, 'stm')

# Extract advanced features for all activities
print("Extracting advanced features...\n")

advanced_features_ms = {}
advanced_features_stm = {}
advanced_features_ms_gyro = {}
advanced_features_stm_gyro = {}

for act in activities:
    if not ms_data[act].empty:
        advanced_features_ms[act] = extract_advanced_features(ms_data[act])
    if not stm_data[act].empty:
        advanced_features_stm[act] = extract_advanced_features(stm_data[act])
    if ms_gyro_data and not ms_gyro_data[act].empty:
        advanced_features_ms_gyro[act] = extract_advanced_features(ms_gyro_data[act])
    if stm_gyro_data and not stm_gyro_data[act].empty:
        advanced_features_stm_gyro[act] = extract_advanced_features(stm_gyro_data[act])

# Save results to CSV and text files
output_file = os.path.join(results_dir, 'advanced_features_analysis.txt')
with open(output_file, 'w') as f:
    for act in activities:
        f.write(f"\n{'='*80}\n")
        f.write(f"Activity: {act}\n")
        f.write(f"{'='*80}\n")
        
        if act in advanced_features_ms:
            f.write(f"\nMotionSense Accelerometer Features:\n")
            f.write("-" * 80 + "\n")
            for k, v in sorted(advanced_features_ms[act].items()):
                f.write(f"{k:30s}: {v:15.6f}\n")
        
        if act in advanced_features_ms_gyro:
            f.write(f"\nMotionSense Gyroscope Features:\n")
            f.write("-" * 80 + "\n")
            for k, v in sorted(advanced_features_ms_gyro[act].items()):
                f.write(f"{k:30s}: {v:15.6f}\n")
        
        if act in advanced_features_stm:
            f.write(f"\nSTM_MotionSense Accelerometer Features:\n")
            f.write("-" * 80 + "\n")
            for k, v in sorted(advanced_features_stm[act].items()):
                f.write(f"{k:30s}: {v:15.6f}\n")
        
        if act in advanced_features_stm_gyro:
            f.write(f"\nSTM_MotionSense Gyroscope Features:\n")
            f.write("-" * 80 + "\n")
            for k, v in sorted(advanced_features_stm_gyro[act].items()):
                f.write(f"{k:30s}: {v:15.6f}\n")

print(f"Advanced features analysis saved to: {output_file}")

# Create comparison DataFrames and save to CSV
comparison_data = []
for act in activities:
    row = {'Activity': act}
    if act in advanced_features_ms:
        row.update({f'MS_Accel_{k}': v for k, v in advanced_features_ms[act].items()})
    if act in advanced_features_stm:
        row.update({f'STM_Accel_{k}': v for k, v in advanced_features_stm[act].items()})
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
csv_file = os.path.join(results_dir, 'features_comparison.csv')
comparison_df.to_csv(csv_file, index=False)
print(f"Features comparison saved to: {csv_file}")

# Create feature similarity analysis
similarity_file = os.path.join(results_dir, 'feature_similarity_analysis.txt')
with open(similarity_file, 'w') as f:
    f.write("Feature Transferability Analysis\n")
    f.write("="*80 + "\n\n")
    for act in activities:
        f.write(f"\nActivity: {act}\n")
        f.write("-"*80 + "\n")
        if act in advanced_features_ms and act in advanced_features_stm:
            ms_feat = advanced_features_ms[act]
            stm_feat = advanced_features_stm[act]
            
            f.write(f"{'Feature':<30} {'MotionSense':<20} {'STM_MotionSense':<20} {'Diff %':<15}\n")
            f.write("-"*85 + "\n")
            
            for key in sorted(ms_feat.keys()):
                if key in stm_feat:
                    ms_val = ms_feat[key]
                    stm_val = stm_feat[key]
                    if ms_val != 0:
                        diff_pct = abs((stm_val - ms_val) / ms_val) * 100
                    else:
                        diff_pct = 0
                    f.write(f"{key:<30} {ms_val:<20.6f} {stm_val:<20.6f} {diff_pct:<15.2f}%\n")

print(f"Feature similarity analysis saved to: {similarity_file}")

print("\nAll results saved to:", results_dir)
