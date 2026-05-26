"""
game-2-gyro-cross_dataset_inference.py
========================================
Adversary Game 2: Gyroscope Only + dec_tree_out_1

This script trains an adversarial activity recognition classifier using ONLY
gyroscope data (no accelerometer) + dec_tree_out_1 features.

Training: MotionSense (gyroscope only)
Testing: STM_MotionSense (gyroscope only)

This simulates a scenario where the adversary has access to:
- 3-axis gyroscope data
- On-device shake/no-shake binary classifier output (dec_tree_out_1)

The goal is to infer user activity across datasets with limited sensor modality.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Create results directory
results_dir = 'results/game-2-gyro_cross_dataset_inference'
os.makedirs(results_dir, exist_ok=True)

# Paths
motionsense_gyro_root = 'MotionSense/C_Gyroscope_data/C_Gyroscope_data'
stm_root = 'STM_MotionSense'

# Activities
activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']
activity_to_label = {act: idx for idx, act in enumerate(activities)}
label_to_activity = {idx: act for act, idx in activity_to_label.items()}

# Helper to collect files
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
                for f in os.listdir(act_path):
                    fpath = os.path.join(act_path, f)
                    if os.path.isfile(fpath) and f.endswith('.csv'):
                        files.append((fpath, act))
                    elif os.path.isdir(fpath):
                        for subf in os.listdir(fpath):
                            subfpath = os.path.join(fpath, subf)
                            if os.path.isfile(subfpath) and subf.endswith('.csv'):
                                files.append((subfpath, act))
    return files

# Load data - GYROSCOPE ONLY
def load_data_gyro(files, dataset):
    dfs = {act: [] for act in activities}
    for idx, (f, act) in enumerate(files):
        try:
            df = pd.read_csv(f)
            if df is None or df.empty:
                continue
            # Load gyroscope + dec_tree_out_1 only
            cols = ['x', 'y', 'z']
            
            # Handle both naming conventions
            if 'gyro_x' in df.columns:
                df = df.rename(columns={'gyro_x': 'x', 'gyro_y': 'y', 'gyro_z': 'z'})
            
            if all(c in df.columns for c in cols):
                gyro_df = df[cols].copy()
                if 'dec_tree_out_1' in df.columns:
                    gyro_df['dec_tree_out_1'] = df['dec_tree_out_1']
                dfs[act].append(gyro_df)
        except Exception as e:
            pass
    
    # Concatenate all files per activity
    result = {}
    for act in activities:
        if dfs[act]:
            result[act] = pd.concat(dfs[act], ignore_index=True)
        else:
            result[act] = pd.DataFrame()
    return result

# Extract gyroscope features
def extract_gyro_features(gyro_data, window_size=50):
    """Extract statistical and spectral features from gyroscope window"""
    features = {}
    
    if len(gyro_data) < window_size or gyro_data.empty:
        return None
    
    for axis in ['x', 'y', 'z']:
        if axis not in gyro_data.columns:
            continue
        signal = gyro_data[axis].values
        
        # Basic statistical features
        features[f'{axis}_mean'] = np.mean(signal)
        features[f'{axis}_std'] = np.std(signal)
        features[f'{axis}_min'] = np.min(signal)
        features[f'{axis}_max'] = np.max(signal)
        features[f'{axis}_range'] = features[f'{axis}_max'] - features[f'{axis}_min']
        features[f'{axis}_median'] = np.median(signal)
        features[f'{axis}_rms'] = np.sqrt(np.mean(signal ** 2))
        
        # Distribution features
        features[f'{axis}_skew'] = stats.skew(signal)
        features[f'{axis}_kurtosis'] = stats.kurtosis(signal)
        
        # Spectral features
        fft_vals = np.abs(fft(signal))
        fft_vals = fft_vals[:len(fft_vals)//2]
        if len(fft_vals) > 1:
            features[f'{axis}_spectral_energy'] = np.sum(fft_vals**2)
            features[f'{axis}_dominant_freq'] = np.argmax(fft_vals)
        
        # Peak detection
        signal_1d = np.asarray(signal).flatten()
        peaks, _ = find_peaks(np.abs(signal_1d), height=np.std(signal_1d))
        features[f'{axis}_peak_count'] = len(peaks)
        features[f'{axis}_peak_avg_height'] = np.mean(np.abs(signal_1d[peaks])) if len(peaks) > 0 else 0
        
        # Energy and motion
        features[f'{axis}_energy'] = np.sum(signal**2)
        features[f'{axis}_zero_crossing'] = np.sum(np.abs(np.diff(np.sign(signal))))
    
    # Magnitude (rotational velocity magnitude)
    x_signal = gyro_data['x'].values
    y_signal = gyro_data['y'].values
    z_signal = gyro_data['z'].values
    
    mag_signal = np.sqrt(x_signal**2 + y_signal**2 + z_signal**2)
    features['mag_mean'] = np.mean(mag_signal)
    features['mag_std'] = np.std(mag_signal)
    features['mag_rms'] = np.sqrt(np.mean(mag_signal ** 2))
    features['mag_energy'] = np.sum(mag_signal**2)
    
    # dec_tree_out_1 features - strong static/dynamic separator
    if 'dec_tree_out_1' in gyro_data.columns:
        dec_tree_vals = gyro_data['dec_tree_out_1'].values
        # Handle case where values might be arrays
        try:
            dec_tree_vals = pd.to_numeric(dec_tree_vals, errors='coerce')
            dec_tree_vals = dec_tree_vals[~np.isnan(dec_tree_vals)]
            
            if len(dec_tree_vals) > 0:
                dec_tree_mean = np.mean(dec_tree_vals)
                features['dec_tree_out_1_mean'] = float(dec_tree_mean)
                features['dec_tree_out_1_sum'] = float(np.sum(dec_tree_vals))
                features['dec_tree_out_1_max'] = float(np.max(dec_tree_vals))
                features['dec_tree_out_1_ratio'] = float(np.sum(dec_tree_vals) / len(dec_tree_vals))
                # Binary indicator: is this window dynamic or static?
                features['is_dynamic'] = 1.0 if dec_tree_mean >= 0.5 else 0.0
        except (ValueError, TypeError):
            # If dec_tree_out_1 can't be converted to numeric, skip it
            pass
    
    return features

# Create windowed feature dataset
def create_feature_dataset(data_dict, window_size=50, step=25):
    X = []
    y = []
    
    for activity, df in data_dict.items():
        if df.empty:
            continue
        
        for i in range(0, len(df) - window_size, step):
            window = df.iloc[i:i+window_size].copy()
            
            # Ensure numeric columns - convert each element
            for col in window.columns:
                try:
                    # Apply numeric conversion to each element
                    window[col] = window[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                except (ValueError, TypeError):
                    pass
            
            features = extract_gyro_features(window, window_size)
            if features:
                X.append(features)
                y.append(activity_to_label[activity])
    
    return pd.DataFrame(X), np.array(y)

# Collect and load data
print("="*80)
print("GAME 2: Gyroscope-Only Cross-Dataset Activity Inference")
print("="*80)
print("\nLoading MotionSense data (gyroscope only)...")
ms_gyro_files = collect_files(motionsense_gyro_root, 'motionsense')
ms_data = load_data_gyro(ms_gyro_files, 'motionsense')

print("Loading STM_MotionSense data (gyroscope only)...")
stm_files = collect_files(stm_root, 'stm')
stm_data = load_data_gyro(stm_files, 'stm')

# Create feature datasets
print("Creating feature datasets...")
X_train, y_train = create_feature_dataset(ms_data)
X_test, y_test = create_feature_dataset(stm_data)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Align features
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
common_cols = sorted(train_cols.intersection(test_cols))

X_train = X_train[common_cols].copy()
X_test = X_test[common_cols].copy()

# Ensure all values are numeric
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

# Fill any NaN values with 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f"Features used: {X_train.shape[1]}")
if 'dec_tree_out_1_mean' in common_cols:
    print("✓ dec_tree_out_1 features included (5): dec_tree_out_1_mean, dec_tree_out_1_sum, dec_tree_out_1_max, dec_tree_out_1_ratio, is_dynamic")
    print("  Strong static/dynamic context for gyroscope-only adversary")
else:
    print("⚠ dec_tree_out_1 features NOT included")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=100, gamma='scale', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
}

# Train and evaluate models
results_summary = []

print("\n" + "="*80)
print("Training Multiple Models - Gyroscope Only")
print("="*80)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results_summary.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'F1-Score': f1
    })
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"  F1-Score: {f1:.4f}")
    
    # Save detailed results for each model
    cm = confusion_matrix(y_test, y_pred)
    
    model_results_file = os.path.join(results_dir, f'results_{model_name.replace(" ", "_")}.txt')
    with open(model_results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Game 2 - Gyroscope Only Cross-Dataset HAR Inference: {model_name}\n")
        f.write("Training: MotionSense | Test: STM_MotionSense\n")
        f.write("Sensor Modality: Gyroscope Only + dec_tree_out_1\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Activities: {', '.join(activities)}\n\n")
        
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
        
        f.write("Per-Activity Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=activities))
        
        f.write("\n" + "="*80 + "\n")
        f.write("Confusion Matrix\n")
        f.write("="*80 + "\n")
        
        cm_df = pd.DataFrame(cm, index=activities, columns=activities)
        f.write(cm_df.to_string())
        f.write("\n\n")
        
        f.write("Per-Class Accuracy:\n")
        for i, activity in enumerate(activities):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
                f.write(f"  {activity}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=activities, yticklabels=activities, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Game 2 (Gyro Only) - {model_name}\nAccuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot_file = os.path.join(results_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
    plt.close()

# Save summary comparison
summary_df = pd.DataFrame(results_summary).sort_values('Accuracy', ascending=False)

summary_file = os.path.join(results_dir, 'model_comparison.txt')
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Game 2 - Model Comparison: Gyroscope-Only Cross-Dataset HAR Inference\n")
    f.write("="*80 + "\n\n")
    f.write("Training Dataset: MotionSense\n")
    f.write("Test Dataset: STM_MotionSense\n")
    f.write("Sensor Modality: Gyroscope Only + dec_tree_out_1\n")
    f.write(f"Number of Features: {X_train.shape[1]}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    
    best_model = summary_df.iloc[0]
    f.write(f"Best Model: {best_model['Model']}\n")
    f.write(f"  Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']:.2%})\n")
    f.write(f"  F1-Score: {best_model['F1-Score']:.4f}\n")

print(f"\nSummary saved to: {summary_file}")

# Plot model comparison
plt.figure(figsize=(10, 6))
colors_list = ['green', 'blue', 'orange', 'red']
plt.barh(summary_df['Model'], summary_df['Accuracy'], color=colors_list[:len(summary_df)])
plt.xlabel('Accuracy')
plt.title('Game 2: Gyroscope-Only - Model Comparison')
plt.xlim(0, 1)
for i, (model, acc) in enumerate(zip(summary_df['Model'], summary_df['Accuracy'])):
    plt.text(acc + 0.02, i, f'{acc:.2%}', va='center')
plt.tight_layout()
comparison_plot_file = os.path.join(results_dir, 'model_comparison.png')
plt.savefig(comparison_plot_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Model comparison plot saved to: {comparison_plot_file}")
print("\n" + "="*80)
print("All results saved to:", results_dir)
print("="*80)
