"""
game-1-accel-gyro-reconstruction-inference.py
==============================================
Adversary Game 1: IMU Reconstruction (Accel + Gyro) from Binary Output

Strategy:
1. Load train dataset with full accel + gyro + binary output
2. For each test sample's binary, reconstruct by sampling from matching train class
3. Save reconstructed time-series data to STM_MotionSense_Reconstructed (same structure)
4. Train model on full MotionSense (accel + gyro)
5. Test on reconstructed STM_MotionSense with TRUE test labels
6. Evaluate test accuracy and detailed metrics

Training: MotionSense (full accel + gyro)
Testing: STM_MotionSense_Reconstructed (reconstructed from binary + train data)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Directories
results_dir = 'results/game-1-accel-gyro-reconstruction-inference'
os.makedirs(results_dir, exist_ok=True)

reconstructed_root = 'STM_MotionSense_Reconstructed'
os.makedirs(reconstructed_root, exist_ok=True)

# Data paths
motionsense_root = 'MotionSense'
stm_root = 'STM_MotionSense'

activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']
activity_to_label = {act: idx for idx, act in enumerate(activities)}

print("="*80)
print("GAME 1: Accel + Gyro Reconstruction from Binary Output")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL TRAIN DATA (MotionSense with accel + gyro)
# ============================================================================

print("\n[STEP 1] Loading TRAIN data (MotionSense)...")

train_data = []

# Load MotionSense Accelerometer
accel_root = os.path.join(motionsense_root, 'B_Accelerometer_data', 'B_Accelerometer_data')
activity_map = {'wlk': 'Walking', 'jog': 'Jogging', 'std': 'Standing', 
                'sit': 'Sitting', 'ups': 'Upstairs', 'dws': 'Downstairs'}

for folder in os.listdir(accel_root):
    folder_path = os.path.join(accel_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    activity = None
    for prefix, act in activity_map.items():
        if folder.lower().startswith(prefix):
            activity = act
            break
    if not activity:
        continue
    
    for csv_file in os.listdir(folder_path):
        if csv_file.endswith('.csv'):
            filepath = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                df['activity'] = activity
                df['modality'] = 'accel'
                df['source'] = 'MotionSense'
                train_data.append(df)
            except:
                pass

# Load MotionSense Gyroscope
gyro_root = os.path.join(motionsense_root, 'C_Gyroscope_data', 'C_Gyroscope_data')
for folder in os.listdir(gyro_root):
    folder_path = os.path.join(gyro_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    activity = None
    for prefix, act in activity_map.items():
        if folder.lower().startswith(prefix):
            activity = act
            break
    if not activity:
        continue
    
    for csv_file in os.listdir(folder_path):
        if csv_file.endswith('.csv'):
            filepath = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(filepath)
                df['activity'] = activity
                df['modality'] = 'gyro'
                df['source'] = 'MotionSense'
                train_data.append(df)
            except:
                pass

print(f"Loaded {len(train_data)} train files")

# ============================================================================
# STEP 2: LOAD ALL TEST DATA (STM_MotionSense)
# ============================================================================

print("\n[STEP 2] Loading TEST data (STM_MotionSense)...")

test_data = []
test_file_map = {}  # Map to save files with same structure

for user_folder in sorted(os.listdir(stm_root)):
    user_path = os.path.join(stm_root, user_folder, 'Processed')
    if not os.path.isdir(user_path):
        continue
    
    for activity in activities:
        activity_path = os.path.join(user_path, activity)
        if not os.path.isdir(activity_path):
            continue
        
        # Recursively find all CSV files
        for root, dirs, files in os.walk(activity_path):
            for csv_file in files:
                if csv_file.endswith('.csv'):
                    filepath = os.path.join(root, csv_file)
                    try:
                        df = pd.read_csv(filepath)
                        
                        # Determine modality
                        has_accel = any(col in df.columns for col in ['x', 'y', 'z', 'Ax', 'Ay', 'Az'])
                        has_gyro = any(col in df.columns for col in ['Gx', 'Gy', 'Gz', 'gyro_x', 'gyro_y', 'gyro_z'])
                        
                        if has_accel or has_gyro:
                            df['activity'] = activity
                            df['user'] = user_folder
                            df['source'] = 'STM_MotionSense'
                            
                            test_data.append(df)
                            
                            # Store relative path for reconstruction
                            rel_path = os.path.relpath(filepath, stm_root)
                            test_file_map[len(test_data)-1] = (filepath, rel_path, activity, user_folder)
                    except:
                        pass

print(f"Loaded {len(test_data)} test files")

# ============================================================================
# STEP 3: CREATE WINDOWED FEATURES WITH BINARY
# ============================================================================

print("\n[STEP 3] Creating windowed features with binary output...")

def extract_features_window(df, window_size=50):
    """Extract accel and gyro features from a window"""
    features = {}
    
    # Try to extract accel features
    accel_cols = None
    for col_set in [['x', 'y', 'z'], ['Ax', 'Ay', 'Az'], ['accel_x', 'accel_y', 'accel_z']]:
        if all(c in df.columns for c in col_set):
            accel_cols = col_set
            break
    
    if accel_cols:
        for i, col in enumerate(accel_cols):
            signal = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(signal) > 0:
                axis = ['x', 'y', 'z'][i]
                features[f'accel_{axis}_mean'] = np.mean(signal)
                features[f'accel_{axis}_std'] = np.std(signal)
                features[f'accel_{axis}_min'] = np.min(signal)
                features[f'accel_{axis}_max'] = np.max(signal)
                features[f'accel_{axis}_rms'] = np.sqrt(np.mean(signal**2))
    
    # Try to extract gyro features
    gyro_cols = None
    for col_set in [['Gx', 'Gy', 'Gz'], ['gFx', 'gFy', 'gFz'], ['gyro_x', 'gyro_y', 'gyro_z']]:
        if all(c in df.columns for c in col_set):
            gyro_cols = col_set
            break
    
    if gyro_cols:
        for i, col in enumerate(gyro_cols):
            signal = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(signal) > 0:
                axis = ['x', 'y', 'z'][i]
                features[f'gyro_{axis}_mean'] = np.mean(signal)
                features[f'gyro_{axis}_std'] = np.std(signal)
                features[f'gyro_{axis}_min'] = np.min(signal)
                features[f'gyro_{axis}_max'] = np.max(signal)
                features[f'gyro_{axis}_rms'] = np.sqrt(np.mean(signal**2))
    
    return features if features else None

def create_feature_dataset(data_list, window_size=50, step=25):
    """Create windowed features"""
    X = []
    y = []
    binary_vals = []
    original_data = []
    
    for df in data_list:
        # Extract binary value
        binary_val = 0
        if 'dec_tree_out_1' in df.columns:
            binary_arr = pd.to_numeric(df['dec_tree_out_1'], errors='coerce').dropna().values
            if len(binary_arr) > 0:
                binary_val = int(np.median(binary_arr))
        
        # Create windows
        for start in range(0, len(df) - window_size, step):
            window = df.iloc[start:start+window_size]
            features = extract_features_window(window)
            
            if features:
                X.append(features)
                y.append(activity_to_label[df['activity'].iloc[0]])
                binary_vals.append(binary_val)
                original_data.append(window)
    
    X_df = pd.DataFrame(X)
    return X_df, np.array(y), np.array(binary_vals), original_data

X_train, y_train, train_binary, _ = create_feature_dataset(train_data)
X_test_original, y_test, test_binary, test_windows = create_feature_dataset(test_data)

# Store TRUE test labels (from folder structure, NOT from reconstructed data)
y_test_true = y_test.copy()

print(f"Train windows: {len(X_train)}")
print(f"Test windows: {len(X_test_original)}")
print(f"Train features: {X_train.shape[1]}")

# ============================================================================
# STEP 4: BUILD BINARY CLASS MAPPINGS FROM TRAIN DATA
# ============================================================================

print("\n[STEP 4] Building binary class mappings...")

feature_cols = X_train.columns.tolist()

# Get samples for each binary class from train data
binary_0_samples = X_train[train_binary == 0].values
binary_1_samples = X_train[train_binary == 1].values
binary_other_samples = X_train[(train_binary != 0) & (train_binary != 1)].values

print(f"  Binary 0 train samples: {len(binary_0_samples)}")
print(f"  Binary 1 train samples: {len(binary_1_samples)}")
print(f"  Binary other train samples: {len(binary_other_samples)}")

# ============================================================================
# STEP 5: RECONSTRUCT TEST DATA
# ============================================================================

print("\n[STEP 5] Reconstructing test data from binary output...")

X_test_reconstructed = []
reconstructed_time_series = []  # Store full time-series reconstructions

for file_idx in range(len(test_data)):
    original_df = test_data[file_idx]
    binary_val = 0
    if 'dec_tree_out_1' in original_df.columns:
        binary_arr = pd.to_numeric(original_df['dec_tree_out_1'], errors='coerce').dropna().values
        if len(binary_arr) > 0:
            binary_val = int(np.median(binary_arr))
    
    # Reconstruct time-series: sample from matching binary class in train data
    reconstructed_df = original_df.copy()
    
    # Identify IMU columns to reconstruct
    imu_cols_to_reconstruct = []
    
    # Check for accel columns
    for col_set in [['x', 'y', 'z'], ['Ax', 'Ay', 'Az']]:
        if all(c in reconstructed_df.columns for c in col_set):
            imu_cols_to_reconstruct.extend(col_set)
    
    # Check for gyro columns
    for col_set in [['Gx', 'Gy', 'Gz'], ['gFx', 'gFy', 'gFz'], ['gyro_x', 'gyro_y', 'gyro_z']]:
        if all(c in reconstructed_df.columns for c in col_set):
            imu_cols_to_reconstruct.extend(col_set)
    
    # For each time step, find a matching train sample and use its values
    if len(imu_cols_to_reconstruct) > 0:
        # Choose which pool to sample from
        if binary_val == 0 and len(binary_0_samples) > 0:
            pool = binary_0_samples
        elif binary_val == 1 and len(binary_1_samples) > 0:
            pool = binary_1_samples
        else:
            pool = np.vstack([binary_0_samples, binary_1_samples]) if len(binary_0_samples) > 0 and len(binary_1_samples) > 0 else binary_0_samples if len(binary_0_samples) > 0 else binary_1_samples
        
    # Randomly sample a train window and use its reconstructed values
            sampled_train_window = pool[np.random.randint(0, len(pool))]
            sampled_train_df = pd.DataFrame([sampled_train_window], columns=feature_cols)
            
            # Map reconstructed features back to time-series columns
            for col in imu_cols_to_reconstruct:
                axis = col[-1].lower() if col[-1].lower() in ['x', 'y', 'z'] else 'x'
                
                # Try to find the corresponding mean feature
                if 'accel' in col.lower() or col in ['x', 'y', 'z']:
                    feat_name = f'accel_{axis}_mean'
                else:
                    feat_name = f'gyro_{axis}_mean'
                
                if feat_name in sampled_train_df.columns:
                    reconstructed_df[col] = sampled_train_df[feat_name].values[0]
    
    reconstructed_time_series.append(reconstructed_df)

print(f"  Reconstructed {len(reconstructed_time_series)} test files")

# ============================================================================
# STEP 5B: SAVE RECONSTRUCTED TIME-SERIES DATA
# ============================================================================

print("\n[STEP 5B] Saving reconstructed time-series data to disk...")

# Save reconstructed data maintaining original structure
saved_count = 0
for file_idx in test_file_map.keys():
    if file_idx < len(reconstructed_time_series):
        original_filepath, rel_path, activity, user = test_file_map[file_idx]
        
        # Create reconstructed directory structure
        recon_filepath = os.path.join(reconstructed_root, rel_path)
        os.makedirs(os.path.dirname(recon_filepath), exist_ok=True)
        
        # Get reconstructed time-series
        reconstructed_ts = reconstructed_time_series[file_idx]
        
        # Remove metadata columns that shouldn't be in reconstructed data
        cols_to_keep = [c for c in reconstructed_ts.columns if c not in ['activity', 'modality', 'source']]
        
        # Save as CSV
        reconstructed_ts[cols_to_keep].to_csv(recon_filepath, index=False)
        saved_count += 1

print(f"  Saved {saved_count} reconstructed files to {reconstructed_root}")

# Now create feature dataset from reconstructed time-series for testing
X_test_real, y_test, test_binary, _ = create_feature_dataset(reconstructed_time_series)
X_test_recon_df = X_test_real.copy()

# ============================================================================
# STEP 6: PREPARE DATA FOR MODELING
# ============================================================================

print("\n[STEP 6] Preparing data for modeling...")

# Fill NaN and ensure numeric
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_test_recon_df[col] = pd.to_numeric(X_test_recon_df[col], errors='coerce').fillna(0)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[feature_cols])
X_test_scaled = scaler.transform(X_test_recon_df[feature_cols])

print(f"Training samples: {len(X_train_scaled)}")
print(f"Test samples: {len(X_test_scaled)}")
print(f"Features: {len(feature_cols)}")

# ============================================================================
# STEP 7: TRAIN AND EVALUATE MODELS ON TEST DATA
# ============================================================================

print("\n[STEP 7] Training and evaluating models...")
print("="*80)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=100, gamma='scale', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
}

results_summary = []
activity_names = list(activity_to_label.keys())

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    # TEST PREDICTIONS
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_true, y_pred)
    f1 = f1_score(y_test_true, y_pred, average='weighted')
    
    results_summary.append({
        'Model': model_name,
        'Test_Accuracy': accuracy,
        'Test_F1': f1
    })
    
    print(f"  TEST Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"  TEST F1-Score: {f1:.4f}")
    
    # SAVE DETAILED RESULTS
    cm = confusion_matrix(y_test_true, y_pred)
    
    results_file = os.path.join(results_dir, f'results_{model_name.replace(" ", "_")}.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Game 1 - Accel+Gyro Reconstruction: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("SETUP:\n")
        f.write("  Training: MotionSense (full accel + gyro features)\n")
        f.write("  Testing: STM_MotionSense reconstructed from binary output\n")
        f.write("  Reconstruction: Samples from matching binary class in train set\n")
        f.write("  Test Labels: TRUE labels (NOT used during reconstruction)\n\n")
        
        f.write("DATA:\n")
        f.write(f"  Train samples: {len(X_train_scaled):,}\n")
        f.write(f"  Test samples: {len(X_test_scaled):,}\n")
        f.write(f"  Features: {len(feature_cols)}\n")
        f.write(f"  Activities: {', '.join(activity_names)}\n\n")
        
        f.write("TEST RESULTS:\n")
        f.write(f"  Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})\n")
        f.write(f"  Weighted F1-Score: {f1:.4f}\n\n")
        
        f.write("Per-Activity Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_test_true, y_pred, target_names=activity_names))
        
        f.write("\n" + "="*80 + "\n")
        f.write("Confusion Matrix\n")
        f.write("="*80 + "\n")
        cm_df = pd.DataFrame(cm, index=activity_names, columns=activity_names)
        f.write(cm_df.to_string())
        f.write("\n\n")
        
        f.write("Per-Class Test Accuracy:\n")
        for i, activity in enumerate(activity_names):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
                f.write(f"  {activity}: {class_acc:.2%} ({cm[i, i]}/{cm[i].sum()})\n")
    
    # PLOT CONFUSION MATRIX
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_names, yticklabels=activity_names, cbar_kws={'label': 'Count'})
    plt.title(f'Test Confusion Matrix - Game 1 (Reconstructed) - {model_name}\nTest Accuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot = os.path.join(results_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Results saved to {results_file}")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================

summary_df = pd.DataFrame(results_summary).sort_values('Test_Accuracy', ascending=False)

summary_file = os.path.join(results_dir, 'model_comparison.txt')
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Game 1: Accel + Gyro Reconstruction - TEST RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("SETUP:\n")
    f.write("  Train Data: MotionSense (full accel + gyro)\n")
    f.write("  Test Data: STM_MotionSense reconstructed from binary output\n")
    f.write("  Reconstruction Method: Sample from matching binary class in train set\n")
    f.write("  Binary Access: YES (used during reconstruction)\n")
    f.write("  Test Labels: NOT used during reconstruction or training\n\n")
    
    f.write(f"Training Samples: {len(X_train_scaled):,}\n")
    f.write(f"Test Samples: {len(X_test_scaled):,}\n")
    f.write(f"Features: {len(feature_cols)}\n")
    f.write(f"Reconstructed Files Saved: {saved_count}\n\n")
    
    f.write(summary_df.to_string(index=False))
    f.write("\n\n")
    
    best = summary_df.iloc[0]
    f.write(f"BEST MODEL: {best['Model']}\n")
    f.write(f"  Test Accuracy: {best['Test_Accuracy']:.4f} ({best['Test_Accuracy']:.2%})\n")
    f.write(f"  Test F1-Score: {best['Test_F1']:.4f}\n")

print(f"\n\nSummary saved to: {summary_file}")

# PLOT MODEL COMPARISON
plt.figure(figsize=(10, 6))
colors_list = ['green', 'blue', 'orange', 'red']
plt.barh(summary_df['Model'], summary_df['Test_Accuracy'], color=colors_list[:len(summary_df)])
plt.xlabel('Test Accuracy')
plt.title('Game 1: Accel+Gyro Reconstruction - Model Comparison (TEST DATA)')
plt.xlim(0, 1)
for i, (model, acc) in enumerate(zip(summary_df['Model'], summary_df['Test_Accuracy'])):
    plt.text(acc + 0.02, i, f'{acc:.2%}', va='center', fontsize=10)
plt.tight_layout()
comp_plot = os.path.join(results_dir, 'model_comparison.png')
plt.savefig(comp_plot, dpi=300, bbox_inches='tight')
plt.close()

print(f"Model comparison plot saved to: {comp_plot}")

print("\n" + "="*80)
print(f"All results saved to: {results_dir}")
print(f"Reconstructed data saved to: {reconstructed_root}")
print("="*80)
