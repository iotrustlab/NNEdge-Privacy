"""
game-2-gyro-reconstruction-based.py
====================================
Adversary Game 2 (Advanced): Gyroscope-Based Accelerometer Reconstruction

Strategy:
1. Train a model (Gyro -> Accel) using full MotionSense dataset
2. Use this model to reconstruct test accelerometer from test gyroscope (STM_MotionSense)
3. Train activity recognition on reconstructed accelerometer (which may be more discriminative than gyro)
4. Test activity recognition on reconstructed accelerometer
5. Save reconstructed test data maintaining original folder structure
6. dec_tree_out_1 can be used as auxiliary feature but not required

This simulates a sophisticated adversary who:
- Has full access to training data (can learn gyro->accel mapping)
- Knows test gyroscope signals
- Can reconstruct plausible accelerometer signals from gyroscope
- Uses reconstructed accelerometer for better activity recognition

"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Directories
results_dir = 'results/game-2-gyro-reconstruction-based'
os.makedirs(results_dir, exist_ok=True)

reconstructed_root = 'STM_MotionSense_Reconstructed_Accel_from_Gyro'
os.makedirs(reconstructed_root, exist_ok=True)

# Data paths
motionsense_root = 'MotionSense'
stm_root = 'STM_MotionSense'

activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']
activity_to_label = {act: idx for idx, act in enumerate(activities)}
label_to_activity = {idx: act for act, idx in activity_to_label.items()}

print("="*80)
print("GAME 2 (ADVANCED): Gyroscope-Based Accelerometer Reconstruction")
print("="*80)

# ============================================================================
# STEP 1: LOAD ALL TRAINING DATA (MotionSense with accel + gyro)
# ============================================================================

print("\n[STEP 1] Loading TRAIN data (MotionSense - full accel + gyro)...")

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
                train_data.append(df)
            except:
                pass

print(f"Loaded {len(train_data)} train files")

# ============================================================================
# STEP 2: LOAD ALL TEST DATA (STM_MotionSense)
# ============================================================================

print("\n[STEP 2] Loading TEST data (STM_MotionSense - gyro)...")

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
                        
                        # Check if has gyro data
                        has_gyro = any(col in df.columns for col in ['Gx', 'Gy', 'Gz', 'gyro_x', 'gyro_y', 'gyro_z', 'x', 'y', 'z'])
                        
                        if has_gyro:
                            df['activity'] = activity
                            df['user'] = user_folder
                            test_data.append(df)
                            
                            # Store relative path for reconstruction
                            rel_path = os.path.relpath(filepath, stm_root)
                            test_file_map[len(test_data)-1] = (filepath, rel_path, activity, user_folder)
                    except:
                        pass

print(f"Loaded {len(test_data)} test files")

# ============================================================================
# STEP 3: CREATE WINDOWED FEATURES FOR GYRO->ACCEL MAPPING
# ============================================================================

print("\n[STEP 3] Creating windowed features for gyro->accel regression model...")

def extract_gyro_features_for_regression(df, window_size=50):
    """Extract gyroscope features for predicting accelerometer"""
    features = {}
    
    # Try to extract gyro columns
    gyro_cols = None
    for col_set in [['Gx', 'Gy', 'Gz'], ['gFx', 'gFy', 'gFz'], ['gyro_x', 'gyro_y', 'gyro_z']]:
        if all(c in df.columns for c in col_set):
            gyro_cols = col_set
            break
    
    # If no gyro found, try x, y, z (could be gyro)
    if not gyro_cols and all(c in df.columns for c in ['x', 'y', 'z']):
        gyro_cols = ['x', 'y', 'z']
    
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
                features[f'gyro_{axis}_energy'] = np.sum(signal**2)
                features[f'gyro_{axis}_skew'] = stats.skew(signal)
                features[f'gyro_{axis}_kurtosis'] = stats.kurtosis(signal)
    
    # Magnitude of gyroscope
    if gyro_cols:
        gyro_x = pd.to_numeric(df[gyro_cols[0]], errors='coerce').dropna().values
        gyro_y = pd.to_numeric(df[gyro_cols[1]], errors='coerce').dropna().values
        gyro_z = pd.to_numeric(df[gyro_cols[2]], errors='coerce').dropna().values
        if len(gyro_x) > 0 and len(gyro_y) > 0 and len(gyro_z) > 0:
            mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            features['gyro_mag_mean'] = np.mean(mag)
            features['gyro_mag_std'] = np.std(mag)
            features['gyro_mag_rms'] = np.sqrt(np.mean(mag**2))
    
    return features if features else None

def extract_accel_targets(df, window_size=50):
    """Extract accelerometer targets for regression"""
    targets = {}
    
    # Try to extract accel columns
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
                targets[f'accel_{axis}_mean'] = np.mean(signal)
                targets[f'accel_{axis}_std'] = np.std(signal)
                targets[f'accel_{axis}_min'] = np.min(signal)
                targets[f'accel_{axis}_max'] = np.max(signal)
                targets[f'accel_{axis}_rms'] = np.sqrt(np.mean(signal**2))
                targets[f'accel_{axis}_energy'] = np.sum(signal**2)
    
    return targets if targets else None

def create_regression_dataset(data_list, window_size=50, step=25):
    """Create windowed gyro->accel regression dataset"""
    X = []
    y_dict = {}
    
    for df in data_list:
        # Create windows
        for start in range(0, len(df) - window_size, step):
            window = df.iloc[start:start+window_size]
            gyro_features = extract_gyro_features_for_regression(window)
            accel_targets = extract_accel_targets(window)
            
            if gyro_features and accel_targets:
                X.append(gyro_features)
                for target_name, target_val in accel_targets.items():
                    if target_name not in y_dict:
                        y_dict[target_name] = []
                    y_dict[target_name].append(target_val)
    
    X_df = pd.DataFrame(X)
    return X_df, y_dict

# Create regression dataset from train data
X_gyro_train, y_accel_train = create_regression_dataset(train_data)

print(f"Gyroscope features (input): {X_gyro_train.shape}")
print(f"Accelerometer targets: {len(y_accel_train)} signals")

# ============================================================================
# STEP 4: TRAIN GYRO->ACCEL REGRESSION MODELS
# ============================================================================

print("\n[STEP 4] Training gyro->accel regression models...")

# Train separate regression models for each accelerometer signal
regression_models = {}
for target_name in y_accel_train.keys():
    y_target = np.array(y_accel_train[target_name])
    
    # Fill NaN and standardize
    X_gyro_train_clean = X_gyro_train.fillna(0)
    y_target_clean = np.nan_to_num(y_target)
    
    # Train model
    reg_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    reg_model.fit(X_gyro_train_clean, y_target_clean.astype(int))  # Simplified: treat as classification
    
    regression_models[target_name] = reg_model
    print(f"  Trained model for {target_name}")

print(f"Total regression models trained: {len(regression_models)}")

# ============================================================================
# STEP 5: RECONSTRUCT TEST ACCELEROMETER FROM TEST GYROSCOPE
# ============================================================================

print("\n[STEP 5] Reconstructing test accelerometer from test gyroscope...")

reconstructed_time_series = []

for file_idx in range(len(test_data)):
    original_df = test_data[file_idx]
    
    # Reconstruct accelerometer
    reconstructed_df = original_df.copy()
    
    # Identify gyro columns
    gyro_cols_source = None
    for col_set in [['Gx', 'Gy', 'Gz'], ['gFx', 'gFy', 'gFz'], ['gyro_x', 'gyro_y', 'gyro_z']]:
        if all(c in reconstructed_df.columns for c in col_set):
            gyro_cols_source = col_set
            break
    
    if not gyro_cols_source and all(c in reconstructed_df.columns for c in ['x', 'y', 'z']):
        gyro_cols_source = ['x', 'y', 'z']
    
    # Identify accel columns (if they exist)
    accel_cols_target = None
    for col_set in [['x', 'y', 'z'], ['Ax', 'Ay', 'Az']]:
        if all(c in reconstructed_df.columns for c in col_set):
            accel_cols_target = col_set
            break
    
    # If test file only has gyro, we need to add accel columns
    if not accel_cols_target:
        accel_cols_target = ['x_recon', 'y_recon', 'z_recon']
        for col in accel_cols_target:
            reconstructed_df[col] = 0.0

    window_size = 50  # Same as training window size
    # For each time step, extract gyro window and predict accel
    if gyro_cols_source:
        for t in range(window_size, len(reconstructed_df)):
            window = original_df.iloc[t-window_size:t]
            gyro_features = extract_gyro_features_for_regression(window)
            
            if gyro_features:
                gyro_df = pd.DataFrame([gyro_features])
                gyro_df = gyro_df.fillna(0)
                
                # Predict accelerometer values using regression models
                for i, target_col in enumerate(['accel_x_mean', 'accel_y_mean', 'accel_z_mean']):
                    if target_col in regression_models:
                        try:
                            pred = regression_models[target_col].predict(gyro_df)[0]
                            reconstructed_df.loc[t, accel_cols_target[i]] = pred
                        except:
                            reconstructed_df.loc[t, accel_cols_target[i]] = 0.0
    
    reconstructed_time_series.append(reconstructed_df)

print(f"Reconstructed {len(reconstructed_time_series)} test files")

# ============================================================================
# STEP 5B: SAVE RECONSTRUCTED TIME-SERIES DATA
# ============================================================================

print("\n[STEP 5B] Saving reconstructed time-series data to disk...")

saved_count = 0
for file_idx in test_file_map.keys():
    if file_idx < len(reconstructed_time_series):
        original_filepath, rel_path, activity, user = test_file_map[file_idx]
        
        # Create reconstructed directory structure
        recon_filepath = os.path.join(reconstructed_root, rel_path)
        os.makedirs(os.path.dirname(recon_filepath), exist_ok=True)
        
        # Get reconstructed time-series
        reconstructed_ts = reconstructed_time_series[file_idx]
        
        # Remove metadata columns
        cols_to_keep = [c for c in reconstructed_ts.columns if c not in ['activity', 'modality', 'user']]
        
        # Save as CSV
        reconstructed_ts[cols_to_keep].to_csv(recon_filepath, index=False)
        saved_count += 1

print(f"Saved {saved_count} reconstructed files to {reconstructed_root}")

# ============================================================================
# STEP 6: CREATE WINDOWED FEATURES FOR ACTIVITY RECOGNITION
# ============================================================================

print("\n[STEP 6] Creating windowed features for activity recognition...")

def extract_accel_features(df, window_size=50):
    """Extract accelerometer features for activity recognition"""
    features = {}
    
    # Try to extract accel columns
    accel_cols = None
    for col_set in [['x', 'y', 'z'], ['Ax', 'Ay', 'Az'], ['accel_x', 'accel_y', 'accel_z'], ['x_recon', 'y_recon', 'z_recon']]:
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
    
    # Magnitude
    if accel_cols:
        accel_x = pd.to_numeric(df[accel_cols[0]], errors='coerce').dropna().values
        accel_y = pd.to_numeric(df[accel_cols[1]], errors='coerce').dropna().values
        accel_z = pd.to_numeric(df[accel_cols[2]], errors='coerce').dropna().values
        if len(accel_x) > 0 and len(accel_y) > 0 and len(accel_z) > 0:
            mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
            features['accel_mag_mean'] = np.mean(mag)
            features['accel_mag_std'] = np.std(mag)
            features['accel_mag_rms'] = np.sqrt(np.mean(mag**2))
    
    # dec_tree_out_1 if available
    if 'dec_tree_out_1' in df.columns:
        dec_tree_vals = pd.to_numeric(df['dec_tree_out_1'], errors='coerce').dropna().values
        if len(dec_tree_vals) > 0:
            features['dec_tree_out_1_mean'] = np.mean(dec_tree_vals)
            features['dec_tree_out_1_sum'] = np.sum(dec_tree_vals)
            features['is_dynamic'] = 1.0 if np.mean(dec_tree_vals) >= 0.5 else 0.0
    
    return features if features else None

def create_activity_feature_dataset(data_list, window_size=50, step=25):
    """Create windowed features for activity classification"""
    X = []
    y = []
    
    for df in data_list:
        if 'activity' not in df.columns:
            continue
        
        activity = df['activity'].iloc[0]
        
        for start in range(0, len(df) - window_size, step):
            window = df.iloc[start:start+window_size]
            features = extract_accel_features(window)
            
            if features:
                X.append(features)
                y.append(activity_to_label[activity])
    
    X_df = pd.DataFrame(X)
    return X_df, np.array(y)

# Create training dataset (original accelerometer from MotionSense)
# Filter train_data to only accelerometer
train_data_accel = [df for df in train_data if df['modality'].iloc[0] == 'accel']
X_train, y_train = create_activity_feature_dataset(train_data_accel)

# Create test dataset (reconstructed accelerometer)
X_test, y_test = create_activity_feature_dataset(reconstructed_time_series)

print(f"Train windows: {len(X_train)}")
print(f"Test windows: {len(X_test)}")
print(f"Train features: {X_train.shape[1]}")

# ============================================================================
# STEP 7: PREPARE DATA FOR MODELING
# ============================================================================

print("\n[STEP 7] Preparing data for modeling...")

feature_cols = X_train.columns.tolist()

# Fill NaN and ensure numeric
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[feature_cols])
X_test_scaled = scaler.transform(X_test[feature_cols])

print(f"Training samples: {len(X_train_scaled)}")
print(f"Test samples: {len(X_test_scaled)}")
print(f"Features: {len(feature_cols)}")

# ============================================================================
# STEP 8: TRAIN AND EVALUATE MODELS
# ============================================================================

print("\n[STEP 8] Training and evaluating models...")
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
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results_summary.append({
        'Model': model_name,
        'Test_Accuracy': accuracy,
        'Test_F1': f1
    })
    
    print(f"  TEST Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"  TEST F1-Score: {f1:.4f}")
    
    # SAVE DETAILED RESULTS
    cm = confusion_matrix(y_test, y_pred)
    
    results_file = os.path.join(results_dir, f'results_{model_name.replace(" ", "_")}.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Game 2 (Advanced) - Gyro-Based Accel Reconstruction: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("SETUP:\n")
        f.write("  Training: MotionSense (original accelerometer)\n")
        f.write("  Testing: STM_MotionSense (reconstructed accelerometer from gyroscope)\n")
        f.write("  Reconstruction: Gyro->Accel regression model trained on full MotionSense\n")
        f.write("  Test Labels: TRUE labels (NOT used during reconstruction or training)\n\n")
        
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
        f.write(classification_report(y_test, y_pred, target_names=activity_names))
        
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
    plt.title(f'Test Confusion Matrix - Game 2 (Gyro Reconstruction) - {model_name}\nTest Accuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot = os.path.join(results_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.savefig(cm_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Results saved to {results_file}")

# ============================================================================
# STEP 9: SUMMARY
# ============================================================================

summary_df = pd.DataFrame(results_summary).sort_values('Test_Accuracy', ascending=False)

summary_file = os.path.join(results_dir, 'model_comparison.txt')
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Game 2 (Advanced): Gyro-Based Accel Reconstruction - TEST RESULTS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("SETUP:\n")
    f.write("  Train Data: MotionSense (original accelerometer)\n")
    f.write("  Test Data: STM_MotionSense reconstructed accelerometer from gyroscope\n")
    f.write("  Reconstruction Method: Gyro->Accel regression model (trained on full MotionSense)\n")
    f.write("  Adversary Knowledge: Full training data access + test gyroscope signals\n")
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
plt.title('Game 2 (Advanced): Gyro-Based Accel Reconstruction - Model Comparison')
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
