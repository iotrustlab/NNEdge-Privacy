"""
game-1-accel-reconstruction-inference.py
==========================================
Adversary Game 1: Accelerometer Only Reconstruction from Binary Output

Strategy:
1. Load train dataset with accelerometer only from MotionSense
2. Test on pre-reconstructed accelerometer data from STM_MotionSense_Reconstructed
3. Extract transferable accel features (magnitude, z-axis, RMS, etc.)
4. Train model on MotionSense accelerometer
5. Test on reconstructed STM accelerometer with TRUE test labels
6. Evaluate test accuracy and detailed metrics

Training: MotionSense (accelerometer only)
Testing: STM_MotionSense_Reconstructed (reconstructed accel from binary + train data)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Directories
results_dir = 'results/game-1-accel-reconstruction-inference'
os.makedirs(results_dir, exist_ok=True)

reconstructed_root = 'STM_MotionSense_Reconstructed'
motionsense_root = 'MotionSense'
stm_root = 'STM_MotionSense'

activities = ['Walking', 'Jogging', 'Standing', 'Sitting', 'Upstairs', 'Downstairs']
activity_to_label = {act: idx for idx, act in enumerate(activities)}
label_to_activity = {idx: act for act, idx in activity_to_label.items()}

print("="*80)
print("GAME 1: Accelerometer-Only Reconstruction from Binary Output")
print("="*80)

# ============================================================================
# STEP 1: LOAD TRAIN DATA (MotionSense - Accelerometer Only)
# ============================================================================

print("\n[STEP 1] Loading TRAIN data (MotionSense - Accel Only)...")

train_data = []

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
                train_data.append(df)
            except:
                pass

print(f"Loaded {len(train_data)} train files")

# ============================================================================
# STEP 2: LOAD TEST DATA (Reconstructed STM - Accelerometer Only)
# ============================================================================

print("\n[STEP 2] Loading TEST data (Reconstructed STM - Accel Only)...")

test_data = []
test_file_map = {}

for user_folder in sorted(os.listdir(reconstructed_root)):
    user_path = os.path.join(reconstructed_root, user_folder, 'Processed')
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
                        
                        # Check for accelerometer columns
                        has_accel = any(col in df.columns for col in ['x', 'y', 'z', 'Ax', 'Ay', 'Az', 'accel_x', 'accel_y', 'accel_z'])
                        
                        if has_accel:
                            df['activity'] = activity
                            df['user'] = user_folder
                            test_data.append(df)
                            test_file_map[len(test_data)-1] = (filepath, activity, user_folder)
                    except:
                        pass

print(f"Loaded {len(test_data)} test files")

# ============================================================================
# STEP 3: CREATE WINDOWED FEATURES
# ============================================================================

print("\n[STEP 3] Creating windowed features...")

def extract_accel_features(df, window_size=50):
    """Extract transferable accelerometer features from a window"""
    features = {}
    
    # Try to extract accel features
    accel_cols = None
    for col_set in [['x', 'y', 'z'], ['Ax', 'Ay', 'Az'], ['accel_x', 'accel_y', 'accel_z']]:
        if all(c in df.columns for c in col_set):
            accel_cols = col_set
            break
    
    if accel_cols is None:
        return None
    
    x_signal = df[accel_cols[0]].values
    y_signal = df[accel_cols[1]].values
    z_signal = df[accel_cols[2]].values
    
    # Magnitude features (orientation-independent)
    magnitude = np.sqrt(x_signal**2 + y_signal**2 + z_signal**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_energy'] = np.sum(magnitude ** 2)
    features['magnitude_rms'] = np.sqrt(np.mean(magnitude ** 2))
    features['magnitude_max'] = np.max(magnitude)
    features['magnitude_min'] = np.min(magnitude)
    
    # Z-axis features (gravity-aligned)
    features['z_mean'] = np.mean(z_signal)
    features['z_std'] = np.std(z_signal)
    features['z_rms'] = np.sqrt(np.mean(z_signal ** 2))
    features['z_min'] = np.min(z_signal)
    features['z_max'] = np.max(z_signal)
    features['z_range'] = features['z_max'] - features['z_min']
    features['z_energy'] = np.sum(z_signal ** 2)
    features['z_q25'] = np.percentile(z_signal, 25)
    features['z_q75'] = np.percentile(z_signal, 75)
    features['z_skew'] = stats.skew(z_signal)
    features['z_kurtosis'] = stats.kurtosis(z_signal)
    
    # X and Y RMS and std (normalized)
    features['x_rms'] = np.sqrt(np.mean(x_signal ** 2))
    features['x_std'] = np.std(x_signal)
    features['y_rms'] = np.sqrt(np.mean(y_signal ** 2))
    features['y_std'] = np.std(y_signal)
    
    return features

def create_feature_dataset(data_list, window_size=50, step=25):
    X = []
    y = []
    
    for df in data_list:
        if df.empty or len(df) < window_size:
            continue
        
        activity = df['activity'].iloc[0]
        activity_label = activity_to_label[activity]
        
        for i in range(0, len(df) - window_size, step):
            window = df.iloc[i:i+window_size]
            features = extract_accel_features(window, window_size)
            if features:
                X.append(features)
                y.append(activity_label)
    
    return pd.DataFrame(X), np.array(y)

print("Creating train features...")
X_train, y_train = create_feature_dataset(train_data, window_size=50, step=25)
print(f"Train features shape: {X_train.shape}, labels shape: {y_train.shape}")

print("Creating test features...")
X_test, y_test = create_feature_dataset(test_data, window_size=50, step=25)
print(f"Test features shape: {X_test.shape}, labels shape: {y_test.shape}")

# ============================================================================
# STEP 4: NORMALIZE AND TRAIN MODELS
# ============================================================================

print("\n[STEP 4] Training models...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

results = {}

for model_name, model in models.items():
    print(f"  Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"    Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n[STEP 5] Saving results...")

# Model comparison
comparison_txt = f"""================================================================================
Game 1 - Model Comparison: Accelerometer-Only Reconstruction HAR Inference
================================================================================

Training Dataset: MotionSense
Test Dataset: STM_MotionSense (Reconstructed - Accel Only)
Reconstruction Method: Binary output-based sampling
Sensor Modality: Accelerometer Only
Number of Features: {X_train.shape[1]}
Training Samples: {len(X_train)}
Test Samples: {len(X_test)}

"""

# Sort by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

comparison_txt += "            Model  Accuracy  F1-Score\n"
for model_name, res in sorted_results:
    comparison_txt += f"{model_name:>20} {res['accuracy']:>9.6f} {res['f1']:>9.6f}\n"

best_model_name, best_result = sorted_results[0]
comparison_txt += f"\nBest Model: {best_model_name}\n"
comparison_txt += f"  Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)\n"
comparison_txt += f"  F1-Score: {best_result['f1']:.4f}\n"

with open(os.path.join(results_dir, 'model_comparison.txt'), 'w') as f:
    f.write(comparison_txt)

print(comparison_txt)

# Detailed results for each model
for model_name, res in results.items():
    y_pred = res['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    results_txt = f"""================================================================================
Game 1 - Accelerometer-Only Reconstruction: {model_name} Results
================================================================================

Test Accuracy: {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)
Test F1-Score: {res['f1']:.4f}

Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_test, y_pred, target_names=activities)}
"""
    
    filename = f"results_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.txt"
    with open(os.path.join(results_dir, filename), 'w') as f:
        f.write(results_txt)
    
    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=activities, yticklabels=activities)
    plt.title(f'Game 1 - Accel Only - Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'))
    plt.close()

# Overall model comparison plot
fig, ax = plt.subplots(figsize=(10, 6))
model_names = [name for name, _ in sorted_results]
accuracies = [res['accuracy'] for _, res in sorted_results]
f1_scores = [res['f1'] for _, res in sorted_results]

x = np.arange(len(model_names))
width = 0.35

ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Game 1 - Accelerometer-Only Reconstruction: Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
plt.close()

print(f"\nResults saved to {results_dir}")
print("="*80)
print("Game 1 - Accelerometer-Only Reconstruction: COMPLETE")
print("="*80)
