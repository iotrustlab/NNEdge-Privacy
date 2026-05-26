"""
game-2-accel-cross_dataset_inference.py
========================================
Adversary Game 2: Accelerometer Only + dec_tree_out_1

This script trains an adversarial activity recognition classifier using ONLY
accelerometer data (no gyroscope) + dec_tree_out_1 features.

Training: MotionSense (accelerometer only)
Testing: STM_MotionSense (accelerometer only)

This simulates a scenario where the adversary has access to:
- 3-axis accelerometer data
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
import warnings
warnings.filterwarnings('ignore')

# Create results directory
results_dir = 'results/game-2-accel_cross_dataset_inference'
os.makedirs(results_dir, exist_ok=True)

# Paths
motionsense_accel_root = 'MotionSense/B_Accelerometer_data/B_Accelerometer_data'
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
                    if os.path.isfile(fpath) and f.endswith('.csv') and 'pocket' in f.lower():
                        files.append((fpath, act))
                pocket_dir = os.path.join(act_path, 'right-pocket')
                if os.path.isdir(pocket_dir):
                    for f in os.listdir(pocket_dir):
                        fpath = os.path.join(pocket_dir, f)
                        if os.path.isfile(fpath) and f.endswith('.csv'):
                            files.append((fpath, act))
    return files

# Load data - ACCELEROMETER ONLY
def load_data_accel(files, dataset):
    dfs = {act: [] for act in activities}
    for idx, (f, act) in enumerate(files):
        try:
            df = pd.read_csv(f)
            if df is None or df.empty:
                continue
            # Load accelerometer + dec_tree_out_1 only
            cols = ['x', 'y', 'z']
            if 'dec_tree_out_1' in df.columns:
                cols.append('dec_tree_out_1')
            if all(col in df.columns for col in ['x', 'y', 'z']):
                dfs[act].append(df[cols])
        except:
            pass
    for act in activities:
        if dfs[act]:
            dfs[act] = pd.concat(dfs[act], ignore_index=True)
        else:
            dfs[act] = pd.DataFrame()
    return dfs

# Extract accelerometer-only transferable features
def extract_accel_features(accel_data, window_size=50):
    """
    Extract features from accelerometer data only.
    Focus on orientation-independent and transferable features.
    Based on feature_similarity_analysis.txt, best transferable features are:
    - Magnitude-based (orientation-independent)
    - Z-axis (gravity-aligned)
    - RMS values (normalized)
    - Energy metrics
    """
    if accel_data.empty or len(accel_data) < window_size:
        return None
    
    features = {}
    
    if 'x' in accel_data.columns and 'y' in accel_data.columns and 'z' in accel_data.columns:
        # Magnitude features (most transferable - orientation independent)
        magnitude = np.sqrt(accel_data['x']**2 + accel_data['y']**2 + accel_data['z']**2)
        features['magnitude_mean'] = np.mean(magnitude)
        features['magnitude_std'] = np.std(magnitude)
        features['magnitude_energy'] = np.sum(magnitude ** 2)
        features['magnitude_rms'] = np.sqrt(np.mean(magnitude ** 2))
        features['magnitude_max'] = np.max(magnitude)
        features['magnitude_min'] = np.min(magnitude)
        
        # Z-axis features (gravity-aligned, good for activity distinction)
        z_signal = accel_data['z'].values
        features['z_mean'] = np.mean(z_signal)
        features['z_std'] = np.std(z_signal)
        features['z_rms'] = np.sqrt(np.mean(z_signal ** 2))
        features['z_min'] = np.min(z_signal)
        features['z_max'] = np.max(z_signal)
        features['z_range'] = features['z_max'] - features['z_min']
        features['z_energy'] = np.sum(z_signal ** 2)
        features['z_q25'] = np.percentile(z_signal, 25)
        features['z_q75'] = np.percentile(z_signal, 75)
        
        # RMS features (normalized, transferable)
        x_signal = accel_data['x'].values
        y_signal = accel_data['y'].values
        features['x_rms'] = np.sqrt(np.mean(x_signal ** 2))
        features['x_std'] = np.std(x_signal)
        features['y_rms'] = np.sqrt(np.mean(y_signal ** 2))
        features['y_std'] = np.std(y_signal)
        
        # Statistical features
        features['z_skew'] = stats.skew(z_signal)
        features['z_kurtosis'] = stats.kurtosis(z_signal)
        
        # dec_tree_out_1 features - strong static/dynamic separator
        if 'dec_tree_out_1' in accel_data.columns:
            dec_tree_vals = accel_data['dec_tree_out_1'].values
            dec_tree_mean = np.mean(dec_tree_vals)
            features['dec_tree_out_1_mean'] = dec_tree_mean
            features['dec_tree_out_1_sum'] = np.sum(dec_tree_vals)
            features['dec_tree_out_1_max'] = np.max(dec_tree_vals)
            features['dec_tree_out_1_ratio'] = np.sum(dec_tree_vals) / len(dec_tree_vals)
            # Binary indicator: is this window dynamic or static?
            features['is_dynamic'] = 1.0 if dec_tree_mean >= 0.5 else 0.0
    
    return features

# Create windowed feature dataset
def create_feature_dataset(data_dict, window_size=50, step=25):
    X = []
    y = []
    
    for activity, df in data_dict.items():
        if df.empty:
            continue
        
        for i in range(0, len(df) - window_size, step):
            window = df.iloc[i:i+window_size]
            features = extract_accel_features(window, window_size)
            if features:
                X.append(features)
                y.append(activity_to_label[activity])
    
    return pd.DataFrame(X), np.array(y)

# Collect and load data
print("="*80)
print("GAME 2: Accelerometer-Only Cross-Dataset Activity Inference")
print("="*80)
print("\nLoading MotionSense data (accelerometer only)...")
ms_accel_files = collect_files(motionsense_accel_root, 'motionsense')
ms_data = load_data_accel(ms_accel_files, 'motionsense')

print("Loading STM_MotionSense data (accelerometer only)...")
stm_files = collect_files(stm_root, 'stm')
stm_data = load_data_accel(stm_files, 'stm')

# Create feature datasets
print("Creating feature datasets...")
X_train, y_train = create_feature_dataset(ms_data)
X_test, y_test = create_feature_dataset(stm_data)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Align features
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
common_cols = sorted(train_cols.intersection(test_cols))

X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"Features used: {X_train.shape[1]}")
if 'dec_tree_out_1_mean' in common_cols:
    print("✓ dec_tree_out_1 features included (5): dec_tree_out_1_mean, dec_tree_out_1_sum, dec_tree_out_1_max, dec_tree_out_1_ratio, is_dynamic")
    print("  Strong static/dynamic context for accelerometer-only adversary")
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
print("Training Multiple Models - Accelerometer Only")
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
        f.write(f"Game 2 - Accelerometer Only Cross-Dataset HAR Inference: {model_name}\n")
        f.write("Training: MotionSense | Test: STM_MotionSense\n")
        f.write("Sensor Modality: Accelerometer Only + dec_tree_out_1\n")
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
    plt.title(f'Confusion Matrix - Game 2 (Accel Only) - {model_name}\nAccuracy: {accuracy:.2%}')
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
    f.write("Game 2 - Model Comparison: Accelerometer-Only Cross-Dataset HAR Inference\n")
    f.write("="*80 + "\n\n")
    f.write("Training Dataset: MotionSense\n")
    f.write("Test Dataset: STM_MotionSense\n")
    f.write("Sensor Modality: Accelerometer Only + dec_tree_out_1\n")
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
plt.title('Game 2: Accelerometer-Only - Model Comparison')
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
