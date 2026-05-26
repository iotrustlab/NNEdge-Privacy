#!/usr/bin/env python3
"""
Game-2 Cross-Sensor Generalization Study with Accelerometer Data
================================================================

This script implements a placement-specific Game-2 adversarial HAR study to test
cross-placement generalization capabilities using accelerometer + binary decision tree
data with user identity available.

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Raw accelerometer data: (acc_x, acc_y, acc_z)
- Binary decision tree outputs from specific known placements
- Temporal and statistical patterns from accelerometer signals

❌ REMOVED SEMANTIC INFORMATION:
- Gyroscope data (Game-2 constraint: accelerometer only)
- Cross-sensor coordination (training on single placement)

🎯 RESEARCH QUESTION:
How well do models trained on one specific placement generalize to other placements
when using accelerometer + binary data with user identity (Game-2 constraint)?

🔬 EXPERIMENTAL DESIGN:
- Train models on ONE specific placement using accelerometer features
- Test on ALL 5 placements individually
- Compare same-placement vs cross-placement performance
- Hypothesis: Accelerometer features should show placement specificity vs binary-only

📊 RESULTS STRUCTURE:
- For each split ratio (8:3, 6:5, 4:7, 1:10):
  - 25 result files (5 train placements × 5 test placements)
  - 5 best models (one per training placement)
- Files named: train_{placement}_test_{placement}_{split}_avg.json

📈 EVALUATION METRICS:
- Traditional: Accuracy, F1-score, Precision, Recall
- Information-Theoretic (Class-Agnostic):
  * Normalized Mutual Information (NMI %): Information preservation measure
  * Reverse KL Divergence (%): Model calibration quality measure
  * Both metrics are comparable across different numbers of classes

METHODOLOGY:
- Train on specific placement data with accelerometer features + user identity
- Test cross-placement generalization with rich accelerometer feature set
- Compare against Game-1 (binary-only) and Game-3 (full sensor) results
- Evaluate whether accelerometer improves placement specificity vs binary alone
- Calculate class-agnostic information-theoretic metrics for cross-dataset comparison
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import json
from itertools import combinations
import warnings
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')

from cross_sensor_ablation_common import (
    DEFAULT_SPLIT_SPECS,
    build_results_dir,
    build_run_config,
    calculate_vulnerability,
    sample_combinations,
)

# ============================================================================
# INFORMATION-THEORETIC METRICS FOR CROSS-SENSOR PRIVACY ANALYSIS
# ============================================================================

def calculate_normalized_mutual_information(y_true, y_pred):
    """
    Calculate Normalized Mutual Information (NMI) as a percentage.
    Class-agnostic metric suitable for cross-dataset comparison.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred: Predicted labels (encoded integers)
    
    Returns:
        float: NMI percentage [0, 100] where 100% = perfect prediction
    """
    try:
        # Calculate mutual information using sklearn
        mi = mutual_info_score(y_true, y_pred)
        
        # Calculate entropies
        h_true = entropy(np.bincount(y_true), base=2)
        h_pred = entropy(np.bincount(y_pred), base=2)
        
        # Avoid division by zero
        if h_true == 0 or h_pred == 0:
            return 0.0
        
        # Normalized MI using geometric mean normalization
        nmi = mi / np.sqrt(h_true * h_pred)
        
        # Convert to percentage
        nmi_percentage = nmi * 100.0
        
        return min(100.0, max(0.0, nmi_percentage))
        
    except Exception as e:
        print(f"⚠️ Warning: Error calculating NMI: {e}")
        return 0.0

def calculate_kl_divergence_from_predictions(y_true, y_pred_proba):
    """
    Calculate normalized KL divergence as Reverse KL Percentage.
    Class-agnostic metric suitable for cross-dataset comparison.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
    
    Returns:
        float: Reverse KL percentage [0, 100] where 100% = perfect calibration
    """
    try:
        n_classes = y_pred_proba.shape[1]
        n_samples = len(y_true)
        
        # Create one-hot encoded true labels
        y_true_one_hot = np.zeros((n_samples, n_classes))
        y_true_one_hot[np.arange(n_samples), y_true] = 1.0
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_proba_safe = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
        
        # Calculate KL divergence for each sample
        kl_divs = []
        for i in range(n_samples):
            true_dist = y_true_one_hot[i]
            pred_dist = y_pred_proba_safe[i]
            
            # KL(P||Q) = sum(P * log(P/Q))
            # For one-hot true distribution, this simplifies to -log(Q[true_class])
            true_class = y_true[i]
            kl_div = -np.log(pred_dist[true_class])
            kl_divs.append(kl_div)
        
        # Average KL divergence
        kl_avg = np.mean(kl_divs)
        
        # Theoretical maximum KL divergence (uniform prediction for one-hot true)
        kl_max = np.log(n_classes)
        
        # Normalize KL divergence
        kl_normalized = min(1.0, kl_avg / kl_max)
        
        # Reverse KL percentage (higher = better)
        reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
        
        return max(0.0, min(100.0, reverse_kl_percentage))
        
    except Exception as e:
        print(f"⚠️ Warning: Error calculating KL divergence: {e}")
        return 0.0

def calculate_information_theoretic_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate both NMI and KL divergence metrics for comprehensive evaluation.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred: Predicted labels (encoded integers)  
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
    
    Returns:
        dict: Dictionary containing NMI percentage and Reverse KL percentage
    """
    nmi_percentage = calculate_normalized_mutual_information(y_true, y_pred)
    reverse_kl_percentage = calculate_kl_divergence_from_predictions(y_true, y_pred_proba)
    
    return {
        'nmi_percentage': nmi_percentage,
        'reverse_kl_percentage': reverse_kl_percentage,
        'nmi_interpretation': f"{nmi_percentage:.1f}% information preservation",
        'kl_interpretation': f"{reverse_kl_percentage:.1f}% calibration quality"
    }

# ============================================================================

class Game2CrossSensorAnalyzer:
    """
    Analyzer for Game-2 accelerometer cross-sensor generalization study
    """
    
    def __init__(self):
        self.run_config = build_run_config("game-2-cross-sensor")
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS
        self.all_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Game-2 constraint: Accelerometer + Binary only (NO gyroscope)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.binary_col = 'dec_tree_out_1'
        self.available_cols = self.accel_cols + [self.binary_col]
        
        # This execution path is intentionally restricted to the requested split only.
        self.split_ratios = {
            self.run_config.split_name: DEFAULT_SPLIT_SPECS[self.run_config.split_name]
        }
        
        # Results storage
        self.base_results_dir = str(build_results_dir(self.run_config.results_root, "game-2-cross-sensor"))
        
        print(f"🎯 Target: Game-2 Cross-Sensor Generalization with Accelerometer Data")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📈 Sensors: Accelerometer + Binary (Game-2 constraint)")
        print(f"🚫 Excluded: Gyroscope data (Game-2 constraint)")
        print(f"⏱️  Sampling frequency: {self.run_config.frequency_hz} Hz (native source data)")
        print(f"🧪 Split: {self.run_config.split_name} | Max combinations: {self.run_config.max_combinations}")
        print(f"📊 Metrics: Traditional + Information-Theoretic (NMI %, RKL %)")
        print(f"📊 Added metric: Vulnerability")
        print(f"🔬 Class-agnostic evaluation for cross-dataset comparison")
        print(f"💾 Results directory: {self.base_results_dir}")
        print()
    
    def generate_all_combinations(self, split_name):
        """Generate all possible train/test combinations for a split ratio"""
        config = self.split_ratios[split_name]
        train_size = config['train_size']
        test_size = config['test_size']
        
        # Generate all possible combinations of test users
        test_combinations = list(combinations(self.all_users, test_size))
        
        train_test_pairs = []
        for test_users in test_combinations:
            train_users = [u for u in self.all_users if u not in test_users]
            train_test_pairs.append((list(train_users), list(test_users)))

        sampled_pairs = sample_combinations(
            train_test_pairs,
            self.run_config.max_combinations,
            self.run_config.random_seed,
        )
        print(
            f"Split {split_name}: Using {len(sampled_pairs)}/{len(train_test_pairs)} "
            f"combinations (seed={self.run_config.random_seed})"
        )
        return sampled_pairs
    
    def load_user_data_single_placement(self, user_id, placement):
        """
        Load accelerometer + binary data from SPECIFIC placement
        This is for placement-specific training
        """
        user_dir = f"Data/User {user_id}/Processed"
        
        if not os.path.exists(user_dir):
            return {}
        
        user_data = {}
        
        for activity in self.activities:
            activity_dir = os.path.join(user_dir, activity)
            
            if not os.path.exists(activity_dir):
                continue
            
            # Load data from specific placement only
            if activity in ['Standing', 'Sitting', 'Laying', 'Walking', 'Jogging']:
                # Single file per placement
                file_path = os.path.join(activity_dir, f"{placement}.csv")
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Check if required columns exist
                        missing_cols = [col for col in self.available_cols if col not in df.columns]
                        if missing_cols:
                            continue
                        
                        # Keep accelerometer + binary columns
                        df_clean = df[self.available_cols].copy()
                        user_data[activity] = df_clean
                        
                    except Exception as e:
                        continue
            
            elif activity in ['Upstairs', 'Downstairs']:
                # Multiple files per placement
                placement_subdir = os.path.join(activity_dir, placement)
                
                if os.path.exists(placement_subdir):
                    csv_files = sorted([f for f in os.listdir(placement_subdir) if f.endswith('.csv')])
                    
                    activity_data = []
                    for csv_file in csv_files:
                        file_path = os.path.join(placement_subdir, csv_file)
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Check if required columns exist
                            missing_cols = [col for col in self.available_cols if col not in df.columns]
                            if missing_cols:
                                continue
                            
                            df_clean = df[self.available_cols].copy()
                            activity_data.append(df_clean)
                            
                        except Exception as e:
                            continue
                    
                    if activity_data:
                        user_data[activity] = pd.concat(activity_data, ignore_index=True)
        
        return user_data
    
    def extract_accelerometer_features(self, df, user_id):
        """
        Extract rich accelerometer features for Game-2
        Focus on accelerometer patterns + binary decision tree + user identity
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # 1. ACCELEROMETER STATISTICAL FEATURES
        for axis_col in self.accel_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                
                # Basic statistics
                features[f'accel_{axis}_mean'] = np.mean(data)
                features[f'accel_{axis}_std'] = np.std(data)
                features[f'accel_{axis}_max'] = np.max(data)
                features[f'accel_{axis}_min'] = np.min(data)
                features[f'accel_{axis}_median'] = np.median(data)
                features[f'accel_{axis}_range'] = np.max(data) - np.min(data)
                features[f'accel_{axis}_energy'] = np.sum(data**2)
                
                # Advanced statistics
                if len(data) > 1:
                    features[f'accel_{axis}_skew'] = pd.Series(data).skew()
                    features[f'accel_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                    features[f'accel_{axis}_rms'] = np.sqrt(np.mean(data**2))
                    
                    # Temporal features
                    diff_data = np.diff(data)
                    features[f'accel_{axis}_diff_mean'] = np.mean(diff_data)
                    features[f'accel_{axis}_diff_std'] = np.std(diff_data)
                    features[f'accel_{axis}_diff_max'] = np.max(np.abs(diff_data))
                    
                    # Percentiles
                    features[f'accel_{axis}_q25'] = np.percentile(data, 25)
                    features[f'accel_{axis}_q75'] = np.percentile(data, 75)
                    features[f'accel_{axis}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
                    
                    # Zero crossings
                    features[f'accel_{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
        
        # 2. ACCELEROMETER MAGNITUDE FEATURES
        if all(col in df.columns for col in self.accel_cols):
            accel_data = df[self.accel_cols].values
            magnitude = np.linalg.norm(accel_data, axis=1)
            
            features['accel_magnitude_mean'] = np.mean(magnitude)
            features['accel_magnitude_std'] = np.std(magnitude)
            features['accel_magnitude_max'] = np.max(magnitude)
            features['accel_magnitude_min'] = np.min(magnitude)
            features['accel_magnitude_median'] = np.median(magnitude)
            features['accel_magnitude_range'] = np.max(magnitude) - np.min(magnitude)
            features['accel_magnitude_energy'] = np.sum(magnitude**2)
            
            if len(magnitude) > 1:
                features['accel_magnitude_skew'] = pd.Series(magnitude).skew()
                features['accel_magnitude_kurtosis'] = pd.Series(magnitude).kurtosis()
                features['accel_magnitude_rms'] = np.sqrt(np.mean(magnitude**2))
        
        # 3. CROSS-AXIS CORRELATION FEATURES
        if all(col in df.columns for col in self.accel_cols):
            acc_x = df[self.accel_cols[0]].values
            acc_y = df[self.accel_cols[1]].values
            acc_z = df[self.accel_cols[2]].values
            
            if len(acc_x) > 1:
                features['accel_xy_corr'] = np.corrcoef(acc_x, acc_y)[0, 1] if not np.isnan(np.corrcoef(acc_x, acc_y)[0, 1]) else 0
                features['accel_xz_corr'] = np.corrcoef(acc_x, acc_z)[0, 1] if not np.isnan(np.corrcoef(acc_x, acc_z)[0, 1]) else 0
                features['accel_yz_corr'] = np.corrcoef(acc_y, acc_z)[0, 1] if not np.isnan(np.corrcoef(acc_y, acc_z)[0, 1]) else 0
        
        # 4. BINARY DECISION TREE FEATURES
        if self.binary_col in df.columns:
            binary_data = df[self.binary_col].values
            
            features['binary_activity_ratio'] = np.mean(binary_data)
            features['binary_total_samples'] = len(binary_data)
            features['binary_active_samples'] = np.sum(binary_data)
            
            if len(binary_data) > 1:
                transitions = np.sum(np.abs(np.diff(binary_data.astype(int))))
                features['binary_transitions'] = transitions
                features['binary_transition_rate'] = transitions / len(binary_data)
            else:
                features['binary_transitions'] = 0
                features['binary_transition_rate'] = 0
        
        # 5. USER IDENTITY SEMANTIC INFORMATION
        features['user_id'] = user_id
        
        # Clean NaN and inf values
        cleaned_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = value
            else:
                cleaned_features[key] = value
        
        return cleaned_features
    
    def create_dataset_single_placement(self, user_ids, placement):
        """Create dataset from single placement for training"""
        all_features = []
        all_labels = []
        all_metadata = []
        
        for user_id in user_ids:
            user_data = self.load_user_data_single_placement(user_id, placement)
            
            for activity in self.activities:
                if activity in user_data:
                    df = user_data[activity]
                    if len(df) > 0:
                        # Extract features for this activity
                        features = self.extract_accelerometer_features(df, user_id)
                        
                        if features:
                            all_features.append(features)
                            all_labels.append(activity)
                            all_metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'placement': placement,
                                'n_samples': len(df)
                            })
        
        if not all_features:
            return None, None, None
        
        return all_features, all_labels, all_metadata
    
    def analyze_placement_specific_split(self, split_name, train_placement):
        """
        Analyze specific split ratio for specific training placement
        Train on train_placement, test on ALL placements
        """
        print(f"🔬 ANALYZING {split_name} - Training on {train_placement}")
        print(f"{'='*80}")
        
        # Generate all user combinations for this split
        combinations_list = self.generate_all_combinations(split_name)
        
        # Store results for each test placement
        placement_results = {}
        # Track BEST model for each train-test placement pair
        best_models_per_test_placement = {}
        
        # Initialize results and best model tracking for each test placement
        for test_placement in self.all_placements:
            placement_results[test_placement] = []
            best_models_per_test_placement[test_placement] = {
                'best_accuracy': 0,
                'best_model': None,
                'best_combination': None,
                'best_results': None
            }
        
        # Check for existing progress and resume if needed
        progress_file = os.path.join(self.base_results_dir, f"progress_{train_placement}_{split_name}.json")
        completed_combinations = self._load_progress(progress_file)
        
        # Evaluate each user combination
        successful_runs = 0
        for i, (train_users, test_users) in enumerate(combinations_list):
            combination_id = f"{sorted(train_users)}_{sorted(test_users)}"
            
            # Skip if already completed
            if combination_id in completed_combinations:
                continue
            
            if (i + 1) % 10 == 0:
                print(f"🔄 Progress: {i+1}/{len(combinations_list)} combinations...")
            
            # Train model on train_placement and test on ALL test placements
            result_set = self.evaluate_single_combination_all_placements(
                train_users, test_users, train_placement, split_name
            )
            
            if result_set:
                # Store results for each test placement
                for test_placement, result in result_set.items():
                    if result:
                        placement_results[test_placement].append(result)
                        
                        # Track best model for this train-test pair
                        if result['accuracy'] > best_models_per_test_placement[test_placement]['best_accuracy']:
                            best_models_per_test_placement[test_placement] = {
                                'best_accuracy': result['accuracy'],
                                'best_model': result.get('model_package'),
                                'best_combination': combination_id,
                                'best_results': result
                            }
                
                successful_runs += 1
                
                # Save progress
                completed_combinations.add(combination_id)
                self._save_progress(progress_file, completed_combinations)
        
        print(f"✅ Completed: {successful_runs}/{len(combinations_list)} successful runs")
        
        # Calculate and save aggregate results for each test placement
        for test_placement in self.all_placements:
            if placement_results[test_placement]:
                results = placement_results[test_placement]
                
                # Calculate statistics
                accuracies = [r['accuracy'] for r in results]
                f1_scores = [r['f1_score'] for r in results]
                train_accs = [r['train_accuracy'] for r in results]
                nmi_percentages = [r.get('nmi_percentage', 0.0) for r in results]
                reverse_kl_percentages = [r.get('reverse_kl_percentage', 0.0) for r in results]
                vulnerabilities = [r.get('vulnerability', 0.0) for r in results]
                
                stats = {
                    'train_placement': train_placement,
                    'test_placement': test_placement,
                    'split_ratio': split_name,
                    'n_combinations': len(results),
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'accuracy_min': np.min(accuracies),
                    'accuracy_max': np.max(accuracies),
                    'f1_mean': np.mean(f1_scores),
                    'f1_std': np.std(f1_scores),
                    'train_accuracy_mean': np.mean(train_accs),
                    'precision_mean': np.mean([r['precision'] for r in results]),
                    'recall_mean': np.mean([r['recall'] for r in results]),
                    'nmi_mean': np.mean(nmi_percentages),
                    'nmi_std': np.std(nmi_percentages),
                    'nmi_min': np.min(nmi_percentages),
                    'nmi_max': np.max(nmi_percentages),
                    'reverse_kl_mean': np.mean(reverse_kl_percentages),
                    'reverse_kl_std': np.std(reverse_kl_percentages),
                    'reverse_kl_min': np.min(reverse_kl_percentages),
                    'reverse_kl_max': np.max(reverse_kl_percentages),
                    'vulnerability_mean': np.mean(vulnerabilities),
                    'vulnerability_std': np.std(vulnerabilities),
                    'vulnerability_min': np.min(vulnerabilities),
                    'vulnerability_max': np.max(vulnerabilities),
                    'avg_train_samples': np.mean([r['n_train_samples'] for r in results]),
                    'avg_test_samples': np.mean([r['n_test_samples'] for r in results]),
                    'avg_features': np.mean([r['n_features'] for r in results]),
                    'frequency_hz': self.run_config.frequency_hz,
                    'max_combinations_requested': self.run_config.max_combinations,
                    'random_seed': self.run_config.random_seed,
                }
                
                # Print summary
                print(f"\\n📊 GAME-2 {train_placement}→{test_placement} RESULTS:")
                print(f"   Acc: {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f} | "
                      f"NMI: {stats['nmi_mean']:.1f}% | RKL: {stats['reverse_kl_mean']:.1f}%")
                print(f"   F1-Score:  {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
                print(f"   Vulnerability: {stats['vulnerability_mean']:.3f} ± {stats['vulnerability_std']:.3f}")
                print(f"   Features:  {stats['avg_features']:.0f} (accelerometer + binary + user identity)")
                
                # Save results
                results_file = os.path.join(self.base_results_dir, f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json")
                
                # Clean results for JSON serialization (remove model_package)
                json_safe_results = []
                for result in results:
                    clean_result = {k: v for k, v in result.items() if k != 'model_package'}
                    json_safe_results.append(clean_result)
                
                save_data = {
                    'summary_stats': stats,
                    'individual_results': json_safe_results,
                    'timestamp': datetime.now().isoformat(),
                    'sensor_types': 'accelerometer_binary',
                    'game_type': 'game-2',
                    'frequency_hz': self.run_config.frequency_hz,
                    'max_combinations_requested': self.run_config.max_combinations,
                }
                
                with open(results_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                print(f"💾 Results saved to: {results_file}")
                
                # Save best model for this train-test pair
                if best_models_per_test_placement[test_placement]['best_model']:
                    self._save_best_model(
                        best_models_per_test_placement[test_placement]['best_model'],
                        best_models_per_test_placement[test_placement]['best_results'],
                        train_placement, test_placement, split_name
                    )
        
        return placement_results, best_models_per_test_placement
    
    def evaluate_single_combination_all_placements(self, train_users, test_users, train_placement, split_name):
        """
        Train on specific placement and test on ALL placements
        Returns results dictionary with all test placement results
        """
        try:
            # Train model on train_placement
            X_train, y_train, train_meta = self.create_dataset_single_placement(train_users, train_placement)
            
            if X_train is None or len(X_train) == 0:
                return None
            
            # Prepare training data
            train_df = pd.DataFrame(X_train)
            train_df = train_df.fillna(0)
            
            # One-hot encode user identity
            if 'user_id' in train_df.columns:
                user_dummies = pd.get_dummies(train_df['user_id'], prefix='user')
                train_df = pd.concat([train_df.drop('user_id', axis=1), user_dummies], axis=1)
            
            train_df = train_df.fillna(0)
            feature_columns = train_df.columns.tolist()
            
            # Encode labels and scale features
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_df.values)
            
            # Train model - Game-2 optimized Random Forest
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train_encoded)
            
            # Get training accuracy
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
            
            # Create model package for testing
            model_package = {
                'model': model,
                'scaler': scaler,
                'label_encoder': le,
                'feature_columns': feature_columns,
                'train_placement': train_placement
            }
            
            # Test on ALL placements
            results = {}
            
            for test_placement in self.all_placements:
                # Test on this placement
                test_accuracy, test_f1, test_precision, test_recall, nmi_percentage, reverse_kl_percentage, vulnerability = self._test_model_on_placement(
                    model_package, test_users, test_placement
                )
                
                if test_accuracy is not None:
                    combination_id = f"{sorted(train_users)}_{sorted(test_users)}"
                    
                    results[test_placement] = {
                        'train_users': train_users,
                        'test_users': test_users,
                        'train_placement': train_placement,
                        'test_placement': test_placement,
                        'accuracy': test_accuracy,
                        'f1_score': test_f1,
                        'precision': test_precision,
                        'recall': test_recall,
                        'nmi_percentage': nmi_percentage,
                        'reverse_kl_percentage': reverse_kl_percentage,
                        'vulnerability': vulnerability,
                        'train_accuracy': train_accuracy,
                        'n_train_samples': len(X_train),
                        'n_test_samples': 0,  # Will be updated in test function
                        'n_features': len(feature_columns),
                        'combination_id': combination_id,
                        'has_user_identity': True,
                        'sensor_types': 'accelerometer_binary',
                        'model_package': model_package  # Include for best model saving
                    }
            
            return results
            
        except Exception as e:
            print(f"❌ Error in combination {train_placement}: {e}")
            return None
    
    def _test_model_on_placement(self, model_package, test_users, test_placement):
        """Test trained model on specific placement
        Now includes information-theoretic metrics (NMI and KL divergence)
        """
        try:
            # Load test data from specific placement
            test_features = []
            test_labels = []
            
            for user_id in test_users:
                user_data = self.load_user_data_single_placement(user_id, test_placement)
                
                for activity in self.activities:
                    if activity in user_data:
                        df = user_data[activity]
                        if len(df) > 0:
                            # Extract features for this activity
                            features = self.extract_accelerometer_features(df, user_id)
                            
                            if features:
                                test_features.append(features)
                                test_labels.append(activity)
            
            if not test_features:
                return None, None, None, None, None, None
            
            # Convert to DataFrame and align with training features
            test_features_df = pd.DataFrame(test_features)
            
            # Align columns with training data
            feature_columns = model_package['feature_columns']
            for col in feature_columns:
                if col not in test_features_df.columns:
                    test_features_df[col] = 0
            
            test_features_df = test_features_df[feature_columns]
            test_features_df = test_features_df.fillna(0)
            
            # Scale features using training scaler
            X_test_scaled = model_package['scaler'].transform(test_features_df)
            
            # Encode labels using training encoder
            test_labels_encoded = model_package['label_encoder'].transform(test_labels)
            
            # Make predictions
            y_pred = model_package['model'].predict(X_test_scaled)
            y_pred_proba = model_package['model'].predict_proba(X_test_scaled)
            
            # Calculate traditional metrics
            accuracy = accuracy_score(test_labels_encoded, y_pred)
            f1 = f1_score(test_labels_encoded, y_pred, average='weighted')
            precision = precision_score(test_labels_encoded, y_pred, average='weighted')
            recall = recall_score(test_labels_encoded, y_pred, average='weighted')
            
            # Calculate information-theoretic metrics (class-agnostic)
            info_metrics = calculate_information_theoretic_metrics(
                test_labels_encoded, y_pred, y_pred_proba
            )
            nmi_percentage = info_metrics['nmi_percentage']
            reverse_kl_percentage = info_metrics['reverse_kl_percentage']
            vulnerability = calculate_vulnerability(y_pred_proba)
            
            return accuracy, f1, precision, recall, nmi_percentage, reverse_kl_percentage, vulnerability
            
        except Exception as e:
            print(f"❌ Error testing on placement {test_placement}: {e}")
            return None, None, None, None, None, None, None
    
    def _load_progress(self, progress_file):
        """Load completed combinations from progress file"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                return set(data.get('completed_combinations', []))
            except Exception as e:
                print(f"⚠️  Warning: Could not load progress file {progress_file}: {e}")
        return set()
    
    def _save_progress(self, progress_file, completed_combinations):
        """Save progress to checkpoint file"""
        try:
            progress_data = {
                'completed_combinations': list(completed_combinations),
                'total_completed': len(completed_combinations),
                'last_updated': datetime.now().isoformat()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save progress: {e}")
    
    def _save_best_model(self, model_package, result_data, train_placement, test_placement, split_name):
        """Save best model for specific train-test placement combination"""
        try:
            # Create best models directory structure
            models_dir = os.path.join(self.base_results_dir, "best_models", split_name)
            os.makedirs(models_dir, exist_ok=True)
            
            # Model filename for best model
            model_filename = f"best_model_{train_placement}_to_{test_placement}.joblib"
            model_filepath = os.path.join(models_dir, model_filename)
            
            # Save model
            joblib.dump(model_package, model_filepath)
            
            # Save model metadata
            metadata_filename = f"best_model_{train_placement}_to_{test_placement}_metadata.json"
            metadata_filepath = os.path.join(models_dir, metadata_filename)
            
            metadata = {
                'train_placement': train_placement,
                'test_placement': test_placement,
                'split_ratio': split_name,
                'combination_id': result_data['combination_id'],
                'train_users': result_data['train_users'],
                'test_users': result_data['test_users'],
                'accuracy': result_data['accuracy'],
                'f1_score': result_data['f1_score'],
                'precision': result_data['precision'],
                'recall': result_data['recall'],
                'train_accuracy': result_data['train_accuracy'],
                'n_features': result_data['n_features'],
                'sensor_types': result_data['sensor_types'],
                'saved_timestamp': datetime.now().isoformat(),
                'game_type': 'game-2'
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"🏆 Best model saved: {model_filename} (acc: {result_data['accuracy']:.3f})")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save best model: {e}")
    
    def _analyze_detailed_progress(self):
        """Analyze progress at the most granular level possible"""
        print(f"\\n📊 DETAILED PROGRESS ANALYSIS (Game-2)")
        print("=" * 80)
        
        for split_name in self.split_ratios.keys():
            config = self.split_ratios[split_name]
            actual_combinations = len(list(combinations(self.all_users, config['test_size'])))
            
            print(f"\\n🔬 SPLIT: {split_name} ({actual_combinations} combinations per placement)")
            
            split_fully_complete = self._is_split_fully_completed(split_name)
            if split_fully_complete:
                print(f"   ✅ FULLY COMPLETE - All placements done")
                continue
            
            for train_placement in self.all_placements:
                placement_complete = self._is_placement_completed(split_name, train_placement)
                
                if placement_complete:
                    print(f"   {train_placement:>12}: ✅ COMPLETE (5/5 test placements)")
                else:
                    # Count completed combinations for this placement
                    completed_combinations = self._count_completed_combinations(split_name, train_placement)
                    print(f"   {train_placement:>12}: 🔄 {completed_combinations}/{actual_combinations} combinations")
                    
                    # Show test placement breakdown
                    for test_placement in self.all_placements:
                        pair_complete = self._is_train_test_pair_completed(split_name, train_placement, test_placement)
                        status = "✅" if pair_complete else "⏳"
                        print(f"      → {test_placement:>12}: {status}")
    
    def _is_split_fully_completed(self, split_name):
        """Check if entire split is fully completed at all placement levels"""
        # Check if ALL required result files exist (5 train placements × 5 test placements = 25 files)
        expected_files = 0
        existing_files = 0
        
        for train_placement in self.all_placements:
            for test_placement in self.all_placements:
                expected_files += 1
                avg_file = os.path.join(self.base_results_dir, f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json")
                if os.path.exists(avg_file):
                    existing_files += 1
        
        return existing_files == expected_files == 25
    
    def _is_placement_completed(self, split_name, train_placement):
        """Check if a specific training placement is completed for all test placements"""
        # Check if ALL 5 result files exist for this training placement
        for test_placement in self.all_placements:
            avg_file = os.path.join(self.base_results_dir, f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json")
            if not os.path.exists(avg_file):
                return False
        return True
    
    def _is_train_test_pair_completed(self, split_name, train_placement, test_placement):
        """Check if a specific train-test placement pair has results saved"""
        avg_file = os.path.join(self.base_results_dir, f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json")
        return os.path.exists(avg_file)
    
    def _count_completed_combinations(self, split_name, train_placement):
        """Count how many combinations are completed for a specific placement"""
        progress_file = os.path.join(self.base_results_dir, f"progress_{train_placement}_{split_name}.json")
        completed_combinations = self._load_progress(progress_file)
        return len(completed_combinations)
    
    def _load_placement_results(self, split_name, train_placement):
        """Load results for a specific completed placement"""
        placement_file = os.path.join(self.base_results_dir, f"placement_{train_placement}_{split_name}_complete.json")
        
        if not os.path.exists(placement_file):
            return {}, {}
        
        try:
            with open(placement_file, 'r') as f:
                data = json.load(f)
            
            return data.get('placement_results', {}), data.get('placement_best_models', {})
        except Exception as e:
            print(f"⚠️  Warning: Could not load placement results for {train_placement}_{split_name}: {e}")
            return {}, {}
    
    def _save_placement_results_immediately(self, split_name, train_placement, placement_results, placement_best_models):
        """Save placement results immediately after completion"""
        try:
            placement_file = os.path.join(self.base_results_dir, f"placement_{train_placement}_{split_name}_complete.json")
            
            # Clean placement_results for JSON serialization (remove model_package)
            clean_placement_results = {}
            for test_placement, results_list in placement_results.items():
                clean_results_list = []
                for result in results_list:
                    clean_result = {k: v for k, v in result.items() if k != 'model_package'}
                    clean_results_list.append(clean_result)
                clean_placement_results[test_placement] = clean_results_list
            
            save_data = {
                'split_ratio': split_name,
                'train_placement': train_placement,
                'placement_results': clean_placement_results,
                'placement_best_models': {k: {
                    'best_accuracy': v['best_accuracy'],
                    'best_combination': v['best_combination'],
                    # Don't save the actual model object in JSON
                } for k, v in placement_best_models.items()},
                'completed_timestamp': datetime.now().isoformat(),
                'game_type': 'game-2'
            }
            
            with open(placement_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Placement results saved: {placement_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save placement results: {e}")
    
    def _save_split_completion_marker(self, split_name, split_results, split_best_models):
        """Mark entire split as completed"""
        try:
            split_complete_file = os.path.join(self.base_results_dir, f"split_{split_name}_fully_complete.json")
            
            save_data = {
                'split_ratio': split_name,
                'completed_timestamp': datetime.now().isoformat(),
                'total_placements': len(self.all_placements),
                'game_type': 'game-2',
                'status': 'fully_complete'
            }
            
            with open(split_complete_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"✅ Split completion marker saved: {split_complete_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save split completion marker: {e}")
    
    def _save_split_results_immediately(self, split_name, split_results, split_best_models):
        """Save additional summary results immediately after a split completes"""
        try:
            print(f"💾 Creating summary files for {split_name} (individual files already saved)...")
            
            # Count existing individual files to verify completion
            individual_files_count = 0
            for train_placement in self.all_placements:
                for test_placement in self.all_placements:
                    filename = f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json"
                    filepath = os.path.join(self.base_results_dir, filename)
                    if os.path.exists(filepath):
                        individual_files_count += 1
            
            print(f"✅ Verified: {individual_files_count}/25 individual result files exist")
            
            # Save summary
            split_summary = {
                'split_ratio': split_name,
                'game_type': 'game-2',
                'sensor_types': 'accelerometer_binary',
                'individual_files_verified': individual_files_count,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save split summary
            summary_file = os.path.join(self.base_results_dir, f"split_{split_name}_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(split_summary, f, indent=2)
            
            print(f"✅ Split {split_name} summary saved: split_{split_name}_summary.json")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save split summary: {e}")
    
    def _analyze_placement_with_granular_restart(self, split_name, train_placement):
        """Analyze placement with ultra-granular restart from exact combination"""
        print(f"   🔍 Analyzing {train_placement} with granular restart...")
        
        # Use the existing placement analysis method
        placement_results, placement_best_models = self.analyze_placement_specific_split(split_name, train_placement)
        
        return placement_results, placement_best_models
    
    def run_comprehensive_analysis(self):
        """
        Run the comprehensive placement-specific analysis with ultra-granular restart.
        
        🎯 EXPERIMENTAL DESIGN:
        - For each split ratio (8:3, 6:5, 4:7, 1:10)
        - For each train placement (5 placements)
        - Train on specific placement and test on ALL placements
        - Save results with ultra-granular checkpointing
        
        🔄 RESTART CAPABILITIES:
        - Resumes from exact combination if interrupted
        - Placement-level completion tracking
        - Split-level completion markers
        - Best model preservation
        """
        print("🚀 STARTING COMPREHENSIVE GAME-2 CROSS-SENSOR ANALYSIS")
        print("=" * 80)
        print(f"📂 Results directory: {self.base_results_dir}")
        print(f"📊 Split ratios to analyze: {list(self.split_ratios.keys())}")
        print(f"📍 Placements: {self.all_placements}")
        print(f"👥 Users: {len(self.all_users)} users")
        print(f"🏃 Activities: {self.activities}")
        print(f"🎯 ACCELEROMETER FEATURES: Accelerometer + Binary + User Identity")
        print(f"📈 EVALUATION: Traditional + Information-Theoretic Metrics (NMI %, RKL %)")
        print("=" * 80)
        
        # Analyze current progress status
        self._analyze_detailed_progress()
        
        # Process each split ratio
        for split_name in self.split_ratios.keys():
            print(f"\\n🔬 PROCESSING SPLIT: {split_name}")
            print("-" * 50)
            
            # Check if split is already fully completed
            if self._is_split_fully_completed(split_name):
                print(f"✅ Split {split_name} already fully completed. Skipping...")
                continue
            
            split_results = {}
            split_best_models = {}
            
            # Process each training placement
            for train_placement in self.all_placements:
                print(f"\\n📍 Processing train placement: {train_placement}")
                
                # Check if this specific placement is already completed
                if self._is_placement_completed(split_name, train_placement):
                    print(f"✅ Placement {train_placement} for {split_name} already completed")
                    # Load existing results
                    placement_results, placement_best_models = self._load_placement_results(split_name, train_placement)
                    split_results.update(placement_results)
                    split_best_models.update(placement_best_models)
                    continue
                
                # Analyze this placement with granular restart
                placement_results, placement_best_models = self._analyze_placement_with_granular_restart(
                    split_name, train_placement
                )
                
                if placement_results:
                    # Save placement results immediately
                    self._save_placement_results_immediately(
                        split_name, train_placement, placement_results, placement_best_models
                    )
                    
                    # Add to split results
                    split_results.update(placement_results)
                    split_best_models.update(placement_best_models)
                    
                    print(f"✅ Completed placement {train_placement} for {split_name}")
                else:
                    print(f"❌ Failed to process placement {train_placement} for {split_name}")
            
            # Mark split as fully completed
            if len(split_results) > 0:
                # Save all results immediately
                self._save_split_results_immediately(split_name, split_results, split_best_models)
                
                # Mark split as completed
                self._save_split_completion_marker(split_name, split_results, split_best_models)
                print(f"🎉 Split {split_name} fully completed!")
            
        print("\\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("=" * 80)
        
        # Final progress summary
        self._analyze_detailed_progress()
        
        print("\\n💾 All results saved with ultra-granular checkpointing")
        print("🔄 Script can be safely restarted from any interruption point")
        print("🏆 Best models saved for each train-test placement combination")
        print("📊 Ready for results aggregation and analysis")


def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = Game2CrossSensorAnalyzer()
        
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis()
        
    except KeyboardInterrupt:
        print("\\n⚠️  Analysis interrupted by user. Progress has been saved.")
        print("🔄 Restart script to continue from where you left off.")
    except Exception as e:
        print(f"\\n❌ Error during analysis: {e}")
        print("🔄 Check error and restart script to continue from checkpoint.")


if __name__ == "__main__":
    main()
