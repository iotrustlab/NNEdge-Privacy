#!/usr/bin/env python3
"""
Game-1 User Identity Only: Placement-Specific Cross-Generalization Study
========================================================================

This script implements a placement-specific Game-1 adversarial HAR study to test
cross-placement generalization capabilities when user identity is available.

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Binary decision tree outputs from specific known placements
- Temporal patterns and activity signatures

🎯 RESEARCH QUESTION:
How well do models trained on one specific placement generalize to other placements
when user identity information is available?

🔬 EXPERIMENTAL DESIGN:
- Train models on ONE specific placement (e.g., left-wrist only)
- Test on ALL 5 placements individually
- Compare same-placement vs cross-placement performance
- Hypothesis: Same-placement performance > Cross-placement performance

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
- Train on specific placement data with user identity features
- Test cross-placement generalization capability
- Save placement-specific models for future use
- Evaluate user identity effectiveness in cross-placement scenarios
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
from datetime import datetime
from itertools import combinations
import json
import joblib  # For saving actual model objects
import warnings
warnings.filterwarnings('ignore')

from cross_sensor_ablation_common import (
    DEFAULT_SPLIT_SPECS,
    build_results_dir,
    build_run_config,
    calculate_vulnerability,
    sample_combinations,
)

print("🎭 GAME-1 PLACEMENT-SPECIFIC CROSS-GENERALIZATION STUDY")
print("=" * 65)
print("Testing cross-placement generalization WITH user identity semantic information")
print("Training on single placements | Testing on all placements individually")
print("📊 Evaluation: Traditional + Information-Theoretic Metrics (NMI %, RKL %)")
print()

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

class Game1UserIdentityOnlyAnalyzer:
    """
    Analyzer that removes placement information while using user identity
    for adversarial HAR attacks with semantic information
    """
    
    def __init__(self):
        self.run_config = build_run_config("game-1-cross-sensor")
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        
        # This execution path is intentionally restricted to the requested split only.
        self.split_ratios = {
            self.run_config.split_name: DEFAULT_SPLIT_SPECS[self.run_config.split_name]
        }
        
        # Results storage
        self.base_results_dir = str(build_results_dir(self.run_config.results_root, "game-1-cross-sensor"))
        
        # Initialize parallel processing
        self.gpu_available = self._enable_gpu_acceleration()
        
        print(f"🎯 Target: Placement-specific adversarial HAR with user identity")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"🧪 Cross-placement generalization study")
        print(f"🌲 Model: Random Forest with parallel CPU processing")
        print(f"🏆 Model selection: Best 25 models per split (5×5 train-test combinations)")
        print(f"⏱️  Sampling frequency: {self.run_config.frequency_hz} Hz (native source data)")
        print(f"🧪 Split: {self.run_config.split_name} | Max combinations: {self.run_config.max_combinations}")
        print(f"📈 Metrics: Traditional + Information-Theoretic (NMI %, RKL %)")
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
    
    def load_user_data_placement_anonymous(self, user_id):
        """
        Load binary data from ALL placements but anonymize placement information
        This simulates an adversary who has access to binary signals but doesn't 
        know which signal comes from which body location
        """
        user_dir = f"Data/User {user_id}/Processed"
        
        if not os.path.exists(user_dir):
            print(f"⚠️  Warning: User {user_id} directory not found: {user_dir}")
            return {}
        
        user_data = {}
        
        for activity in self.activities:
            activity_data = []
            activity_dir = os.path.join(user_dir, activity)
            
            if not os.path.exists(activity_dir):
                continue
            
            # Load data from ALL placements but anonymize them
            for placement_idx, placement in enumerate(self.all_placements):
                
                if activity in ['Standing', 'Sitting', 'Laying', 'Walking', 'Jogging']:
                    # Single file per placement: Data/User X/Processed/ACTIVITY/PLACEMENT.csv
                    file_path = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if self.binary_col in df.columns:
                                # Create anonymized placement identifier
                                df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep binary column + metadata (no placement name)
                                df_clean = df[['anonymous_placement_id', 'user_id', 'activity', self.binary_col]].copy()
                                activity_data.append(df_clean)
                                
                        except Exception as e:
                            print(f"⚠️  Error loading {file_path}: {e}")
                            continue
                
                elif activity in ['Upstairs', 'Downstairs']:
                    # Multiple files per placement: Data/User X/Processed/ACTIVITY/PLACEMENT/PLACEMENT_activityN.csv
                    placement_subdir = os.path.join(activity_dir, placement)
                    
                    if os.path.exists(placement_subdir):
                        csv_files = sorted([f for f in os.listdir(placement_subdir) if f.endswith('.csv')])
                        
                        for csv_file in csv_files:
                            file_path = os.path.join(placement_subdir, csv_file)
                            try:
                                df = pd.read_csv(file_path)
                                if self.binary_col in df.columns:
                                    # Create anonymized placement identifier
                                    df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    # Only keep binary column + metadata (no placement name)
                                    df_clean = df[['anonymous_placement_id', 'user_id', 'activity', self.binary_col]].copy()
                                    activity_data.append(df_clean)
                                    
                            except Exception as e:
                                print(f"⚠️  Error loading {file_path}: {e}")
                                continue
            
            if activity_data:
                # Concatenate all placement data for this activity
                user_data[activity] = pd.concat(activity_data, ignore_index=True)
                # Debug: Show successful data loading
                if activity == 'Walking':  # Only show for one activity
                    print(f"✅ User {user_id} {activity}: {len(user_data[activity])} samples from {len(activity_data)} placements")
            
        return user_data
    
    def load_user_data_single_placement(self, user_id, target_placement):
        """
        Load binary data from ONE SPECIFIC placement for placement-specific training
        """
        user_dir = f"Data/User {user_id}/Processed"
        
        if not os.path.exists(user_dir):
            print(f"⚠️  Warning: User {user_id} directory not found: {user_dir}")
            return {}
        
        user_data = {}
        
        for activity in self.activities:
            activity_data = []
            activity_dir = os.path.join(user_dir, activity)
            
            if not os.path.exists(activity_dir):
                continue
            
            # Load data from ONLY the specified placement
            placement = target_placement
                
            if activity in ['Standing', 'Sitting', 'Laying', 'Walking', 'Jogging']:
                # Single file per placement: Data/User X/Processed/ACTIVITY/PLACEMENT.csv
                file_path = os.path.join(activity_dir, f"{placement}.csv")
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        if self.binary_col in df.columns:
                            # Keep original placement name for single placement training
                            df['placement'] = placement
                            df['user_id'] = user_id
                            df['activity'] = activity
                            
                            # Keep binary column + metadata with placement info
                            df_clean = df[['placement', 'user_id', 'activity', self.binary_col]].copy()
                            activity_data.append(df_clean)
                            
                    except Exception as e:
                        print(f"⚠️  Error loading {file_path}: {e}")
                        continue
            
            elif activity in ['Upstairs', 'Downstairs']:
                # Multiple files per placement: Data/User X/Processed/ACTIVITY/PLACEMENT/PLACEMENT_activityN.csv
                placement_subdir = os.path.join(activity_dir, placement)
                
                if os.path.exists(placement_subdir):
                    csv_files = sorted([f for f in os.listdir(placement_subdir) if f.endswith('.csv')])
                    
                    for csv_file in csv_files:
                        file_path = os.path.join(placement_subdir, csv_file)
                        try:
                            df = pd.read_csv(file_path)
                            if self.binary_col in df.columns:
                                # Keep original placement name for single placement training
                                df['placement'] = placement
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Keep binary column + metadata with placement info
                                df_clean = df[['placement', 'user_id', 'activity', self.binary_col]].copy()
                                activity_data.append(df_clean)
                                
                        except Exception as e:
                            print(f"⚠️  Error loading {file_path}: {e}")
                            continue
            
            if activity_data:
                # Concatenate data for this activity from this specific placement
                user_data[activity] = pd.concat(activity_data, ignore_index=True)
            
        return user_data
    
    def extract_placement_anonymous_features(self, df, user_id):
        """
        Extract features WITHOUT using placement information
        Focus ONLY on temporal patterns - NO activity labels used during inference
        
        CRITICAL: This function MUST NOT use activity labels for feature extraction
        to ensure no cheating during inference
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique anonymous sensors for this user's data
        unique_sensors = df['anonymous_placement_id'].unique()
        n_sensors = len(unique_sensors)
        
        # GLOBAL TEMPORAL FEATURES (aggregated across all anonymous sensors)
        # IMPORTANT: Only using binary sensor data, NOT activity labels
        all_binary_data = df[self.binary_col].values
        
        # 1. OVERALL ACTIVITY PATTERNS (placement-agnostic, activity-agnostic)
        features['global_activity_ratio'] = np.mean(all_binary_data)
        features['global_total_samples'] = len(all_binary_data)
        features['global_active_samples'] = np.sum(all_binary_data)
        features['global_inactive_samples'] = np.sum(1 - all_binary_data)
        
        # 2. GLOBAL TEMPORAL DYNAMICS (from binary patterns only)
        if len(all_binary_data) > 1:
            # Transitions across all sensors
            global_transitions = np.sum(np.abs(np.diff(all_binary_data)))
            features['global_transitions'] = global_transitions
            features['global_transition_rate'] = global_transitions / len(all_binary_data)
            
            # Run-length encoding for entire sequence
            global_runs = self._get_run_lengths(all_binary_data)
            if global_runs:
                features.update(self._extract_run_length_features(global_runs, 'global'))
        
        # 3. USER-SPECIFIC PATTERN ENCODING (using user identity semantic info)
        features['user_id'] = user_id  # Use user identity as semantic information
        
        # 4. TEMPORAL PATTERN COMPLEXITY (placement-independent, activity-independent)
        features['pattern_complexity'] = len(np.unique(all_binary_data))
        features['temporal_variance'] = np.var(all_binary_data)
        
        # 5. BASIC SENSOR COUNT (without coordination features)
        features['n_anonymous_sensors'] = n_sensors
        features['avg_sensor_length'] = np.mean([len(df[df['anonymous_placement_id'] == s]) for s in unique_sensors])
        
        # Clean NaN and inf values more robustly
        cleaned_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays
                if len(value) == 0 or np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value[0]) if len(value) == 1 else float(np.mean(value))
            else:
                cleaned_features[key] = value
        
        return cleaned_features
    
    def extract_features(self, df, user_id):
        """
        Extract features for placement-specific training using single placement data
        
        This function is used for placement-specific models (not placement-anonymous)
        Focus on temporal patterns from binary sensor data + user identity
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # BINARY SENSOR FEATURES from single placement
        binary_data = df[self.binary_col].values
        
        # 1. BASIC ACTIVITY PATTERNS
        features['activity_ratio'] = np.mean(binary_data)
        features['total_samples'] = len(binary_data)
        features['active_samples'] = np.sum(binary_data)
        features['inactive_samples'] = np.sum(1 - binary_data)
        
        # 2. TEMPORAL DYNAMICS
        if len(binary_data) > 1:
            # Transitions
            transitions = np.sum(np.abs(np.diff(binary_data)))
            features['transitions'] = transitions
            features['transition_rate'] = transitions / len(binary_data)
            
            # Run-length encoding
            runs = self._get_run_lengths(binary_data)
            if runs:
                features.update(self._extract_run_length_features(runs, 'placement'))
        
        # 3. USER IDENTITY (semantic information)
        features['user_id'] = user_id
        
        # 4. PATTERN COMPLEXITY
        features['pattern_complexity'] = len(np.unique(binary_data))
        features['temporal_variance'] = np.var(binary_data)
        
        # 5. STATISTICAL FEATURES
        if len(binary_data) > 0:
            features['mean_activity'] = np.mean(binary_data)
            features['std_activity'] = np.std(binary_data)
            features['activity_energy'] = np.sum(binary_data**2)
            
            if len(binary_data) > 1:
                # Advanced temporal features
                diff_data = np.diff(binary_data.astype(float))
                features['diff_mean'] = np.mean(diff_data)
                features['diff_std'] = np.std(diff_data)
                features['diff_max'] = np.max(np.abs(diff_data))
                
                # Periodicity detection (simple)
                features['zero_crossings'] = np.sum(np.diff(np.signbit(binary_data - np.mean(binary_data))))
        
        # Clean NaN and inf values
        cleaned_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            elif isinstance(value, np.ndarray):
                if len(value) == 0 or np.any(np.isnan(value)) or np.any(np.isinf(value)):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value[0]) if len(value) == 1 else float(np.mean(value))
            else:
                cleaned_features[key] = value
        
        return cleaned_features
    
    def _get_run_lengths(self, binary_sequence):
        """Extract run lengths from binary sequence"""
        if len(binary_sequence) == 0:
            return []
        
        runs = []
        current_val = binary_sequence[0]
        current_length = 1
        
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] == current_val:
                current_length += 1
            else:
                runs.append((current_val, current_length))
                current_val = binary_sequence[i]
                current_length = 1
        
        runs.append((current_val, current_length))
        return runs
    
    def _extract_run_length_features(self, runs, prefix):
        """Extract statistical features from run lengths"""
        if not runs:
            return {}
        
        active_runs = [length for val, length in runs if val == 1]
        inactive_runs = [length for val, length in runs if val == 0]
        all_runs = [length for val, length in runs]
        
        features = {}
        
        # Active period features
        if active_runs:
            features[f'{prefix}_active_runs_count'] = len(active_runs)
            features[f'{prefix}_active_runs_mean'] = np.mean(active_runs)
            features[f'{prefix}_active_runs_std'] = np.std(active_runs)
            features[f'{prefix}_active_runs_max'] = np.max(active_runs)
        
        # Inactive period features
        if inactive_runs:
            features[f'{prefix}_inactive_runs_count'] = len(inactive_runs)
            features[f'{prefix}_inactive_runs_mean'] = np.mean(inactive_runs)
            features[f'{prefix}_inactive_runs_std'] = np.std(inactive_runs)
            features[f'{prefix}_inactive_runs_max'] = np.max(inactive_runs)
        
        # Overall pattern features
        features[f'{prefix}_total_runs'] = len(runs)
        features[f'{prefix}_run_variance'] = np.var(all_runs)
        
        return features
    
    def create_dataset(self, user_list):
        """
        Create dataset from specified users WITHOUT placement information
        WITH user identity as semantic information
        
        CRITICAL: Activity labels are used ONLY as targets, NEVER as features
        """
        X = []
        y = []
        metadata = []
        
        for user_id in user_list:
            print(f"📊 Processing User {user_id} (placement-anonymous)...", end="")
            user_data = self.load_user_data_placement_anonymous(user_id)
            
            if not user_data:
                print(" ❌ No data")
                continue
            
            user_samples = 0
            for activity in self.activities:
                if activity in user_data:
                    df = user_data[activity]
                    if len(df) > 0:
                        # Extract placement-anonymous features
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_anonymous_features(df, user_id)
                        
                        if features:
                            X.append(features)
                            y.append(activity)  # Activity as TARGET, not feature
                            metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'n_sensors': features.get('n_anonymous_sensors', 0),
                                'total_samples': features.get('global_total_samples', 0)
                            })
                            user_samples += 1
            
            print(f" ✅ {user_samples} activities")
        
        if not X:
            print("❌ No data could be loaded!")
            return None, None, None
        
        # Convert to DataFrame for easier handling
        X_df = pd.DataFrame(X)
        
        # Handle NaN values by filling with 0
        X_df = X_df.fillna(0)
        
        # One-hot encode user identity as features (user semantic information)
        if 'user_id' in X_df.columns:
            user_dummies = pd.get_dummies(X_df['user_id'], prefix='user')
            X_df = pd.concat([X_df.drop('user_id', axis=1), user_dummies], axis=1)
        
        # Final NaN check and replacement
        X_df = X_df.fillna(0)
        
        print(f"📈 Dataset created: {len(X_df)} samples, {len(X_df.columns)} features")
        print(f"🔑 User identity encoded as {len([c for c in X_df.columns if c.startswith('user_')])} user features")
        return X_df.values, np.array(y), metadata
    
    def evaluate_single_combination(self, train_users, test_users):
        """Evaluate a single train/test combination with user identity semantic information"""
        try:
            # Create datasets
            X_train, y_train, train_meta = self.create_dataset(train_users)
            X_test, y_test, test_meta = self.create_dataset(test_users)
            
            if X_train is None or X_test is None:
                return None
            
            if len(X_train) == 0 or len(X_test) == 0:
                return None
            
            # Ensure same feature dimensions
            train_df = pd.DataFrame(X_train)
            test_df = pd.DataFrame(X_test)
            
            # Align features (handle missing columns)
            all_features = set(train_df.columns) | set(test_df.columns)
            
            for col in all_features:
                if col not in train_df.columns:
                    train_df[col] = 0
                if col not in test_df.columns:
                    test_df[col] = 0
            
            # Reorder columns to match
            common_cols = sorted(all_features)
            X_train_aligned = train_df[common_cols].values
            X_test_aligned = test_df[common_cols].values
            
            # Encode labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aligned)
            X_test_scaled = scaler.transform(X_test_aligned)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train_scaled, y_train_encoded)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            vulnerability = calculate_vulnerability(y_pred_proba)
            
            return {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'vulnerability': vulnerability,
                'n_train_samples': len(X_train_aligned),
                'n_test_samples': len(X_test_aligned),
                'n_features': len(common_cols),
                'has_user_identity': True
            }
            
        except Exception as e:
            print(f"❌ Error in combination {train_users} -> {test_users}: {e}")
            return None
    
    def analyze_split_ratio(self, split_name):
        """Analyze all combinations for a specific split ratio with user identity"""
        print(f"\n{'='*80}")
        print(f"🎭 ANALYZING SPLIT RATIO: {split_name} (WITH user identity)")
        print(f"{'='*80}")
        
        # Generate all combinations
        combinations_list = self.generate_all_combinations(split_name)
        
        # Evaluate each combination
        results = []
        successful_runs = 0
        
        for i, (train_users, test_users) in enumerate(combinations_list):
            if (i + 1) % 10 == 0:
                print(f"🔄 Progress: {i+1}/{len(combinations_list)} combinations...")
            
            result = self.evaluate_single_combination(train_users, test_users)
            if result:
                results.append(result)
                successful_runs += 1
        
        print(f"✅ Completed: {successful_runs}/{len(combinations_list)} successful runs")
        
        if not results:
            print("❌ No successful runs!")
            return None
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        vulnerabilities = [r.get('vulnerability', 0.0) for r in results]
        
        stats = {
            'split_ratio': split_name,
            'has_user_identity': True,
            'placement_anonymous': True,
            'n_combinations': len(results),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_mean': np.mean(precisions),
            'recall_mean': np.mean(recalls),
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
        print(f"\n📊 PLACEMENT-ANONYMOUS GAME-1 RESULTS (WITH user identity):")
        print(f"   Accuracy:  {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
        print(f"   F1-Score:  {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
        print(f"   Precision: {stats['precision_mean']:.3f}")
        print(f"   Recall:    {stats['recall_mean']:.3f}")
        print(f"   Vulnerability: {stats['vulnerability_mean']:.3f} ± {stats['vulnerability_std']:.3f}")
        print(f"   Features:  {stats['avg_features']:.0f} (placement-anonymous + user identity)")
        
        # Save results
        results_file = os.path.join(self.base_results_dir, f"{split_name}_with_user_identity.json")
        
        save_data = {
            'summary_stats': stats,
            'individual_results': results,
            'timestamp': datetime.now().isoformat(),
            'frequency_hz': self.run_config.frequency_hz,
            'max_combinations_requested': self.run_config.max_combinations,
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return stats
    
    def run_user_identity_study(self):
        """
        Run complete placement-specific user identity study 
        ULTRA-GRANULAR RESTART: Resumes from exact placement and combination
        Saves maximum time by tracking progress at the finest level
        """
        print("🚀 Starting Game-1 Placement-Specific User Identity Study...")
        print("Training Random Forest models on single placements, testing on all placements")
        print("Testing cross-placement generalization with user identity")
        print("✅ ULTRA-GRANULAR RESUMPTION: Resumes from exact placement-combination")
        print("✅ COMBINATION COUNTING: Tracks individual combination progress")
        print("✅ PLACEMENT-LEVEL RESTART: Skips completed placement pairs")
        print("🌲 Using Random Forest with parallel CPU processing")
        print("💾 IMMEDIATE SAVING: Results saved after each placement completes")
        print()
        
        # Detailed progress analysis
        self._analyze_detailed_progress()
        print("\n🚦 Auto-confirm enabled for scripted execution.")
        
        all_results = {}
        best_models = {}
        
        # Test each split ratio with granular restart
        for split_name in self.split_ratios.keys():
            print(f"\n🔬 Testing {split_name} placement-specific study...")
            
            # Check if entire split is completed
            if self._is_split_fully_completed(split_name):
                print(f"✅ Split {split_name} FULLY completed, loading all results...")
                split_results, split_best_models = self._load_completed_split_results(split_name)
                all_results[split_name] = split_results
                best_models[split_name] = split_best_models
                continue
            
            # Initialize results for this split
            split_results = {}
            split_best_models = {}
            
            # GRANULAR: Check each training placement individually
            for train_placement in self.all_placements:
                print(f"\n🎯 Training placement: {train_placement}")
                
                # Check if this specific placement is completed
                if self._is_placement_completed(split_name, train_placement):
                    print(f"   ✅ Placement {train_placement} already completed, loading results...")
                    placement_results, placement_best_models = self._load_placement_results(split_name, train_placement)
                    split_results[train_placement] = placement_results
                    split_best_models[train_placement] = placement_best_models
                    continue
                
                # ULTRA-GRANULAR: Resume from exact combination within placement
                print(f"   🔍 Analyzing combination progress for {train_placement}...")
                placement_results, placement_best_models = self._analyze_placement_with_granular_restart(
                    split_name, train_placement
                )
                
                # Save placement results immediately
                self._save_placement_results_immediately(split_name, train_placement, placement_results, placement_best_models)
                
                split_results[train_placement] = placement_results
                split_best_models[train_placement] = placement_best_models
                
                print(f"   ✅ Placement {train_placement} COMPLETE and SAVED!")
            
            # Mark entire split as completed and save
            print(f"\n💾 SAVING COMPLETE SPLIT {split_name}...")
            self._save_split_completion_marker(split_name, split_results, split_best_models)
            
            all_results[split_name] = split_results
            best_models[split_name] = split_best_models
            
            print(f"✅ Split {split_name} COMPLETE and SAVED!")
        
        # Generate final summary
        self._generate_final_overall_summary(all_results)
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
        print(f"� Maximum efficiency achieved with granular restart!")
        
        return all_results, best_models
        
        return all_results, best_models
    
    def analyze_placement_specific_split(self, split_name, train_placement):
        """
        Analyze one training placement across all test placements for a specific split ratio
        Returns results for all test placements and tracks best model for each train-test pair
        """
        print(f"\n{'='*80}")
        print(f"🎭 TRAINING ON: {train_placement} | SPLIT: {split_name}")
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
                print(f"⏭️  Skipping completed combination {i+1}/{len(combinations_list)}")
                # Load existing results for this combination
                self._load_existing_combination_results(
                    combination_id, train_placement, split_name, 
                    placement_results, best_models_per_test_placement
                )
                successful_runs += 1
                continue
            
            if (i + 1) % 10 == 0:
                print(f"🔄 Progress: {i+1}/{len(combinations_list)} combinations...")
            
            # Train model on training placement data
            model, train_accuracy = self.train_placement_specific_model_gpu(
                train_users, train_placement
            )
            
            if model is None:
                continue
                
            # Test on all placements and track best models
            combination_results = {}
            
            for test_placement in self.all_placements:
                accuracy, f1, precision, recall, nmi_percentage, reverse_kl_percentage, vulnerability = self.test_on_placement(
                    model, test_users, test_placement
                )
                
                if accuracy is not None:
                    result_data = {
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall,
                        'nmi_percentage': nmi_percentage,
                        'reverse_kl_percentage': reverse_kl_percentage,
                        'vulnerability': vulnerability,
                        'train_users': train_users,
                        'test_users': test_users,
                        'train_placement': train_placement,
                        'test_placement': test_placement,
                        'combination_id': combination_id,
                        'train_accuracy': train_accuracy
                    }
                    
                    combination_results[test_placement] = result_data
                    placement_results[test_placement].append(result_data)
                    
                    # Check if this is the best model for this train-test placement pair
                    if accuracy > best_models_per_test_placement[test_placement]['best_accuracy']:
                        best_models_per_test_placement[test_placement] = {
                            'best_accuracy': accuracy,
                            'best_model': model,
                            'best_combination': (train_users, test_users),
                            'best_results': result_data
                        }
                        
                        # Save this best model immediately
                        self._save_best_placement_model(
                            model, result_data, train_placement, test_placement, split_name
                        )
            
            # Check if this combination was successful across all test placements
            if len(combination_results) == len(self.all_placements):
                successful_runs += 1
                
                # Save progress checkpoint
                completed_combinations.add(combination_id)
                self._save_progress(progress_file, completed_combinations)
                
                # Save combination results checkpoint
                self._save_combination_checkpoint(
                    combination_results, train_placement, split_name, combination_id
                )
        
        print(f"✅ Completed: {successful_runs}/{len(combinations_list)} successful runs")
        
        # Calculate statistics for each test placement
        final_results = {}
        for test_placement in self.all_placements:
            if placement_results[test_placement]:
                results = placement_results[test_placement]
                accuracies = [r['accuracy'] for r in results]
                f1_scores = [r['f1_score'] for r in results]
                precisions = [r['precision'] for r in results]
                recalls = [r['recall'] for r in results]
                nmi_percentages = [r['nmi_percentage'] for r in results]
                reverse_kl_percentages = [r['reverse_kl_percentage'] for r in results]
                vulnerabilities = [r.get('vulnerability', 0.0) for r in results]
                
                final_results[test_placement] = {
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
                    'precision_mean': np.mean(precisions),
                    'recall_mean': np.mean(recalls),
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
                    'individual_results': results,
                    'best_model_accuracy': best_models_per_test_placement[test_placement]['best_accuracy'],
                    'frequency_hz': self.run_config.frequency_hz,
                    'max_combinations_requested': self.run_config.max_combinations,
                    'random_seed': self.run_config.random_seed,
                }
                
                print(f"📊 {train_placement} → {test_placement}: "
                      f"Acc: {final_results[test_placement]['accuracy_mean']:.3f} ± "
                      f"{final_results[test_placement]['accuracy_std']:.3f} | "
                      f"NMI: {final_results[test_placement]['nmi_mean']:.1f}% | "
                      f"RKL: {final_results[test_placement]['reverse_kl_mean']:.1f}% | "
                      f"V: {final_results[test_placement]['vulnerability_mean']:.3f} "
                      f"(best: {final_results[test_placement]['best_model_accuracy']:.3f})")
        
        return final_results, best_models_per_test_placement
    
    def train_placement_specific_model(self, train_users, train_placement):
        """
        Train a model on data from a specific placement and training users
        """
        try:
            # Extract features for training data (one feature vector per activity instance)
            train_features = []
            train_labels = []
            
            for user_id in train_users:
                user_data = self.load_user_data_single_placement(user_id, train_placement)
                
                # Process each activity separately to create feature-label pairs
                for activity, df in user_data.items():
                    if len(df) > 0:
                        # Add user_id and activity columns to the dataframe
                        df = df.copy()
                        df['user_id'] = user_id
                        df['activity'] = activity
                        
                        # Extract features for this activity
                        features = self.extract_features(df, user_id)
                        
                        if features:
                            train_features.append(features)
                            train_labels.append(activity)
            
            if not train_features:
                print(f"⚠️  No training data for placement {train_placement}")
                return None, None
            
            # Convert to arrays
            train_features_df = pd.DataFrame(train_features)
            train_features_df = train_features_df.fillna(0)
            
            # Encode labels
            label_encoder = LabelEncoder()
            train_labels_encoded = label_encoder.fit_transform(train_labels)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_features_df)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, train_labels_encoded)
            
            # Calculate training accuracy
            train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(train_labels_encoded, train_pred)
            
            # Store preprocessing objects with model
            model_package = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_columns': train_features_df.columns.tolist(),
                'train_placement': train_placement,
                'train_accuracy': train_accuracy
            }
            
            return model_package, train_accuracy
            
        except Exception as e:
            print(f"❌ Error training model for placement {train_placement}: {e}")
            return None, None
    
    def test_on_placement(self, model_package, test_users, test_placement):
        """
        Test a trained Random Forest model on data from a specific placement and test users
        Now includes information-theoretic metrics (NMI and KL divergence)
        """
        try:
            if model_package is None:
                return None, None, None, None, None, None
            
            # Extract features for test data (one feature vector per activity instance)
            test_features = []
            test_labels = []
            
            for user_id in test_users:
                user_data = self.load_user_data_single_placement(user_id, test_placement)
                
                # Process each activity separately to create feature-label pairs
                for activity, df in user_data.items():
                    if len(df) > 0:
                        # Add user_id and activity columns to the dataframe
                        df = df.copy()
                        df['user_id'] = user_id
                        df['activity'] = activity
                        
                        # Extract features for this activity
                        features = self.extract_features(df, user_id)
                        
                        if features:
                            test_features.append(features)
                            test_labels.append(activity)
            
            if not test_features:
                return None, None, None, None, None, None, None
            
            # Convert to arrays and align with training features
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
            
            # Make predictions using Random Forest model
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
    
    def _save_individual_model(self, model_package, result_data, train_placement, test_placement, split_name, combination_id):
        """Save individual model for specific train-test placement combination"""
        try:
            # Create models directory structure
            models_dir = os.path.join(self.base_results_dir, "all_models", split_name)
            os.makedirs(models_dir, exist_ok=True)
            
            # Model filename includes train placement, test placement, and combination
            model_filename = f"model_{train_placement}_to_{test_placement}_{combination_id}.joblib"
            model_filepath = os.path.join(models_dir, model_filename)
            
            # Save model
            joblib.dump(model_package, model_filepath)
            
            # Save model metadata
            metadata_filename = f"model_{train_placement}_to_{test_placement}_{combination_id}_metadata.json"
            metadata_filepath = os.path.join(models_dir, metadata_filename)
            
            metadata = {
                'train_placement': train_placement,
                'test_placement': test_placement,
                'split_ratio': split_name,
                'combination_id': combination_id,
                'train_users': result_data['train_users'],
                'test_users': result_data['test_users'],
                'accuracy': result_data['accuracy'],
                'f1_score': result_data['f1_score'],
                'precision': result_data['precision'],
                'recall': result_data['recall'],
                'train_accuracy': result_data['train_accuracy'],
                'model_file': model_filename,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save model for {train_placement}→{test_placement}: {e}")
    
    def _save_best_placement_model(self, model_package, result_data, train_placement, test_placement, split_name):
        """Save the best model for a specific train-test placement pair"""
        try:
            # Create best models directory structure
            best_models_dir = os.path.join(self.base_results_dir, "best_placement_models", split_name)
            os.makedirs(best_models_dir, exist_ok=True)
            
            # Model filename for train-test placement pair
            model_filename = f"best_{train_placement}_to_{test_placement}.joblib"
            model_filepath = os.path.join(best_models_dir, model_filename)
            
            # Save model (overwrite previous best)
            joblib.dump(model_package, model_filepath)
            
            # Save model metadata
            metadata_filename = f"best_{train_placement}_to_{test_placement}_metadata.json"
            metadata_filepath = os.path.join(best_models_dir, metadata_filename)
            
            metadata = {
                'train_placement': train_placement,
                'test_placement': test_placement,
                'split_ratio': split_name,
                'best_accuracy': result_data['accuracy'],
                'f1_score': result_data['f1_score'],
                'precision': result_data['precision'],
                'recall': result_data['recall'],
                'train_users': result_data['train_users'],
                'test_users': result_data['test_users'],
                'train_accuracy': result_data['train_accuracy'],
                'combination_id': result_data['combination_id'],
                'model_file': model_filename,
                'model_type': f'Best {train_placement} → {test_placement} model',
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"🏆 New best {train_placement}→{test_placement}: {result_data['accuracy']:.3f}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save best model for {train_placement}→{test_placement}: {e}")
    
    def _enable_gpu_acceleration(self):
        """Check for parallel processing capabilities (Random Forest only)"""
        import multiprocessing
        
        n_cores = multiprocessing.cpu_count()
        print(f"� Random Forest with parallel processing: {n_cores} CPU cores available")
        print("ℹ️  Using scikit-learn Random Forest (CPU-optimized)")
        
        return False  # No GPU needed for Random Forest
    
    def train_placement_specific_model_gpu(self, train_users, train_placement):
        """
        Train a Random Forest model on data from a specific placement and training users
        """
        try:
            # Extract features for training data (one feature vector per activity instance)
            train_features = []
            train_labels = []
            
            for user_id in train_users:
                user_data = self.load_user_data_single_placement(user_id, train_placement)
                
                # Process each activity separately to create feature-label pairs
                for activity, df in user_data.items():
                    if len(df) > 0:
                        # Add user_id and activity columns to the dataframe
                        df = df.copy()
                        df['user_id'] = user_id
                        df['activity'] = activity
                        
                        # Extract features for this activity
                        features = self.extract_features(df, user_id)
                        
                        if features:
                            train_features.append(features)
                            train_labels.append(activity)
            
            if not train_features:
                print(f"⚠️  No training data for placement {train_placement}")
                return None, None
            
            # Convert to arrays
            train_features_df = pd.DataFrame(train_features)
            train_features_df = train_features_df.fillna(0)
            
            # Encode labels
            label_encoder = LabelEncoder()
            train_labels_encoded = label_encoder.fit_transform(train_labels)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_features_df)
            
            # Train Random Forest model with all available cores
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
            
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=n_jobs  # Use all available CPU cores
            )
            
            model.fit(X_train_scaled, train_labels_encoded)
            
            # Calculate training accuracy
            train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(train_labels_encoded, train_pred)
            
            # Store preprocessing objects with model
            model_package = {
                'model': model,
                'model_type': 'random_forest',
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_columns': train_features_df.columns.tolist(),
                'train_placement': train_placement,
                'train_accuracy': train_accuracy
            }
            
            return model_package, train_accuracy
            
        except Exception as e:
            print(f"❌ Error training model for placement {train_placement}: {e}")
            return None, None
    
    def _save_combination_checkpoint(self, combination_results, train_placement, split_name, combination_id):
        """Save results checkpoint for a completed combination"""
        try:
            checkpoints_dir = os.path.join(self.base_results_dir, "checkpoints", split_name, train_placement)
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(checkpoints_dir, f"combination_{combination_id}.json")
            
            checkpoint_data = {
                'combination_id': combination_id,
                'train_placement': train_placement,
                'split_ratio': split_name,
                'results': combination_results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save combination checkpoint: {e}")
    
    def _load_existing_combination_results(self, combination_id, train_placement, split_name, placement_results, best_models_per_test_placement):
        """Load existing results for a completed combination"""
        try:
            checkpoints_dir = os.path.join(self.base_results_dir, "checkpoints", split_name, train_placement)
            checkpoint_file = os.path.join(checkpoints_dir, f"combination_{combination_id}.json")
            
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Load results back into placement_results
                for test_placement, result_data in checkpoint_data['results'].items():
                    placement_results[test_placement].append(result_data)
                    
                    # Update best models tracking if this result is better
                    accuracy = result_data.get('accuracy', 0)
                    if accuracy > best_models_per_test_placement[test_placement]['best_accuracy']:
                        # Note: We don't load the actual model to save memory, just track the best result
                        best_models_per_test_placement[test_placement]['best_accuracy'] = accuracy
                        best_models_per_test_placement[test_placement]['best_results'] = result_data
                        best_models_per_test_placement[test_placement]['best_combination'] = (
                            result_data.get('train_users', []), 
                            result_data.get('test_users', [])
                        )
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing combination {combination_id}: {e}")
    
    def get_experiment_status(self):
        """Get detailed status of experiment progress"""
        print(f"\n📊 EXPERIMENT STATUS REPORT")
        print("=" * 60)
        
        total_combinations_per_split = len(list(combinations(self.all_users, 3)))  # Example for 8:3
        
        for split_name in self.split_ratios.keys():
            print(f"\n🔬 SPLIT RATIO: {split_name}")
            
            total_expected = len(self.all_placements) * total_combinations_per_split
            total_completed = 0
            
            for train_placement in self.all_placements:
                progress_file = os.path.join(self.base_results_dir, f"progress_{train_placement}_{split_name}.json")
                completed = self._load_progress(progress_file)
                
                print(f"   {train_placement:>12}: {len(completed):>3}/{total_combinations_per_split} combinations")
                total_completed += len(completed)
            
            completion_pct = (total_completed / total_expected) * 100 if total_expected > 0 else 0
            print(f"   {'TOTAL':>12}: {total_completed:>3}/{total_expected} ({completion_pct:.1f}%)")
        
        return total_completed, total_expected
    
    def _analyze_detailed_progress(self):
        """Analyze progress at the most granular level possible"""
        print(f"\n📊 DETAILED PROGRESS ANALYSIS")
        print("=" * 80)
        
        total_combinations_per_split = len(list(combinations(self.all_users, 3)))  # Base for counting
        
        for split_name in self.split_ratios.keys():
            config = self.split_ratios[split_name]
            actual_combinations = len(list(combinations(self.all_users, config['test_size'])))
            
            print(f"\n🔬 SPLIT: {split_name} ({actual_combinations} combinations per placement)")
            
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
        split_complete_file = os.path.join(self.base_results_dir, f"split_{split_name}_fully_complete.json")
        return os.path.exists(split_complete_file)
    
    def _is_placement_completed(self, split_name, train_placement):
        """Check if a specific training placement is completed for all test placements"""
        placement_complete_file = os.path.join(self.base_results_dir, f"placement_{train_placement}_{split_name}_complete.json")
        return os.path.exists(placement_complete_file)
    
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
    
    def _analyze_placement_with_granular_restart(self, split_name, train_placement):
        """Analyze placement with ultra-granular restart from exact combination"""
        print(f"   🔍 Analyzing {train_placement} with granular restart...")
        
        # Generate all combinations for this split
        combinations_list = self.generate_all_combinations(split_name)
        progress_file = os.path.join(self.base_results_dir, f"progress_{train_placement}_{split_name}.json")
        completed_combinations = self._load_progress(progress_file)
        
        # Calculate remaining work
        total_combinations = len(combinations_list)
        completed_count = len(completed_combinations)
        remaining = total_combinations - completed_count
        
        print(f"   📊 Progress: {completed_count}/{total_combinations} combinations ({remaining} remaining)")
        
        if remaining == 0:
            print(f"   ✅ All combinations complete, loading existing results...")
            return self._reconstruct_placement_results_from_checkpoints(split_name, train_placement)
        
        print(f"   🚀 Resuming from combination {completed_count + 1}...")
        
        # Use existing analyze_placement_specific_split but with detailed progress
        return self.analyze_placement_specific_split(split_name, train_placement)
    
    def _reconstruct_placement_results_from_checkpoints(self, split_name, train_placement):
        """Reconstruct placement results from existing checkpoint files"""
        placement_results = {}
        best_models_per_test_placement = {}
        
        # Initialize for each test placement
        for test_placement in self.all_placements:
            placement_results[test_placement] = []
            best_models_per_test_placement[test_placement] = {
                'best_accuracy': 0,
                'best_model': None,
                'best_combination': None,
                'best_results': None
            }
        
        # Load from checkpoint directory
        checkpoints_dir = os.path.join(self.base_results_dir, "checkpoints", split_name, train_placement)
        
        if os.path.exists(checkpoints_dir):
            checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.json')]
            
            for checkpoint_file in checkpoint_files:
                try:
                    with open(os.path.join(checkpoints_dir, checkpoint_file), 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    # Reconstruct results
                    for test_placement, result_data in checkpoint_data.get('results', {}).items():
                        placement_results[test_placement].append(result_data)
                        
                        # Update best model if this is better
                        if result_data['accuracy'] > best_models_per_test_placement[test_placement]['best_accuracy']:
                            best_models_per_test_placement[test_placement]['best_accuracy'] = result_data['accuracy']
                            best_models_per_test_placement[test_placement]['best_results'] = result_data
                
                except Exception as e:
                    print(f"⚠️  Warning loading checkpoint {checkpoint_file}: {e}")
        
        # Calculate final statistics
        final_results = {}
        for test_placement in self.all_placements:
            if placement_results[test_placement]:
                results = placement_results[test_placement]
                accuracies = [r['accuracy'] for r in results]
                f1_scores = [r['f1_score'] for r in results]
                precisions = [r['precision'] for r in results]
                recalls = [r['recall'] for r in results]
                # Handle backward compatibility for existing results without new metrics
                nmi_percentages = [r.get('nmi_percentage', 0.0) for r in results]
                reverse_kl_percentages = [r.get('reverse_kl_percentage', 0.0) for r in results]
                vulnerabilities = [r.get('vulnerability', 0.0) for r in results]
                
                final_results[test_placement] = {
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
                    'precision_mean': np.mean(precisions),
                    'recall_mean': np.mean(recalls),
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
                    'individual_results': results,
                    'best_model_accuracy': best_models_per_test_placement[test_placement]['best_accuracy'],
                    'frequency_hz': self.run_config.frequency_hz,
                    'max_combinations_requested': self.run_config.max_combinations,
                    'random_seed': self.run_config.random_seed,
                }
        
        return final_results, best_models_per_test_placement
    
    def _save_placement_results_immediately(self, split_name, train_placement, placement_results, placement_best_models):
        """Save results immediately after a placement completes"""
        try:
            # Save individual avg files for this placement (5 files)
            print(f"      💾 Saving avg files for {train_placement}...")
            for test_placement, test_stats in placement_results.items():
                filename = f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json"
                filepath = os.path.join(self.base_results_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump({
                        'summary_stats': test_stats,
                        'study_type': 'Game-1 Placement-Specific User Identity Study',
                        'semantic_info_available': ['user_identity'],
                        'placement_specific_training': True,
                        'cross_placement_testing': True,
                        'granular_restart_enabled': True,
                        'frequency_hz': self.run_config.frequency_hz,
                        'max_combinations_requested': self.run_config.max_combinations,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            
            # Save best models for this placement
            models_dir = os.path.join(self.base_results_dir, "best_placement_models", split_name)
            os.makedirs(models_dir, exist_ok=True)
            
            for test_placement, model_info in placement_best_models.items():
                if model_info.get('best_model') is not None:
                    model_filename = f"best_{train_placement}_to_{test_placement}.joblib"
                    model_filepath = os.path.join(models_dir, model_filename)
                    joblib.dump(model_info['best_model'], model_filepath)
            
            # Mark placement as complete
            placement_complete_file = os.path.join(self.base_results_dir, f"placement_{train_placement}_{split_name}_complete.json")
            with open(placement_complete_file, 'w') as f:
                json.dump({
                    'train_placement': train_placement,
                    'split_name': split_name,
                    'completed_timestamp': datetime.now().isoformat(),
                    'placement_results': placement_results,
                    'placement_best_models': {
                        test_placement: {
                            'best_accuracy': model_info.get('best_accuracy', 0),
                            'best_combination': model_info.get('best_combination', [])
                        } for test_placement, model_info in placement_best_models.items()
                    }
                }, f, indent=2)
            
            print(f"      ✅ Placement {train_placement} results saved!")
            
        except Exception as e:
            print(f"❌ Error saving placement results for {train_placement}: {e}")
    
    def _save_split_completion_marker(self, split_name, split_results, split_best_models):
        """Mark entire split as fully completed"""
        try:
            split_complete_file = os.path.join(self.base_results_dir, f"split_{split_name}_fully_complete.json")
            with open(split_complete_file, 'w') as f:
                json.dump({
                    'split_name': split_name,
                    'completed_timestamp': datetime.now().isoformat(),
                    'total_placements': len(self.all_placements),
                    'total_avg_files': len(self.all_placements) * len(self.all_placements),  # 5x5 = 25
                    'granular_restart_enabled': True,
                    'ultra_efficient_completion': True
                }, f, indent=2)
            
            print(f"   ✅ Split {split_name} marked as fully complete!")
        except Exception as e:
            print(f"❌ Error marking split completion: {e}")
    
    def _is_split_completed(self, split_name):
        """Check if a split is already completed by looking for saved results"""
        split_results_file = os.path.join(self.base_results_dir, f"split_{split_name}_complete.json")
        return os.path.exists(split_results_file)
    
    def _load_completed_split_results(self, split_name):
        """Load results for a completed split"""
        split_results_file = os.path.join(self.base_results_dir, f"split_{split_name}_complete.json")
        
        if not os.path.exists(split_results_file):
            return {}, {}
        
        try:
            with open(split_results_file, 'r') as f:
                saved_data = json.load(f)
            
            return saved_data.get('split_results', {}), saved_data.get('split_best_models', {})
        except Exception as e:
            print(f"⚠️  Warning: Could not load split results for {split_name}: {e}")
            return {}, {}
    
    def _save_split_results_immediately(self, split_name, split_results, split_best_models):
        """Save results immediately after a split completes"""
        try:
            # Save individual placement combination avg files (25 per split)
            print(f"💾 Saving individual avg.json files for {split_name}...")
            for train_placement, train_results in split_results.items():
                for test_placement, test_stats in train_results.items():
                    # Save individual placement combination results
                    filename = f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json"
                    filepath = os.path.join(self.base_results_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump({
                            'summary_stats': test_stats,
                            'study_type': 'Game-1 Placement-Specific User Identity Study',
                            'semantic_info_available': ['user_identity'],
                            'placement_specific_training': True,
                            'cross_placement_testing': True,
                            'split_completed_immediately': True,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                    
                    print(f"   ✅ Saved: {filename}")
            
            # Save best models for this split
            print(f"💾 Saving best models for {split_name}...")
            models_dir = os.path.join(self.base_results_dir, "best_models")
            os.makedirs(models_dir, exist_ok=True)
            
            for train_placement, model_info in split_best_models.items():
                if model_info.get('best_model') is not None:
                    # Save actual model object using joblib
                    model_filename = f"best_model_{train_placement}_{split_name}.joblib"
                    model_filepath = os.path.join(models_dir, model_filename)
                    
                    joblib.dump(model_info['best_model'], model_filepath)
                    
                    # Also save model metadata
                    metadata_filename = f"best_model_{train_placement}_{split_name}_metadata.json"
                    metadata_filepath = os.path.join(models_dir, metadata_filename)
                    
                    model_metadata = {
                        'train_placement': train_placement,
                        'split_ratio': split_name,
                        'model_type': 'random_forest',
                        'best_accuracy': model_info.get('best_accuracy', 0),
                        'best_combination': model_info.get('best_combination', []),
                        'model_file': model_filename,
                        'timestamp': datetime.now().isoformat(),
                        'note': 'Random Forest model - best performing for this training placement',
                        'immediate_save': True,
                        'usage_instructions': {
                            'load_model': f"model_package = joblib.load('{model_filename}')",
                            'components': ['model (RandomForest)', 'scaler', 'label_encoder', 'feature_columns'],
                            'prediction_example': "predictions = model_package['model'].predict(scaled_features)"
                        }
                    }
                    
                    with open(metadata_filepath, 'w') as f:
                        json.dump(model_metadata, f, indent=2)
                    
                    print(f"   ✅ Saved model: {model_filename}")
            
            # Save split completion marker with all data
            split_completion_file = os.path.join(self.base_results_dir, f"split_{split_name}_complete.json")
            completion_data = {
                'split_name': split_name,
                'completed_timestamp': datetime.now().isoformat(),
                'split_results': split_results,
                'split_best_models': {
                    train_placement: {
                        'best_accuracy': model_info.get('best_accuracy', 0),
                        'best_combination': model_info.get('best_combination', []),
                        'model_file': f"best_model_{train_placement}_{split_name}.joblib" if model_info.get('best_model') else None
                    } for train_placement, model_info in split_best_models.items()
                },
                'files_saved': {
                    'avg_files': 25,  # 5x5 train-test combinations
                    'best_models': len([m for m in split_best_models.values() if m.get('best_model') is not None])
                }
            }
            
            with open(split_completion_file, 'w') as f:
                json.dump(completion_data, f, indent=2)
            
            print(f"✅ Split {split_name} completion marker saved!")
            
        except Exception as e:
            print(f"❌ Error saving split results for {split_name}: {e}")
    
    def _generate_final_overall_summary(self, all_results):
        """Generate final overall summary across all completed splits"""
        print(f"\n📊 GENERATING FINAL OVERALL SUMMARY...")
        
        # Analyze same-placement vs cross-placement performance across all splits
        same_placement_scores = []
        cross_placement_scores = []
        same_placement_nmi = []
        cross_placement_nmi = []
        same_placement_rkl = []
        cross_placement_rkl = []
        same_placement_vulnerability = []
        cross_placement_vulnerability = []
        
        for split_name, split_results in all_results.items():
            for train_placement, train_results in split_results.items():
                for test_placement, test_stats in train_results.items():
                    if train_placement == test_placement:
                        same_placement_scores.append(test_stats['accuracy_mean'])
                        same_placement_nmi.append(test_stats.get('nmi_mean', 0.0))
                        same_placement_rkl.append(test_stats.get('reverse_kl_mean', 0.0))
                        same_placement_vulnerability.append(test_stats.get('vulnerability_mean', 0.0))
                    else:
                        cross_placement_scores.append(test_stats['accuracy_mean'])
                        cross_placement_nmi.append(test_stats.get('nmi_mean', 0.0))
                        cross_placement_rkl.append(test_stats.get('reverse_kl_mean', 0.0))
                        cross_placement_vulnerability.append(test_stats.get('vulnerability_mean', 0.0))
        
        if same_placement_scores and cross_placement_scores:
            print(f"📈 CROSS-PLACEMENT GENERALIZATION SUMMARY:")
            print(f"   Same placement avg:  {np.mean(same_placement_scores):.3f} ± {np.std(same_placement_scores):.3f}")
            print(f"   Cross placement avg: {np.mean(cross_placement_scores):.3f} ± {np.std(cross_placement_scores):.3f}")
            print(f"   Performance drop:    {np.mean(same_placement_scores) - np.mean(cross_placement_scores):.3f}")
            
            print(f"📊 INFORMATION-THEORETIC METRICS SUMMARY:")
            print(f"   Same placement NMI:   {np.mean(same_placement_nmi):.1f}% ± {np.std(same_placement_nmi):.1f}%")
            print(f"   Cross placement NMI:  {np.mean(cross_placement_nmi):.1f}% ± {np.std(cross_placement_nmi):.1f}%")
            print(f"   Same placement RKL:   {np.mean(same_placement_rkl):.1f}% ± {np.std(same_placement_rkl):.1f}%")
            print(f"   Cross placement RKL:  {np.mean(cross_placement_rkl):.1f}% ± {np.std(cross_placement_rkl):.1f}%")
            print(f"   Same placement Vuln:  {np.mean(same_placement_vulnerability):.3f} ± {np.std(same_placement_vulnerability):.3f}")
            print(f"   Cross placement Vuln: {np.mean(cross_placement_vulnerability):.3f} ± {np.std(cross_placement_vulnerability):.3f}")
        
        # Save overall summary
        overall_summary = {
            'study_type': 'Game-1 Placement-Specific User Identity Study',
            'immediate_saving_enabled': True,
            'information_theoretic_metrics_included': True,
            'same_placement_performance': {
                'accuracy_mean': np.mean(same_placement_scores) if same_placement_scores else 0,
                'accuracy_std': np.std(same_placement_scores) if same_placement_scores else 0,
                'nmi_mean': np.mean(same_placement_nmi) if same_placement_nmi else 0,
                'nmi_std': np.std(same_placement_nmi) if same_placement_nmi else 0,
                'reverse_kl_mean': np.mean(same_placement_rkl) if same_placement_rkl else 0,
                'reverse_kl_std': np.std(same_placement_rkl) if same_placement_rkl else 0,
                'vulnerability_mean': np.mean(same_placement_vulnerability) if same_placement_vulnerability else 0,
                'vulnerability_std': np.std(same_placement_vulnerability) if same_placement_vulnerability else 0,
                'scores': same_placement_scores
            },
            'cross_placement_performance': {
                'accuracy_mean': np.mean(cross_placement_scores) if cross_placement_scores else 0,
                'accuracy_std': np.std(cross_placement_scores) if cross_placement_scores else 0,
                'nmi_mean': np.mean(cross_placement_nmi) if cross_placement_nmi else 0,
                'nmi_std': np.std(cross_placement_nmi) if cross_placement_nmi else 0,
                'reverse_kl_mean': np.mean(cross_placement_rkl) if cross_placement_rkl else 0,
                'reverse_kl_std': np.std(cross_placement_rkl) if cross_placement_rkl else 0,
                'vulnerability_mean': np.mean(cross_placement_vulnerability) if cross_placement_vulnerability else 0,
                'vulnerability_std': np.std(cross_placement_vulnerability) if cross_placement_vulnerability else 0,
                'scores': cross_placement_scores
            },
            'performance_drop': np.mean(same_placement_scores) - np.mean(cross_placement_scores) if same_placement_scores and cross_placement_scores else 0,
            'nmi_drop': np.mean(same_placement_nmi) - np.mean(cross_placement_nmi) if same_placement_nmi and cross_placement_nmi else 0,
            'reverse_kl_drop': np.mean(same_placement_rkl) - np.mean(cross_placement_rkl) if same_placement_rkl and cross_placement_rkl else 0,
            'vulnerability_drop': np.mean(same_placement_vulnerability) - np.mean(cross_placement_vulnerability) if same_placement_vulnerability and cross_placement_vulnerability else 0,
            'completed_splits': list(all_results.keys()),
            'total_avg_files': sum(len(split_results) * 5 for split_results in all_results.values()),
            'frequency_hz': self.run_config.frequency_hz,
            'max_combinations_requested': self.run_config.max_combinations,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.base_results_dir, "overall_summary_final.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"💾 Final overall summary saved to: {summary_file}")
    
    def _generate_placement_specific_summary(self, all_results, best_models):
        """Generate comprehensive summary and save all results"""
        print(f"\n{'='*100}")
        print("🎭 GAME-1 PLACEMENT-SPECIFIC USER IDENTITY STUDY - SUMMARY REPORT")
        print(f"{'='*100}")
        
        print("\n📊 CROSS-PLACEMENT GENERALIZATION WITH USER IDENTITY:")
        print("   Training on single placements | Testing on all placements")
        print("   User identity AVAILABLE | Placement-specific patterns")
        print()
        
        # Save individual result files (25 per split)
        for split_name, split_results in all_results.items():
            print(f"\n🔬 SPLIT RATIO: {split_name}")
            
            for train_placement, train_results in split_results.items():
                for test_placement, test_stats in train_results.items():
                    # Save individual placement combination results
                    filename = f"train_{train_placement}_test_{test_placement}_{split_name}_avg.json"
                    filepath = os.path.join(self.base_results_dir, filename)
                    
                    with open(filepath, 'w') as f:
                        json.dump({
                            'summary_stats': test_stats,
                            'study_type': 'Game-1 Placement-Specific User Identity Study',
                            'semantic_info_available': ['user_identity'],
                            'placement_specific_training': True,
                            'cross_placement_testing': True,
                            'timestamp': datetime.now().isoformat()
                        }, f, indent=2)
                    
                    # Print summary
                    print(f"   {train_placement:>12} → {test_placement:>12}: "
                          f"{test_stats['accuracy_mean']:.3f} ± {test_stats['accuracy_std']:.3f}")
        
        # Save best models (5 per split) - keep for backward compatibility
        models_dir = os.path.join(self.base_results_dir, "best_models")
        os.makedirs(models_dir, exist_ok=True)
        
        for split_name, split_models in best_models.items():
            print(f"\n💾 Saving best models for {split_name}:")
            
            for train_placement, model_info in split_models.items():
                if model_info['model'] is not None:
                    # Save actual model object using joblib
                    model_filename = f"best_model_{train_placement}_{split_name}.joblib"
                    model_filepath = os.path.join(models_dir, model_filename)
                    
                    # Save the complete model package (model + preprocessing)
                    joblib.dump(model_info['model'], model_filepath)
                    
                    # Also save model metadata for easy reference
                    metadata_filename = f"best_model_{train_placement}_{split_name}_metadata.json"
                    metadata_filepath = os.path.join(models_dir, metadata_filename)
                    
                    model_metadata = {
                        'train_placement': train_placement,
                        'split_ratio': split_name,
                        'model_type': 'random_forest',
                        'best_accuracy': model_info['best_accuracy'],
                        'best_combination': model_info['combination'],
                        'train_accuracy': model_info['model']['train_accuracy'],
                        'feature_columns': model_info['model']['feature_columns'],
                        'test_results': model_info.get('results', {}),
                        'model_file': model_filename,
                        'timestamp': datetime.now().isoformat(),
                        'note': 'Random Forest model - best performing for this training placement',
                        'all_models_location': f'all_models/{split_name}/',
                        'usage_instructions': {
                            'load_model': f"model_package = joblib.load('{model_filename}')",
                            'components': ['model (RandomForest)', 'scaler', 'label_encoder', 'feature_columns'],
                            'prediction_example': "predictions = model_package['model'].predict(scaled_features)"
                        }
                    }
                    
                    with open(metadata_filepath, 'w') as f:
                        json.dump(model_metadata, f, indent=2)
                    
                    print(f"   ✅ {train_placement}: {model_info['best_accuracy']:.3f} avg accuracy")
                    print(f"      📁 Best model: {model_filename}")
                    print(f"      📄 Metadata: {metadata_filename}")
        
        # Report on best placement models (25 per split)
        print(f"\n📊 BEST PLACEMENT MODEL SUMMARY:")
        total_best_models = 0
        for split_name in self.split_ratios.keys():
            best_models_dir = os.path.join(self.base_results_dir, "best_placement_models", split_name)
            if os.path.exists(best_models_dir):
                model_files = [f for f in os.listdir(best_models_dir) if f.endswith('.joblib')]
                total_best_models += len(model_files)
                print(f"   {split_name}: {len(model_files)} best models (target: 25)")
                
                # Show some examples
                if model_files:
                    print(f"      Examples:")
                    for i, model_file in enumerate(sorted(model_files)[:3]):
                        model_name = model_file.replace('.joblib', '').replace('best_', '')
                        print(f"        • {model_name}")
                    if len(model_files) > 3:
                        print(f"        • ... and {len(model_files) - 3} more")
        
        print(f"   TOTAL: {total_best_models} best placement models")
        print(f"   TARGET: {len(self.all_placements) * len(self.all_placements) * len(self.split_ratios)} models")
        print(f"   STRUCTURE: 5 train placements × 5 test placements × 4 splits = 100 models")
        
        # Generate overall summary statistics
        self._generate_overall_summary(all_results)
        
        # Generate overall summary statistics
        self._generate_overall_summary(all_results)
        
        print(f"\n💾 Results saved to: {self.base_results_dir}")
        print(f"📊 Total files: {len(self.all_placements) * len(self.all_placements) * len(self.split_ratios)} average result files")
        print(f"🤖 Total models: {len(self.all_placements) * len(self.split_ratios)} best models saved")
    
    def _generate_overall_summary(self, all_results):
        """Generate overall summary statistics across all placements and splits"""
        print(f"\n📈 OVERALL CROSS-PLACEMENT GENERALIZATION SUMMARY:")
        
        # Analyze same-placement vs cross-placement performance
        same_placement_scores = []
        cross_placement_scores = []
        
        for split_name, split_results in all_results.items():
            for train_placement, train_results in split_results.items():
                for test_placement, test_stats in train_results.items():
                    if train_placement == test_placement:
                        same_placement_scores.append(test_stats['accuracy_mean'])
                    else:
                        cross_placement_scores.append(test_stats['accuracy_mean'])
        
        if same_placement_scores and cross_placement_scores:
            print(f"   Same placement avg:  {np.mean(same_placement_scores):.3f} ± {np.std(same_placement_scores):.3f}")
            print(f"   Cross placement avg: {np.mean(cross_placement_scores):.3f} ± {np.std(cross_placement_scores):.3f}")
            print(f"   Performance drop:    {np.mean(same_placement_scores) - np.mean(cross_placement_scores):.3f}")
        
        # Save overall summary
        overall_summary = {
            'study_type': 'Game-1 Placement-Specific User Identity Study',
            'same_placement_performance': {
                'mean': np.mean(same_placement_scores) if same_placement_scores else 0,
                'std': np.std(same_placement_scores) if same_placement_scores else 0,
                'scores': same_placement_scores
            },
            'cross_placement_performance': {
                'mean': np.mean(cross_placement_scores) if cross_placement_scores else 0,
                'std': np.std(cross_placement_scores) if cross_placement_scores else 0,
                'scores': cross_placement_scores
            },
            'performance_drop': np.mean(same_placement_scores) - np.mean(cross_placement_scores) if same_placement_scores and cross_placement_scores else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.base_results_dir, "overall_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
    
    def load_saved_model(self, train_placement, split_ratio):
        """
        Utility function to load a saved model and demonstrate usage
        
        Args:
            train_placement (str): Training placement (e.g., 'left-wrist')
            split_ratio (str): Split ratio (e.g., '8_3')
            
        Returns:
            dict: Model package with all preprocessing components
        """
        models_dir = os.path.join(self.base_results_dir, "best_models")
        model_filename = f"best_model_{train_placement}_{split_ratio}.joblib"
        model_filepath = os.path.join(models_dir, model_filename)
        
        if not os.path.exists(model_filepath):
            print(f"❌ Model not found: {model_filepath}")
            return None
        
        try:
            # Load the complete model package
            model_package = joblib.load(model_filepath)
            
            print(f"✅ Loaded model trained on: {train_placement} (split: {split_ratio})")
            print(f"📊 Components: {list(model_package.keys())}")
            print(f"🎯 Training accuracy: {model_package['train_accuracy']:.3f}")
            print(f"📈 Features: {len(model_package['feature_columns'])}")
            
            return model_package
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def predict_with_saved_model(self, model_package, test_user_id, test_placement):
        """
        Demonstrate how to use a saved model for prediction
        
        Args:
            model_package (dict): Loaded model package
            test_user_id (int): User ID for testing
            test_placement (str): Placement for testing
            
        Returns:
            dict: Predictions and probabilities
        """
        if model_package is None:
            return None
        
        try:
            # Load test data
            user_data = self.load_user_data_single_placement(test_user_id, test_placement)
            
            if not user_data:
                print(f"❌ No test data found for User {test_user_id}, {test_placement}")
                return None
            
            # Extract features for all activities
            test_features = []
            test_activities = []
            
            for activity, df in user_data.items():
                if len(df) > 0:
                    features = self.extract_features(df, test_user_id)
                    test_features.append(features)
                    test_activities.append(activity)
            
            if not test_features:
                return None
            
            # Convert to DataFrame and align with model features
            test_features_df = pd.DataFrame(test_features)
            
            # Align columns with training data
            feature_columns = model_package['feature_columns']
            for col in feature_columns:
                if col not in test_features_df.columns:
                    test_features_df[col] = 0
            
            test_features_df = test_features_df[feature_columns]
            test_features_df = test_features_df.fillna(0)
            
            # Scale features using saved scaler
            X_test_scaled = model_package['scaler'].transform(test_features_df)
            
            # Make predictions
            predictions = model_package['model'].predict(X_test_scaled)
            probabilities = model_package['model'].predict_proba(X_test_scaled)
            
            # Decode predictions
            predicted_activities = model_package['label_encoder'].inverse_transform(predictions)
            
            results = {
                'user_id': test_user_id,
                'test_placement': test_placement,
                'train_placement': model_package['train_placement'],
                'true_activities': test_activities,
                'predicted_activities': predicted_activities.tolist(),
                'prediction_probabilities': probabilities.tolist(),
                'activity_labels': model_package['label_encoder'].classes_.tolist()
            }
            
            print(f"🎯 Predictions for User {test_user_id} on {test_placement}:")
            for true_act, pred_act in zip(test_activities, predicted_activities):
                print(f"   {true_act:>10} → {pred_act:>10}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            return None
    
    def _generate_summary_report(self, results):
        """Generate a summary report of user identity study results"""
        print(f"\n{'='*100}")
        print("🎭 GAME-1 USER IDENTITY STUDY - SUMMARY REPORT")
        print(f"{'='*100}")
        
        print("\n📊 ADVERSARIAL HAR EFFECTIVENESS WITH USER IDENTITY:")
        print("   Placement information ANONYMIZED | User identity AVAILABLE")
        print()
        
        for split_name, stats in results.items():
            if stats:
                print(f"🎯 SPLIT RATIO {split_name}:")
                print(f"   📊 Accuracy:  {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
                print(f"   📊 F1-Score:  {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
                print(f"   📊 Precision: {stats['precision_mean']:.3f}")
                print(f"   📊 Recall:    {stats['recall_mean']:.3f}")
                print(f"   📊 Features:  {stats['avg_features']:.0f}")
                print()
        
        print("🔍 KEY FINDINGS:")
        print("   • Placement information has been ANONYMIZED")
        print("   • User identity is available as semantic information")
        print("   • Results show adversarial effectiveness with user-specific modeling")
        print("   • Cross-sensor coordination features have been omitted")
        print("   • No activity labels used during feature extraction (no cheating)")
        
        # Save comprehensive report
        report_file = os.path.join(self.base_results_dir, "user_identity_study_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'study_type': 'Game-1 User Identity Study',
                'semantic_info_removed': ['sensor_placement_identity', 'cross_sensor_coordination'],
                'semantic_info_available': ['user_identity'],
                'placement_anonymous': True,
                'activity_labels_in_features': False,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Complete report saved to: {report_file}")


if __name__ == "__main__":
    analyzer = Game1UserIdentityOnlyAnalyzer()
    results, best_models = analyzer.run_user_identity_study()
