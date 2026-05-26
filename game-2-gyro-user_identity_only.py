#!/usr/bin/env python3
"""
Game-2-Gyro User Identity Only Analysis
=======================================

This script implements Game-2 gyroscope attacks using ONLY user identity semantic information.
This is a semantic ablation study that tests adversarial effectiveness when:
- Placement information is ANONYMIZED (no body location knowledge)
- User identity is AVAILABLE as semantic information  
- Only gyroscope + binary data is used (Game-2 constraint)

The goal is to test if user-specific gyroscope patterns can be exploited for activity inference
without any placement-specific semantic advantages.

Research Question:
Can adversaries leverage user identity to improve gyroscope-based activity inference 
when placement information is completely anonymized?

Semantic Information:
✅ User identity (one-hot encoded user features)  
❌ Sensor placement information (anonymized as sensor_0, sensor_1, etc.)
❌ Cross-sensor coordination features
❌ Placement-specific feature engineering
"""

import os
import pandas as pd
import numpy as np
import glob
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from project_paths import (
    SOURCE_FREQUENCY_HZ,
    build_frequency_results_dir,
    load_experiment_csv,
    resolve_identity_experiment_config,
    sample_experiment_combinations,
    user_processed_dir,
)
from identity_experiment_runner import build_dataset_frame_from_rows, run_identity_experiment_suite

print("🎯 GAME-2-GYRO USER IDENTITY ONLY ANALYSIS")
print("=" * 48)
print("Semantic ablation study with placement anonymization")
print("Testing user identity semantic information effectiveness")
print("Gyroscope data available | Placement information ANONYMIZED")
print()

def calculate_entropy(labels):
    """Calculate entropy of a label distribution"""
    if len(labels) == 0:
        return 0.0
    
    # Get probability distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Calculate entropy using natural logarithm
    entropy_val = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return entropy_val

def calculate_mutual_information(y_true, y_pred):
    """Calculate mutual information between true and predicted labels"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Create contingency table
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # Joint probability distribution
    contingency = np.zeros((len(unique_true), len(unique_pred)))
    
    for i, true_val in enumerate(unique_true):
        for j, pred_val in enumerate(unique_pred):
            contingency[i, j] = np.sum((y_true == true_val) & (y_pred == pred_val))
    
    # Convert to probabilities
    joint_prob = contingency / len(y_true)
    
    # Marginal probabilities
    prob_true = np.sum(joint_prob, axis=1)
    prob_pred = np.sum(joint_prob, axis=0)
    
    # Calculate mutual information
    mi = 0.0
    for i in range(len(unique_true)):
        for j in range(len(unique_pred)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (prob_true[i] * prob_pred[j] + 1e-10) + 1e-10)
    
    return max(0.0, mi)  # Ensure non-negative

def calculate_normalized_mutual_information(y_true, y_pred):
    """Calculate normalized mutual information (NMI) percentage - class-agnostic"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Calculate mutual information
    mi = calculate_mutual_information(y_true, y_pred)
    
    # Calculate entropies
    h_true = calculate_entropy(y_true)
    h_pred = calculate_entropy(y_pred)
    
    # Normalize using geometric mean
    if h_true > 0 and h_pred > 0:
        nmi = mi / np.sqrt(h_true * h_pred)
    else:
        nmi = 0.0
    
    # Convert to percentage and ensure it's bounded [0, 100]
    nmi_percentage = min(100.0, max(0.0, nmi * 100.0))
    
    return nmi_percentage

def calculate_kl_divergence_from_predictions(y_true, y_pred_proba):
    """Calculate average KL divergence from prediction probabilities"""
    if len(y_true) == 0 or y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0
    
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else len(np.unique(y_true))
    
    total_kl = 0.0
    
    for i in range(n_samples):
        # True distribution (one-hot)
        true_dist = np.zeros(n_classes)
        if y_true[i] < n_classes:
            true_dist[y_true[i]] = 1.0
        
        # Predicted distribution
        if len(y_pred_proba.shape) > 1:
            pred_dist = y_pred_proba[i]
        else:
            # If we only have predictions, create uniform distribution
            pred_dist = np.ones(n_classes) / n_classes
        
        # Add small epsilon to avoid log(0)
        pred_dist = pred_dist + 1e-10
        pred_dist = pred_dist / np.sum(pred_dist)  # Normalize
        
        # Calculate KL divergence for this sample
        kl = 0.0
        for j in range(n_classes):
            if true_dist[j] > 0:
                kl += true_dist[j] * np.log(true_dist[j] / pred_dist[j])
        
        total_kl += kl
    
    return total_kl / n_samples

def calculate_normalized_kl_divergence(y_true, y_pred_proba):
    """Calculate normalized KL divergence percentage - class-agnostic"""
    if len(y_true) == 0 or y_pred_proba is None:
        return 0.0
    
    # Calculate average KL divergence
    avg_kl = calculate_kl_divergence_from_predictions(y_true, y_pred_proba)
    
    # Calculate theoretical maximum KL (uniform prediction)
    n_classes = len(np.unique(y_true))
    kl_max = np.log(n_classes)  # Theoretical maximum
    
    # Normalize
    if kl_max > 0:
        kl_normalized = min(1.0, avg_kl / kl_max)
    else:
        kl_normalized = 0.0
    
    # Convert to reverse KL percentage (higher = better)
    reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
    
    return max(0.0, min(100.0, reverse_kl_percentage))

class Game2GyroUserIdentityOnlyAnalyzer:
    """
    Analyzer for Game-2 gyroscope attacks with user identity but
    without placement-specific or cross-sensor semantic information
    """
    
    def __init__(self, config):
        self.config = config
        self.experiment_name = "game-2-gyro-user_identity_only"
        self.game_type = "Game-2 Gyroscope"
        self.study_type = "Game-2 Gyroscope User Identity Study"
        self.data_available = ["gyroscope", "binary_decision_tree"]
        self.semantic_info_removed = [
            "sensor_placement_identity",
            "cross_sensor_coordination",
            "placement_specific_features",
        ]
        self.rf_description = "VotingClassifier"
        self.split_results_filename = f"{self.config.split_name}_detailed_results.json"
        self.report_filename = "game2_gyro_user_identity_study_report.json"
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        self._user_feature_row_cache = {}
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        
        # Game-2 constraint: Gyroscope + Binary only (NO accelerometer)
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        self.available_cols = self.gyro_cols + [self.binary_col]
        
        # Split ratios to test
        self.split_ratios = {
            '8_3': {'train_size': 8, 'test_size': 3},
        }
        
        # Results storage
        self.base_results_dir = str(
            build_frequency_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                self.experiment_name,
            )
        )
        
        print(f"🎯 Target: Game-2 gyroscope attacks with user identity semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (ANONYMIZED)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📈 Sensors: Gyroscope + Binary (Game-2 constraint)")
        print(f"⏱️  Frequency: {self.config.frequency_hz} Hz (source: {SOURCE_FREQUENCY_HZ} Hz)")
        print(f"🔢 Max combinations per split: {self.config.max_combinations}")
        print(f"📁 Data root: {self.config.data_root}")
        print(f"📁 Results dir: {self.base_results_dir}")
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
            train_test_pairs.append({
                'train_users': list(train_users),
                'test_users': list(test_users)
            })
        
        return train_test_pairs

    def load_user_data_placement_anonymous(self, user_id):
        """
        Load gyroscope + binary data from ALL placements but anonymize placement information
        This simulates an adversary who has access to gyroscope signals but doesn't 
        know which signal comes from which body location
        """
        user_dir = user_processed_dir(self.config.data_root, user_id)
        
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
                            df = load_experiment_csv(file_path, self.config.frequency_hz)
                            # Check if required columns exist
                            if any(col in df.columns for col in self.available_cols):
                                # Create anonymized placement identifier
                                df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep gyroscope + binary columns + metadata
                                keep_cols = ['anonymous_placement_id', 'user_id', 'activity']
                                for col in self.available_cols:
                                    if col in df.columns:
                                        keep_cols.append(col)
                                
                                df_clean = df[keep_cols].copy()
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
                                df = load_experiment_csv(file_path, self.config.frequency_hz)
                                # Check if required columns exist
                                if any(col in df.columns for col in self.available_cols):
                                    # Create anonymized placement identifier
                                    df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    # Only keep gyroscope + binary columns + metadata
                                    keep_cols = ['anonymous_placement_id', 'user_id', 'activity']
                                    for col in self.available_cols:
                                        if col in df.columns:
                                            keep_cols.append(col)
                                    
                                    df_clean = df[keep_cols].copy()
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
    
    def extract_placement_anonymous_gyroscope_features(self, df, user_id):
        """
        Extract gyroscope features WITHOUT using placement information
        Focus on global gyroscope patterns that don't rely on placement context
        
        CRITICAL: This function MUST NOT use activity labels for feature extraction
        to ensure no cheating during inference
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique anonymous sensors for this user's data
        unique_sensors = df['anonymous_placement_id'].unique()
        n_sensors = len(unique_sensors)
        
        # GLOBAL GYROSCOPE FEATURES (aggregated across all anonymous sensors)
        # IMPORTANT: Only using gyroscope + binary data, NOT activity labels
        
        # 1. GLOBAL GYROSCOPE STATISTICAL FEATURES
        for axis_col in self.gyro_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                
                # Basic statistics (placement-agnostic)
                features[f'global_{axis}_mean'] = np.mean(data)
                features[f'global_{axis}_std'] = np.std(data)
                features[f'global_{axis}_max'] = np.max(data)
                features[f'global_{axis}_min'] = np.min(data)
                features[f'global_{axis}_median'] = np.median(data)
                features[f'global_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_{axis}_energy'] = np.sum(data**2)
                
                # Advanced statistics
                if len(data) > 1:
                    features[f'global_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                    features[f'global_{axis}_rms'] = np.sqrt(np.mean(data**2))
                    
                    # Temporal features
                    diff_data = np.diff(data)
                    features[f'global_{axis}_diff_mean'] = np.mean(diff_data)
                    features[f'global_{axis}_diff_std'] = np.std(diff_data)
                    features[f'global_{axis}_diff_max'] = np.max(np.abs(diff_data))
                    
                    # Percentiles
                    features[f'global_{axis}_q25'] = np.percentile(data, 25)
                    features[f'global_{axis}_q75'] = np.percentile(data, 75)
                    features[f'global_{axis}_iqr'] = np.percentile(data, 75) - np.percentile(data, 25)
                    
                    # Zero crossings (rotational direction changes)
                    features[f'global_{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
        
        # 2. GLOBAL ANGULAR MAGNITUDE FEATURES (if all gyroscope axes available)
        if all(col in df.columns for col in self.gyro_cols):
            gyro_data = df[self.gyro_cols].values
            angular_magnitude = np.linalg.norm(gyro_data, axis=1)
            
            features['global_angular_magnitude_mean'] = np.mean(angular_magnitude)
            features['global_angular_magnitude_std'] = np.std(angular_magnitude)
            features['global_angular_magnitude_max'] = np.max(angular_magnitude)
            features['global_angular_magnitude_min'] = np.min(angular_magnitude)
            features['global_angular_magnitude_median'] = np.median(angular_magnitude)
            features['global_angular_magnitude_range'] = np.max(angular_magnitude) - np.min(angular_magnitude)
            features['global_angular_magnitude_energy'] = np.sum(angular_magnitude**2)
            
            if len(angular_magnitude) > 1:
                features['global_angular_magnitude_skew'] = pd.Series(angular_magnitude).skew()
                features['global_angular_magnitude_kurtosis'] = pd.Series(angular_magnitude).kurtosis()
                features['global_angular_magnitude_rms'] = np.sqrt(np.mean(angular_magnitude**2))
        
        # 3. GLOBAL BINARY FEATURES (from anonymous sensors)
        if self.binary_col in df.columns:
            binary_data = df[self.binary_col].values
            
            features['global_binary_activity_ratio'] = np.mean(binary_data)
            features['global_binary_total_samples'] = len(binary_data)
            features['global_binary_active_samples'] = np.sum(binary_data)
            
            if len(binary_data) > 1:
                transitions = np.sum(np.abs(np.diff(binary_data.astype(int))))
                features['global_binary_transitions'] = transitions
                features['global_binary_transition_rate'] = transitions / len(binary_data)
            else:
                features['global_binary_transitions'] = 0
                features['global_binary_transition_rate'] = 0
        
        # 4. USER-SPECIFIC PATTERN ENCODING (using user identity semantic info)
        features['user_id'] = user_id  # Use user identity as semantic information
        
        # 5. SENSOR COUNT METADATA (without placement identity)
        features['n_anonymous_sensors'] = n_sensors
        features['avg_sensor_length'] = np.mean([len(df[df['anonymous_placement_id'] == s]) for s in unique_sensors])
        
        # Clean NaN and inf values more robustly
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
    
    def build_rf_model(self, random_state):
        """Build the branch baseline model, stored under the RF slot."""
        rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
        lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        svm_model = SVC(random_state=random_state, probability=True, kernel='rbf')

        return VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lr', lr_model),
                ('svm', svm_model)
            ],
            voting='soft'
        )

    def _build_user_feature_rows(self, user_id):
        """Load and cache feature rows for a single user."""
        if user_id in self._user_feature_row_cache:
            return self._user_feature_row_cache[user_id]

        print(f"📊 Processing User {user_id} (placement-anonymous)...", end="")
        user_data = self.load_user_data_placement_anonymous(user_id)

        if not user_data:
            print(" ❌ No data")
            cached_rows = ([], [], [])
            self._user_feature_row_cache[user_id] = cached_rows
            return cached_rows

        feature_rows = []
        labels = []
        metadata = []
        user_samples = 0

        for activity in self.activities:
            if activity not in user_data:
                continue

            df = user_data[activity]
            if len(df) == 0:
                continue

            features = self.extract_placement_anonymous_gyroscope_features(df, user_id)
            if not features:
                continue

            feature_rows.append(features)
            labels.append(activity)
            metadata.append({
                'user_id': user_id,
                'activity': activity,
                'n_sensors': features.get('n_anonymous_sensors', 0),
                'total_samples': features.get('global_binary_total_samples', 0)
            })
            user_samples += 1

        print(f" ✅ {user_samples} samples")
        cached_rows = (feature_rows, labels, metadata)
        self._user_feature_row_cache[user_id] = cached_rows
        return cached_rows

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
            user_rows, user_labels, user_metadata = self._build_user_feature_rows(user_id)
            X.extend(user_rows)
            y.extend(user_labels)
            metadata.extend(user_metadata)
        
        if not X:
            print("❌ No data could be loaded!")
            return None, None, None

        X_df, y_array, metadata_rows = build_dataset_frame_from_rows(X, y, metadata, self.all_users)
        
        print(f"📈 Dataset created: {len(X_df)} samples, {len(X_df.columns)} features")
        print(f"🔑 User identity encoded as {len([c for c in X_df.columns if c.startswith('user_')])} user features")
        return X_df, y_array, metadata_rows
    
    def evaluate_single_combination(self, train_users, test_users):
        """Evaluate a single train/test combination with user identity semantic information"""
        try:
            # Create datasets
            print(f"\n🔄 Training Users: {train_users}")
            print(f"🧪 Testing Users: {test_users}")
            
            X_train, y_train, train_metadata = self.create_dataset(train_users)
            X_test, y_test, test_metadata = self.create_dataset(test_users)
            
            if X_train is None or X_test is None:
                return None
            
            # Feature alignment between train and test
            train_df = pd.DataFrame(X_train)
            test_df = pd.DataFrame(X_test)
            
            # Get common features
            common_cols = list(set(train_df.columns) & set(test_df.columns))
            common_cols.sort()  # Ensure consistent ordering
            
            X_train_aligned = train_df[common_cols].values
            X_test_aligned = test_df[common_cols].values
            
            print(f"📊 Training: {len(X_train_aligned)} samples, {len(common_cols)} features")
            print(f"🧪 Testing: {len(X_test_aligned)} samples, {len(common_cols)} features")
            
            # Handle labels
            label_encoder = LabelEncoder()
            
            # Fit on training labels only
            y_train_encoded = label_encoder.fit_transform(y_train)
            
            # Check if test labels exist in training
            test_classes = set(y_test)
            train_classes = set(y_train)
            unseen_classes = test_classes - train_classes
            
            if unseen_classes:
                print(f"⚠️  Unseen classes in test: {unseen_classes}")
                return None
            
            y_test_encoded = label_encoder.transform(y_test)
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aligned)
            X_test_scaled = scaler.transform(X_test_aligned)
            
            # Create ensemble model (same as original Game-2-Gyro)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            svm_model = SVC(random_state=42, probability=True, kernel='rbf')
            
            ensemble_model = VotingClassifier(
                estimators=[
                    ('rf', rf_model),
                    ('lr', lr_model),
                    ('svm', svm_model)
                ],
                voting='soft'
            )
            
            # Train and predict
            ensemble_model.fit(X_train_scaled, y_train_encoded)
            y_pred = ensemble_model.predict(X_test_scaled)
            y_pred_proba = ensemble_model.predict_proba(X_test_scaled)
            
            # Calculate traditional metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # Calculate information-theoretic metrics (class-agnostic)
            nmi_percentage = calculate_normalized_mutual_information(y_test_encoded, y_pred)
            reverse_kl_percentage = calculate_normalized_kl_divergence(y_test_encoded, y_pred_proba)
            
            return {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'nmi_percentage': nmi_percentage,
                'reverse_kl_percentage': reverse_kl_percentage,
                'n_train_samples': len(X_train_aligned),
                'n_test_samples': len(X_test_aligned),
                'n_features': len(common_cols),
                'train_classes': sorted(train_classes),
                'test_classes': sorted(test_classes)
            }
            
        except Exception as e:
            print(f"❌ Error in combination: {e}")
            return None
    
    def analyze_split_ratio(self, split_name):
        """Analyze all combinations for a specific split ratio"""
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING SPLIT RATIO: {split_name}")
        print(f"{'='*60}")
        
        all_combinations = self.generate_all_combinations(split_name)
        combinations = sample_experiment_combinations(
            all_combinations,
            max_combinations=self.config.max_combinations,
            seed=self.config.random_seed,
        )
        print(
            f"📊 Testing {len(combinations)} of {len(all_combinations)} available combinations "
            f"at {self.config.frequency_hz} Hz"
        )
        
        results = []
        successful_combinations = 0
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n--- Combination {i}/{len(combinations)} ---")
            result = self.evaluate_single_combination(combo['train_users'], combo['test_users'])
            
            if result is not None:
                results.append(result)
                successful_combinations += 1
                print(f"✅ Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f}")
            else:
                print("❌ Failed")
        
        if results:
            # Calculate statistics
            accuracies = [r['accuracy'] for r in results]
            f1_scores = [r['f1_score'] for r in results]
            precisions = [r['precision'] for r in results]
            recalls = [r['recall'] for r in results]
            nmi_percentages = [r['nmi_percentage'] for r in results]
            reverse_kl_percentages = [r['reverse_kl_percentage'] for r in results]
            
            summary = {
                'split_ratio': split_name,
                'frequency_hz': self.config.frequency_hz,
                'source_frequency_hz': SOURCE_FREQUENCY_HZ,
                'total_combinations_available': len(all_combinations),
                'total_combinations': len(combinations),
                'max_combinations_requested': self.config.max_combinations,
                'successful_combinations': successful_combinations,
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_min': np.min(accuracies),
                'accuracy_max': np.max(accuracies),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'f1_min': np.min(f1_scores),
                'f1_max': np.max(f1_scores),
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
                'data_root': str(self.config.data_root),
                'results_dir': self.base_results_dir,
                'detailed_results': results
            }
            
            print(f"\n📈 SUMMARY FOR {split_name}:")
            print(f"   Successful combinations: {successful_combinations}/{len(combinations)}")
            print(f"   Accuracy:     {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
            print(f"   Range:        [{summary['accuracy_min']:.4f}, {summary['accuracy_max']:.4f}]")
            print(f"   F1-Score:     {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
            print(f"   📈 NMI (%):      {summary['nmi_mean']:.1f} ± {summary['nmi_std']:.1f} (min: {summary['nmi_min']:.1f}, max: {summary['nmi_max']:.1f})")
            print(f"   📈 Reverse KL (%): {summary['reverse_kl_mean']:.1f} ± {summary['reverse_kl_std']:.1f} (min: {summary['reverse_kl_min']:.1f}, max: {summary['reverse_kl_max']:.1f})")
            print(f"   💡 NMI & Reverse KL are class-agnostic and comparable across datasets")
            
            # Save results
            results_file = os.path.join(self.base_results_dir, f"{split_name}_detailed_results.json")
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"💾 Results saved to: {results_file}")
            
            return summary
        else:
            print("❌ No successful combinations!")
            return None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across all split ratios"""
        return run_identity_experiment_suite(self)

if __name__ == "__main__":
    config = resolve_identity_experiment_config("Run the Game-2 gyroscope user identity only experiment.")
    analyzer = Game2GyroUserIdentityOnlyAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    print("\n🎉 Game-2-Gyro User Identity Analysis Complete!")
