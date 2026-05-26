#!/usr/bin/env python3
"""
Game-1 User Identity Only: Placement-Anonymous Binary-Only Adversary
===================================================================

This script implements Game-1 adversarial HAR with user identity semantic information
but WITHOUT placement information. The adversary has access to:

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Binary decision tree outputs from unknown placements
- Temporal patterns and activity signatures

❌ REMOVED SEMANTIC INFORMATION:
- Sensor placement information (anonymized)
- Placement-specific feature engineering
- Location-based semantic understanding
- Cross-sensor coordination features (omitted)

RESEARCH QUESTION:
How effective are adversarial HAR attacks when placement information 
is anonymized but user identity is available?

METHODOLOGY:
- Pool all binary signals from all placements without placement labels
- Extract placement-agnostic temporal features
- Use user identity for proper train/test splits AND as model features
- Evaluate adversarial effectiveness with user semantic information
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from itertools import combinations
import json
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

print("🎭 GAME-1 USER IDENTITY ONLY: PLACEMENT-ANONYMOUS ADVERSARY")
print("=" * 65)
print("Testing adversarial effectiveness WITH user identity semantic information")
print("Placement information ANONYMIZED | User identity AVAILABLE")
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

class Game1UserIdentityOnlyAnalyzer:
    """
    Analyzer that removes placement information while using user identity
    for adversarial HAR attacks with semantic information
    """
    
    def __init__(self, config):
        self.config = config
        self.experiment_name = "game-1-user_identity_only"
        self.game_type = "Game-1"
        self.study_type = "Game-1 User Identity Study"
        self.data_available = ["binary_decision_tree"]
        self.semantic_info_removed = ["sensor_placement_identity", "cross_sensor_coordination"]
        self.rf_description = "RandomForestClassifier"
        self.split_results_filename = f"{self.config.split_name}_with_user_identity.json"
        self.report_filename = "user_identity_study_report.json"
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        self._user_feature_row_cache = {}
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        
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
        
        print(f"🎯 Target: Adversarial HAR with user identity semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (ANONYMIZED)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
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
            train_test_pairs.append((list(train_users), list(test_users)))
        
        print(f"Split {split_name}: Generated {len(train_test_pairs)} combinations")
        return train_test_pairs
    
    def load_user_data_placement_anonymous(self, user_id):
        """
        Load binary data from ALL placements but anonymize placement information
        This simulates an adversary who has access to binary signals but doesn't 
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
                                df = load_experiment_csv(file_path, self.config.frequency_hz)
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
    
    def build_rf_model(self, random_state):
        """Build the branch baseline model, stored under the RF slot."""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
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

            features = self.extract_placement_anonymous_features(df, user_id)
            if not features:
                continue

            feature_rows.append(features)
            labels.append(activity)
            metadata.append({
                'user_id': user_id,
                'activity': activity,
                'n_sensors': features.get('n_anonymous_sensors', 0),
                'total_samples': features.get('global_total_samples', 0)
            })
            user_samples += 1

        print(f" ✅ {user_samples} activities")
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
            
            # Calculate traditional metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # Calculate information-theoretic metrics (class-agnostic)
            nmi_percentage = calculate_normalized_mutual_information(y_test_encoded, y_pred)
            reverse_kl_percentage = calculate_normalized_kl_divergence(y_test_encoded, y_pred_proba)
            
            return {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'nmi_percentage': nmi_percentage,
                'reverse_kl_percentage': reverse_kl_percentage,
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
        
        # Generate and sample combinations
        all_combinations = self.generate_all_combinations(split_name)
        combinations_list = sample_experiment_combinations(
            all_combinations,
            max_combinations=self.config.max_combinations,
            seed=self.config.random_seed,
        )
        print(
            f"📊 Testing {len(combinations_list)} of {len(all_combinations)} available combinations "
            f"at {self.config.frequency_hz} Hz"
        )
        
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
        nmi_percentages = [r['nmi_percentage'] for r in results]
        reverse_kl_percentages = [r['reverse_kl_percentage'] for r in results]
        
        stats = {
            'split_ratio': split_name,
            'has_user_identity': True,
            'placement_anonymous': True,
            'frequency_hz': self.config.frequency_hz,
            'source_frequency_hz': SOURCE_FREQUENCY_HZ,
            'total_combinations_available': len(all_combinations),
            'max_combinations_requested': self.config.max_combinations,
            'n_combinations': len(results),
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
            'avg_train_samples': np.mean([r['n_train_samples'] for r in results]),
            'avg_test_samples': np.mean([r['n_test_samples'] for r in results]),
            'avg_features': np.mean([r['n_features'] for r in results]),
            'data_root': str(self.config.data_root),
            'results_dir': self.base_results_dir,
        }
        
        # Print summary
        print(f"\n📊 PLACEMENT-ANONYMOUS GAME-1 RESULTS (WITH user identity):")
        print(f"   Accuracy:     {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f} (min: {stats['accuracy_min']:.3f}, max: {stats['accuracy_max']:.3f})")
        print(f"   F1-Score:     {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f} (min: {stats['f1_min']:.3f}, max: {stats['f1_max']:.3f})")
        print(f"   Precision:    {stats['precision_mean']:.3f}")
        print(f"   Recall:       {stats['recall_mean']:.3f}")
        print(f"   📈 NMI (%):      {stats['nmi_mean']:.1f} ± {stats['nmi_std']:.1f} (min: {stats['nmi_min']:.1f}, max: {stats['nmi_max']:.1f})")
        print(f"   📈 Reverse KL (%): {stats['reverse_kl_mean']:.1f} ± {stats['reverse_kl_std']:.1f} (min: {stats['reverse_kl_min']:.1f}, max: {stats['reverse_kl_max']:.1f})")
        print(f"   Features:     {stats['avg_features']:.0f} (placement-anonymous + user identity)")
        print(f"   💡 NMI & Reverse KL are class-agnostic and comparable across datasets")
        
        # Save results
        results_file = os.path.join(self.base_results_dir, f"{split_name}_with_user_identity.json")
        
        save_data = {
            'summary_stats': stats,
            'individual_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return stats
    
    def run_user_identity_study(self):
        """
        Run complete user identity study with placement-anonymous adversarial HAR
        """
        return run_identity_experiment_suite(self)
    
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
                print(f"   📊 Accuracy:     {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
                print(f"   📊 F1-Score:     {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
                print(f"   📊 Precision:    {stats['precision_mean']:.3f}")
                print(f"   📊 Recall:       {stats['recall_mean']:.3f}")
                print(f"   � NMI (%):      {stats['nmi_mean']:.1f} ± {stats['nmi_std']:.1f}")
                print(f"   📈 Reverse KL (%): {stats['reverse_kl_mean']:.1f} ± {stats['reverse_kl_std']:.1f}")
                print(f"   �📊 Features:     {stats['avg_features']:.0f}")
                print()
        
        print("🔍 KEY FINDINGS:")
        print("   • Placement information has been ANONYMIZED")
        print("   • User identity is available as semantic information")
        print("   • Results show adversarial effectiveness with user-specific modeling")
        print("   • Cross-sensor coordination features have been omitted")
        print("   • No activity labels used during feature extraction (no cheating)")
        print("   📈 NMI & Reverse KL metrics are class-agnostic and comparable across datasets")
        print("   📈 Higher NMI = better information preservation (0-100%)")
        print("   📈 Higher Reverse KL = better model calibration (0-100%)")
        
        # Save comprehensive report
        report_file = os.path.join(self.base_results_dir, "user_identity_study_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'study_type': 'Game-1 User Identity Study',
                'semantic_info_removed': ['sensor_placement_identity', 'cross_sensor_coordination'],
                'semantic_info_available': ['user_identity'],
                'placement_anonymous': True,
                'activity_labels_in_features': False,
                'frequency_hz': self.config.frequency_hz,
                'source_frequency_hz': SOURCE_FREQUENCY_HZ,
                'max_combinations_requested': self.config.max_combinations,
                'data_root': str(self.config.data_root),
                'results_dir': self.base_results_dir,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Complete report saved to: {report_file}")


if __name__ == "__main__":
    config = resolve_identity_experiment_config("Run the Game-1 user identity only experiment.")
    analyzer = Game1UserIdentityOnlyAnalyzer(config)
    results = analyzer.run_user_identity_study()
