#!/usr/bin/env python3
"""
Game-1 Cross-Sensor Cross-Dataset HAR: File-Wise Implementation
==============================================================

This script implements cross-sensor cross-dataset HAR using file-wise processing 
and robust temporal features. It trains on one STM-UC sensor placement and tests 
on entire UCI_HAR dataset, and vice versa.

Key Improvements:
1. File-wise processing for complete activity sessions
2. Robust temporal feature engineering (29 features)
3. Cross-sensor cross-dataset evaluation
4. Advanced binary pattern analysis
5. Comprehensive placement vs smartphone comparison

EXPERIMENTAL DESIGN:
Case 1: Train on each STM-UC placement → Test on entire UCI_HAR (smartphone)
Case 2: Train on entire UCI_HAR (smartphone) → Test on each STM-UC placement

Total experiments: 5 placements × 2 directions = 10 experiments

Placements: left-ankle, left-wrist, right-ankle, right-pocket, right-wrist
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib
import itertools
from cross_sensor_path_utils import get_stm_uc_root, get_uci_root
import warnings
warnings.filterwarnings('ignore')

print("🎭 GAME-1 CROSS-SENSOR CROSS-DATASET HAR: FILE-WISE IMPLEMENTATION")
print("=" * 70)
print("🔧 Key Improvements:")
print("   ✅ File-wise processing for proper activity sessions")
print("   ✅ Robust temporal feature engineering (29 features)")
print("   ✅ Cross-sensor cross-dataset evaluation")
print("   ✅ STM-UC placements vs UCI_HAR smartphone")
print("   ✅ Advanced binary pattern analysis")
print()

class FileWiseCrossSensorCrossDatasetAnalyzer:
    """
    Cross-sensor cross-dataset analyzer using file-wise processing
    Train on STM-UC placement → Test on UCI_HAR (and vice versa)
    """
    
    def __init__(self, window_size=100):
        # Activities available in both datasets
        self.activities = ['Downstairs', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # STM sensor placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        
        # Window size for temporal analysis
        self.window_size = window_size
        
        # Results directory
        self.results_dir = "results/game-1-cross-sensor-cross-dataset-file-wise"
        self.stm_root = str(get_stm_uc_root())
        self.uci_root = str(get_uci_root())
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🎯 Cross-sensor cross-dataset evaluation")
        print(f"📍 STM-UC placements: {', '.join(self.stm_placements)}")
        print(f"📱 UCI_HAR: smartphone placement")
        print(f"🪟 Window size: {self.window_size} samples")
        print(f"📁 Results: {self.results_dir}")
        print()
    
    def load_stm_placement_files(self, placement, max_users=11):
        """
        Load STM-UC data file-wise for a specific placement
        """
        print(f"📊 Loading STM-UC files for placement: {placement}")
        all_files = []
        
        for user_id in range(1, max_users + 1):
            user_dir = os.path.join(self.stm_root, f"User {user_id}", "Processed")
            
            if not os.path.exists(user_dir):
                continue
            
            user_files = 0
            for activity in self.activities:
                activity_dir = os.path.join(user_dir, activity)
                
                if not os.path.exists(activity_dir):
                    continue
                
                if activity in ['Standing', 'Sitting', 'Walking', 'Laying']:
                    # Single file structure
                    file_path = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if self.binary_col in df.columns and len(df) > 0:
                                file_info = {
                                    'user_id': user_id,
                                    'activity': activity,
                                    'placement': placement,
                                    'dataset': 'STM-UC',
                                    'file_path': file_path,
                                    'binary_data': df[self.binary_col].values,
                                    'file_size': len(df)
                                }
                                all_files.append(file_info)
                                user_files += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {file_path}: {e}")
                
                elif activity in ['Upstairs', 'Downstairs']:
                    # Multiple files structure for stairs
                    placement_dir = os.path.join(activity_dir, placement)
                    
                    if os.path.exists(placement_dir):
                        csv_files = [f for f in os.listdir(placement_dir) if f.endswith('.csv')]
                        
                        for csv_file in csv_files:
                            file_path = os.path.join(placement_dir, csv_file)
                            try:
                                df = pd.read_csv(file_path)
                                if self.binary_col in df.columns and len(df) > 0:
                                    file_info = {
                                        'user_id': user_id,
                                        'activity': activity,
                                        'placement': placement,
                                        'dataset': 'STM-UC',
                                        'file_path': file_path,
                                        'binary_data': df[self.binary_col].values,
                                        'file_size': len(df)
                                    }
                                    all_files.append(file_info)
                                    user_files += 1
                            except Exception as e:
                                print(f"⚠️ Error loading {file_path}: {e}")
            
            if user_files > 0:
                print(f"   👤 User {user_id}: {user_files} files loaded")
        
        print(f"✅ {placement} loaded: {len(all_files)} files from {len(set([f['user_id'] for f in all_files]))} users")
        return all_files
    
    def load_uci_files(self, max_subjects=30):
        """
        Load UCI_HAR data file-wise
        """
        print(f"📱 Loading UCI_HAR files...")
        all_files = []
        
        activity_mapping = {
            'activity_WALKING_DOWNSTAIRS': 'Downstairs',
            'activity_LAYING': 'Laying', 
            'activity_SITTING': 'Sitting',
            'activity_STANDING': 'Standing',
            'activity_WALKING_UPSTAIRS': 'Upstairs',
            'activity_WALKING': 'Walking'
        }
        
        for subject_id in range(1, max_subjects + 1):
            subject_dir = os.path.join(self.uci_root, f"subject_{subject_id}")
            
            if not os.path.exists(subject_dir):
                continue
            
            subject_files = 0
            for activity_folder in os.listdir(subject_dir):
                activity_path = os.path.join(subject_dir, activity_folder)
                
                if not os.path.isdir(activity_path) or activity_folder not in activity_mapping:
                    continue
                
                activity_name = activity_mapping[activity_folder]
                if activity_name not in self.activities:
                    continue
                
                csv_file = os.path.join(activity_path, 'data.csv')
                
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if self.binary_col in df.columns and len(df) > 0:
                            file_info = {
                                'user_id': subject_id,
                                'activity': activity_name,
                                'placement': 'smartphone',
                                'dataset': 'UCI_HAR',
                                'file_path': csv_file,
                                'binary_data': df[self.binary_col].values,
                                'file_size': len(df)
                            }
                            all_files.append(file_info)
                            subject_files += 1
                    except Exception as e:
                        print(f"⚠️ Error loading {csv_file}: {e}")
            
            if subject_files > 0:
                print(f"   👤 Subject {subject_id}: {subject_files} files loaded")
        
        print(f"✅ UCI_HAR loaded: {len(all_files)} files from {len(set([f['user_id'] for f in all_files]))} subjects")
        return all_files
    
    def extract_robust_file_features(self, file_info):
        """
        Extract robust temporal features from a complete activity file
        """
        binary_data = file_info['binary_data']
        
        if len(binary_data) == 0:
            return None
        
        features = {
            'user_id': file_info['user_id'],
            'activity': file_info['activity'],
            'placement': file_info['placement'],
            'file_size': file_info['file_size']
        }
        
        # 1. BASIC ACTIVITY STATISTICS
        features['activity_ratio'] = np.mean(binary_data)
        features['total_samples'] = len(binary_data)
        features['active_samples'] = np.sum(binary_data)
        features['inactive_samples'] = np.sum(1 - binary_data)
        
        # 2. TEMPORAL DYNAMICS
        if len(binary_data) > 1:
            # State transitions
            transitions = np.sum(np.abs(np.diff(binary_data)))
            features['transitions'] = transitions
            features['transition_rate'] = transitions / len(binary_data)
            
            # First and second derivatives (change patterns)
            if len(binary_data) > 2:
                first_diff = np.diff(binary_data.astype(float))
                features['mean_change'] = np.mean(np.abs(first_diff))
                features['change_variance'] = np.var(first_diff)
        else:
            features['transitions'] = 0
            features['transition_rate'] = 0
            features['mean_change'] = 0
            features['change_variance'] = 0
        
        # 3. RUN-LENGTH ANALYSIS
        runs = self._get_runs(binary_data)
        if runs:
            active_runs = [length for value, length in runs if value == 1]
            inactive_runs = [length for value, length in runs if value == 0]
            
            # Active run statistics
            if active_runs:
                features['active_runs_count'] = len(active_runs)
                features['active_runs_mean'] = np.mean(active_runs)
                features['active_runs_std'] = np.std(active_runs)
                features['active_runs_max'] = np.max(active_runs)
                features['active_runs_min'] = np.min(active_runs)
            else:
                features['active_runs_count'] = 0
                features['active_runs_mean'] = 0
                features['active_runs_std'] = 0
                features['active_runs_max'] = 0
                features['active_runs_min'] = 0
            
            # Inactive run statistics
            if inactive_runs:
                features['inactive_runs_count'] = len(inactive_runs)
                features['inactive_runs_mean'] = np.mean(inactive_runs)
                features['inactive_runs_std'] = np.std(inactive_runs)
                features['inactive_runs_max'] = np.max(inactive_runs)
                features['inactive_runs_min'] = np.min(inactive_runs)
            else:
                features['inactive_runs_count'] = 0
                features['inactive_runs_mean'] = 0
                features['inactive_runs_std'] = 0
                features['inactive_runs_max'] = 0
                features['inactive_runs_min'] = 0
            
            # Overall run pattern
            all_runs = [length for _, length in runs]
            features['total_runs'] = len(runs)
            features['run_length_variance'] = np.var(all_runs)
            features['run_length_mean'] = np.mean(all_runs)
        else:
            # No runs (all same value)
            features.update({
                'active_runs_count': 0, 'active_runs_mean': 0, 'active_runs_std': 0, 
                'active_runs_max': 0, 'active_runs_min': 0,
                'inactive_runs_count': 0, 'inactive_runs_mean': 0, 'inactive_runs_std': 0,
                'inactive_runs_max': 0, 'inactive_runs_min': 0,
                'total_runs': 1, 'run_length_variance': 0, 'run_length_mean': len(binary_data)
            })
        
        # 4. WINDOW-BASED TEMPORAL FEATURES
        window_features = self._extract_window_features(binary_data)
        features.update(window_features)
        
        # 5. PATTERN COMPLEXITY FEATURES
        features['entropy'] = self._calculate_entropy(binary_data)
        features['pattern_complexity'] = len(np.unique(binary_data))
        features['temporal_variance'] = np.var(binary_data)
        
        # 6. ACTIVITY-SPECIFIC FEATURES
        features['burst_intensity'] = self._calculate_burst_intensity(binary_data)
        features['regularity_score'] = self._calculate_regularity(binary_data)
        features['activity_density'] = self._calculate_activity_density(binary_data)
        
        return features
    
    def calculate_information_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate information-theoretic metrics: NMI and reverse KL divergence
        """
        # Calculate NMI (Normalized Mutual Information)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        nmi_percentage = nmi * 100
        
        # Calculate reverse KL divergence
        reverse_kl_percentage = 0.0
        if y_prob is not None:
            try:
                # Get unique labels and create distributions
                unique_labels = np.unique(y_true)
                n_classes = len(unique_labels)
                
                # Create label encoder for consistent indexing
                label_encoder = LabelEncoder()
                label_encoder.fit(unique_labels)
                
                y_true_encoded = label_encoder.transform(y_true)
                y_pred_encoded = label_encoder.transform(y_pred)
                
                # Calculate true distribution (from actual labels)
                true_dist = np.bincount(y_true_encoded, minlength=n_classes) / len(y_true)
                
                # Calculate predicted distribution (from predicted probabilities)
                if y_prob.shape[1] >= n_classes:
                    pred_dist = np.mean(y_prob[:, :n_classes], axis=0)
                else:
                    # If probability shape doesn't match, fall back to predicted labels
                    pred_dist = np.bincount(y_pred_encoded, minlength=n_classes) / len(y_pred)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                true_dist = true_dist + epsilon
                pred_dist = pred_dist + epsilon
                
                # Normalize to ensure they sum to 1
                true_dist = true_dist / np.sum(true_dist)
                pred_dist = pred_dist / np.sum(pred_dist)
                
                # Calculate reverse KL divergence: KL(P_true || P_pred)
                reverse_kl = entropy(true_dist, pred_dist)
                
                # Convert to percentage (lower is better, so we use 100 - normalized_kl)
                # Normalize by maximum possible KL divergence
                max_kl = entropy(true_dist, np.ones(n_classes) / n_classes)
                if max_kl > 0:
                    normalized_kl = reverse_kl / max_kl
                    reverse_kl_percentage = max(0, 100 - (normalized_kl * 100))
                else:
                    reverse_kl_percentage = 100.0
                
            except Exception as e:
                print(f"⚠️ Warning: Could not calculate reverse KL divergence: {e}")
                reverse_kl_percentage = 0.0
        
        return nmi_percentage, reverse_kl_percentage
    
    def _get_runs(self, sequence):
        """Get run-length encoding of binary sequence"""
        if len(sequence) == 0:
            return []
        
        runs = []
        current_value = sequence[0]
        current_length = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_value:
                current_length += 1
            else:
                runs.append((current_value, current_length))
                current_value = sequence[i]
                current_length = 1
        
        runs.append((current_value, current_length))
        return runs
    
    def _extract_window_features(self, binary_data):
        """Extract features from sliding windows to capture temporal dynamics"""
        if len(binary_data) < self.window_size:
            window_size = len(binary_data) // 2 if len(binary_data) > 1 else 1
        else:
            window_size = self.window_size
        
        if window_size < 1:
            return {
                'window_activity_std': 0, 'window_transition_std': 0,
                'window_activity_range': 0, 'window_count': 0
            }
        
        window_activities = []
        window_transitions = []
        
        # Sliding window analysis
        for i in range(0, len(binary_data) - window_size + 1, window_size // 2):
            window = binary_data[i:i + window_size]
            
            # Window activity ratio
            window_activities.append(np.mean(window))
            
            # Window transitions
            if len(window) > 1:
                window_transitions.append(np.sum(np.abs(np.diff(window))) / len(window))
            else:
                window_transitions.append(0)
        
        return {
            'window_activity_std': np.std(window_activities) if window_activities else 0,
            'window_transition_std': np.std(window_transitions) if window_transitions else 0,
            'window_activity_range': np.ptp(window_activities) if window_activities else 0,
            'window_count': len(window_activities)
        }
    
    def _calculate_entropy(self, sequence):
        """Calculate Shannon entropy of binary sequence"""
        if len(sequence) == 0:
            return 0
        
        unique, counts = np.unique(sequence, return_counts=True)
        probabilities = counts / len(sequence)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _calculate_burst_intensity(self, sequence):
        """Calculate burst intensity (consecutive 1s patterns)"""
        if len(sequence) == 0:
            return 0
        
        bursts = []
        current_burst = 0
        
        for val in sequence:
            if val == 1:
                current_burst += 1
            else:
                if current_burst > 0:
                    bursts.append(current_burst)
                    current_burst = 0
        
        if current_burst > 0:
            bursts.append(current_burst)
        
        return np.mean(bursts) if bursts else 0
    
    def _calculate_regularity(self, sequence):
        """Calculate regularity score based on pattern repetition"""
        if len(sequence) < 4:
            return 0
        
        # Look for repeating patterns of length 2-4
        pattern_scores = []
        
        for pattern_len in range(2, min(5, len(sequence) // 2)):
            pattern_counts = {}
            
            for i in range(len(sequence) - pattern_len + 1):
                pattern = tuple(sequence[i:i + pattern_len])
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            if pattern_counts:
                max_count = max(pattern_counts.values())
                total_patterns = len(sequence) - pattern_len + 1
                pattern_scores.append(max_count / total_patterns)
        
        return np.mean(pattern_scores) if pattern_scores else 0
    
    def _calculate_activity_density(self, sequence):
        """Calculate activity density (distribution of active periods)"""
        if len(sequence) == 0:
            return 0
        
        # Divide sequence into quarters and calculate activity in each
        quarter_size = len(sequence) // 4
        if quarter_size == 0:
            return np.mean(sequence)
        
        quarters = []
        for i in range(4):
            start_idx = i * quarter_size
            end_idx = start_idx + quarter_size if i < 3 else len(sequence)
            quarter = sequence[start_idx:end_idx]
            quarters.append(np.mean(quarter))
        
        # Return variance in activity across quarters (lower = more uniform)
        return 1 - np.var(quarters)  # Invert so higher = more uniform
    
    def create_feature_dataset(self, file_list):
        """Create feature dataset from file list"""
        if not file_list:
            return None, None, None
        
        print(f"🔧 Extracting features from {len(file_list)} files...")
        
        features_list = []
        for i, file_info in enumerate(file_list):
            features = self.extract_robust_file_features(file_info)
            if features:
                features_list.append(features)
        
        if not features_list:
            print("❌ No features extracted!")
            return None, None, None
        
        df = pd.DataFrame(features_list)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['user_id', 'activity', 'placement']]
        X = df[feature_cols].values
        y = df['activity'].values
        meta = df[['user_id', 'activity', 'placement']].copy()
        
        print(f"✅ Features extracted: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, meta
    
    def run_stm_to_uci_experiment(self, stm_placement):
        """
        Run STM-UC placement → UCI_HAR cross-sensor cross-dataset experiment
        """
        print(f"\n📍➡️📱 Cross-Sensor Cross-Dataset: {stm_placement} → UCI_HAR")
        print("-" * 60)
        
        # Load STM-UC training data (specific placement)
        print(f"🏋️ Loading training data: STM-UC {stm_placement}")
        train_files = self.load_stm_placement_files(stm_placement)
        X_train, y_train, train_meta = self.create_feature_dataset(train_files)
        
        # Load UCI_HAR test data (smartphone)
        print(f"🧪 Loading test data: UCI_HAR (smartphone)")
        test_files = self.load_uci_files()
        X_test, y_test, test_meta = self.create_feature_dataset(test_files)
        
        if X_train is None or X_test is None:
            print("❌ Failed to create datasets!")
            return None
        
        return self._train_and_evaluate(X_train, y_train, X_test, y_test, 
                                      f"STM-UC_{stm_placement}", "UCI_HAR_smartphone")
    
    def run_uci_to_stm_experiment(self, stm_placement):
        """
        Run UCI_HAR → STM-UC placement cross-sensor cross-dataset experiment
        """
        print(f"\n📱➡️📍 Cross-Dataset Cross-Sensor: UCI_HAR → {stm_placement}")
        print("-" * 60)
        
        # Load UCI_HAR training data (smartphone)
        print(f"🏋️ Loading training data: UCI_HAR (smartphone)")
        train_files = self.load_uci_files()
        X_train, y_train, train_meta = self.create_feature_dataset(train_files)
        
        # Load STM-UC test data (specific placement)
        print(f"🧪 Loading test data: STM-UC {stm_placement}")
        test_files = self.load_stm_placement_files(stm_placement)
        X_test, y_test, test_meta = self.create_feature_dataset(test_files)
        
        if X_train is None or X_test is None:
            print("❌ Failed to create datasets!")
            return None
        
        return self._train_and_evaluate(X_train, y_train, X_test, y_test, 
                                      "UCI_HAR_smartphone", f"STM-UC_{stm_placement}")
    
    def _train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                           train_name, test_name):
        """
        Train model and evaluate performance
        """
        print(f"🌲 Training Random Forest model...")
        print(f"   Training: {train_name} ({X_train.shape[0]} samples)")
        print(f"   Testing: {test_name} ({X_test.shape[0]} samples)")
        print(f"   Features: {X_train.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = rf.predict(X_test_scaled)
        y_pred_proba = rf.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Calculate information-theoretic metrics
        nmi_percentage, reverse_kl_percentage = self.calculate_information_metrics(
            y_test, y_pred, y_pred_proba)
        
        print(f"📊 Results:")
        print(f"   🎯 Accuracy: {accuracy:.3f}")
        print(f"   🎯 F1-Score: {f1:.3f}")
        print(f"   🎯 Precision: {precision:.3f}")
        print(f"   🎯 Recall: {recall:.3f}")
        print(f"   📊 NMI: {nmi_percentage:.3f}%")
        print(f"   📊 Reverse KL: {reverse_kl_percentage:.3f}%")
        
        # Activity-specific performance
        report = classification_report(y_test, y_pred, output_dict=True)
        activity_performance = {}
        for activity in self.activities:
            if activity in report:
                activity_performance[activity] = {
                    'precision': report[activity]['precision'],
                    'recall': report[activity]['recall'],
                    'f1-score': report[activity]['f1-score']
                }
        
        # Results
        results = {
            'train_source': train_name,
            'test_source': test_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'nmi_percentage': nmi_percentage,
            'reverse_kl_percentage': reverse_kl_percentage,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features_count': X_train.shape[1],
            'activity_performance': activity_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def run_all_cross_sensor_cross_dataset_experiments(self):
        """
        Run all cross-sensor cross-dataset experiments
        """
        print(f"🚀 RUNNING ALL CROSS-SENSOR CROSS-DATASET EXPERIMENTS")
        print("=" * 60)
        print(f"📍 STM-UC placements: {', '.join(self.stm_placements)}")
        print(f"� UCI_HAR: smartphone placement")
        print(f"�🔄 Total experiments: {len(self.stm_placements)} × 2 directions = {len(self.stm_placements) * 2}")
        print()
        
        all_results = []
        
        # Direction 1: STM-UC placement → UCI_HAR smartphone
        print(f"\n{'🔄' * 20} DIRECTION 1: STM-UC → UCI_HAR {'🔄' * 20}")
        for i, placement in enumerate(self.stm_placements, 1):
            print(f"\n🔄 Experiment {i}/{len(self.stm_placements)}")
            result = self.run_stm_to_uci_experiment(placement)
            if result:
                result['direction'] = 'STM_to_UCI'
                result['stm_placement'] = placement
                all_results.append(result)
            print(f"✅ Completed: {placement} → UCI_HAR")
        
        # Direction 2: UCI_HAR smartphone → STM-UC placement
        print(f"\n{'🔄' * 20} DIRECTION 2: UCI_HAR → STM-UC {'🔄' * 20}")
        for i, placement in enumerate(self.stm_placements, 1):
            print(f"\n🔄 Experiment {i}/{len(self.stm_placements)}")
            result = self.run_uci_to_stm_experiment(placement)
            if result:
                result['direction'] = 'UCI_to_STM'
                result['stm_placement'] = placement
                all_results.append(result)
            print(f"✅ Completed: UCI_HAR → {placement}")
        
        # Save and analyze results
        self.save_and_analyze_results(all_results)
        
        return all_results
    
    def save_and_analyze_results(self, all_results):
        """
        Save results and create comprehensive analysis
        """
        print(f"\n💾 SAVING AND ANALYZING RESULTS")
        print("=" * 40)
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'cross_sensor_cross_dataset_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create results DataFrame for analysis
        df_results = pd.DataFrame(all_results)
        
        # Summary statistics
        print(f"📊 CROSS-SENSOR CROSS-DATASET PERFORMANCE SUMMARY:")
        print(f"   Total experiments: {len(all_results)}")
        print(f"   Mean accuracy: {df_results['accuracy'].mean():.3f} ± {df_results['accuracy'].std():.3f}")
        print(f"   Mean F1-score: {df_results['f1_score'].mean():.3f} ± {df_results['f1_score'].std():.3f}")
        print(f"   Mean NMI: {df_results['nmi_percentage'].mean():.3f}% ± {df_results['nmi_percentage'].std():.3f}%")
        print(f"   Mean Reverse KL: {df_results['reverse_kl_percentage'].mean():.3f}% ± {df_results['reverse_kl_percentage'].std():.3f}%")
        print(f"   Best accuracy: {df_results['accuracy'].max():.3f}")
        print(f"   Worst accuracy: {df_results['accuracy'].min():.3f}")
        
        # Direction-specific analysis
        stm_to_uci = df_results[df_results['direction'] == 'STM_to_UCI']
        uci_to_stm = df_results[df_results['direction'] == 'UCI_to_STM']
        
        print(f"\n🔍 DIRECTION-SPECIFIC ANALYSIS:")
        print(f"   STM-UC → UCI_HAR (N={len(stm_to_uci)}):")
        print(f"     Mean accuracy: {stm_to_uci['accuracy'].mean():.3f} ± {stm_to_uci['accuracy'].std():.3f}")
        print(f"     Mean F1-score: {stm_to_uci['f1_score'].mean():.3f} ± {stm_to_uci['f1_score'].std():.3f}")
        print(f"     Mean NMI: {stm_to_uci['nmi_percentage'].mean():.3f}% ± {stm_to_uci['nmi_percentage'].std():.3f}%")
        print(f"     Mean Reverse KL: {stm_to_uci['reverse_kl_percentage'].mean():.3f}% ± {stm_to_uci['reverse_kl_percentage'].std():.3f}%")
        print(f"   UCI_HAR → STM-UC (N={len(uci_to_stm)}):")
        print(f"     Mean accuracy: {uci_to_stm['accuracy'].mean():.3f} ± {uci_to_stm['accuracy'].std():.3f}")
        print(f"     Mean F1-score: {uci_to_stm['f1_score'].mean():.3f} ± {uci_to_stm['f1_score'].std():.3f}")
        print(f"     Mean NMI: {uci_to_stm['nmi_percentage'].mean():.3f}% ± {uci_to_stm['nmi_percentage'].std():.3f}%")
        print(f"     Mean Reverse KL: {uci_to_stm['reverse_kl_percentage'].mean():.3f}% ± {uci_to_stm['reverse_kl_percentage'].std():.3f}%")
        
        # Placement-specific analysis
        print(f"\n📍 PLACEMENT-SPECIFIC ANALYSIS:")
        for placement in self.stm_placements:
            placement_stm_to_uci = stm_to_uci[stm_to_uci['stm_placement'] == placement]
            placement_uci_to_stm = uci_to_stm[uci_to_stm['stm_placement'] == placement]
            
            print(f"   {placement}:")
            if len(placement_stm_to_uci) > 0:
                acc = placement_stm_to_uci['accuracy'].iloc[0]
                f1 = placement_stm_to_uci['f1_score'].iloc[0]
                nmi = placement_stm_to_uci['nmi_percentage'].iloc[0]
                rkl = placement_stm_to_uci['reverse_kl_percentage'].iloc[0]
                print(f"     → UCI_HAR: Acc={acc:.3f}, F1={f1:.3f}, NMI={nmi:.3f}%, RKL={rkl:.3f}%")
            if len(placement_uci_to_stm) > 0:
                acc = placement_uci_to_stm['accuracy'].iloc[0]
                f1 = placement_uci_to_stm['f1_score'].iloc[0]
                nmi = placement_uci_to_stm['nmi_percentage'].iloc[0]
                rkl = placement_uci_to_stm['reverse_kl_percentage'].iloc[0]
                print(f"     ← UCI_HAR: Acc={acc:.3f}, F1={f1:.3f}, NMI={nmi:.3f}%, RKL={rkl:.3f}%")
        
        # Best and worst performing experiments
        best_exp = df_results.loc[df_results['accuracy'].idxmax()]
        worst_exp = df_results.loc[df_results['accuracy'].idxmin()]
        
        print(f"\n🏆 BEST PERFORMING EXPERIMENT:")
        print(f"   {best_exp['train_source']} → {best_exp['test_source']}")
        print(f"   Accuracy: {best_exp['accuracy']:.3f}, F1: {best_exp['f1_score']:.3f}")
        print(f"   NMI: {best_exp['nmi_percentage']:.3f}%, Reverse KL: {best_exp['reverse_kl_percentage']:.3f}%")
        
        print(f"\n💥 WORST PERFORMING EXPERIMENT:")
        print(f"   {worst_exp['train_source']} → {worst_exp['test_source']}")
        print(f"   Accuracy: {worst_exp['accuracy']:.3f}, F1: {worst_exp['f1_score']:.3f}")
        print(f"   NMI: {worst_exp['nmi_percentage']:.3f}%, Reverse KL: {worst_exp['reverse_kl_percentage']:.3f}%")
        
        # Create performance visualization
        self.create_performance_visualization(df_results)
        
        # Save summary
        summary = {
            'total_experiments': len(all_results),
            'mean_accuracy': df_results['accuracy'].mean(),
            'std_accuracy': df_results['accuracy'].std(),
            'mean_f1_score': df_results['f1_score'].mean(),
            'std_f1_score': df_results['f1_score'].std(),
            'mean_nmi_percentage': df_results['nmi_percentage'].mean(),
            'std_nmi_percentage': df_results['nmi_percentage'].std(),
            'mean_reverse_kl_percentage': df_results['reverse_kl_percentage'].mean(),
            'std_reverse_kl_percentage': df_results['reverse_kl_percentage'].std(),
            'best_experiment': {
                'train': best_exp['train_source'],
                'test': best_exp['test_source'],
                'accuracy': best_exp['accuracy'],
                'f1_score': best_exp['f1_score'],
                'nmi_percentage': best_exp['nmi_percentage'],
                'reverse_kl_percentage': best_exp['reverse_kl_percentage']
            },
            'worst_experiment': {
                'train': worst_exp['train_source'],
                'test': worst_exp['test_source'],
                'accuracy': worst_exp['accuracy'],
                'f1_score': worst_exp['f1_score'],
                'nmi_percentage': worst_exp['nmi_percentage'],
                'reverse_kl_percentage': worst_exp['reverse_kl_percentage']
            },
            'stm_to_uci_performance': {
                'mean_accuracy': stm_to_uci['accuracy'].mean(),
                'mean_f1_score': stm_to_uci['f1_score'].mean(),
                'mean_nmi_percentage': stm_to_uci['nmi_percentage'].mean(),
                'mean_reverse_kl_percentage': stm_to_uci['reverse_kl_percentage'].mean()
            },
            'uci_to_stm_performance': {
                'mean_accuracy': uci_to_stm['accuracy'].mean(),
                'mean_f1_score': uci_to_stm['f1_score'].mean(),
                'mean_nmi_percentage': uci_to_stm['nmi_percentage'].mean(),
                'mean_reverse_kl_percentage': uci_to_stm['reverse_kl_percentage'].mean()
            }
        }
        
        summary_file = os.path.join(self.results_dir, 'cross_sensor_cross_dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"   📁 Detailed results: {results_file}")
        print(f"   📁 Summary: {summary_file}")
        
        # Print comprehensive summary
        print(f"\n{'='*50}")
        print(f"CROSS-SENSOR CROSS-DATASET SUMMARY")
        print(f"{'='*50}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Overall Performance:")
        print(f"  Accuracy: {summary['mean_accuracy']:.1f}% ± {summary['std_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['mean_f1_score']:.1f}% ± {summary['std_f1_score']:.1f}%")
        print(f"  NMI: {summary['mean_nmi_percentage']:.1f}% ± {summary['std_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['mean_reverse_kl_percentage']:.2f}% ± {summary['std_reverse_kl_percentage']:.2f}%")
        
        print(f"\nBest Experiment: {summary['best_experiment']['train']} → {summary['best_experiment']['test']}")
        print(f"  Accuracy: {summary['best_experiment']['accuracy']:.1f}%")
        print(f"  F1-Score: {summary['best_experiment']['f1_score']:.1f}%")
        print(f"  NMI: {summary['best_experiment']['nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['best_experiment']['reverse_kl_percentage']:.2f}%")
        
        print(f"\nWorst Experiment: {summary['worst_experiment']['train']} → {summary['worst_experiment']['test']}")
        print(f"  Accuracy: {summary['worst_experiment']['accuracy']:.1f}%")
        print(f"  F1-Score: {summary['worst_experiment']['f1_score']:.1f}%")
        print(f"  NMI: {summary['worst_experiment']['nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['worst_experiment']['reverse_kl_percentage']:.2f}%")
        
        print(f"\nSTM → UCI Performance:")
        print(f"  Accuracy: {summary['stm_to_uci_performance']['mean_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['stm_to_uci_performance']['mean_f1_score']:.1f}%")
        print(f"  NMI: {summary['stm_to_uci_performance']['mean_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['stm_to_uci_performance']['mean_reverse_kl_percentage']:.2f}%")
        
        print(f"\nUCI → STM Performance:")
        print(f"  Accuracy: {summary['uci_to_stm_performance']['mean_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['uci_to_stm_performance']['mean_f1_score']:.1f}%")
        print(f"  NMI: {summary['uci_to_stm_performance']['mean_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['uci_to_stm_performance']['mean_reverse_kl_percentage']:.2f}%")
        
        return all_results
        
    def create_performance_visualization(self, df_results):
        """
        Create performance visualization plots
        """
        print(f"\n📊 Creating performance visualizations...")
        
        # Separate by direction
        stm_to_uci = df_results[df_results['direction'] == 'STM_to_UCI']
        uci_to_stm = df_results[df_results['direction'] == 'UCI_to_STM']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cross-Sensor Cross-Dataset HAR Performance', fontsize=16)
        
        # 1. Accuracy by placement and direction
        ax1 = axes[0, 0]
        placement_data = []
        for placement in self.stm_placements:
            stm_acc = stm_to_uci[stm_to_uci['stm_placement'] == placement]['accuracy'].iloc[0] if len(stm_to_uci[stm_to_uci['stm_placement'] == placement]) > 0 else 0
            uci_acc = uci_to_stm[uci_to_stm['stm_placement'] == placement]['accuracy'].iloc[0] if len(uci_to_stm[uci_to_stm['stm_placement'] == placement]) > 0 else 0
            placement_data.append({'placement': placement, 'STM→UCI': stm_acc, 'UCI→STM': uci_acc})
        
        placement_df = pd.DataFrame(placement_data)
        placement_df.set_index('placement')[['STM→UCI', 'UCI→STM']].plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Accuracy by Placement and Direction')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. F1-score by placement and direction
        ax2 = axes[0, 1]
        f1_data = []
        for placement in self.stm_placements:
            stm_f1 = stm_to_uci[stm_to_uci['stm_placement'] == placement]['f1_score'].iloc[0] if len(stm_to_uci[stm_to_uci['stm_placement'] == placement]) > 0 else 0
            uci_f1 = uci_to_stm[uci_to_stm['stm_placement'] == placement]['f1_score'].iloc[0] if len(uci_to_stm[uci_to_stm['stm_placement'] == placement]) > 0 else 0
            f1_data.append({'placement': placement, 'STM→UCI': stm_f1, 'UCI→STM': uci_f1})
        
        f1_df = pd.DataFrame(f1_data)
        f1_df.set_index('placement')[['STM→UCI', 'UCI→STM']].plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('F1-Score by Placement and Direction')
        ax2.set_ylabel('F1-Score')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Direction comparison boxplot
        ax3 = axes[1, 0]
        direction_data = []
        for _, row in df_results.iterrows():
            direction_label = 'STM→UCI' if row['direction'] == 'STM_to_UCI' else 'UCI→STM'
            direction_data.append({'Direction': direction_label, 'Accuracy': row['accuracy']})
        
        direction_df = pd.DataFrame(direction_data)
        direction_df.boxplot(column='Accuracy', by='Direction', ax=ax3)
        ax3.set_title('Accuracy Distribution by Direction')
        ax3.set_xlabel('Transfer Direction')
        ax3.set_ylabel('Accuracy')
        
        # 4. Performance scatter plot
        ax4 = axes[1, 1]
        ax4.scatter(stm_to_uci['accuracy'], stm_to_uci['f1_score'], 
                   color='blue', alpha=0.7, s=100, label='STM→UCI')
        ax4.scatter(uci_to_stm['accuracy'], uci_to_stm['f1_score'], 
                   color='red', alpha=0.7, s=100, label='UCI→STM')
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Accuracy vs F1-Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.results_dir, 'cross_sensor_cross_dataset_performance.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Performance visualization saved: {plot_file}")

def main():
    """
    Run the complete cross-sensor cross-dataset experiment
    """
    print("🚀 FILE-WISE CROSS-SENSOR CROSS-DATASET HAR EXPERIMENT")
    print("=" * 60)
    print("🔧 Implementation Features:")
    print("   ✅ File-wise processing (complete activity sessions)")
    print("   ✅ Robust temporal features (29 features)")
    print("   ✅ Cross-sensor cross-dataset evaluation")
    print("   ✅ STM-UC placements ↔ UCI_HAR smartphone")
    print("   ✅ Advanced pattern recognition")
    print()
    
    # Create analyzer
    analyzer = FileWiseCrossSensorCrossDatasetAnalyzer()
    
    # Run all experiments
    results = analyzer.run_all_cross_sensor_cross_dataset_experiments()
    
    print(f"\n🎉 CROSS-SENSOR CROSS-DATASET EXPERIMENT COMPLETED!")
    print(f"📊 {len(results)} experiments completed successfully")
    print(f"📁 Results saved in: {analyzer.results_dir}")
    print(f"\n💡 Expected: Cross-sensor cross-dataset generalization evaluation")
    print(f"🎯 Target: Better performance than original ~16% F1-score")

if __name__ == "__main__":
    main()
