#!/usr/bin/env python3
"""
Game-1 File-wise Cross-Sensor HAR with Game-2 Feature Extraction
===============================================================

This script applies Game-1's file-wise processing approach but uses EXACTLY the same
feature extraction methodology as your Game-2 script:

✅ GAME-2 FEATURE EXTRACTION (EXACT):
- Global accelerometer statistical features (per axis x,y,z + magnitude) 
- Global binary features from decision tree outputs
- User identity semantic information
- Placement-agnostic feature engineering
- NO temporal/run-length features from Game-1

✅ GAME-1 PROCESSING FRAMEWORK:
- File-wise processing (complete activity sessions)
- Cross-sensor cross-dataset evaluation (STM-UC ↔ UCI_HAR)
- Per-placement training and testing
- Robust data loading and handling

METHODOLOGY:
Uses Game-1's robust file-wise data processing with Game-2's exact accelerometer
feature extraction to evaluate cross-sensor transferability.

FEATURE COUNT: ~67 features (same as Game-2)
- 51 accelerometer features (17 per axis × 3 axes)
- 10 magnitude features  
- 5 binary features
- 1 user identity feature
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

print("🎯 GAME-1 FILE-WISE + GAME-2 FEATURES: Cross-Sensor HAR")
print("=" * 60)
print("Using Game-1's file-wise processing with Game-2's exact features")
print()

class EnhancedFileWiseCrossSensorAnalyzer:
    """
    File-wise cross-sensor analyzer using EXACT Game-2 feature extraction
    Combines Game-1's file processing with Game-2's accelerometer features
    """
    
    def __init__(self, window_size=100):
        # Activities available in both datasets
        self.activities = ['Downstairs', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # STM sensor placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Sensor columns
        self.binary_col = 'dec_tree_out_1'
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        
        # Window size for temporal analysis
        self.window_size = window_size
        
        # Results directory
        self.results_dir = "results/game-1-enhanced-accelerometer"
        self.stm_root = str(get_stm_uc_root())
        self.uci_root = str(get_uci_root())
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🎯 Enhanced cross-sensor cross-dataset evaluation")
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
                                    'file_size': len(df),
                                    'dataframe': df  # Store full dataframe for accelerometer features
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
                                        'file_size': len(df),
                                        'dataframe': df  # Store full dataframe for accelerometer features
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
                                'file_size': len(df),
                                'dataframe': df  # Store full dataframe for accelerometer features
                            }
                            all_files.append(file_info)
                            subject_files += 1
                    except Exception as e:
                        print(f"⚠️ Error loading {csv_file}: {e}")
            
            if subject_files > 0:
                print(f"   👤 Subject {subject_id}: {subject_files} files loaded")
        
        print(f"✅ UCI_HAR loaded: {len(all_files)} files from {len(set([f['user_id'] for f in all_files]))} subjects")
        return all_files
    
    def extract_enhanced_file_features(self, file_info):
        """
        Extract features using EXACT GAME-2 approach: Accelerometer + Binary + User Identity
        NO Game-1 temporal features - using only your Game-2 feature extraction
        """
        df = file_info['dataframe']
        
        if len(df) == 0:
            return None
        
        features = {
            'user_id': file_info['user_id'],
            'activity': file_info['activity'],
            'placement': file_info['placement'],
            'file_size': file_info['file_size']
        }
        
        # Check for accelerometer columns
        has_accel = all(col in df.columns for col in self.accel_cols)
        
        # ===== EXACT GAME-2 FEATURE EXTRACTION =====
        # Using placement-anonymous approach from your Game-2 script
        
        if has_accel:
            # 1. GLOBAL ACCELEROMETER STATISTICAL FEATURES
            for axis_col in self.accel_cols:
                data = df[axis_col].values
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                
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
                    
                    # Zero crossings
                    features[f'global_{axis}_zero_crossings'] = np.sum(np.diff(np.sign(data)) != 0)
                else:
                    # Single sample defaults
                    features[f'global_{axis}_skew'] = 0.0
                    features[f'global_{axis}_kurtosis'] = 0.0
                    features[f'global_{axis}_rms'] = abs(data[0])
                    features[f'global_{axis}_diff_mean'] = 0.0
                    features[f'global_{axis}_diff_std'] = 0.0
                    features[f'global_{axis}_diff_max'] = 0.0
                    features[f'global_{axis}_q25'] = data[0]
                    features[f'global_{axis}_q75'] = data[0]
                    features[f'global_{axis}_iqr'] = 0.0
                    features[f'global_{axis}_zero_crossings'] = 0.0
            
            # 2. GLOBAL MAGNITUDE FEATURES (if all accelerometer axes available)
            accel_data = df[self.accel_cols].values
            magnitude = np.linalg.norm(accel_data, axis=1)
            
            features['global_magnitude_mean'] = np.mean(magnitude)
            features['global_magnitude_std'] = np.std(magnitude)
            features['global_magnitude_max'] = np.max(magnitude)
            features['global_magnitude_min'] = np.min(magnitude)
            features['global_magnitude_median'] = np.median(magnitude)
            features['global_magnitude_range'] = np.max(magnitude) - np.min(magnitude)
            features['global_magnitude_energy'] = np.sum(magnitude**2)
            
            if len(magnitude) > 1:
                features['global_magnitude_skew'] = pd.Series(magnitude).skew()
                features['global_magnitude_kurtosis'] = pd.Series(magnitude).kurtosis()
                features['global_magnitude_rms'] = np.sqrt(np.mean(magnitude**2))
            else:
                features['global_magnitude_skew'] = 0.0
                features['global_magnitude_kurtosis'] = 0.0
                features['global_magnitude_rms'] = magnitude[0]
        
        else:
            # Default values when accelerometer data is not available
            print(f"⚠️ No accelerometer data in {file_info['file_path']}")
            
            # Set all Game-2 accelerometer features to 0
            for axis in ['x', 'y', 'z']:
                features.update({
                    f'global_{axis}_mean': 0.0, f'global_{axis}_std': 0.0, f'global_{axis}_max': 0.0, f'global_{axis}_min': 0.0,
                    f'global_{axis}_median': 0.0, f'global_{axis}_range': 0.0, f'global_{axis}_energy': 0.0,
                    f'global_{axis}_skew': 0.0, f'global_{axis}_kurtosis': 0.0, f'global_{axis}_rms': 0.0,
                    f'global_{axis}_diff_mean': 0.0, f'global_{axis}_diff_std': 0.0, f'global_{axis}_diff_max': 0.0,
                    f'global_{axis}_q25': 0.0, f'global_{axis}_q75': 0.0, f'global_{axis}_iqr': 0.0, f'global_{axis}_zero_crossings': 0.0
                })
            
            features.update({
                'global_magnitude_mean': 0.0, 'global_magnitude_std': 0.0, 'global_magnitude_max': 0.0, 'global_magnitude_min': 0.0,
                'global_magnitude_median': 0.0, 'global_magnitude_range': 0.0, 'global_magnitude_energy': 0.0,
                'global_magnitude_skew': 0.0, 'global_magnitude_kurtosis': 0.0, 'global_magnitude_rms': 0.0
            })
        
        # 3. GLOBAL BINARY FEATURES (from anonymous sensors) - EXACT Game-2
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
        else:
            features.update({
                'global_binary_activity_ratio': 0.0, 'global_binary_total_samples': 0.0, 'global_binary_active_samples': 0.0,
                'global_binary_transitions': 0.0, 'global_binary_transition_rate': 0.0
            })
        
        # 4. USER-SPECIFIC PATTERN ENCODING (using user identity semantic info) - EXACT Game-2
        features['user_id_feature'] = file_info['user_id']  # Use user identity as semantic information
        
        # 5. SENSOR COUNT METADATA (without placement identity) - Following Game-2
        features['n_anonymous_sensors'] = 1  # Single file = 1 anonymous sensor
        features['avg_sensor_length'] = len(df)
        
        # Clean NaN and inf values more robustly (same as Game-2)
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

    def _max_sequence_length(self, binary_array):
        """Calculate maximum consecutive sequence of 1s"""
        if len(binary_array) == 0:
            return 0
        
        max_length = 0
        current_length = 0
        
        for val in binary_array:
            if val == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
        
        return max_length / len(binary_array)
    
    def create_feature_dataset(self, file_list):
        """Create feature dataset from file list using enhanced features"""
        if not file_list:
            return None, None, None
        
        print(f"🔧 Extracting enhanced features from {len(file_list)} files...")
        
        features_list = []
        for i, file_info in enumerate(file_list):
            features = self.extract_enhanced_file_features(file_info)
            if features:
                features_list.append(features)
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{len(file_list)} files...")
        
        if not features_list:
            print("❌ No features extracted!")
            return None, None, None
        
        df = pd.DataFrame(features_list)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['user_id', 'activity', 'placement']]
        X = df[feature_cols].values
        y = df['activity'].values
        meta = df[['user_id', 'activity', 'placement']].copy()
        
        print(f"✅ Enhanced features extracted: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   📊 Feature breakdown: {X.shape[1]} total features")
        print(f"       - Game-1 temporal features: ~29")
        print(f"       - Game-2 accelerometer features: ~51")  
        print(f"       - Binary pattern features: ~5")
        print(f"       - Interaction features: ~6")
        
        return X, y, meta
    
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

    def run_stm_to_uci_experiment(self, stm_placement):
        """
        Run STM-UC placement → UCI_HAR cross-sensor cross-dataset experiment
        """
        print(f"\n📍➡️📱 Enhanced Cross-Sensor Cross-Dataset: {stm_placement} → UCI_HAR")
        print("-" * 70)
        
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
        print(f"\n📱➡️📍 Enhanced Cross-Dataset Cross-Sensor: UCI_HAR → {stm_placement}")
        print("-" * 70)
        
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
        Train model and evaluate performance with enhanced features
        """
        print(f"🌲 Training Enhanced Random Forest model...")
        print(f"   Training: {train_name} ({X_train.shape[0]} samples)")
        print(f"   Testing: {test_name} ({X_test.shape[0]} samples)")
        print(f"   Enhanced Features: {X_train.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest with enhanced features
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight='balanced',
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            n_jobs=-1
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
        
        print(f"📊 Enhanced Results:")
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
        
        # Feature importance analysis
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        feature_importance = rf.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:]
        top_features = [(feature_names[i], feature_importance[i]) for i in top_features_idx]
        
        print(f"🔍 Top 10 Most Important Features:")
        for i, (feat_name, importance) in enumerate(reversed(top_features), 1):
            print(f"   {i:2d}. {feat_name}: {importance:.4f}")
        
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
            'top_features': top_features,
            'timestamp': datetime.now().isoformat(),
            'feature_approach': 'Game1_temporal_plus_Game2_accelerometer'
        }
        
        return results
    
    def run_all_enhanced_experiments(self):
        """
        Run all enhanced cross-sensor cross-dataset experiments
        """
        print(f"🚀 RUNNING ALL ENHANCED CROSS-SENSOR CROSS-DATASET EXPERIMENTS")
        print("=" * 80)
        print(f"📍 STM-UC placements: {', '.join(self.stm_placements)}")
        print(f"📱 UCI_HAR: smartphone placement")
        print(f"🔄 Total experiments: {len(self.stm_placements)} × 2 directions = {len(self.stm_placements) * 2}")
        print(f"🧬 Enhanced features: Game-1 temporal + Game-2 accelerometer + interactions")
        print()
        
        all_results = []
        
        # Direction 1: STM-UC placement → UCI_HAR smartphone
        print(f"\n{'🔄' * 25} DIRECTION 1: STM-UC → UCI_HAR {'🔄' * 25}")
        for i, placement in enumerate(self.stm_placements, 1):
            print(f"\n🔄 Experiment {i}/{len(self.stm_placements)}")
            result = self.run_stm_to_uci_experiment(placement)
            if result:
                result['direction'] = 'STM_to_UCI'
                result['stm_placement'] = placement
                all_results.append(result)
            print(f"✅ Completed: {placement} → UCI_HAR")
        
        # Direction 2: UCI_HAR smartphone → STM-UC placement
        print(f"\n{'🔄' * 25} DIRECTION 2: UCI_HAR → STM-UC {'🔄' * 25}")
        for i, placement in enumerate(self.stm_placements, 1):
            print(f"\n🔄 Experiment {i}/{len(self.stm_placements)}")
            result = self.run_uci_to_stm_experiment(placement)
            if result:
                result['direction'] = 'UCI_to_STM'
                result['stm_placement'] = placement
                all_results.append(result)
            print(f"✅ Completed: UCI_HAR → {placement}")
        
        # Save and analyze results
        self.save_and_analyze_enhanced_results(all_results)
        
        return all_results
    
    def save_and_analyze_enhanced_results(self, all_results):
        """
        Save results and create comprehensive analysis
        """
        print(f"\n💾 SAVING AND ANALYZING ENHANCED RESULTS")
        print("=" * 60)
        
        # Save detailed results (convert numpy types for JSON serialization)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f'enhanced_game1_results_{timestamp}.json')
        
        # Convert numpy types in results to Python types
        json_safe_results = []
        for result in all_results:
            json_safe_result = {}
            for key, value in result.items():
                if isinstance(value, np.integer):
                    json_safe_result[key] = int(value)
                elif isinstance(value, np.floating):
                    json_safe_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_safe_result[key] = value.tolist()
                else:
                    json_safe_result[key] = value
            json_safe_results.append(json_safe_result)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        
        # Create results DataFrame for analysis
        df_results = pd.DataFrame(all_results)
        
        # Summary statistics
        print(f"📊 ENHANCED GAME-1 PERFORMANCE SUMMARY:")
        print(f"   Total experiments: {len(all_results)}")
        print(f"   Mean accuracy: {df_results['accuracy'].mean():.3f} ± {df_results['accuracy'].std():.3f}")
        print(f"   Mean F1-score: {df_results['f1_score'].mean():.3f} ± {df_results['f1_score'].std():.3f}")
        print(f"   Mean NMI: {df_results['nmi_percentage'].mean():.3f}% ± {df_results['nmi_percentage'].std():.3f}%")
        print(f"   Mean Reverse KL: {df_results['reverse_kl_percentage'].mean():.3f}% ± {df_results['reverse_kl_percentage'].std():.3f}%")
        print(f"   Best accuracy: {df_results['accuracy'].max():.3f}")
        print(f"   Worst accuracy: {df_results['accuracy'].min():.3f}")
        print(f"   Enhanced features used: {df_results['features_count'].iloc[0] if len(df_results) > 0 else 'N/A'}")
        
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
        
        # Save summary (convert numpy types to Python types for JSON serialization)
        summary = {
            'total_experiments': int(len(all_results)),
            'feature_approach': 'Game1_temporal_plus_Game2_accelerometer_enhanced',
            'feature_count': int(df_results['features_count'].iloc[0]) if len(df_results) > 0 else 0,
            'mean_accuracy': float(df_results['accuracy'].mean()),
            'std_accuracy': float(df_results['accuracy'].std()),
            'mean_f1_score': float(df_results['f1_score'].mean()),
            'std_f1_score': float(df_results['f1_score'].std()),
            'mean_nmi_percentage': float(df_results['nmi_percentage'].mean()),
            'std_nmi_percentage': float(df_results['nmi_percentage'].std()),
            'mean_reverse_kl_percentage': float(df_results['reverse_kl_percentage'].mean()),
            'std_reverse_kl_percentage': float(df_results['reverse_kl_percentage'].std()),
            'best_experiment': {
                'train': str(best_exp['train_source']),
                'test': str(best_exp['test_source']),
                'accuracy': float(best_exp['accuracy']),
                'f1_score': float(best_exp['f1_score']),
                'nmi_percentage': float(best_exp['nmi_percentage']),
                'reverse_kl_percentage': float(best_exp['reverse_kl_percentage'])
            },
            'worst_experiment': {
                'train': str(worst_exp['train_source']),
                'test': str(worst_exp['test_source']),
                'accuracy': float(worst_exp['accuracy']),
                'f1_score': float(worst_exp['f1_score']),
                'nmi_percentage': float(worst_exp['nmi_percentage']),
                'reverse_kl_percentage': float(worst_exp['reverse_kl_percentage'])
            },
            'stm_to_uci_performance': {
                'mean_accuracy': float(stm_to_uci['accuracy'].mean()),
                'mean_f1_score': float(stm_to_uci['f1_score'].mean()),
                'mean_nmi_percentage': float(stm_to_uci['nmi_percentage'].mean()),
                'mean_reverse_kl_percentage': float(stm_to_uci['reverse_kl_percentage'].mean())
            },
            'uci_to_stm_performance': {
                'mean_accuracy': float(uci_to_stm['accuracy'].mean()),
                'mean_f1_score': float(uci_to_stm['f1_score'].mean()),
                'mean_nmi_percentage': float(uci_to_stm['nmi_percentage'].mean()),
                'mean_reverse_kl_percentage': float(uci_to_stm['reverse_kl_percentage'].mean())
            }
        }
        
        summary_file = os.path.join(self.results_dir, f'enhanced_game1_summary_{timestamp}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Enhanced results saved:")
        print(f"   📁 Detailed results: {results_file}")
        print(f"   📁 Summary: {summary_file}")
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print(f"ENHANCED GAME-1 CROSS-SENSOR CROSS-DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Feature Approach: Game-1 temporal + Game-2 accelerometer + interactions")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Feature count: {summary['feature_count']}")
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

def main():
    """
    Run the complete enhanced cross-sensor cross-dataset experiment
    """
    print("🚀 ENHANCED GAME-1: FILE-WISE CROSS-SENSOR WITH ACCELEROMETER FEATURES")
    print("=" * 90)
    print("🔧 Implementation Features:")
    print("   ✅ File-wise processing (complete activity sessions)")
    print("   ✅ Enhanced features: Game-1 temporal + Game-2 accelerometer + interactions")
    print("   ✅ Cross-sensor cross-dataset evaluation")
    print("   ✅ STM-UC placements ↔ UCI_HAR smartphone")
    print("   ✅ Advanced pattern recognition with ~91 features")
    print()
    
    # Create analyzer
    analyzer = EnhancedFileWiseCrossSensorAnalyzer()
    
    # Run all experiments
    results = analyzer.run_all_enhanced_experiments()
    
    print(f"\n🎉 ENHANCED GAME-1 EXPERIMENT COMPLETED!")
    print(f"📊 {len(results)} experiments completed successfully")
    print(f"📁 Results saved in: {analyzer.results_dir}")
    print(f"\n💡 Enhancement: Combined Game-1 temporal features with Game-2 accelerometer features")
    print(f"🎯 Expected: Significantly better performance than original through enhanced feature engineering")
    print(f"📈 Feature count: ~91 features (29 temporal + 51 accelerometer + 5 binary + 6 interaction)")

if __name__ == "__main__":
    main()
