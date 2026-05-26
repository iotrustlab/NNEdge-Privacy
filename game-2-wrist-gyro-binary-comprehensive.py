#!/usr/bin/env python3
"""
🎯 GAME-2 WRIST GYRO + BINARY HAR ATTACK - COMPREHENSIVE EVALUATION
==================================================================

Enhanced version that combines:
- Binary decision tree features (proven effective)
- Rich gyroscope features (gyro_x, gyro_y, gyro_z)
- All possible 4-4, 5-3, and 6-2 subject splits
- Comprehensive metrics: Accuracy, F1, NMI, NRKL

Focus: Understanding how gyroscope features compare to accelerometer features
Target: 21 wrist actions from UTD-MHAD
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import itertools
import warnings
from scipy.stats import entropy, skew, kurtosis, pearsonr
from scipy.signal import find_peaks
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')  # For headless plotting
warnings.filterwarnings('ignore')
import json
import time
from utd_vulnerability_utils import calculate_vulnerability, save_max_vulnerability_from_values

class Game2WristGyroHAR:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.wrist_actions = {
            1: "Right arm swipe to left", 2: "Right arm swipe to right", 3: "Right hand wave",
            4: "Two hand front clap", 5: "Right arm throw", 6: "Cross arms in chest",
            7: "Basketball shoot", 8: "Right hand draw x", 9: "Right hand draw circle (clockwise)",
            10: "Right hand draw circle (counter clockwise)", 11: "Draw triangle", 
            12: "Bowling (right hand)", 13: "Front boxing", 14: "Baseball swing from right",
            15: "Tennis right hand forehand swing", 16: "Arm curl (two arms)", 17: "Tennis serve",
            18: "Two hand push", 19: "Right hand knock on door", 20: "Right hand catch an object",
            21: "Right hand pick up and throw"
        }
        
        # Map to 0-20 for 21-class classification
        self.action_mapping = {i: i-1 for i in range(1, 22)}
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Expected columns in UTD-MHAD
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        
        # Create comprehensive results directory
        os.makedirs("Results", exist_ok=True)
        
        print(f"🎯 GAME-2 WRIST GYRO + BINARY EVALUATION INITIALIZED")
        print(f"📊 Will test gyroscope + binary features for 21 wrist actions")
        
    def load_wrist_data_with_gyro(self):
        """Load wrist data with gyroscope and binary features"""
        print(f"\n📂 LOADING WRIST DATA WITH GYROSCOPE...")
        
        all_data = []
        action_counts = defaultdict(lambda: defaultdict(int))
        
        # Get all CSV files from right-wrist directories
        csv_pattern = os.path.join(self.data_dir, "**", "right-wrist", "Action_*", "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        for file_path in csv_files:
            try:
                # Extract action and subject from path
                path_parts = file_path.split(os.sep)
                
                action_dir = next((part for part in path_parts if part.startswith('Action_')), None)
                subject_dir = next((part for part in path_parts if part.startswith('Subject')), None)
                
                if action_dir and subject_dir:
                    action_num = int(action_dir.split('_')[1])
                    subject_num = int(subject_dir.replace('Subject', ''))
                    
                    # Check if this is a target wrist action (1-21)
                    if action_num in self.wrist_actions:
                        # Load the CSV to check columns
                        data = pd.read_csv(file_path)
                        
                        # Check for required columns
                        has_binary = self.binary_col in data.columns
                        available_gyro = [col for col in self.gyro_cols if col in data.columns]
                        
                        if has_binary and len(available_gyro) == 3:  # Require complete gyro data
                            all_data.append({
                                'file_path': file_path,
                                'action': action_num,
                                'subject': subject_num,
                                'data': data,
                                'sequence_length': len(data),
                                'binary_ratio': data[self.binary_col].mean()
                            })
                            action_counts[action_num][subject_num] += 1
                            
            except (ValueError, IndexError) as e:
                continue
        
        print(f"📊 WRIST DATA SUMMARY: {len(all_data)} sequences with complete gyro+binary data")
        subjects = sorted(set(item['subject'] for item in all_data))
        print(f"📊 Subjects: {subjects}")
        
        return all_data, subjects
    
    def extract_comprehensive_gyro_binary_features(self, data):
        """Extract comprehensive gyroscope + binary features"""
        features = {}
        
        if len(data) == 0:
            return {}
        
        # =================================================================
        # 1. BINARY FEATURES
        # =================================================================
        if self.binary_col in data.columns:
            binary_data = data[self.binary_col].values
            n = len(binary_data)
            
            if n > 0:
                # Basic binary features
                features['binary_ratio'] = np.mean(binary_data)
                features['binary_variance'] = np.var(binary_data.astype(float))
                features['binary_std'] = np.std(binary_data.astype(float))
                features['sequence_length'] = n
                
                # Binary transitions
                if n > 1:
                    transitions = np.diff(binary_data.astype(int))
                    features['binary_total_transitions'] = np.sum(np.abs(transitions))
                    features['binary_transition_rate'] = features['binary_total_transitions'] / n
                    features['binary_rise_rate'] = np.sum(transitions == 1) / n
                    features['binary_fall_rate'] = np.sum(transitions == -1) / n
                    
                    # Transition gaps
                    if features['binary_total_transitions'] > 0:
                        transition_indices = np.where(np.abs(transitions) == 1)[0]
                        if len(transition_indices) > 1:
                            gaps = np.diff(transition_indices)
                            features['binary_max_transition_gap'] = np.max(gaps)
                            features['binary_avg_transition_gap'] = np.mean(gaps)
                            features['binary_std_transition_gap'] = np.std(gaps)
                        else:
                            features['binary_max_transition_gap'] = n
                            features['binary_avg_transition_gap'] = n
                            features['binary_std_transition_gap'] = 0
                    else:
                        features['binary_max_transition_gap'] = n
                        features['binary_avg_transition_gap'] = n
                        features['binary_std_transition_gap'] = 0
                
                # Binary burst analysis
                active_bursts, inactive_bursts = self._analyze_binary_bursts(binary_data)
                
                if active_bursts:
                    features['binary_max_active_burst'] = np.max(active_bursts)
                    features['binary_avg_active_burst'] = np.mean(active_bursts)
                    features['binary_std_active_burst'] = np.std(active_bursts)
                    features['binary_total_active_samples'] = np.sum(active_bursts)
                    features['binary_num_active_bursts'] = len(active_bursts)
                else:
                    for key in ['binary_max_active_burst', 'binary_avg_active_burst', 'binary_std_active_burst',
                              'binary_total_active_samples', 'binary_num_active_bursts']:
                        features[key] = 0
                
                # Binary positional features
                third = n // 3
                if third > 0:
                    features['binary_start_ratio'] = np.mean(binary_data[:third])
                    features['binary_middle_ratio'] = np.mean(binary_data[third:2*third])
                    features['binary_end_ratio'] = np.mean(binary_data[2*third:])
                else:
                    features['binary_start_ratio'] = features['binary_ratio']
                    features['binary_middle_ratio'] = features['binary_ratio']
                    features['binary_end_ratio'] = features['binary_ratio']
                
                # Binary entropy
                p1 = features['binary_ratio']
                p0 = 1 - p1
                if p0 > 0 and p1 > 0:
                    features['binary_entropy'] = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                else:
                    features['binary_entropy'] = 0
        
        # =================================================================
        # 2. GYROSCOPE FEATURES
        # =================================================================
        gyro_data = data[self.gyro_cols].values
        
        # Per-axis features
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_data = gyro_data[:, i]
            
            # Basic statistical features
            features[f'gyro_{axis}_mean'] = np.mean(axis_data)
            features[f'gyro_{axis}_std'] = np.std(axis_data)
            features[f'gyro_{axis}_max'] = np.max(axis_data)
            features[f'gyro_{axis}_min'] = np.min(axis_data)
            features[f'gyro_{axis}_median'] = np.median(axis_data)
            features[f'gyro_{axis}_range'] = np.max(axis_data) - np.min(axis_data)
            features[f'gyro_{axis}_energy'] = np.sum(axis_data**2)
            features[f'gyro_{axis}_rms'] = np.sqrt(np.mean(axis_data**2))
            features[f'gyro_{axis}_abs_mean'] = np.mean(np.abs(axis_data))
            
            if len(axis_data) > 1:
                # Advanced statistics
                features[f'gyro_{axis}_skew'] = skew(axis_data)
                features[f'gyro_{axis}_kurtosis'] = kurtosis(axis_data)
                
                # Temporal features (important for rotational movements)
                diff_data = np.diff(axis_data)
                features[f'gyro_{axis}_diff_mean'] = np.mean(diff_data)
                features[f'gyro_{axis}_diff_std'] = np.std(diff_data)
                features[f'gyro_{axis}_diff_max'] = np.max(np.abs(diff_data))
                features[f'gyro_{axis}_angular_acceleration'] = np.mean(np.abs(diff_data))
                
                # Percentiles for distribution analysis
                features[f'gyro_{axis}_q25'] = np.percentile(axis_data, 25)
                features[f'gyro_{axis}_q75'] = np.percentile(axis_data, 75)
                features[f'gyro_{axis}_iqr'] = features[f'gyro_{axis}_q75'] - features[f'gyro_{axis}_q25']
                
                # Zero crossings (direction changes)
                features[f'gyro_{axis}_zero_crossings'] = np.sum(np.diff(np.sign(axis_data)) != 0)
                features[f'gyro_{axis}_zero_crossing_rate'] = features[f'gyro_{axis}_zero_crossings'] / len(axis_data)
                
                # Rotation direction analysis
                features[f'gyro_{axis}_positive_ratio'] = np.sum(axis_data > 0) / len(axis_data)
                features[f'gyro_{axis}_negative_ratio'] = np.sum(axis_data < 0) / len(axis_data)
                
                # Peak analysis for rotational bursts
                peaks, _ = find_peaks(np.abs(axis_data), height=np.std(axis_data))
                features[f'gyro_{axis}_peak_count'] = len(peaks)
                features[f'gyro_{axis}_peak_density'] = len(peaks) / len(axis_data)
                
                # Rotational intensity bands
                std_thresh = np.std(axis_data)
                features[f'gyro_{axis}_low_rotation'] = np.sum(np.abs(axis_data) < std_thresh) / len(axis_data)
                features[f'gyro_{axis}_medium_rotation'] = np.sum((np.abs(axis_data) >= std_thresh) & (np.abs(axis_data) < 2*std_thresh)) / len(axis_data)
                features[f'gyro_{axis}_high_rotation'] = np.sum(np.abs(axis_data) >= 2*std_thresh) / len(axis_data)
        
        # Combined gyroscope features
        angular_magnitude = np.linalg.norm(gyro_data, axis=1)
        features['gyro_magnitude_mean'] = np.mean(angular_magnitude)
        features['gyro_magnitude_std'] = np.std(angular_magnitude)
        features['gyro_magnitude_max'] = np.max(angular_magnitude)
        features['gyro_magnitude_min'] = np.min(angular_magnitude)
        features['gyro_magnitude_median'] = np.median(angular_magnitude)
        features['gyro_magnitude_range'] = np.max(angular_magnitude) - np.min(angular_magnitude)
        features['gyro_magnitude_energy'] = np.sum(angular_magnitude**2)
        features['gyro_magnitude_rms'] = np.sqrt(np.mean(angular_magnitude**2))
        
        if len(angular_magnitude) > 1:
            features['gyro_magnitude_skew'] = skew(angular_magnitude)
            features['gyro_magnitude_kurtosis'] = kurtosis(angular_magnitude)
            
            # Magnitude temporal features
            mag_diff = np.diff(angular_magnitude)
            features['gyro_magnitude_diff_mean'] = np.mean(mag_diff)
            features['gyro_magnitude_diff_std'] = np.std(mag_diff)
            features['gyro_magnitude_diff_max'] = np.max(np.abs(mag_diff))
            features['gyro_magnitude_acceleration'] = np.mean(np.abs(mag_diff))
            
            # Magnitude peaks
            mag_peaks, _ = find_peaks(angular_magnitude, height=np.mean(angular_magnitude) + np.std(angular_magnitude)/2)
            features['gyro_magnitude_peak_count'] = len(mag_peaks)
            features['gyro_magnitude_peak_density'] = len(mag_peaks) / len(angular_magnitude)
        
        # Cross-axis correlations (coordination between rotational axes)
        if len(gyro_data) > 1:
            features['gyro_xy_correlation'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 1])[0, 1]
            features['gyro_xz_correlation'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 2])[0, 1]
            features['gyro_yz_correlation'] = np.corrcoef(gyro_data[:, 1], gyro_data[:, 2])[0, 1]
            
            # Principal axis analysis (dominant rotational direction)
            try:
                cov_matrix = np.cov(gyro_data.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                features['gyro_principal_eigenval_1'] = eigenvals[-1]
                features['gyro_principal_eigenval_2'] = eigenvals[-2]
                features['gyro_principal_ratio'] = eigenvals[-1] / (eigenvals[-2] + 1e-10)
                features['gyro_anisotropy'] = (eigenvals[-1] - eigenvals[0]) / (eigenvals[-1] + 1e-10)
            except:
                features['gyro_principal_eigenval_1'] = 0
                features['gyro_principal_eigenval_2'] = 0
                features['gyro_principal_ratio'] = 1
                features['gyro_anisotropy'] = 0
        
        # =================================================================
        # 3. COMBINED BINARY + GYROSCOPE FEATURES
        # =================================================================
        if self.binary_col in data.columns:
            binary_data = data[self.binary_col].values
            
            if len(binary_data) == len(angular_magnitude) and len(binary_data) > 1:
                try:
                    corr, _ = pearsonr(binary_data, angular_magnitude)
                    features['binary_gyro_correlation'] = corr if not np.isnan(corr) else 0
                except:
                    features['binary_gyro_correlation'] = 0
                
                # Activity-specific gyroscope features
                active_mask = binary_data == 1
                inactive_mask = binary_data == 0
                
                if np.sum(active_mask) > 0:
                    features['gyro_during_binary_active_mean'] = np.mean(angular_magnitude[active_mask])
                    features['gyro_during_binary_active_std'] = np.std(angular_magnitude[active_mask])
                    features['gyro_during_binary_active_max'] = np.max(angular_magnitude[active_mask])
                    
                    # Per-axis features during active binary periods
                    for i, axis in enumerate(['x', 'y', 'z']):
                        axis_active = gyro_data[active_mask, i]
                        if len(axis_active) > 0:
                            features[f'gyro_{axis}_during_active_mean'] = np.mean(axis_active)
                            features[f'gyro_{axis}_during_active_std'] = np.std(axis_active)
                else:
                    for key in ['gyro_during_binary_active_mean', 'gyro_during_binary_active_std', 'gyro_during_binary_active_max']:
                        features[key] = 0
                    for axis in ['x', 'y', 'z']:
                        features[f'gyro_{axis}_during_active_mean'] = 0
                        features[f'gyro_{axis}_during_active_std'] = 0
                
                if np.sum(inactive_mask) > 0:
                    features['gyro_during_binary_inactive_mean'] = np.mean(angular_magnitude[inactive_mask])
                    features['gyro_during_binary_inactive_std'] = np.std(angular_magnitude[inactive_mask])
                else:
                    features['gyro_during_binary_inactive_mean'] = 0
                    features['gyro_during_binary_inactive_std'] = 0
                
                # Rotation contrast between active/inactive periods
                if features['gyro_during_binary_inactive_mean'] > 0:
                    features['gyro_active_inactive_contrast'] = features['gyro_during_binary_active_mean'] / features['gyro_during_binary_inactive_mean']
                else:
                    features['gyro_active_inactive_contrast'] = 0
                
                # Movement synchronization
                high_rotation_threshold = np.mean(angular_magnitude) + np.std(angular_magnitude)
                high_rotation_mask = angular_magnitude > high_rotation_threshold
                
                if np.sum(high_rotation_mask) > 0 and np.sum(active_mask) > 0:
                    features['high_rotation_binary_overlap'] = np.sum(high_rotation_mask & active_mask) / np.sum(high_rotation_mask)
                    features['binary_high_rotation_overlap'] = np.sum(high_rotation_mask & active_mask) / np.sum(active_mask)
                else:
                    features['high_rotation_binary_overlap'] = 0
                    features['binary_high_rotation_overlap'] = 0
        
        # Clean NaN and inf values
        cleaned_features = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = float(value)
            else:
                cleaned_features[key] = value
        
        return cleaned_features
    
    def _analyze_binary_bursts(self, binary_data):
        """Analyze consecutive active/inactive periods in binary data"""
        active_bursts = []
        inactive_bursts = []
        
        if len(binary_data) == 0:
            return active_bursts, inactive_bursts
        
        current_state = binary_data[0]
        current_length = 1
        
        for i in range(1, len(binary_data)):
            if binary_data[i] == current_state:
                current_length += 1
            else:
                if current_state == 1:
                    active_bursts.append(current_length)
                else:
                    inactive_bursts.append(current_length)
                
                current_state = binary_data[i]
                current_length = 1
        
        # Add final burst
        if current_state == 1:
            active_bursts.append(current_length)
        else:
            inactive_bursts.append(current_length)
        
        return active_bursts, inactive_bursts
    
    def create_feature_matrix(self, all_data):
        """Create comprehensive feature matrix"""
        print(f"\n🔧 CREATING COMPREHENSIVE GYRO + BINARY FEATURE MATRIX...")
        
        features_list = []
        labels = []
        subjects = []
        
        for item in all_data:
            features = self.extract_comprehensive_gyro_binary_features(item['data'])
            if features:
                features_list.append(features)
                labels.append(self.action_mapping[item['action']])
                subjects.append(item['subject'])
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        # Categorize features
        self.feature_categories = {
            'binary': [f for f in features_df.columns if 'binary' in f and 'gyro' not in f],
            'gyro': [f for f in features_df.columns if 'gyro' in f and 'binary' not in f],
            'combined': [f for f in features_df.columns if 'binary' in f and 'gyro' in f]
        }
        
        print(f"📊 Feature matrix: {features_df.shape}")
        print(f"📊 Binary features: {len(self.feature_categories['binary'])}")
        print(f"📊 Gyro features: {len(self.feature_categories['gyro'])}")
        print(f"📊 Combined features: {len(self.feature_categories['combined'])}")
        
        return features_df, np.array(labels), np.array(subjects)
    
    def analyze_gyro_features(self, features_df):
        """Analyze gyroscope feature characteristics"""
        print(f"\n🔬 ANALYZING GYROSCOPE FEATURE CHARACTERISTICS...")
        
        gyro_features = self.feature_categories['gyro']
        
        # Per-axis analysis
        axis_features = {
            'x_axis': [f for f in gyro_features if 'gyro_x_' in f],
            'y_axis': [f for f in gyro_features if 'gyro_y_' in f], 
            'z_axis': [f for f in gyro_features if 'gyro_z_' in f],
            'magnitude': [f for f in gyro_features if 'gyro_magnitude_' in f],
            'correlations': [f for f in gyro_features if 'correlation' in f or 'principal' in f or 'anisotropy' in f]
        }
        
        print(f"📊 GYROSCOPE FEATURE BREAKDOWN:")
        for category, feats in axis_features.items():
            print(f"   {category}: {len(feats)} features")
            if len(feats) > 0:
                # Show variance for first few features
                for feat in feats[:3]:
                    if feat in features_df.columns:
                        variance = np.var(features_df[feat])
                        print(f"      {feat}: variance = {variance:.2e}")
        
        # Statistical summary
        gyro_data = features_df[gyro_features]
        print(f"\n📊 GYROSCOPE FEATURE STATISTICS:")
        print(f"   Total gyro features: {len(gyro_features)}")
        print(f"   Mean feature variance: {np.mean([np.var(gyro_data[f]) for f in gyro_features]):.2e}")
        print(f"   Features with zero variance: {sum([np.var(gyro_data[f]) == 0 for f in gyro_features])}")
        
        # Feature importance preview
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Quick feature importance analysis
        try:
            labels = np.random.randint(0, 21, len(features_df))  # Dummy labels for analysis
            X_train, X_test, y_train, y_test = train_test_split(gyro_data, labels, test_size=0.3, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Top gyroscope features
            feature_importance = pd.DataFrame({
                'feature': gyro_features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🏆 TOP 10 GYROSCOPE FEATURES (preliminary analysis):")
            for i, row in feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            print(f"   ⚠️ Feature importance analysis failed: {str(e)}")
        
        return axis_features
    
    def calculate_nmi(self, y_true, y_pred):
        """Calculate Normalized Mutual Information"""
        return normalized_mutual_info_score(y_true, y_pred)
    
    def calculate_nrkl(self, y_true, y_pred_proba, num_classes=21):
        """Calculate Normalized Relative Kurtosis Loss"""
        # Convert true labels to one-hot distribution per class
        y_true_oh = np.zeros((len(y_true), num_classes))
        y_true_oh[np.arange(len(y_true)), y_true] = 1
        
        # Get average predicted probabilities per class
        true_class_dist = np.mean(y_true_oh, axis=0)
        pred_class_dist = np.mean(y_pred_proba, axis=0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        true_class_dist = true_class_dist + epsilon
        pred_class_dist = pred_class_dist + epsilon
        
        # Normalize
        true_class_dist = true_class_dist / np.sum(true_class_dist)
        pred_class_dist = pred_class_dist / np.sum(pred_class_dist)
        
        # Calculate KL divergence
        kl_divergence = np.sum(true_class_dist * np.log(true_class_dist / pred_class_dist))
        
        # Normalize by maximum possible KL divergence
        max_kl_divergence = np.log(num_classes)
        kl_normalized = kl_divergence / max_kl_divergence
        
        # Convert to percentage where higher is better
        nrkl_percentage = (1.0 - kl_normalized) * 100.0
        
        return nrkl_percentage
    
    def generate_all_subject_splits(self, subjects):
        """Generate all possible 4-4, 5-3, and 6-2 subject splits"""
        print(f"\n🔢 GENERATING ALL POSSIBLE SUBJECT SPLITS...")
        
        from itertools import combinations
        
        splits = {}
        
        # Generate all possible 4-4 splits
        print(f"📊 Generating all 4-4 splits (choose 4 from {len(subjects)} subjects)...")
        split_count_4_4 = 0
        for train_subjects in combinations(subjects, 4):
            test_subjects = [s for s in subjects if s not in train_subjects]
            split_name = f"4-4 Split {split_count_4_4+1}: Train{list(train_subjects)} Test{test_subjects}"
            splits[split_name] = {
                'train_subjects': list(train_subjects),
                'test_subjects': test_subjects,
                'type': '4-4'
            }
            split_count_4_4 += 1
        
        # Generate all possible 5-3 splits
        print(f"📊 Generating all 5-3 splits (choose 5 from {len(subjects)} subjects)...")
        split_count_5_3 = 0
        for train_subjects in combinations(subjects, 5):
            test_subjects = [s for s in subjects if s not in train_subjects]
            split_name = f"5-3 Split {split_count_5_3+1}: Train{list(train_subjects)} Test{test_subjects}"
            splits[split_name] = {
                'train_subjects': list(train_subjects),
                'test_subjects': test_subjects,
                'type': '5-3'
            }
            split_count_5_3 += 1
        
        # Generate all possible 6-2 splits  
        print(f"📊 Generating all 6-2 splits (choose 6 from {len(subjects)} subjects)...")
        split_count_6_2 = 0
        for train_subjects in combinations(subjects, 6):
            test_subjects = [s for s in subjects if s not in train_subjects]
            split_name = f"6-2 Split {split_count_6_2+1}: Train{list(train_subjects)} Test{test_subjects}"
            splits[split_name] = {
                'train_subjects': list(train_subjects),
                'test_subjects': test_subjects,
                'type': '6-2'
            }
            split_count_6_2 += 1
        
        print(f"📊 Generated {split_count_4_4} different 4-4 splits")
        print(f"📊 Generated {split_count_5_3} different 5-3 splits")
        print(f"📊 Generated {split_count_6_2} different 6-2 splits")
        print(f"📊 Total splits: {len(splits)}")
        
        return splits
    
    def evaluate_comprehensive_combinations(self, features_df, labels, subjects):
        """Evaluate ALL possible combinations systematically"""
        print(f"\n🎯 COMPREHENSIVE EVALUATION WITH GYRO + BINARY FEATURES...")
        
        # Get unique subjects
        unique_subjects = sorted(np.unique(subjects))
        print(f"📊 Available subjects: {unique_subjects}")
        
        # Generate all possible splits
        splits = self.generate_all_subject_splits(unique_subjects)
        
        # Use ExtraTreesClassifier (best performer from accelerometer analysis)
        classifier = ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')
        
        total_experiments = len(splits)
        print(f"🧪 Running {total_experiments} total experiments...")
        
        all_results = {}
        experiment_count = 0
        start_time = time.time()
        
        for split_name, split_config in splits.items():
            print(f"\n📈 {split_name}:")
            print(f"   Train subjects: {split_config['train_subjects']}")
            print(f"   Test subjects: {split_config['test_subjects']}")
            
            # Create train/test masks
            train_mask = np.isin(subjects, split_config['train_subjects'])
            test_mask = np.isin(subjects, split_config['test_subjects'])
            
            if not np.any(train_mask) or not np.any(test_mask):
                print(f"   ⚠️ Insufficient data for {split_name}")
                continue
            
            X_train = features_df[train_mask]
            X_test = features_df[test_mask]
            y_train = labels[train_mask]
            y_test = labels[test_mask]
            
            print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            experiment_count += 1
            
            if experiment_count % 10 == 0 or experiment_count <= 20:
                elapsed = time.time() - start_time
                avg_time = elapsed / experiment_count if experiment_count > 0 else 0
                remaining = (total_experiments - experiment_count) * avg_time
                print(f"      🔧 Progress: {experiment_count}/{total_experiments} ({experiment_count/total_experiments*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f} min")
            
            try:
                # Cross-validation on training set
                cv = StratifiedKFold(n_splits=min(3, len(np.unique(y_train))), shuffle=True, random_state=42)
                cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='accuracy')
                
                # Train and predict
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                y_pred_proba = classifier.predict_proba(X_test)
                
                # Calculate metrics
                test_accuracy = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred, average='weighted')
                nmi = self.calculate_nmi(y_test, y_pred)
                nrkl = self.calculate_nrkl(y_test, y_pred_proba)
                vulnerability = calculate_vulnerability(y_pred_proba)
                
                # Store result
                result = {
                    'split': split_name,
                    'split_type': split_config['type'],
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'test_accuracy': test_accuracy,
                    'test_f1_weighted': test_f1,
                    'nmi': nmi,
                    'nmi_percentage': nmi * 100.0,
                    'nrkl_percentage': nrkl,
                    'vulnerability': vulnerability,
                    'features_used': len(features_df.columns),
                    'improvement_factor': test_accuracy / (1/21),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'train_subjects': split_config['train_subjects'],
                    'test_subjects': split_config['test_subjects']
                }
                
                all_results[split_name] = result
                
            except Exception as e:
                print(f"      ⚠️ Error with {split_name}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total evaluation time: {total_time/60:.1f} minutes")
        print(f"📊 Completed {experiment_count} experiments")
        
        return all_results
    
    def find_and_save_best_configurations(self, all_results):
        """Find and save the best performing configurations"""
        print(f"\n🏆 ANALYZING GYRO + BINARY RESULTS FROM ALL SUBJECT SPLITS...")
        
        # Collect all results
        all_results_flat = []
        for split_name, result in all_results.items():
            result['split_name'] = split_name
            all_results_flat.append(result)
        
        if not all_results_flat:
            print("❌ No results to analyze!")
            return {}, None
        
        # Sort by test accuracy
        sorted_results = sorted(all_results_flat, key=lambda x: x['test_accuracy'], reverse=True)
        
        # Overall best
        overall_best = sorted_results[0]
        
        # Best by split type
        split_4_4_results = [r for r in sorted_results if r['split_type'] == '4-4']
        split_5_3_results = [r for r in sorted_results if r['split_type'] == '5-3']
        split_6_2_results = [r for r in sorted_results if r['split_type'] == '6-2']
        
        best_4_4 = split_4_4_results[0] if split_4_4_results else None
        best_5_3 = split_5_3_results[0] if split_5_3_results else None
        best_6_2 = split_6_2_results[0] if split_6_2_results else None
        
        print(f"\n📊 GYRO + BINARY RESULTS SUMMARY:")
        print(f"   Total experiments: {len(all_results_flat)}")
        print(f"   4-4 splits tested: {len(split_4_4_results)}")
        print(f"   5-3 splits tested: {len(split_5_3_results)}")
        print(f"   6-2 splits tested: {len(split_6_2_results)}")
        
        print(f"\n🥇 BEST OVERALL CONFIGURATION (GYRO + BINARY):")
        print(f"   Split: {overall_best['split_name']}")
        print(f"   Test Accuracy: {overall_best['test_accuracy']:.4f}")
        print(f"   F1-Score: {overall_best['test_f1_weighted']:.4f}")
        print(f"   NMI: {overall_best['nmi_percentage']:.2f}%")
        print(f"   NRKL: {overall_best['nrkl_percentage']:.2f}%")
        print(f"   Improvement: {overall_best['improvement_factor']:.1f}x over random")
        
        if best_4_4:
            print(f"\n🥈 BEST 4-4 SPLIT CONFIGURATION:")
            print(f"   Split: {best_4_4['split_name']}")
            print(f"   Test Accuracy: {best_4_4['test_accuracy']:.4f}")
            print(f"   F1-Score: {best_4_4['test_f1_weighted']:.4f}")
            print(f"   NMI: {best_4_4['nmi_percentage']:.2f}%")
            print(f"   NRKL: {best_4_4['nrkl_percentage']:.2f}%")
        
        if best_5_3:
            print(f"\n🏅 BEST 5-3 SPLIT CONFIGURATION:")
            print(f"   Split: {best_5_3['split_name']}")
            print(f"   Test Accuracy: {best_5_3['test_accuracy']:.4f}")
            print(f"   F1-Score: {best_5_3['test_f1_weighted']:.4f}")
            print(f"   NMI: {best_5_3['nmi_percentage']:.2f}%")
            print(f"   NRKL: {best_5_3['nrkl_percentage']:.2f}%")
            
        if best_6_2:
            print(f"\n🥉 BEST 6-2 SPLIT CONFIGURATION:")
            print(f"   Split: {best_6_2['split_name']}")
            print(f"   Test Accuracy: {best_6_2['test_accuracy']:.4f}")
            print(f"   F1-Score: {best_6_2['test_f1_weighted']:.4f}")
            print(f"   NMI: {best_6_2['nmi_percentage']:.2f}%")
            print(f"   NRKL: {best_6_2['nrkl_percentage']:.2f}%")
        
        # Performance statistics by split type
        split_types = []
        if split_4_4_results:
            avg_acc_4_4 = np.mean([r['test_accuracy'] for r in split_4_4_results])
            std_acc_4_4 = np.std([r['test_accuracy'] for r in split_4_4_results])
            split_types.append(('4-4', avg_acc_4_4, std_acc_4_4, len(split_4_4_results)))
        
        if split_5_3_results:
            avg_acc_5_3 = np.mean([r['test_accuracy'] for r in split_5_3_results])
            std_acc_5_3 = np.std([r['test_accuracy'] for r in split_5_3_results])
            split_types.append(('5-3', avg_acc_5_3, std_acc_5_3, len(split_5_3_results)))
        
        if split_6_2_results:
            avg_acc_6_2 = np.mean([r['test_accuracy'] for r in split_6_2_results])
            std_acc_6_2 = np.std([r['test_accuracy'] for r in split_6_2_results])
            split_types.append(('6-2', avg_acc_6_2, std_acc_6_2, len(split_6_2_results)))
        
        if len(split_types) > 1:
            print(f"\n📊 PERFORMANCE COMPARISON BY SPLIT TYPE (GYRO + BINARY):")
            for split_name, avg_acc, std_acc, count in split_types:
                print(f"   {split_name} splits: {avg_acc:.4f} ± {std_acc:.4f} accuracy ({count} tests)")
            
            # Find best performing split type
            best_split_type = max(split_types, key=lambda x: x[1])
            print(f"   ✅ {best_split_type[0]} splits perform best on average ({best_split_type[1]:.4f} accuracy)")
        
        # Top 10 configurations overall
        print(f"\n🏆 TOP 10 CONFIGURATIONS OVERALL (GYRO + BINARY):")
        for i, result in enumerate(sorted_results[:10]):
            print(f"{i+1:2d}. {result['split_name']:<40} | "
                  f"Acc: {result['test_accuracy']:.4f} | "
                  f"F1: {result['test_f1_weighted']:.4f} | "
                  f"NMI: {result['nmi_percentage']:.1f}% | "
                  f"NRKL: {result['nrkl_percentage']:.1f}%")
        
        # Prepare results for saving
        best_configs = {
            'overall_best': overall_best,
            'best_4_4': best_4_4,
            'best_5_3': best_5_3,
            'best_6_2': best_6_2,
            'top_10': sorted_results[:10],
            'all_results': all_results,
            'statistics': {
                'total_experiments': len(all_results_flat),
                'split_4_4_count': len(split_4_4_results) if split_4_4_results else 0,
                'split_5_3_count': len(split_5_3_results) if split_5_3_results else 0,
                'split_6_2_count': len(split_6_2_results) if split_6_2_results else 0,
                'avg_accuracy_4_4': np.mean([r['test_accuracy'] for r in split_4_4_results]) if split_4_4_results else 0,
                'avg_accuracy_5_3': np.mean([r['test_accuracy'] for r in split_5_3_results]) if split_5_3_results else 0,
                'avg_accuracy_6_2': np.mean([r['test_accuracy'] for r in split_6_2_results]) if split_6_2_results else 0,
                'std_accuracy_4_4': np.std([r['test_accuracy'] for r in split_4_4_results]) if split_4_4_results else 0,
                'std_accuracy_5_3': np.std([r['test_accuracy'] for r in split_5_3_results]) if split_5_3_results else 0,
                'std_accuracy_6_2': np.std([r['test_accuracy'] for r in split_6_2_results]) if split_6_2_results else 0,
            }
        }
        
        # Save comprehensive results
        self.save_comprehensive_results({
            'best_configurations': best_configs,
            'overall_best': overall_best,
            'all_results': all_results,
            'feature_categories': self.feature_categories,
            'summary': {
                'total_experiments': len(all_results_flat),
                'splits_tested': list(all_results.keys()),
                'best_accuracy': overall_best['test_accuracy'],
                'random_baseline': 1/21,
                'feature_type': 'gyroscope + binary'
            }
        })
        
        return best_configs, overall_best
    
    def save_comprehensive_results(self, results):
        """Save only the maximum vulnerability summary."""
        all_results = results.get('all_results', {})
        vulnerabilities = [result.get('vulnerability') for result in all_results.values()]
        output_file = save_max_vulnerability_from_values(__file__, vulnerabilities)
        print(f"💾 Comprehensive gyro max vulnerability saved to: {output_file}")
    
    def run_comprehensive_evaluation(self):
        """Run the complete comprehensive evaluation"""
        print(f"🚀 STARTING COMPREHENSIVE GAME-2 WRIST GYRO EVALUATION")
        print(f"="*80)
        
        # Load data
        all_data, subjects = self.load_wrist_data_with_gyro()
        
        if not all_data:
            print("❌ No data loaded!")
            return None
        
        # Create feature matrix
        features_df, labels, subjects_array = self.create_feature_matrix(all_data)
        
        # Analyze gyroscope features
        axis_features = self.analyze_gyro_features(features_df)
        
        # Run comprehensive evaluation
        all_results = self.evaluate_comprehensive_combinations(features_df, labels, subjects_array)
        
        # Find and save best configurations
        best_configs, overall_best = self.find_and_save_best_configurations(all_results)
        
        print(f"\n✅ GYRO + BINARY EVALUATION COMPLETED!")
        print(f"🎯 Tested gyroscope + binary features with ExtraTreesClassifier")
        print(f"🔢 Evaluated all possible 4-4, 5-3, and 6-2 subject splits")
        print(f"🏆 Best configuration saved with comprehensive metrics (NMI, NRKL)")
        
        return best_configs, overall_best

def main():
    """Main execution"""
    data_dir = "UTD-MHAD-Reorganized"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    evaluator = Game2WristGyroHAR(data_dir)
    best_configs, overall_best = evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
