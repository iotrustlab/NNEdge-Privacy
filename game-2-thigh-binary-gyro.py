#!/usr/bin/env python3
"""
🎯 GAME-2 THIGH HAR ATTACK: BINARY + GYROSCOPE FEATURES
======================================================

Building on the successful Game-1 thigh attack (70.2% accuracy with binary features only),
this Game-2 implementation combines:
- Binary decision tree features (proven effective for thigh placement)
- Rich gyroscope features (gyro_x, gyro_y, gyro_z)

Target: 6 thigh actions from UTD-MHAD
- Action 22: Right kick
- Action 23: Left kick  
- Action 24: Sit to stand
- Action 25: Stand to sit
- Action 26: Forward lunge
- Action 27: Squat

Comparison study: How do gyroscope features compare to accelerometer features?
Expected: Different characteristics due to rotational vs linear motion sensing
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import json
from datetime import datetime
from itertools import combinations
from utd_vulnerability_utils import calculate_vulnerability, save_max_vulnerability_from_values
from scipy import stats
from scipy.spatial.distance import jensenshannon
warnings.filterwarnings('ignore')

class Game2ThighGyroHARAttack:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # Target thigh actions (same as successful Game-1)
        self.thigh_actions = {
            22: "Right kick",
            23: "Left kick", 
            24: "Sit to stand",
            25: "Stand to sit",
            26: "Forward lunge",
            27: "Squat"
        }
        
        # Map to 0-5 for 6-class classification
        self.action_mapping = {22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5}
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Expected gyroscope columns in UTD-MHAD
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        
        print(f"🎯 GAME-2 THIGH GYRO HAR ATTACK INITIALIZED")
        print(f"📊 6 thigh actions with binary + gyroscope features")
        print(f"🔄 Expected: Different patterns than accelerometer due to rotational sensing")
        
    def load_thigh_data(self):
        """Load thigh placement data for target actions"""
        print(f"\\n📂 LOADING THIGH DATA FOR GYRO ANALYSIS...")
        
        all_files = []
        action_counts = defaultdict(int)
        
        # Get all CSV files from right-thigh directories for target actions
        csv_pattern = os.path.join(self.data_dir, "**", "right-thigh", "Action_*", "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        for file_path in csv_files:
            try:
                # Extract action from path
                path_parts = file_path.split(os.sep)
                action_dir = None
                for part in path_parts:
                    if part.startswith('Action_'):
                        action_dir = part
                        break
                
                if action_dir:
                    action_num = int(action_dir.split('_')[1])
                    
                    # Only include target thigh actions
                    if action_num in self.thigh_actions:
                        all_files.append(file_path)
                        action_counts[action_num] += 1
                            
            except (ValueError, IndexError):
                continue
        
        print(f"📊 THIGH GYRO DATA SUMMARY:")
        total_files = 0
        for action_num in sorted(self.thigh_actions.keys()):
            count = action_counts[action_num]
            total_files += count
            print(f"   Action {action_num}: {self.thigh_actions[action_num]:<20} ({count:2d} files)")
        
        print(f"📈 Total thigh files: {total_files}")
        return all_files
    
    def extract_comprehensive_gyro_features(self, data):
        """Extract both binary and gyroscope features"""
        features = {}
        
        if len(data) == 0:
            return {}
        
        # =================================================================
        # 1. BINARY FEATURES (from successful Game-1 thigh attack)
        # =================================================================
        if self.binary_col in data.columns:
            binary_data = data[self.binary_col].values
            n = len(binary_data)
            
            if n > 0:
                # Basic binary features
                features['binary_activity_ratio'] = np.mean(binary_data)
                features['binary_total_samples'] = n
                features['binary_active_samples'] = np.sum(binary_data)
                features['binary_inactive_samples'] = n - np.sum(binary_data)
                
                # Binary transitions (key for thigh movements)
                if n > 1:
                    transitions = np.sum(np.abs(np.diff(binary_data.astype(int))))
                    features['binary_transitions'] = transitions
                    features['binary_transition_rate'] = transitions / n
                    features['binary_transition_density'] = transitions / max(1, np.sum(binary_data))
                else:
                    features['binary_transitions'] = 0
                    features['binary_transition_rate'] = 0
                    features['binary_transition_density'] = 0
                
                # Binary burst analysis (effective for thigh actions)
                active_bursts, inactive_bursts = self._analyze_binary_bursts(binary_data)
                
                if active_bursts:
                    features['binary_active_burst_count'] = len(active_bursts)
                    features['binary_active_burst_mean'] = np.mean(active_bursts)
                    features['binary_active_burst_std'] = np.std(active_bursts)
                    features['binary_active_burst_max'] = max(active_bursts)
                    features['binary_active_burst_total'] = sum(active_bursts)
                else:
                    features['binary_active_burst_count'] = 0
                    features['binary_active_burst_mean'] = 0
                    features['binary_active_burst_std'] = 0
                    features['binary_active_burst_max'] = 0
                    features['binary_active_burst_total'] = 0
                
                if inactive_bursts:
                    features['binary_inactive_burst_count'] = len(inactive_bursts)
                    features['binary_inactive_burst_mean'] = np.mean(inactive_bursts)
                    features['binary_inactive_burst_max'] = max(inactive_bursts)
                else:
                    features['binary_inactive_burst_count'] = 0
                    features['binary_inactive_burst_mean'] = 0
                    features['binary_inactive_burst_max'] = 0
                
                # Binary temporal positioning (start/middle/end)
                third = n // 3
                if third > 0:
                    features['binary_start_activity'] = np.mean(binary_data[:third])
                    features['binary_middle_activity'] = np.mean(binary_data[third:2*third])
                    features['binary_end_activity'] = np.mean(binary_data[2*third:])
                    features['binary_start_end_diff'] = features['binary_end_activity'] - features['binary_start_activity']
                else:
                    features['binary_start_activity'] = features['binary_activity_ratio']
                    features['binary_middle_activity'] = features['binary_activity_ratio']
                    features['binary_end_activity'] = features['binary_activity_ratio']
                    features['binary_start_end_diff'] = 0
                
                # Binary entropy
                if n > 0:
                    p1 = np.mean(binary_data)
                    p0 = 1 - p1
                    if p0 > 0 and p1 > 0:
                        features['binary_entropy'] = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                    else:
                        features['binary_entropy'] = 0
                else:
                    features['binary_entropy'] = 0
        
        # =================================================================
        # 2. GYROSCOPE FEATURES (Game-2 enhancement)
        # =================================================================
        
        # Check which gyroscope columns are available
        available_gyro_cols = [col for col in self.gyro_cols if col in data.columns]
        
        if available_gyro_cols:
            # Per-axis gyroscope features
            for axis_col in available_gyro_cols:
                axis_data = data[axis_col].values
                axis_name = axis_col.replace('[mdps]', '').replace('gyro_', '')
                
                if len(axis_data) > 0:
                    # Basic statistical features
                    features[f'gyro_{axis_name}_mean'] = np.mean(axis_data)
                    features[f'gyro_{axis_name}_std'] = np.std(axis_data)
                    features[f'gyro_{axis_name}_max'] = np.max(axis_data)
                    features[f'gyro_{axis_name}_min'] = np.min(axis_data)
                    features[f'gyro_{axis_name}_median'] = np.median(axis_data)
                    features[f'gyro_{axis_name}_range'] = np.max(axis_data) - np.min(axis_data)
                    features[f'gyro_{axis_name}_energy'] = np.sum(axis_data**2)
                    features[f'gyro_{axis_name}_abs_mean'] = np.mean(np.abs(axis_data))
                    
                    if len(axis_data) > 1:
                        # Advanced statistics
                        features[f'gyro_{axis_name}_skew'] = pd.Series(axis_data).skew()
                        features[f'gyro_{axis_name}_kurtosis'] = pd.Series(axis_data).kurtosis()
                        features[f'gyro_{axis_name}_rms'] = np.sqrt(np.mean(axis_data**2))
                        
                        # Temporal features (important for rotational movement patterns)
                        diff_data = np.diff(axis_data)
                        features[f'gyro_{axis_name}_diff_mean'] = np.mean(diff_data)
                        features[f'gyro_{axis_name}_diff_std'] = np.std(diff_data)
                        features[f'gyro_{axis_name}_diff_max'] = np.max(np.abs(diff_data))
                        features[f'gyro_{axis_name}_diff_energy'] = np.sum(diff_data**2)
                        
                        # Percentiles
                        features[f'gyro_{axis_name}_q25'] = np.percentile(axis_data, 25)
                        features[f'gyro_{axis_name}_q75'] = np.percentile(axis_data, 75)
                        features[f'gyro_{axis_name}_iqr'] = features[f'gyro_{axis_name}_q75'] - features[f'gyro_{axis_name}_q25']
                        
                        # Zero crossings (rotational direction changes - very important for gyro!)
                        features[f'gyro_{axis_name}_zero_crossings'] = np.sum(np.diff(np.sign(axis_data)) != 0)
                        features[f'gyro_{axis_name}_zero_crossing_rate'] = features[f'gyro_{axis_name}_zero_crossings'] / len(axis_data)
                        
                        # Peak analysis (important for rotational movements)
                        features[f'gyro_{axis_name}_peak_count'] = len(self._find_peaks(axis_data))
                        
                        # Activity bands (rotational intensity levels)
                        std_thresh = np.std(axis_data)
                        features[f'gyro_{axis_name}_low_rotation'] = np.sum(np.abs(axis_data) < std_thresh) / len(axis_data)
                        features[f'gyro_{axis_name}_high_rotation'] = np.sum(np.abs(axis_data) > 2 * std_thresh) / len(axis_data)
                        
                        # Rotational velocity characteristics
                        features[f'gyro_{axis_name}_positive_ratio'] = np.sum(axis_data > 0) / len(axis_data)
                        features[f'gyro_{axis_name}_negative_ratio'] = np.sum(axis_data < 0) / len(axis_data)
                        
                        # Rotational acceleration (second derivative)
                        if len(diff_data) > 1:
                            accel_data = np.diff(diff_data)
                            features[f'gyro_{axis_name}_angular_accel_mean'] = np.mean(accel_data)
                            features[f'gyro_{axis_name}_angular_accel_std'] = np.std(accel_data)
                            features[f'gyro_{axis_name}_angular_accel_max'] = np.max(np.abs(accel_data))
                        else:
                            features[f'gyro_{axis_name}_angular_accel_mean'] = 0
                            features[f'gyro_{axis_name}_angular_accel_std'] = 0
                            features[f'gyro_{axis_name}_angular_accel_max'] = 0
            
            # Combined gyroscope features (if all 3 axes available)
            if len(available_gyro_cols) == 3:
                gyro_data = data[available_gyro_cols].values
                
                # Angular magnitude features (total rotational intensity)
                angular_magnitude = np.linalg.norm(gyro_data, axis=1)
                features['gyro_angular_magnitude_mean'] = np.mean(angular_magnitude)
                features['gyro_angular_magnitude_std'] = np.std(angular_magnitude)
                features['gyro_angular_magnitude_max'] = np.max(angular_magnitude)
                features['gyro_angular_magnitude_min'] = np.min(angular_magnitude)
                features['gyro_angular_magnitude_median'] = np.median(angular_magnitude)
                features['gyro_angular_magnitude_range'] = np.max(angular_magnitude) - np.min(angular_magnitude)
                features['gyro_angular_magnitude_energy'] = np.sum(angular_magnitude**2)
                features['gyro_angular_magnitude_rms'] = np.sqrt(np.mean(angular_magnitude**2))
                
                if len(angular_magnitude) > 1:
                    features['gyro_angular_magnitude_skew'] = pd.Series(angular_magnitude).skew()
                    features['gyro_angular_magnitude_kurtosis'] = pd.Series(angular_magnitude).kurtosis()
                    
                    # Angular magnitude temporal features
                    ang_mag_diff = np.diff(angular_magnitude)
                    features['gyro_angular_magnitude_diff_mean'] = np.mean(ang_mag_diff)
                    features['gyro_angular_magnitude_diff_std'] = np.std(ang_mag_diff)
                    features['gyro_angular_magnitude_diff_max'] = np.max(np.abs(ang_mag_diff))
                    
                    # Angular magnitude peaks (key for detecting rotation bursts)
                    features['gyro_angular_magnitude_peak_count'] = len(self._find_peaks(angular_magnitude))
                    features['gyro_angular_magnitude_peak_prominence'] = np.mean(self._peak_prominence(angular_magnitude))
                
                # Cross-axis correlation features (coordination between rotational axes)
                gyro_x = gyro_data[:, 0]
                gyro_y = gyro_data[:, 1] 
                gyro_z = gyro_data[:, 2]
                
                features['gyro_xy_correlation'] = np.corrcoef(gyro_x, gyro_y)[0, 1] if len(gyro_x) > 1 else 0
                features['gyro_xz_correlation'] = np.corrcoef(gyro_x, gyro_z)[0, 1] if len(gyro_x) > 1 else 0
                features['gyro_yz_correlation'] = np.corrcoef(gyro_y, gyro_z)[0, 1] if len(gyro_y) > 1 else 0
                
                # Principal axis analysis (dominant rotational direction)
                try:
                    if len(gyro_data) > 3:
                        cov_matrix = np.cov(gyro_data.T)
                        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                        features['gyro_principal_eigenval_1'] = eigenvals[-1]  # Largest eigenvalue
                        features['gyro_principal_eigenval_2'] = eigenvals[-2]  # Second largest
                        features['gyro_principal_ratio'] = eigenvals[-1] / (eigenvals[-2] + 1e-10)  # Anisotropy
                    else:
                        features['gyro_principal_eigenval_1'] = 0
                        features['gyro_principal_eigenval_2'] = 0
                        features['gyro_principal_ratio'] = 1
                except:
                    features['gyro_principal_eigenval_1'] = 0
                    features['gyro_principal_eigenval_2'] = 0
                    features['gyro_principal_ratio'] = 1
                
                # Rotational complexity features
                features['gyro_total_rotation'] = np.sum(np.abs(gyro_data))
                features['gyro_mean_rotation_per_axis'] = features['gyro_total_rotation'] / 3
        
        # =================================================================
        # 3. COMBINED BINARY + GYROSCOPE FEATURES (synergy)
        # =================================================================
        
        # Correlation between binary activity and gyroscope magnitude
        if self.binary_col in data.columns and len(available_gyro_cols) == 3:
            binary_data = data[self.binary_col].values
            gyro_data = data[available_gyro_cols].values
            angular_magnitude = np.linalg.norm(gyro_data, axis=1)
            
            if len(binary_data) == len(angular_magnitude) and len(binary_data) > 1:
                features['binary_gyro_correlation'] = np.corrcoef(binary_data, angular_magnitude)[0, 1]
                
                # Activity-specific gyroscope features
                active_mask = binary_data == 1
                inactive_mask = binary_data == 0
                
                if np.sum(active_mask) > 0:
                    features['gyro_during_binary_active_mean'] = np.mean(angular_magnitude[active_mask])
                    features['gyro_during_binary_active_std'] = np.std(angular_magnitude[active_mask])
                    features['gyro_during_binary_active_max'] = np.max(angular_magnitude[active_mask])
                else:
                    features['gyro_during_binary_active_mean'] = 0
                    features['gyro_during_binary_active_std'] = 0
                    features['gyro_during_binary_active_max'] = 0
                
                if np.sum(inactive_mask) > 0:
                    features['gyro_during_binary_inactive_mean'] = np.mean(angular_magnitude[inactive_mask])
                    features['gyro_during_binary_inactive_std'] = np.std(angular_magnitude[inactive_mask])
                else:
                    features['gyro_during_binary_inactive_mean'] = 0
                    features['gyro_during_binary_inactive_std'] = 0
                
                # Rotational contrast ratio
                if features['gyro_during_binary_inactive_mean'] > 0:
                    features['gyro_active_inactive_ratio'] = features['gyro_during_binary_active_mean'] / features['gyro_during_binary_inactive_mean']
                else:
                    features['gyro_active_inactive_ratio'] = 0
            else:
                features['binary_gyro_correlation'] = 0
        
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
        
        current_state = None
        current_length = 0
        
        for val in binary_data:
            if val == current_state:
                current_length += 1
            else:
                if current_state is not None:
                    if current_state == 1:
                        active_bursts.append(current_length)
                    else:
                        inactive_bursts.append(current_length)
                
                current_state = val
                current_length = 1
        
        # Add final burst
        if current_state is not None:
            if current_state == 1:
                active_bursts.append(current_length)
            else:
                inactive_bursts.append(current_length)
        
        return active_bursts, inactive_bursts
    
    def _find_peaks(self, data, min_height_ratio=0.1):
        """Find peaks in data (simple peak detection)"""
        if len(data) < 3:
            return []
        
        peaks = []
        threshold = np.std(data) * min_height_ratio
        
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and 
                abs(data[i]) > threshold):  # Use abs for gyro data
                peaks.append(i)
        
        return peaks
    
    def _peak_prominence(self, data):
        """Calculate prominence of peaks"""
        peaks = self._find_peaks(data)
        if len(peaks) == 0:
            return [0]
        
        prominences = []
        for peak_idx in peaks:
            # Simple prominence calculation
            left_min = np.min(np.abs(data[:peak_idx])) if peak_idx > 0 else abs(data[peak_idx])
            right_min = np.min(np.abs(data[peak_idx+1:])) if peak_idx < len(data)-1 else abs(data[peak_idx])
            prominence = abs(data[peak_idx]) - max(left_min, right_min)
            prominences.append(prominence)
        
        return prominences if prominences else [0]
    
    def calculate_nmi(self, y_true, y_pred):
        """Calculate Normalized Mutual Information"""
        return normalized_mutual_info_score(y_true, y_pred)
    
    def calculate_nrkl(self, y_true, y_pred, num_classes=6):
        """Calculate Normalized KL Divergence (NRKL)"""
        # Get probability distributions
        true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
        pred_dist = np.bincount(y_pred, minlength=num_classes) / len(y_pred)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        true_dist = true_dist + epsilon
        pred_dist = pred_dist + epsilon
        
        # Calculate KL divergence using Jensen-Shannon distance as symmetric measure
        js_distance = jensenshannon(true_dist, pred_dist)
        
        # Normalize to [0,1] range
        nrkl = 1 - js_distance
        
        return nrkl
    
    def process_files_for_game2_gyro(self, file_list):
        """Process all files and extract comprehensive gyroscope features"""
        print(f"\\n🔄 EXTRACTING BINARY + GYROSCOPE FEATURES FROM {len(file_list)} FILES...")
        
        features_list = []
        action_labels = []
        file_info = []
        processed = 0
        
        for file_path in file_list:
            try:
                # Load data
                data = pd.read_csv(file_path)
                
                # Extract action number from path
                path_parts = file_path.split(os.sep)
                action_dir = None
                for part in path_parts:
                    if part.startswith('Action_'):
                        action_dir = part
                        break
                
                if not action_dir:
                    continue
                    
                action_num = int(action_dir.split('_')[1])
                
                # Skip if not target action
                if action_num not in self.thigh_actions:
                    continue
                
                # Extract comprehensive gyroscope features
                features = self.extract_comprehensive_gyro_features(data)
                
                if features:
                    features_list.append(features)
                    action_labels.append(self.action_mapping[action_num])  # Map to 0-5
                    
                    # Extract subject information
                    subject_dir = None
                    for part in path_parts:
                        if part.startswith('Subject'):
                            subject_dir = part
                            break
                    
                    subject_num = int(subject_dir.replace('Subject', '')) if subject_dir else 0
                    
                    filename = os.path.basename(file_path)
                    file_info.append({
                        'file': filename,
                        'action': action_num,
                        'action_name': self.thigh_actions[action_num],
                        'action_class': self.action_mapping[action_num],
                        'subject': subject_num,
                        'path': file_path
                    })
                    processed += 1
                    
                    if processed % 50 == 0:
                        print(f"   Processed {processed}/{len(file_list)} files...")
                        
            except Exception as e:
                continue
        
        print(f"✅ Successfully processed {processed} files")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        action_labels = np.array(action_labels)
        
        print(f"📊 Feature matrix shape: {features_df.shape}")
        print(f"🏷️ Action labels shape: {action_labels.shape}")
        
        return features_df, action_labels, file_info
    
    def analyze_gyro_feature_importance(self, features_df, action_labels):
        """Analyze which gyroscope features are most important"""
        print(f"\\n🔬 ANALYZING GYROSCOPE FEATURE IMPORTANCE...")
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Train a simple classifier to get feature importance
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features_df, action_labels)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Analyze feature categories
        binary_features = feature_importance[feature_importance['feature'].str.contains('binary')]
        gyro_features = feature_importance[feature_importance['feature'].str.contains('gyro')]
        
        print(f"📊 GYROSCOPE FEATURE ANALYSIS SUMMARY:")
        print(f"   Total features: {len(features_df.columns)}")
        print(f"   Binary features: {len(binary_features)}")
        print(f"   Gyroscope features: {len(gyro_features)}")
        
        print(f"\\n🔥 TOP 15 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            feature_type = "🎯 BINARY" if 'binary' in row['feature'] else "🌀 GYRO"
            print(f"{i+1:2d}. {feature_type} {row['feature']:<35} {row['importance']:.4f}")
        
        print(f"\\n📊 BINARY vs GYROSCOPE FEATURE IMPORTANCE:")
        binary_total = binary_features['importance'].sum()
        gyro_total = gyro_features['importance'].sum()
        total_importance = binary_total + gyro_total
        
        if total_importance > 0:
            binary_pct = (binary_total / total_importance) * 100
            gyro_pct = (gyro_total / total_importance) * 100
            print(f"   Binary features:     {binary_pct:.1f}% of total importance")
            print(f"   Gyroscope features:  {gyro_pct:.1f}% of total importance")
        
        return feature_importance
    
    def train_game2_gyro_classifier(self, features_df, action_labels, file_info):
        """Train Game-2 gyroscope classifier with comprehensive evaluation"""
        print(f"\\n🤖 TRAINING GAME-2 THIGH GYRO CLASSIFIER WITH ROBUST EVALUATION...")
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Get feature importance analysis
        feature_importance = self.analyze_gyro_feature_importance(features_df, action_labels)
        
        # Subject-based evaluation (like successful Game-1)
        subjects = sorted(list(set([info['subject'] for info in file_info])))
        print(f"\\n👥 Available subjects: {subjects}")
        
        # Define split configurations like Game-1
        split_configs = {
            '6_2': {
                'train_size': 6,
                'test_size': 2,
                'description': '6 subjects train, 2 subjects test'
            },
            '4_4': {
                'train_size': 4, 
                'test_size': 4,
                'description': '4 subjects train, 4 subjects test'
            }
        }
        
        # Generate all possible combinations for each split
        all_results = {}
        
        for split_name, config in split_configs.items():
            print(f"\\n🧪 EVALUATING {split_name} SPLIT: {config['description']}")
            
            train_size = config['train_size']
            test_size = config['test_size']
            
            # Generate all possible train/test combinations
            test_combinations = list(combinations(subjects, test_size))
            
            split_results = []
            combination_count = 0
            
            for test_subjects in test_combinations:
                train_subjects = [s for s in subjects if s not in test_subjects]
                
                if len(train_subjects) != train_size:
                    continue
                
                combination_count += 1
                
                # Create train/test split based on subjects
                train_indices = []
                test_indices = []
                
                for i, info in enumerate(file_info):
                    if info['subject'] in train_subjects:
                        train_indices.append(i)
                    elif info['subject'] in test_subjects:
                        test_indices.append(i)
                
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                
                X_train = features_df.iloc[train_indices]
                X_test = features_df.iloc[test_indices]
                y_train = action_labels[train_indices]
                y_test = action_labels[test_indices]
                
                # Skip if any class is missing in training or testing
                if len(np.unique(y_train)) < 6 or len(np.unique(y_test)) < 6:
                    continue
                
                # Train Random Forest classifier (primary classifier like Game-1)
                clf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=25,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight='balanced'
                )
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                nmi = self.calculate_nmi(y_test, y_pred)
                nrkl = self.calculate_nrkl(y_test, y_pred)
                vulnerability = calculate_vulnerability(y_pred_proba)
                
                result = {
                    'combination': combination_count,
                    'train_subjects': train_subjects,
                    'test_subjects': list(test_subjects),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'nmi': nmi,
                    'nrkl': nrkl,
                    'vulnerability': vulnerability
                }
                
                split_results.append(result)
                
                if combination_count <= 5:  # Show first 5 combinations
                    print(f"   Combo {combination_count}: Train{train_subjects} → Test{list(test_subjects)} | "
                          f"Acc:{accuracy:.3f} F1:{f1:.3f} NMI:{nmi:.3f} NRKL:{nrkl:.3f}")
                elif combination_count == 6:
                    print(f"   ... continuing evaluation ...")
            
            # Calculate statistics for this split
            if split_results:
                accuracies = [r['accuracy'] for r in split_results]
                f1_scores = [r['f1_score'] for r in split_results]
                nmis = [r['nmi'] for r in split_results]
                nrkls = [r['nrkl'] for r in split_results]
                
                split_stats = {
                    'num_combinations': len(split_results),
                    'accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies),
                        'median': np.median(accuracies)
                    },
                    'f1_score': {
                        'mean': np.mean(f1_scores),
                        'std': np.std(f1_scores),
                        'min': np.min(f1_scores),
                        'max': np.max(f1_scores),
                        'median': np.median(f1_scores)
                    },
                    'nmi': {
                        'mean': np.mean(nmis),
                        'std': np.std(nmis),
                        'min': np.min(nmis),
                        'max': np.max(nmis),
                        'median': np.median(nmis)
                    },
                    'nrkl': {
                        'mean': np.mean(nrkls),
                        'std': np.std(nrkls),
                        'min': np.min(nrkls),
                        'max': np.max(nrkls),
                        'median': np.median(nrkls)
                    },
                    'combinations': split_results
                }
                
                all_results[split_name] = split_stats
                
                print(f"\\n   📊 {split_name} SPLIT STATISTICS ({len(split_results)} combinations):")
                print(f"      Accuracy: {np.mean(accuracies):.3f}±{np.std(accuracies):.3f} [{np.min(accuracies):.3f}-{np.max(accuracies):.3f}] (med: {np.median(accuracies):.3f})")
                print(f"      F1-Score: {np.mean(f1_scores):.3f}±{np.std(f1_scores):.3f} [{np.min(f1_scores):.3f}-{np.max(f1_scores):.3f}] (med: {np.median(f1_scores):.3f})")
                print(f"      NMI:      {np.mean(nmis):.3f}±{np.std(nmis):.3f} [{np.min(nmis):.3f}-{np.max(nmis):.3f}] (med: {np.median(nmis):.3f})")
                print(f"      NRKL:     {np.mean(nrkls):.3f}±{np.std(nrkls):.3f} [{np.min(nrkls):.3f}-{np.max(nrkls):.3f}] (med: {np.median(nrkls):.3f})")
                print(f"      🎲 Random baseline accuracy: {1/6:.3f} ({100/6:.1f}%)")
                print(f"      🚀 Improvement factor: {np.mean(accuracies)/(1/6):.2f}x")
            else:
                print(f"   ⚠️ No valid combinations found for {split_name} split")
                all_results[split_name] = None
        
        # Overall summary comparison with Game-1 and Game-2-Accel
        print(f"\\n🎯 GAME-2 GYRO THIGH HAR ATTACK COMPREHENSIVE SUMMARY:")
        print(f"="*70)
        
        # Calculate overall metrics across all splits
        all_accuracies = []
        all_f1s = []
        all_nmis = []
        all_nrkls = []
        
        for split_name, results in all_results.items():
            if results is not None:
                print(f"\\n📊 {split_name.upper()} SPLIT FINAL RESULTS:")
                print(f"   Mean Accuracy: {results['accuracy']['mean']:.3f} ± {results['accuracy']['std']:.3f}")
                print(f"   Mean F1-Score: {results['f1_score']['mean']:.3f} ± {results['f1_score']['std']:.3f}")
                print(f"   Mean NMI:      {results['nmi']['mean']:.3f} ± {results['nmi']['std']:.3f}")
                print(f"   Mean NRKL:     {results['nrkl']['mean']:.3f} ± {results['nrkl']['std']:.3f}")
                
                all_accuracies.extend([r['accuracy'] for r in results['combinations']])
                all_f1s.extend([r['f1_score'] for r in results['combinations']])
                all_nmis.extend([r['nmi'] for r in results['combinations']])
                all_nrkls.extend([r['nrkl'] for r in results['combinations']])
        
        if all_accuracies:
            overall_accuracy = np.mean(all_accuracies)
            overall_f1 = np.mean(all_f1s)
            overall_nmi = np.mean(all_nmis)
            overall_nrkl = np.mean(all_nrkls)
            
            print(f"\\n🏆 OVERALL GAME-2 GYRO PERFORMANCE:")
            print(f"   Overall Mean Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
            print(f"   Overall Mean F1:       {overall_f1:.3f}")
            print(f"   Overall Mean NMI:      {overall_nmi:.3f}")
            print(f"   Overall Mean NRKL:     {overall_nrkl:.3f}")
            print(f"   🎲 Random baseline:    {1/6:.3f} ({100/6:.1f}%)")
            print(f"   🚀 Improvement:        {overall_accuracy/(1/6):.2f}x vs random")
            
            print(f"\\n📈 COMPARISON WITH OTHER APPROACHES:")
            print(f"   Game-1 (Binary Only):     70.2% accuracy")
            print(f"   Game-2 (Binary + Accel):  98.0% accuracy")
            print(f"   Game-2 (Binary + Gyro):   {overall_accuracy*100:.1f}% accuracy")
            
            if overall_accuracy > 0.702:
                improvement = ((overall_accuracy / 0.702) - 1) * 100
                print(f"   🎊 Gyro vs Game-1:        +{improvement:.1f}% improvement")
            
            # Compare with accelerometer approach
            accel_accuracy = 0.980
            if overall_accuracy > accel_accuracy:
                print(f"   🏆 Gyro vs Accelerometer: +{((overall_accuracy/accel_accuracy)-1)*100:.1f}% improvement")
                print(f"   ✅ Gyroscope features outperform accelerometer!")
            elif overall_accuracy > accel_accuracy * 0.95:
                print(f"   📈 Gyro competitive with accelerometer")
            else:
                print(f"   📉 Accelerometer outperforms gyroscope by {((accel_accuracy/overall_accuracy)-1)*100:.1f}%")
        
        # Save comprehensive results
        self.save_gyro_comprehensive_results(all_results, feature_importance)
        
        return all_results, feature_importance
    
    def save_gyro_comprehensive_results(self, all_results, feature_importance):
        """Save only the maximum vulnerability summary."""
        vulnerabilities = [
            combo.get('vulnerability')
            for results in all_results.values()
            if results is not None
            for combo in results['combinations']
        ]
        output_file = save_max_vulnerability_from_values(__file__, vulnerabilities)
        print(f"\\n💾 Comprehensive gyroscope max vulnerability saved to {output_file}")

def main():
    print("🎯 GAME-2 THIGH HAR ATTACK: BINARY + GYROSCOPE FEATURES")
    print("="*65)
    
    # Set data directory
    data_dir = "UTD-MHAD-Reorganized"
    
    # Initialize Game-2 gyroscope attack system
    game2_gyro_system = Game2ThighGyroHARAttack(data_dir)
    
    # Load thigh data
    thigh_files = game2_gyro_system.load_thigh_data()
    
    if not thigh_files:
        print("❌ No thigh data files found!")
        return
    
    # Extract comprehensive gyroscope features
    features_df, action_labels, file_info = game2_gyro_system.process_files_for_game2_gyro(thigh_files)
    
    if len(features_df) == 0:
        print("❌ No features extracted!")
        return
    
    # Train Game-2 gyroscope classifier with comprehensive evaluation
    results, feature_importance = game2_gyro_system.train_game2_gyro_classifier(features_df, action_labels, file_info)
    
    print(f"\\n🎊 GAME-2 THIGH GYRO HAR ATTACK COMPLETED!")
    print(f"💡 Key findings:")
    print(f"   • Combined binary + gyroscope features")
    print(f"   • Rotational motion sensing vs linear acceleration")
    print(f"   • Comparison with accelerometer-based approach")
    print(f"   • Comprehensive evaluation across multiple splits")

if __name__ == "__main__":
    main()
