#!/usr/bin/env python3
"""
🎯 GAME-3 THIGH HAR ATTACK: BINARY + ACCELEROMETER + GYROSCOPE
==============================================================

The ultimate Game-3 implementation combining ALL sensor modalities:
- Binary decision tree features (proven effective for thigh placement)
- Rich accelerometer features (linear motion sensing)  
- Rich gyroscope features (rotational motion sensing)

Target: 6 thigh actions from UTD-MHAD
- Action 22: Right kick
- Action 23: Left kick  
- Action 24: Sit to stand
- Action 25: Stand to sit
- Action 26: Forward lunge
- Action 27: Squat

Expected: Maximum performance by leveraging ALL available sensor information
Previous results:
- Game-1 (Binary only): 70.2%
- Game-2 (Binary + Accel): 98.0% 
- Game-2 (Binary + Gyro): TBD
- Game-3 (All sensors): Expected >98%
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
from utd_vulnerability_utils import calculate_vulnerability, save_max_vulnerability_from_values
import warnings
import json
from datetime import datetime
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import jensenshannon
warnings.filterwarnings('ignore')

class Game3ThighMultiSensorHARAttack:
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
        
        # All sensor modalities (Game-3 includes everything)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        
        print(f"🎯 GAME-3 THIGH MULTI-SENSOR HAR ATTACK INITIALIZED")
        print(f"📊 6 thigh actions with ALL sensor modalities")
        print(f"🚀 Expected: Maximum performance by combining all sensor types")
        
    def load_thigh_data(self):
        """Load thigh placement data for target actions"""
        print(f"\\n📂 LOADING THIGH DATA FOR MULTI-SENSOR ANALYSIS...")
        
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
        
        print(f"📊 THIGH MULTI-SENSOR DATA SUMMARY:")
        total_files = 0
        for action_num in sorted(self.thigh_actions.keys()):
            count = action_counts[action_num]
            total_files += count
            print(f"   Action {action_num}: {self.thigh_actions[action_num]:<20} ({count:2d} files)")
        
        print(f"📈 Total thigh files: {total_files}")
        return all_files
    
    def extract_ultimate_features(self, data):
        """Extract features from ALL sensor modalities"""
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
                
                # Binary burst analysis
                active_bursts, inactive_bursts = self._analyze_binary_bursts(binary_data)
                
                if active_bursts:
                    features['binary_active_burst_count'] = len(active_bursts)
                    features['binary_active_burst_mean'] = np.mean(active_bursts)
                    features['binary_active_burst_std'] = np.std(active_bursts)
                    features['binary_active_burst_max'] = max(active_bursts)
                    features['binary_active_burst_total'] = sum(active_bursts)
                else:
                    for key in ['binary_active_burst_count', 'binary_active_burst_mean', 
                               'binary_active_burst_std', 'binary_active_burst_max', 'binary_active_burst_total']:
                        features[key] = 0
                
                # Binary temporal positioning
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
                p1 = np.mean(binary_data)
                p0 = 1 - p1
                if p0 > 0 and p1 > 0:
                    features['binary_entropy'] = -(p0 * np.log2(p0) + p1 * np.log2(p1))
                else:
                    features['binary_entropy'] = 0
        
        # =================================================================
        # 2. ACCELEROMETER FEATURES (from successful Game-2)
        # =================================================================
        
        available_accel_cols = [col for col in self.accel_cols if col in data.columns]
        
        if available_accel_cols:
            # Per-axis accelerometer features
            for axis_col in available_accel_cols:
                axis_data = data[axis_col].values
                axis_name = axis_col.replace('[mg]', '').replace('acc_', '')
                
                if len(axis_data) > 0:
                    # Basic statistical features
                    features[f'accel_{axis_name}_mean'] = np.mean(axis_data)
                    features[f'accel_{axis_name}_std'] = np.std(axis_data)
                    features[f'accel_{axis_name}_max'] = np.max(axis_data)
                    features[f'accel_{axis_name}_min'] = np.min(axis_data)
                    features[f'accel_{axis_name}_median'] = np.median(axis_data)
                    features[f'accel_{axis_name}_range'] = np.max(axis_data) - np.min(axis_data)
                    features[f'accel_{axis_name}_energy'] = np.sum(axis_data**2)
                    features[f'accel_{axis_name}_rms'] = np.sqrt(np.mean(axis_data**2))
                    
                    if len(axis_data) > 1:
                        # Advanced statistics
                        features[f'accel_{axis_name}_skew'] = pd.Series(axis_data).skew()
                        features[f'accel_{axis_name}_kurtosis'] = pd.Series(axis_data).kurtosis()
                        
                        # Temporal features
                        diff_data = np.diff(axis_data)
                        features[f'accel_{axis_name}_diff_mean'] = np.mean(diff_data)
                        features[f'accel_{axis_name}_diff_std'] = np.std(diff_data)
                        features[f'accel_{axis_name}_diff_max'] = np.max(np.abs(diff_data))
                        
                        # Percentiles
                        features[f'accel_{axis_name}_q25'] = np.percentile(axis_data, 25)
                        features[f'accel_{axis_name}_q75'] = np.percentile(axis_data, 75)
                        features[f'accel_{axis_name}_iqr'] = features[f'accel_{axis_name}_q75'] - features[f'accel_{axis_name}_q25']
                        
                        # Zero crossings and peaks
                        features[f'accel_{axis_name}_zero_crossings'] = np.sum(np.diff(np.sign(axis_data)) != 0)
                        features[f'accel_{axis_name}_peak_count'] = len(self._find_peaks(axis_data))
            
            # Combined accelerometer features
            if len(available_accel_cols) == 3:
                accel_data = data[available_accel_cols].values
                magnitude = np.linalg.norm(accel_data, axis=1)
                
                features['accel_magnitude_mean'] = np.mean(magnitude)
                features['accel_magnitude_std'] = np.std(magnitude)
                features['accel_magnitude_max'] = np.max(magnitude)
                features['accel_magnitude_min'] = np.min(magnitude)
                features['accel_magnitude_range'] = np.max(magnitude) - np.min(magnitude)
                features['accel_magnitude_energy'] = np.sum(magnitude**2)
                features['accel_magnitude_rms'] = np.sqrt(np.mean(magnitude**2))
                
                if len(magnitude) > 1:
                    features['accel_magnitude_skew'] = pd.Series(magnitude).skew()
                    features['accel_magnitude_kurtosis'] = pd.Series(magnitude).kurtosis()
                    features['accel_magnitude_peak_count'] = len(self._find_peaks(magnitude))
                
                # Cross-axis correlations
                accel_x, accel_y, accel_z = accel_data[:, 0], accel_data[:, 1], accel_data[:, 2]
                if len(accel_x) > 1:
                    features['accel_xy_correlation'] = np.corrcoef(accel_x, accel_y)[0, 1]
                    features['accel_xz_correlation'] = np.corrcoef(accel_x, accel_z)[0, 1]
                    features['accel_yz_correlation'] = np.corrcoef(accel_y, accel_z)[0, 1]
                else:
                    features['accel_xy_correlation'] = 0
                    features['accel_xz_correlation'] = 0
                    features['accel_yz_correlation'] = 0
        
        # =================================================================
        # 3. GYROSCOPE FEATURES (rotational motion sensing)
        # =================================================================
        
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
                    features[f'gyro_{axis_name}_rms'] = np.sqrt(np.mean(axis_data**2))
                    features[f'gyro_{axis_name}_abs_mean'] = np.mean(np.abs(axis_data))
                    
                    if len(axis_data) > 1:
                        # Advanced statistics
                        features[f'gyro_{axis_name}_skew'] = pd.Series(axis_data).skew()
                        features[f'gyro_{axis_name}_kurtosis'] = pd.Series(axis_data).kurtosis()
                        
                        # Temporal features
                        diff_data = np.diff(axis_data)
                        features[f'gyro_{axis_name}_diff_mean'] = np.mean(diff_data)
                        features[f'gyro_{axis_name}_diff_std'] = np.std(diff_data)
                        features[f'gyro_{axis_name}_diff_max'] = np.max(np.abs(diff_data))
                        
                        # Percentiles
                        features[f'gyro_{axis_name}_q25'] = np.percentile(axis_data, 25)
                        features[f'gyro_{axis_name}_q75'] = np.percentile(axis_data, 75)
                        features[f'gyro_{axis_name}_iqr'] = features[f'gyro_{axis_name}_q75'] - features[f'gyro_{axis_name}_q25']
                        
                        # Rotational characteristics
                        features[f'gyro_{axis_name}_zero_crossings'] = np.sum(np.diff(np.sign(axis_data)) != 0)
                        features[f'gyro_{axis_name}_zero_crossing_rate'] = features[f'gyro_{axis_name}_zero_crossings'] / len(axis_data)
                        features[f'gyro_{axis_name}_positive_ratio'] = np.sum(axis_data > 0) / len(axis_data)
                        features[f'gyro_{axis_name}_peak_count'] = len(self._find_peaks(axis_data))
            
            # Combined gyroscope features
            if len(available_gyro_cols) == 3:
                gyro_data = data[available_gyro_cols].values
                angular_magnitude = np.linalg.norm(gyro_data, axis=1)
                
                features['gyro_angular_magnitude_mean'] = np.mean(angular_magnitude)
                features['gyro_angular_magnitude_std'] = np.std(angular_magnitude)
                features['gyro_angular_magnitude_max'] = np.max(angular_magnitude)
                features['gyro_angular_magnitude_min'] = np.min(angular_magnitude)
                features['gyro_angular_magnitude_range'] = np.max(angular_magnitude) - np.min(angular_magnitude)
                features['gyro_angular_magnitude_energy'] = np.sum(angular_magnitude**2)
                features['gyro_angular_magnitude_rms'] = np.sqrt(np.mean(angular_magnitude**2))
                
                if len(angular_magnitude) > 1:
                    features['gyro_angular_magnitude_skew'] = pd.Series(angular_magnitude).skew()
                    features['gyro_angular_magnitude_kurtosis'] = pd.Series(angular_magnitude).kurtosis()
                    features['gyro_angular_magnitude_peak_count'] = len(self._find_peaks(angular_magnitude))
                
                # Cross-axis correlations
                gyro_x, gyro_y, gyro_z = gyro_data[:, 0], gyro_data[:, 1], gyro_data[:, 2]
                if len(gyro_x) > 1:
                    features['gyro_xy_correlation'] = np.corrcoef(gyro_x, gyro_y)[0, 1]
                    features['gyro_xz_correlation'] = np.corrcoef(gyro_x, gyro_z)[0, 1]
                    features['gyro_yz_correlation'] = np.corrcoef(gyro_y, gyro_z)[0, 1]
                else:
                    features['gyro_xy_correlation'] = 0
                    features['gyro_xz_correlation'] = 0
                    features['gyro_yz_correlation'] = 0
        
        # =================================================================
        # 4. CROSS-SENSOR SYNERGY FEATURES (unique to Game-3)
        # =================================================================
        
        # Correlations between different sensor modalities
        if (self.binary_col in data.columns and len(available_accel_cols) == 3 and len(available_gyro_cols) == 3):
            binary_data = data[self.binary_col].values
            accel_data = data[available_accel_cols].values
            gyro_data = data[available_gyro_cols].values
            
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            
            if len(binary_data) == len(accel_magnitude) == len(gyro_magnitude) and len(binary_data) > 1:
                # Binary-Accel synergy
                features['binary_accel_correlation'] = np.corrcoef(binary_data, accel_magnitude)[0, 1]
                
                # Binary-Gyro synergy  
                features['binary_gyro_correlation'] = np.corrcoef(binary_data, gyro_magnitude)[0, 1]
                
                # Accel-Gyro synergy
                features['accel_gyro_correlation'] = np.corrcoef(accel_magnitude, gyro_magnitude)[0, 1]
                
                # Activity-specific sensor characteristics
                active_mask = binary_data == 1
                inactive_mask = binary_data == 0
                
                if np.sum(active_mask) > 0:
                    # During active periods
                    features['accel_during_active_mean'] = np.mean(accel_magnitude[active_mask])
                    features['accel_during_active_std'] = np.std(accel_magnitude[active_mask])
                    features['gyro_during_active_mean'] = np.mean(gyro_magnitude[active_mask])
                    features['gyro_during_active_std'] = np.std(gyro_magnitude[active_mask])
                    
                    # Sensor coordination during activity
                    if np.sum(active_mask) > 2:
                        active_accel = accel_magnitude[active_mask]
                        active_gyro = gyro_magnitude[active_mask]
                        features['accel_gyro_correlation_during_active'] = np.corrcoef(active_accel, active_gyro)[0, 1]
                    else:
                        features['accel_gyro_correlation_during_active'] = 0
                else:
                    features['accel_during_active_mean'] = 0
                    features['accel_during_active_std'] = 0
                    features['gyro_during_active_mean'] = 0
                    features['gyro_during_active_std'] = 0
                    features['accel_gyro_correlation_during_active'] = 0
                
                if np.sum(inactive_mask) > 0:
                    # During inactive periods
                    features['accel_during_inactive_mean'] = np.mean(accel_magnitude[inactive_mask])
                    features['gyro_during_inactive_mean'] = np.mean(gyro_magnitude[inactive_mask])
                else:
                    features['accel_during_inactive_mean'] = 0
                    features['gyro_during_inactive_mean'] = 0
                
                # Sensor contrast ratios
                if features['accel_during_inactive_mean'] > 0:
                    features['accel_active_inactive_ratio'] = features['accel_during_active_mean'] / features['accel_during_inactive_mean']
                else:
                    features['accel_active_inactive_ratio'] = 0
                
                if features['gyro_during_inactive_mean'] > 0:
                    features['gyro_active_inactive_ratio'] = features['gyro_during_active_mean'] / features['gyro_during_inactive_mean']
                else:
                    features['gyro_active_inactive_ratio'] = 0
                
                # Multi-sensor complexity measures
                features['total_sensor_energy'] = np.sum(accel_magnitude**2) + np.sum(gyro_magnitude**2)
                features['sensor_energy_ratio'] = np.sum(accel_magnitude**2) / (np.sum(gyro_magnitude**2) + 1e-10)
                features['multi_sensor_peak_alignment'] = abs(len(self._find_peaks(accel_magnitude)) - len(self._find_peaks(gyro_magnitude)))
        
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
                abs(data[i]) > threshold):
                peaks.append(i)
        
        return peaks
    
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
        
        # Calculate KL divergence using Jensen-Shannon distance
        js_distance = jensenshannon(true_dist, pred_dist)
        nrkl = 1 - js_distance
        
        return nrkl
    
    def process_files_for_game3(self, file_list):
        """Process all files and extract ultimate multi-sensor features"""
        print(f"\\n🔄 EXTRACTING ALL SENSOR FEATURES FROM {len(file_list)} FILES...")
        
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
                
                # Extract ultimate multi-sensor features
                features = self.extract_ultimate_features(data)
                
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
    
    def analyze_ultimate_feature_importance(self, features_df, action_labels):
        """Analyze importance across all sensor modalities"""
        print(f"\\n🔬 ANALYZING ULTIMATE MULTI-SENSOR FEATURE IMPORTANCE...")
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Train a classifier to get feature importance
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features_df, action_labels)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': features_df.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Analyze feature categories
        binary_features = feature_importance[feature_importance['feature'].str.contains('binary')]
        accel_features = feature_importance[feature_importance['feature'].str.contains('accel')]
        gyro_features = feature_importance[feature_importance['feature'].str.contains('gyro')]
        synergy_features = feature_importance[
            (feature_importance['feature'].str.contains('correlation')) |
            (feature_importance['feature'].str.contains('ratio')) |
            (feature_importance['feature'].str.contains('sensor'))
        ]
        
        print(f"📊 ULTIMATE FEATURE ANALYSIS SUMMARY:")
        print(f"   Total features: {len(features_df.columns)}")
        print(f"   Binary features: {len(binary_features)}")
        print(f"   Accelerometer features: {len(accel_features)}")
        print(f"   Gyroscope features: {len(gyro_features)}")
        print(f"   Cross-sensor synergy features: {len(synergy_features)}")
        
        print(f"\\n🔥 TOP 20 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
            if 'binary' in row['feature']:
                feature_type = "🎯 BINARY"
            elif 'accel' in row['feature']:
                feature_type = "📈 ACCEL"
            elif 'gyro' in row['feature']:
                feature_type = "🌀 GYRO"
            else:
                feature_type = "🔗 SYNERGY"
            print(f"{i+1:2d}. {feature_type} {row['feature']:<40} {row['importance']:.4f}")
        
        print(f"\\n📊 FEATURE IMPORTANCE BY SENSOR MODALITY:")
        binary_total = binary_features['importance'].sum()
        accel_total = accel_features['importance'].sum()
        gyro_total = gyro_features['importance'].sum()
        synergy_total = synergy_features['importance'].sum()
        total_importance = binary_total + accel_total + gyro_total + synergy_total
        
        if total_importance > 0:
            binary_pct = (binary_total / total_importance) * 100
            accel_pct = (accel_total / total_importance) * 100
            gyro_pct = (gyro_total / total_importance) * 100
            synergy_pct = (synergy_total / total_importance) * 100
            
            print(f"   Binary features:         {binary_pct:.1f}%")
            print(f"   Accelerometer features:  {accel_pct:.1f}%")
            print(f"   Gyroscope features:      {gyro_pct:.1f}%")
            print(f"   Cross-sensor synergy:    {synergy_pct:.1f}%")
        
        return feature_importance
    
    def train_game3_classifier(self, features_df, action_labels, file_info):
        """Train ultimate Game-3 classifier with all sensors"""
        print(f"\\n🤖 TRAINING ULTIMATE GAME-3 MULTI-SENSOR CLASSIFIER...")
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Get feature importance analysis
        feature_importance = self.analyze_ultimate_feature_importance(features_df, action_labels)
        
        # Subject-based evaluation
        subjects = sorted(list(set([info['subject'] for info in file_info])))
        print(f"\\n👥 Available subjects: {subjects}")
        
        # Define split configurations
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
                
                # Train Random Forest classifier (primary)
                clf = RandomForestClassifier(
                    n_estimators=500,  # More trees for ultimate performance
                    max_depth=30,      # Deeper for complex patterns
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    class_weight='balanced',
                    max_features='sqrt'
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
                
                if combination_count <= 5:
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
                print(f"      🎲 Random baseline: {1/6:.3f} ({100/6:.1f}%)")
                print(f"      🚀 Improvement: {np.mean(accuracies)/(1/6):.2f}x")
            else:
                print(f"   ⚠️ No valid combinations found for {split_name} split")
                all_results[split_name] = None
        
        # Ultimate comparison summary
        print(f"\\n🎯 ULTIMATE GAME-3 MULTI-SENSOR ATTACK SUMMARY:")
        print(f"="*70)
        
        # Calculate overall metrics
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
            
            print(f"\\n🏆 ULTIMATE GAME-3 PERFORMANCE:")
            print(f"   Overall Mean Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
            print(f"   Overall Mean F1:       {overall_f1:.3f}")
            print(f"   Overall Mean NMI:      {overall_nmi:.3f}")
            print(f"   Overall Mean NRKL:     {overall_nrkl:.3f}")
            print(f"   🎲 Random baseline:    {1/6:.3f} ({100/6:.1f}%)")
            print(f"   🚀 Improvement:        {overall_accuracy/(1/6):.2f}x vs random")
            
            print(f"\\n📈 COMPREHENSIVE COMPARISON:")
            print(f"   Game-1 (Binary Only):          70.2% accuracy")
            print(f"   Game-2 (Binary + Accel):       98.0% accuracy") 
            print(f"   Game-2 (Binary + Gyro):        [Previous run]")
            print(f"   Game-3 (ALL Sensors):          {overall_accuracy*100:.1f}% accuracy")
            
            # Compare with Game-2 accelerometer
            accel_accuracy = 0.980
            if overall_accuracy > accel_accuracy:
                improvement = ((overall_accuracy / accel_accuracy) - 1) * 100
                print(f"   🎊 Game-3 vs Game-2-Accel:     +{improvement:.1f}% improvement")
                print(f"   ✅ ULTIMATE SUCCESS: All sensors achieve maximum performance!")
            elif overall_accuracy > accel_accuracy * 0.98:
                print(f"   📈 Game-3 matches Game-2-Accel performance")
            else:
                diff = ((accel_accuracy / overall_accuracy) - 1) * 100
                print(f"   📊 Game-2-Accel outperforms by {diff:.1f}% (sensor redundancy)")
            
            # Game-1 improvement
            game1_improvement = ((overall_accuracy / 0.702) - 1) * 100
            print(f"   🚀 Game-3 vs Game-1:           +{game1_improvement:.1f}% improvement")
        
        # Save ultimate results
        self.save_ultimate_results(all_results, feature_importance)
        
        return all_results, feature_importance
    
    def save_ultimate_results(self, all_results, feature_importance):
        """Save only the maximum vulnerability summary."""
        vulnerabilities = [
            combo.get('vulnerability')
            for results in all_results.values()
            if results is not None
            for combo in results['combinations']
        ]
        output_file = save_max_vulnerability_from_values(__file__, vulnerabilities)
        print(f"\\n💾 Ultimate multi-sensor max vulnerability saved to {output_file}")

def main():
    print("🎯 GAME-3 ULTIMATE THIGH HAR ATTACK: ALL SENSOR MODALITIES")
    print("="*70)
    
    # Set data directory
    data_dir = "UTD-MHAD-Reorganized"
    
    # Initialize ultimate Game-3 attack system
    game3_ultimate_system = Game3ThighMultiSensorHARAttack(data_dir)
    
    # Load thigh data
    thigh_files = game3_ultimate_system.load_thigh_data()
    
    if not thigh_files:
        print("❌ No thigh data files found!")
        return
    
    # Extract ultimate multi-sensor features
    features_df, action_labels, file_info = game3_ultimate_system.process_files_for_game3(thigh_files)
    
    if len(features_df) == 0:
        print("❌ No features extracted!")
        return
    
    # Train ultimate Game-3 classifier
    results, feature_importance = game3_ultimate_system.train_game3_classifier(features_df, action_labels, file_info)
    
    print(f"\\n🎊 ULTIMATE GAME-3 MULTI-SENSOR HAR ATTACK COMPLETED!")
    print(f"💡 Key achievements:")
    print(f"   • Combined ALL sensor modalities (binary + accel + gyro)")
    print(f"   • Cross-sensor synergy features for maximum discrimination")
    print(f"   • Comprehensive evaluation across all game types")
    print(f"   • Ultimate performance benchmark established")

if __name__ == "__main__":
    main()
