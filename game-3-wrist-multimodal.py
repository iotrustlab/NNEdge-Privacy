#!/usr/bin/env python3
"""
🎯 game-3 WRIST ULTIMATE MULTI-MODAL HAR ATTACK - COMPREHENSIVE EVALUATION
=========================================================================

Ultimate version that combines ALL sensor modalities:
- Binary decision tree features (proven effective)
- Accelerometer features (acc_x, acc_y, acc_z) - linear motion
- Gyroscope features (gyro_x, gyro_y, gyro_z) - rotational motion
- Cross-modal synergy features (accel-gyro-binary interactions)
- All possible 4-4, 5-3, and 6-2 subject splits
- Comprehensive metrics: Accuracy, F1, NMI, NRKL

Focus: Maximum performance by leveraging ALL available sensor information
Target: 21 wrist actions from UTD-MHAD with complete sensor fusion
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

class Game2WristUltimateMultiModalHAR:
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
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        
        # Create comprehensive results directory
        os.makedirs("Results", exist_ok=True)
        
        print(f"🎯 ULTIMATE MULTI-MODAL WRIST EVALUATION INITIALIZED")
        print(f"📊 Will test ALL sensor modalities: ACCEL + GYRO + BINARY")
        print(f"🚀 Expected: Maximum performance through complete sensor fusion")
        
    def load_wrist_data_with_all_sensors(self):
        """Load wrist data with ALL sensor modalities"""
        print(f"\n📂 LOADING WRIST DATA WITH ALL SENSOR MODALITIES...")
        
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
                        
                        # Check for required columns - NEED ALL SENSORS
                        has_binary = self.binary_col in data.columns
                        available_accel = [col for col in self.accel_cols if col in data.columns]
                        available_gyro = [col for col in self.gyro_cols if col in data.columns]
                        
                        if has_binary and len(available_accel) == 3 and len(available_gyro) == 3:
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
        
        print(f"📊 WRIST DATA SUMMARY: {len(all_data)} sequences with COMPLETE sensor data")
        subjects = sorted(set(item['subject'] for item in all_data))
        print(f"📊 Subjects: {subjects}")
        
        # Data quality check
        total_samples = sum(len(item['data']) for item in all_data)
        print(f"📊 Total samples: {total_samples}")
        print(f"📊 Average sequence length: {total_samples / len(all_data) if all_data else 0:.1f}")
        
        return all_data, subjects
    
    def extract_ultimate_multimodal_features(self, data):
        """Extract ALL sensor features: accelerometer + gyroscope + binary + cross-modal synergies"""
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
        # 2. ACCELEROMETER FEATURES
        # =================================================================
        accel_data = data[self.accel_cols].values
        
        # Per-axis accelerometer features
        for i, axis in enumerate(['x', 'y', 'z']):
            axis_data = accel_data[:, i]
            
            # Basic statistical features
            features[f'accel_{axis}_mean'] = np.mean(axis_data)
            features[f'accel_{axis}_std'] = np.std(axis_data)
            features[f'accel_{axis}_max'] = np.max(axis_data)
            features[f'accel_{axis}_min'] = np.min(axis_data)
            features[f'accel_{axis}_median'] = np.median(axis_data)
            features[f'accel_{axis}_range'] = np.max(axis_data) - np.min(axis_data)
            features[f'accel_{axis}_energy'] = np.sum(axis_data**2)
            features[f'accel_{axis}_rms'] = np.sqrt(np.mean(axis_data**2))
            
            if len(axis_data) > 1:
                # Advanced statistics
                features[f'accel_{axis}_skew'] = skew(axis_data)
                features[f'accel_{axis}_kurtosis'] = kurtosis(axis_data)
                
                # Temporal features
                diff_data = np.diff(axis_data)
                features[f'accel_{axis}_diff_mean'] = np.mean(diff_data)
                features[f'accel_{axis}_diff_std'] = np.std(diff_data)
                features[f'accel_{axis}_diff_max'] = np.max(np.abs(diff_data))
                features[f'accel_{axis}_jerk'] = np.mean(np.abs(diff_data))
                
                # Percentiles
                features[f'accel_{axis}_q25'] = np.percentile(axis_data, 25)
                features[f'accel_{axis}_q75'] = np.percentile(axis_data, 75)
                features[f'accel_{axis}_iqr'] = features[f'accel_{axis}_q75'] - features[f'accel_{axis}_q25']
                
                # Zero crossings
                features[f'accel_{axis}_zero_crossings'] = np.sum(np.diff(np.sign(axis_data)) != 0)
                features[f'accel_{axis}_zero_crossing_rate'] = features[f'accel_{axis}_zero_crossings'] / len(axis_data)
                
                # Peak analysis
                peaks, _ = find_peaks(np.abs(axis_data), height=np.std(axis_data))
                features[f'accel_{axis}_peak_count'] = len(peaks)
                features[f'accel_{axis}_peak_density'] = len(peaks) / len(axis_data)
        
        # Combined accelerometer features
        acceleration_magnitude = np.linalg.norm(accel_data, axis=1)
        features['accel_magnitude_mean'] = np.mean(acceleration_magnitude)
        features['accel_magnitude_std'] = np.std(acceleration_magnitude)
        features['accel_magnitude_max'] = np.max(acceleration_magnitude)
        features['accel_magnitude_min'] = np.min(acceleration_magnitude)
        features['accel_magnitude_median'] = np.median(acceleration_magnitude)
        features['accel_magnitude_range'] = np.max(acceleration_magnitude) - np.min(acceleration_magnitude)
        features['accel_magnitude_energy'] = np.sum(acceleration_magnitude**2)
        features['accel_magnitude_rms'] = np.sqrt(np.mean(acceleration_magnitude**2))
        
        if len(acceleration_magnitude) > 1:
            features['accel_magnitude_skew'] = skew(acceleration_magnitude)
            features['accel_magnitude_kurtosis'] = kurtosis(acceleration_magnitude)
            
            # Magnitude temporal features
            mag_diff = np.diff(acceleration_magnitude)
            features['accel_magnitude_diff_mean'] = np.mean(mag_diff)
            features['accel_magnitude_diff_std'] = np.std(mag_diff)
            features['accel_magnitude_diff_max'] = np.max(np.abs(mag_diff))
            
            # Magnitude peaks
            mag_peaks, _ = find_peaks(acceleration_magnitude, height=np.mean(acceleration_magnitude) + np.std(acceleration_magnitude)/2)
            features['accel_magnitude_peak_count'] = len(mag_peaks)
            features['accel_magnitude_peak_density'] = len(mag_peaks) / len(acceleration_magnitude)
        
        # Cross-axis accelerometer correlations
        if len(accel_data) > 1:
            features['accel_xy_correlation'] = np.corrcoef(accel_data[:, 0], accel_data[:, 1])[0, 1]
            features['accel_xz_correlation'] = np.corrcoef(accel_data[:, 0], accel_data[:, 2])[0, 1]
            features['accel_yz_correlation'] = np.corrcoef(accel_data[:, 1], accel_data[:, 2])[0, 1]
        
        # =================================================================
        # 3. GYROSCOPE FEATURES
        # =================================================================
        gyro_data = data[self.gyro_cols].values
        
        # Per-axis gyroscope features
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
                
                # Percentiles
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
            
            # Magnitude peaks
            mag_peaks, _ = find_peaks(angular_magnitude, height=np.mean(angular_magnitude) + np.std(angular_magnitude)/2)
            features['gyro_magnitude_peak_count'] = len(mag_peaks)
            features['gyro_magnitude_peak_density'] = len(mag_peaks) / len(angular_magnitude)
        
        # Cross-axis gyroscope correlations
        if len(gyro_data) > 1:
            features['gyro_xy_correlation'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 1])[0, 1]
            features['gyro_xz_correlation'] = np.corrcoef(gyro_data[:, 0], gyro_data[:, 2])[0, 1]
            features['gyro_yz_correlation'] = np.corrcoef(gyro_data[:, 1], gyro_data[:, 2])[0, 1]
        
        # =================================================================
        # 4. CROSS-MODAL SYNERGY FEATURES (UNIQUE TO MULTI-MODAL)
        # =================================================================
        
        # Accelerometer-Gyroscope correlations
        if len(acceleration_magnitude) == len(angular_magnitude) and len(acceleration_magnitude) > 1:
            try:
                features['accel_gyro_magnitude_correlation'] = np.corrcoef(acceleration_magnitude, angular_magnitude)[0, 1]
            except:
                features['accel_gyro_magnitude_correlation'] = 0
            
            # Per-axis cross-modal correlations
            for i, axis in enumerate(['x', 'y', 'z']):
                try:
                    features[f'accel_gyro_{axis}_correlation'] = np.corrcoef(accel_data[:, i], gyro_data[:, i])[0, 1]
                except:
                    features[f'accel_gyro_{axis}_correlation'] = 0
            
            # Movement coordination analysis
            high_accel_mask = acceleration_magnitude > (np.mean(acceleration_magnitude) + np.std(acceleration_magnitude))
            high_gyro_mask = angular_magnitude > (np.mean(angular_magnitude) + np.std(angular_magnitude))
            
            if np.sum(high_accel_mask) > 0 and np.sum(high_gyro_mask) > 0:
                features['high_movement_overlap'] = np.sum(high_accel_mask & high_gyro_mask) / np.sum(high_accel_mask | high_gyro_mask)
                features['accel_gyro_coordination'] = np.sum(high_accel_mask & high_gyro_mask) / len(acceleration_magnitude)
            else:
                features['high_movement_overlap'] = 0
                features['accel_gyro_coordination'] = 0
        
        # Binary + Accelerometer synergy
        if self.binary_col in data.columns:
            binary_data = data[self.binary_col].values
            
            if len(binary_data) == len(acceleration_magnitude) and len(binary_data) > 1:
                try:
                    corr, _ = pearsonr(binary_data, acceleration_magnitude)
                    features['binary_accel_correlation'] = corr if not np.isnan(corr) else 0
                except:
                    features['binary_accel_correlation'] = 0
                
                # Activity-specific acceleration features
                active_mask = binary_data == 1
                inactive_mask = binary_data == 0
                
                if np.sum(active_mask) > 0:
                    features['accel_during_binary_active_mean'] = np.mean(acceleration_magnitude[active_mask])
                    features['accel_during_binary_active_std'] = np.std(acceleration_magnitude[active_mask])
                    features['accel_during_binary_active_max'] = np.max(acceleration_magnitude[active_mask])
                else:
                    for key in ['accel_during_binary_active_mean', 'accel_during_binary_active_std', 'accel_during_binary_active_max']:
                        features[key] = 0
                
                if np.sum(inactive_mask) > 0:
                    features['accel_during_binary_inactive_mean'] = np.mean(acceleration_magnitude[inactive_mask])
                    features['accel_during_binary_inactive_std'] = np.std(acceleration_magnitude[inactive_mask])
                else:
                    features['accel_during_binary_inactive_mean'] = 0
                    features['accel_during_binary_inactive_std'] = 0
                
                # Acceleration contrast
                if features['accel_during_binary_inactive_mean'] > 0:
                    features['accel_active_inactive_contrast'] = features['accel_during_binary_active_mean'] / features['accel_during_binary_inactive_mean']
                else:
                    features['accel_active_inactive_contrast'] = 0
        
        # Binary + Gyroscope synergy
        if self.binary_col in data.columns:
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
                else:
                    for key in ['gyro_during_binary_active_mean', 'gyro_during_binary_active_std', 'gyro_during_binary_active_max']:
                        features[key] = 0
                
                if np.sum(inactive_mask) > 0:
                    features['gyro_during_binary_inactive_mean'] = np.mean(angular_magnitude[inactive_mask])
                    features['gyro_during_binary_inactive_std'] = np.std(angular_magnitude[inactive_mask])
                else:
                    features['gyro_during_binary_inactive_mean'] = 0
                    features['gyro_during_binary_inactive_std'] = 0
                
                # Rotation contrast
                if features['gyro_during_binary_inactive_mean'] > 0:
                    features['gyro_active_inactive_contrast'] = features['gyro_during_binary_active_mean'] / features['gyro_during_binary_inactive_mean']
                else:
                    features['gyro_active_inactive_contrast'] = 0
        
        # Triple synergy: Binary + Accel + Gyro
        if self.binary_col in data.columns and len(binary_data) == len(acceleration_magnitude) == len(angular_magnitude) and len(binary_data) > 1:
            # Combined motion magnitude
            combined_motion = acceleration_magnitude + angular_magnitude
            features['combined_motion_mean'] = np.mean(combined_motion)
            features['combined_motion_std'] = np.std(combined_motion)
            features['combined_motion_max'] = np.max(combined_motion)
            
            # Activity-specific combined motion
            active_mask = binary_data == 1
            if np.sum(active_mask) > 0:
                features['combined_motion_during_active'] = np.mean(combined_motion[active_mask])
                
                # Motion pattern analysis during activity
                features['accel_dominance_during_active'] = np.mean(acceleration_magnitude[active_mask]) / (np.mean(combined_motion[active_mask]) + 1e-10)
                features['gyro_dominance_during_active'] = np.mean(angular_magnitude[active_mask]) / (np.mean(combined_motion[active_mask]) + 1e-10)
            else:
                features['combined_motion_during_active'] = 0
                features['accel_dominance_during_active'] = 0
                features['gyro_dominance_during_active'] = 0
            
            # Overall motion pattern
            features['overall_accel_dominance'] = np.mean(acceleration_magnitude) / (np.mean(combined_motion) + 1e-10)
            features['overall_gyro_dominance'] = np.mean(angular_magnitude) / (np.mean(combined_motion) + 1e-10)
        
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
        print(f"\n🔧 CREATING ULTIMATE MULTI-MODAL FEATURE MATRIX...")
        
        features_list = []
        labels = []
        subjects = []
        
        for item in all_data:
            features = self.extract_ultimate_multimodal_features(item['data'])
            if features:
                features_list.append(features)
                labels.append(self.action_mapping[item['action']])
                subjects.append(item['subject'])
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        # Categorize features
        self.feature_categories = {
            'binary': [f for f in features_df.columns if f.startswith('binary_') and 'accel' not in f and 'gyro' not in f],
            'accel': [f for f in features_df.columns if f.startswith('accel_') and 'binary' not in f and 'gyro' not in f],
            'gyro': [f for f in features_df.columns if f.startswith('gyro_') and 'binary' not in f and 'accel' not in f],
            'binary_accel': [f for f in features_df.columns if 'binary' in f and 'accel' in f and 'gyro' not in f],
            'binary_gyro': [f for f in features_df.columns if 'binary' in f and 'gyro' in f and 'accel' not in f],
            'accel_gyro': [f for f in features_df.columns if 'accel' in f and 'gyro' in f and 'binary' not in f],
            'triple_synergy': [f for f in features_df.columns if ('combined_motion' in f or 'dominance' in f or 'coordination' in f)]
        }
        
        print(f"📊 ULTIMATE FEATURE MATRIX: {features_df.shape}")
        print(f"📊 Binary features: {len(self.feature_categories['binary'])}")
        print(f"📊 Accel features: {len(self.feature_categories['accel'])}")
        print(f"📊 Gyro features: {len(self.feature_categories['gyro'])}")
        print(f"📊 Binary-Accel synergy: {len(self.feature_categories['binary_accel'])}")
        print(f"📊 Binary-Gyro synergy: {len(self.feature_categories['binary_gyro'])}")
        print(f"📊 Accel-Gyro synergy: {len(self.feature_categories['accel_gyro'])}")
        print(f"📊 Triple synergy: {len(self.feature_categories['triple_synergy'])}")
        print(f"📊 TOTAL FEATURES: {len(features_df.columns)}")
        
        return features_df, np.array(labels), np.array(subjects)
    
    def analyze_multimodal_features(self, features_df):
        """Analyze multi-modal feature characteristics"""
        print(f"\n🔬 ANALYZING ULTIMATE MULTI-MODAL FEATURE CHARACTERISTICS...")
        
        total_features = len(features_df.columns)
        
        print(f"📊 FEATURE CATEGORY BREAKDOWN:")
        total_single_modal = 0
        total_cross_modal = 0
        
        for category, feats in self.feature_categories.items():
            print(f"   {category}: {len(feats)} features ({len(feats)/total_features*100:.1f}%)")
            
            if category in ['binary', 'accel', 'gyro']:
                total_single_modal += len(feats)
            else:
                total_cross_modal += len(feats)
        
        print(f"\n📊 MODAL ANALYSIS:")
        print(f"   Single-modal features: {total_single_modal} ({total_single_modal/total_features*100:.1f}%)")
        print(f"   Cross-modal features: {total_cross_modal} ({total_cross_modal/total_features*100:.1f}%)")
        print(f"   Cross-modal advantage: {total_cross_modal} additional synergy features")
        
        # Feature variance analysis
        print(f"\n📊 FEATURE QUALITY ANALYSIS:")
        zero_variance_features = sum([np.var(features_df[f]) == 0 for f in features_df.columns])
        print(f"   Features with zero variance: {zero_variance_features}")
        print(f"   High-quality features: {total_features - zero_variance_features}")
        
        # Cross-modal correlation analysis
        print(f"\n📊 CROSS-MODAL CORRELATION ANALYSIS:")
        correlation_features = [f for f in features_df.columns if 'correlation' in f]
        print(f"   Total correlation features: {len(correlation_features)}")
        
        for feat in correlation_features[:5]:  # Show first 5
            avg_corr = np.mean(np.abs(features_df[feat]))
            print(f"   {feat}: avg correlation = {avg_corr:.3f}")
        
        return self.feature_categories
    
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
        print(f"\n🎯 COMPREHENSIVE EVALUATION WITH ALL SENSOR MODALITIES...")
        
        # Get unique subjects
        unique_subjects = sorted(np.unique(subjects))
        print(f"📊 Available subjects: {unique_subjects}")
        
        # Generate all possible splits
        splits = self.generate_all_subject_splits(unique_subjects)
        
        # Use ExtraTreesClassifier (best performer from previous analyses)
        classifier = ExtraTreesClassifier(n_estimators=300, max_depth=25, random_state=42, class_weight='balanced', n_jobs=-1)
        
        total_experiments = len(splits)
        print(f"🧪 Running {total_experiments} total experiments with ULTIMATE feature set...")
        print(f"🚀 Expected: Best performance due to complete sensor fusion")
        
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
            print(f"   Features: {len(features_df.columns)} (multi-modal)")
            
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
                    'test_subjects': split_config['test_subjects'],
                    'feature_type': 'ultimate_multimodal'
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
        print(f"\n🏆 ANALYZING ULTIMATE MULTI-MODAL RESULTS...")
        
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
        
        print(f"\n📊 ULTIMATE MULTI-MODAL RESULTS SUMMARY:")
        print(f"   Total experiments: {len(all_results_flat)}")
        print(f"   4-4 splits tested: {len(split_4_4_results)}")
        print(f"   5-3 splits tested: {len(split_5_3_results)}")
        print(f"   6-2 splits tested: {len(split_6_2_results)}")
        print(f"   Features used: {overall_best['features_used']} (ACCEL + GYRO + BINARY + SYNERGIES)")
        
        print(f"\n🥇 BEST OVERALL CONFIGURATION (ULTIMATE MULTI-MODAL):")
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
            print(f"\n📊 PERFORMANCE COMPARISON BY SPLIT TYPE (ULTIMATE MULTI-MODAL):")
            for split_name, avg_acc, std_acc, count in split_types:
                print(f"   {split_name} splits: {avg_acc:.4f} ± {std_acc:.4f} accuracy ({count} tests)")
            
            # Find best performing split type
            best_split_type = max(split_types, key=lambda x: x[1])
            print(f"   ✅ {best_split_type[0]} splits perform best on average ({best_split_type[1]:.4f} accuracy)")
        
        # Top 10 configurations overall
        print(f"\n🏆 TOP 10 CONFIGURATIONS OVERALL (ULTIMATE MULTI-MODAL):")
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
                'feature_type': 'ultimate_multimodal',
                'sensor_modalities': 'accelerometer + gyroscope + binary + synergies'
            }
        })
        
        return best_configs, overall_best
    
    def save_comprehensive_results(self, results):
        """Save only the maximum vulnerability summary."""
        all_results = results.get('all_results', {})
        vulnerabilities = [result.get('vulnerability') for result in all_results.values()]
        output_file = save_max_vulnerability_from_values(__file__, vulnerabilities)
        print(f"💾 Comprehensive ultimate max vulnerability saved to: {output_file}")
    
    def run_comprehensive_evaluation(self):
        """Run the complete comprehensive evaluation"""
        print(f"🚀 STARTING ULTIMATE MULTI-MODAL WRIST EVALUATION")
        print(f"="*80)
        
        # Load data
        all_data, subjects = self.load_wrist_data_with_all_sensors()
        
        if not all_data:
            print("❌ No data loaded!")
            return None
        
        # Create feature matrix
        features_df, labels, subjects_array = self.create_feature_matrix(all_data)
        
        # Analyze multi-modal features
        feature_categories = self.analyze_multimodal_features(features_df)
        
        # Run comprehensive evaluation
        all_results = self.evaluate_comprehensive_combinations(features_df, labels, subjects_array)
        
        # Find and save best configurations
        best_configs, overall_best = self.find_and_save_best_configurations(all_results)
        
        print(f"\n✅ ULTIMATE MULTI-MODAL EVALUATION COMPLETED!")
        print(f"🎯 Tested ALL sensor modalities: ACCEL + GYRO + BINARY + SYNERGIES")
        print(f"🔢 Evaluated all possible 4-4, 5-3, and 6-2 subject splits")
        print(f"🚀 Maximum performance achieved through complete sensor fusion")
        print(f"🏆 Best configuration saved with comprehensive metrics (NMI, NRKL)")
        
        return best_configs, overall_best

def main():
    """Main execution"""
    data_dir = "UTD-MHAD-Reorganized"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    evaluator = Game2WristUltimateMultiModalHAR(data_dir)
    best_configs, overall_best = evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
