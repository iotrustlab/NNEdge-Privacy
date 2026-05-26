#!/usr/bin/env python3
"""
Game-3 User Identity + Anthropometrics Analysis
===============================================

This script implements Game-3 attacks using BOTH user identity AND anthropometric
semantic information. This is a semantic ablation study that tests adversarial effectiveness when:
- Placement information is ANONYMIZED (no body location knowledge)
- User identity is AVAILABLE as semantic information  
- Anthropometric measurements are AVAILABLE as semantic information
- All sensor data is used (accelerometer + gyroscope + binary)

The goal is to test if user-specific sensor patterns combined with physical characteristics
can be exploited for activity inference without any placement-specific semantic advantages.

Research Question:
Can adversaries leverage user identity + anthropometric information to improve multi-sensor 
activity inference when placement information is completely anonymized?

Semantic Information:
✅ User identity (one-hot encoded user features)  
✅ Anthropometric measurements (height, limb lengths, dominance, etc.)
❌ Sensor placement information (anonymized as sensor_0, sensor_1, etc.)
❌ Cross-sensor coordination features
❌ Placement-specific feature engineering
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("🎯 GAME-3 USER IDENTITY + ANTHROPOMETRICS ANALYSIS")
print("=" * 52)
print("Semantic ablation study with placement anonymization")
print("Testing user identity + anthropometric semantic information effectiveness")
print("All sensor data available | Placement information ANONYMIZED | Anthropometrics AVAILABLE")
print()

class Game3UserIdentityAnthropometricsAnalyzer:
    """
    Analyzer for Game-3 attacks with user identity + anthropometrics but
    without placement-specific or cross-sensor semantic information
    """
    
    def __init__(self):
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        
        # Game-3: All sensor data (accelerometer + gyroscope + binary)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        self.all_sensor_cols = self.accel_cols + self.gyro_cols + [self.binary_col]
        
        # Split ratios to test
        self.split_ratios = {
            '8_3': {'train_size': 8, 'test_size': 3},
            '6_5': {'train_size': 6, 'test_size': 5},
            '4_7': {'train_size': 4, 'test_size': 7},
            '1_10': {'train_size': 1, 'test_size': 10}
        }
        
        # Results storage
        self.base_results_dir = "results/game-3-user_identity_anthropometrics"
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Load anthropometric data
        self.anthropometric_data = self.load_all_anthropometric_data()
        
        print(f"🎯 Target: Game-3 attacks with user identity + anthropometric semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (ANONYMIZED)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📏 Anthropometrics: AVAILABLE as semantic information")
        print(f"📈 Sensors: Accelerometer + Gyroscope + Binary (Game-3)")
        print(f"📈 Users with anthropometric data: {len(self.anthropometric_data)}")
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

    def load_all_anthropometric_data(self):
        """Load anthropometric measurements for all users"""
        anthropometric_data = {}
        
        for user_id in self.all_users:
            measurements_file = f"Data/User {user_id}/measurements.txt"
            
            if os.path.exists(measurements_file):
                try:
                    with open(measurements_file, 'r') as f:
                        content = f.read()
                    
                    measurements = self.parse_anthropometric_data(content, user_id)
                    if measurements:
                        anthropometric_data[user_id] = measurements
                        print(f"📏 User {user_id}: Loaded anthropometric data")
                    else:
                        print(f"⚠️  User {user_id}: Failed to parse anthropometric data")
                except Exception as e:
                    print(f"❌ User {user_id}: Error loading measurements - {e}")
            else:
                print(f"❌ User {user_id}: No measurements.txt file found")
        
        return anthropometric_data

    def parse_anthropometric_data(self, content, user_id):
        """Parse anthropometric measurements from text content"""
        measurements = {}
        
        try:
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip().lower()
                
                # Height
                if 'height' in line:
                    height_match = re.search(r'(\d+\.?\d*)', line)
                    if height_match:
                        measurements['height'] = float(height_match.group(1))
                
                # Leg length
                elif 'leg' in line and 'length' in line:
                    leg_match = re.search(r'(\d+\.?\d*)', line)
                    if leg_match:
                        measurements['leg_length'] = float(leg_match.group(1))
                
                # Arm length
                elif 'arm' in line and 'length' in line:
                    arm_match = re.search(r'(\d+\.?\d*)', line)
                    if arm_match:
                        measurements['arm_length'] = float(arm_match.group(1))
                
                # Torso length
                elif 'torso' in line:
                    torso_match = re.search(r'(\d+\.?\d*)', line)
                    if torso_match:
                        measurements['torso_length'] = float(torso_match.group(1))
                
                # Shoe size
                elif 'shoe' in line:
                    shoe_match = re.search(r'(\d+\.?\d*)', line)
                    if shoe_match:
                        measurements['shoe_size'] = float(shoe_match.group(1))
                    
                    # Gender indicator from shoe size
                    if '(m)' in line:
                        measurements['gender_indicator'] = 1  # Male
                    elif '(w)' in line:
                        measurements['gender_indicator'] = 0  # Female
                    else:
                        measurements['gender_indicator'] = 0.5  # Unknown
                
                # Dominant hand
                elif 'dominant hand' in line:
                    if 'right' in line:
                        measurements['dominant_hand'] = 1  # Right
                    elif 'left' in line:
                        measurements['dominant_hand'] = 0  # Left
                    else:
                        measurements['dominant_hand'] = 0.5  # Unknown
                
                # Dominant foot
                elif 'dominant foot' in line:
                    if 'right' in line:
                        measurements['dominant_foot'] = 1  # Right
                    elif 'left' in line:
                        measurements['dominant_foot'] = 0  # Left
                    else:
                        measurements['dominant_foot'] = 0.5  # Unknown
            
            # Calculate derived anthropometric features
            if 'height' in measurements and 'leg_length' in measurements:
                measurements['leg_to_height_ratio'] = measurements['leg_length'] / measurements['height']
            
            if 'height' in measurements and 'arm_length' in measurements:
                measurements['arm_to_height_ratio'] = measurements['arm_length'] / measurements['height']
            
            if 'height' in measurements and 'torso_length' in measurements:
                measurements['torso_to_height_ratio'] = measurements['torso_length'] / measurements['height']
            
            if 'arm_length' in measurements and 'leg_length' in measurements:
                measurements['arm_to_leg_ratio'] = measurements['arm_length'] / measurements['leg_length']
            
            # BMI proxy (height-based)
            if 'height' in measurements:
                measurements['height_squared'] = measurements['height'] ** 2
            
            return measurements
            
        except Exception as e:
            print(f"❌ Error parsing anthropometric data for User {user_id}: {e}")
            return {}

    def load_user_data_placement_anonymous(self, user_id):
        """
        Load all sensor data from ALL placements but anonymize placement information
        This simulates an adversary who has access to all sensor signals but doesn't 
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
                            # Check if required columns exist
                            if any(col in df.columns for col in self.all_sensor_cols):
                                # Create anonymized placement identifier
                                df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep sensor columns + metadata
                                keep_cols = ['anonymous_placement_id', 'user_id', 'activity']
                                for col in self.all_sensor_cols:
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
                                df = pd.read_csv(file_path)
                                # Check if required columns exist
                                if any(col in df.columns for col in self.all_sensor_cols):
                                    # Create anonymized placement identifier
                                    df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    # Only keep sensor columns + metadata
                                    keep_cols = ['anonymous_placement_id', 'user_id', 'activity']
                                    for col in self.all_sensor_cols:
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
    
    def extract_placement_anonymous_multi_sensor_features(self, df, user_id):
        """
        Extract multi-sensor features WITHOUT using placement information
        WITH anthropometric semantic information for enhanced feature interpretation
        Focus on global sensor patterns that don't rely on placement context
        
        CRITICAL: This function MUST NOT use activity labels for feature extraction
        to ensure no cheating during inference. Anthropometric features represent
        physical capabilities for both linear and rotational dynamics, not activity-specific information.
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique anonymous sensors for this user's data
        unique_sensors = df['anonymous_placement_id'].unique()
        n_sensors = len(unique_sensors)
        
        # GLOBAL MULTI-SENSOR FEATURES (aggregated across all anonymous sensors)
        # IMPORTANT: Only using sensor data, NOT activity labels
        
        # 1. GLOBAL ACCELEROMETER STATISTICAL FEATURES
        for axis_col in self.accel_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                
                # Basic statistics (placement-agnostic)
                features[f'global_accel_{axis}_mean'] = np.mean(data)
                features[f'global_accel_{axis}_std'] = np.std(data)
                features[f'global_accel_{axis}_max'] = np.max(data)
                features[f'global_accel_{axis}_min'] = np.min(data)
                features[f'global_accel_{axis}_median'] = np.median(data)
                features[f'global_accel_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_accel_{axis}_energy'] = np.sum(data**2)
                
                # Advanced statistics
                if len(data) > 1:
                    features[f'global_accel_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_accel_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                    features[f'global_accel_{axis}_rms'] = np.sqrt(np.mean(data**2))
        
        # 2. GLOBAL GYROSCOPE STATISTICAL FEATURES
        for axis_col in self.gyro_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                
                # Basic statistics (placement-agnostic)
                features[f'global_gyro_{axis}_mean'] = np.mean(data)
                features[f'global_gyro_{axis}_std'] = np.std(data)
                features[f'global_gyro_{axis}_max'] = np.max(data)
                features[f'global_gyro_{axis}_min'] = np.min(data)
                features[f'global_gyro_{axis}_median'] = np.median(data)
                features[f'global_gyro_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_gyro_{axis}_energy'] = np.sum(data**2)
                
                # Advanced statistics
                if len(data) > 1:
                    features[f'global_gyro_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_gyro_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                    features[f'global_gyro_{axis}_rms'] = np.sqrt(np.mean(data**2))
        
        # 3. GLOBAL ACCELEROMETER MAGNITUDE FEATURES
        if all(col in df.columns for col in self.accel_cols):
            accel_data = df[self.accel_cols].values
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            
            features['global_accel_magnitude_mean'] = np.mean(accel_magnitude)
            features['global_accel_magnitude_std'] = np.std(accel_magnitude)
            features['global_accel_magnitude_max'] = np.max(accel_magnitude)
            features['global_accel_magnitude_min'] = np.min(accel_magnitude)
            features['global_accel_magnitude_energy'] = np.sum(accel_magnitude**2)
        
        # 4. GLOBAL GYROSCOPE MAGNITUDE FEATURES
        if all(col in df.columns for col in self.gyro_cols):
            gyro_data = df[self.gyro_cols].values
            gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
            
            features['global_gyro_magnitude_mean'] = np.mean(gyro_magnitude)
            features['global_gyro_magnitude_std'] = np.std(gyro_magnitude)
            features['global_gyro_magnitude_max'] = np.max(gyro_magnitude)
            features['global_gyro_magnitude_min'] = np.min(gyro_magnitude)
            features['global_gyro_magnitude_energy'] = np.sum(gyro_magnitude**2)
        
        # 5. GLOBAL BINARY FEATURES (from anonymous sensors)
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
        
        # 6. USER-SPECIFIC PATTERN ENCODING (using user identity semantic info)
        features['user_id'] = user_id  # Use user identity as semantic information
        
        # 7. ANTHROPOMETRIC FEATURES (comprehensive semantic information for multi-sensor interpretation)
        if user_id in self.anthropometric_data:
            anthropo = self.anthropometric_data[user_id]
            
            # Direct anthropometric measurements
            for measurement_key, measurement_value in anthropo.items():
                if isinstance(measurement_value, (int, float)) and not np.isnan(measurement_value):
                    features[f'anthropo_{measurement_key}'] = float(measurement_value)
            
            # Multi-sensor anthropometric interpretations
            if 'height' in anthropo:
                height = anthropo['height']
                
                # Height-normalized accelerometer features
                for axis_col in self.accel_cols:
                    axis = axis_col.replace('[mg]', '').replace('acc_', '')
                    if f'global_accel_{axis}_mean' in features:
                        features[f'height_normalized_accel_{axis}_mean'] = features[f'global_accel_{axis}_mean'] / height
                        features[f'height_normalized_accel_{axis}_std'] = features[f'global_accel_{axis}_std'] / height
                        features[f'height_normalized_accel_{axis}_energy'] = features[f'global_accel_{axis}_energy'] / (height**2)
                
                # Height-normalized gyroscope features
                for axis_col in self.gyro_cols:
                    axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                    if f'global_gyro_{axis}_mean' in features:
                        features[f'height_normalized_gyro_{axis}_mean'] = features[f'global_gyro_{axis}_mean'] / height
                        features[f'height_normalized_gyro_{axis}_std'] = features[f'global_gyro_{axis}_std'] / height
                        features[f'height_normalized_gyro_{axis}_energy'] = features[f'global_gyro_{axis}_energy'] / (height**2)
                
                # Height-normalized magnitude features
                if 'global_accel_magnitude_mean' in features:
                    features['height_normalized_accel_magnitude_mean'] = features['global_accel_magnitude_mean'] / height
                if 'global_gyro_magnitude_mean' in features:
                    features['height_normalized_gyro_magnitude_mean'] = features['global_gyro_magnitude_mean'] / height
            
            # Limb-specific multi-sensor interpretations
            if 'leg_length' in anthropo and 'arm_length' in anthropo:
                leg_length = anthropo['leg_length']
                arm_length = anthropo['arm_length']
                
                # Limb-normalized features for different body parts
                for axis_col in self.accel_cols:
                    axis = axis_col.replace('[mg]', '').replace('acc_', '')
                    if f'global_accel_{axis}_mean' in features:
                        features[f'leg_normalized_accel_{axis}_mean'] = features[f'global_accel_{axis}_mean'] / leg_length
                        features[f'arm_normalized_accel_{axis}_mean'] = features[f'global_accel_{axis}_mean'] / arm_length
                
                for axis_col in self.gyro_cols:
                    axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                    if f'global_gyro_{axis}_mean' in features:
                        features[f'leg_normalized_gyro_{axis}_mean'] = features[f'global_gyro_{axis}_mean'] / leg_length
                        features[f'arm_normalized_gyro_{axis}_mean'] = features[f'global_gyro_{axis}_mean'] / arm_length
            
            # Comprehensive anthropometric encodings
            if 'dominant_hand' in anthropo:
                features['dominant_hand_encoded'] = anthropo['dominant_hand']
            if 'dominant_foot' in anthropo:
                features['dominant_foot_encoded'] = anthropo['dominant_foot']
            if 'gender_indicator' in anthropo:
                features['gender_encoded'] = anthropo['gender_indicator']
            
            # Multi-sensor body dynamics considerations
            if 'height' in anthropo and 'torso_length' in anthropo:
                # Body mass proxy for both linear and rotational dynamics
                body_mass_proxy = anthropo['height'] * anthropo['torso_length']
                features['body_mass_proxy'] = body_mass_proxy
                
                # Combined sensor interpretation with mass considerations
                if body_mass_proxy > 0:
                    if 'global_accel_magnitude_mean' in features:
                        features['mass_normalized_accel_magnitude'] = features['global_accel_magnitude_mean'] / body_mass_proxy
                    if 'global_gyro_magnitude_mean' in features:
                        features['mass_normalized_gyro_magnitude'] = features['global_gyro_magnitude_mean'] / body_mass_proxy
            
            # Sensor fusion anthropometric features
            if 'leg_to_height_ratio' in anthropo:
                features['anthropo_leg_proportion'] = anthropo['leg_to_height_ratio']
            if 'arm_to_height_ratio' in anthropo:
                features['anthropo_arm_proportion'] = anthropo['arm_to_height_ratio']
        
        # 8. SENSOR COUNT METADATA (without placement identity)
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
                        # Extract placement-anonymous multi-sensor features
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_anonymous_multi_sensor_features(df, user_id)
                        
                        if features:
                            X.append(features)
                            y.append(activity)  # Activity as TARGET, not feature
                            metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'n_sensors': features.get('n_anonymous_sensors', 0),
                                'total_samples': features.get('global_binary_total_samples', 0)
                            })
                            user_samples += 1
            
            print(f" ✅ {user_samples} samples")
        
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
            
            # Create RandomForest model (same as original Game-3)
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            
            # Train and predict
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            return {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
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
        
        combinations = self.generate_all_combinations(split_name)
        print(f"📊 Total combinations to test: {len(combinations)}")
        
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
            
            summary = {
                'split_ratio': split_name,
                'total_combinations': len(combinations),
                'successful_combinations': successful_combinations,
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_min': np.min(accuracies),
                'accuracy_max': np.max(accuracies),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'precision_mean': np.mean(precisions),
                'recall_mean': np.mean(recalls),
                'detailed_results': results
            }
            
            print(f"\n📈 SUMMARY FOR {split_name}:")
            print(f"   Successful combinations: {successful_combinations}/{len(combinations)}")
            print(f"   Accuracy: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
            print(f"   Range: [{summary['accuracy_min']:.4f}, {summary['accuracy_max']:.4f}]")
            print(f"   F1-Score: {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
            
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
        print("🚀 STARTING COMPREHENSIVE GAME-3 USER IDENTITY ANALYSIS")
        print("🔬 Testing user identity semantic information effectiveness")
        print("🎭 Placement information is ANONYMIZED")
        
        results = {}
        
        for split_name in self.split_ratios.keys():
            summary = self.analyze_split_ratio(split_name)
            if summary:
                results[split_name] = summary
        
        # Overall summary
        print(f"\n{'='*80}")
        print("🎯 OVERALL GAME-3 USER IDENTITY STUDY SUMMARY")
        print(f"{'='*80}")
        
        for split_name, summary in results.items():
            if summary:
                print(f"{split_name:>5}: {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} "
                      f"(F1: {summary['f1_mean']:.4f})")
        
        print("\n🔍 KEY FINDINGS:")
        print("   • Placement information has been ANONYMIZED")
        print("   • User identity is available as semantic information")
        print("   • All sensor data available (accelerometer + gyroscope + binary)")
        print("   • Results show adversarial effectiveness with user-specific modeling")
        print("   • Cross-sensor and placement-specific features have been omitted")
        print("   • No activity labels used during feature extraction (no cheating)")
        print("   • RandomForest classifier used (same as original Game-3)")
        
        # Save comprehensive report
        report_file = os.path.join(self.base_results_dir, "game3_user_identity_study_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'study_type': 'Game-3 Multi-sensor User Identity Study',
                'semantic_info_removed': ['sensor_placement_identity', 'cross_sensor_coordination', 'placement_specific_features'],
                'semantic_info_available': ['user_identity'],
                'data_available': ['accelerometer', 'gyroscope', 'binary_decision_tree'],
                'placement_anonymous': True,
                'activity_labels_in_features': False,
                'model_type': 'RandomForest',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Complete report saved to: {report_file}")
        return results

if __name__ == "__main__":
    analyzer = Game3UserIdentityAnthropometricsAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    print("\n🎉 Game-3 User Identity Analysis Complete!")
