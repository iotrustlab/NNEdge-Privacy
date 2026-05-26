#!/usr/bin/env python3
"""
Game-2 User Identity + Anthropometrics: Placement-Anonymous Accelerometer Adversary
===================================================================================

This script implements Game-2 accelerometer adversarial HAR with BOTH user identity AND anthropometric
semantic information but WITHOUT placement-specific or cross-sensor features. The adversary has access to:

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- Accelerometer data (acc_x, acc_y, acc_z) from unknown placements
- Binary decision tree outputs from unknown placements
- Global temporal patterns and activity signatures

❌ REMOVED SEMANTIC INFORMATION:
- Sensor placement information (anonymized)
- Placement-specific feature engineering
- Cross-sensor coordination features
- Location-based semantic understanding

RESEARCH QUESTION:
How effective are Game-2 accelerometer adversarial attacks when placement information 
is anonymized but user identity AND anthropometric information are available?

METHODOLOGY:
- Pool accelerometer data from all placements without placement labels
- Extract placement-agnostic accelerometer features
- Use user identity + anthropometric features for enhanced semantic modeling
- Compare with placement-aware Game-2 performance
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import re
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from datetime import datetime
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("🎭 GAME-2 USER IDENTITY + ANTHROPOMETRICS: PLACEMENT-ANONYMOUS ACCELEROMETER ADVERSARY")
print("=" * 85)
print("Testing adversarial effectiveness WITH user identity + anthropometric semantic information")
print("Accelerometer data available | Placement information ANONYMIZED | Anthropometrics AVAILABLE")
print()

class Game2UserIdentityAnthropometricsAnalyzer:
    """
    Analyzer for Game-2 accelerometer attacks with user identity + anthropometrics but
    without placement-specific or cross-sensor semantic information
    """
    
    def __init__(self):
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        
        # Game-2 constraint: Accelerometer + Binary only (NO gyroscope)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.binary_col = 'dec_tree_out_1'
        self.available_cols = self.accel_cols + [self.binary_col]
        
        # Split ratios to test
        self.split_ratios = {
            '8_3': {'train_size': 8, 'test_size': 3},
            '6_5': {'train_size': 6, 'test_size': 5},
            '4_7': {'train_size': 4, 'test_size': 7},
            '1_10': {'train_size': 1, 'test_size': 10}
        }
        
        # Results storage
        self.base_results_dir = "results/game-2-user_identity_anthropometrics"
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Load anthropometric data
        self.anthropometric_data = self.load_all_anthropometric_data()
        
        print(f"🎯 Target: Game-2 accelerometer attacks with user identity + anthropometric semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (ANONYMIZED)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📏 Anthropometrics: AVAILABLE as semantic information")
        print(f"📈 Sensors: Accelerometer + Binary (Game-2 constraint)")
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
            train_test_pairs.append((list(train_users), list(test_users)))
        
        print(f"Split {split_name}: Generated {len(train_test_pairs)} combinations")
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
        Load accelerometer + binary data from ALL placements but anonymize placement information
        This simulates an adversary who has access to accelerometer signals but doesn't 
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
                            if any(col in df.columns for col in self.available_cols):
                                # Create anonymized placement identifier
                                df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep accelerometer + binary columns + metadata
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
                                df = pd.read_csv(file_path)
                                # Check if required columns exist
                                if any(col in df.columns for col in self.available_cols):
                                    # Create anonymized placement identifier
                                    df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    # Only keep accelerometer + binary columns + metadata
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
    
    def extract_placement_anonymous_accelerometer_features(self, df, user_id):
        """
        Extract accelerometer features WITHOUT using placement information
        WITH anthropometric semantic information for enhanced feature interpretation
        Focus on global accelerometer patterns that don't rely on placement context
        
        CRITICAL: This function MUST NOT use activity labels for feature extraction
        to ensure no cheating during inference. Anthropometric features represent
        physical capabilities, not activity-specific information.
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique anonymous sensors for this user's data
        unique_sensors = df['anonymous_placement_id'].unique()
        n_sensors = len(unique_sensors)
        
        # GLOBAL ACCELEROMETER FEATURES (aggregated across all anonymous sensors)
        # IMPORTANT: Only using accelerometer + binary data, NOT activity labels
        
        # 1. GLOBAL ACCELEROMETER STATISTICAL FEATURES
        for axis_col in self.accel_cols:
            if axis_col in df.columns:
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
        
        # 2. GLOBAL MAGNITUDE FEATURES (if all accelerometer axes available)
        if all(col in df.columns for col in self.accel_cols):
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
        
        # 5. ANTHROPOMETRIC FEATURES (enhanced semantic information)
        if user_id in self.anthropometric_data:
            anthropo = self.anthropometric_data[user_id]
            
            # Direct anthropometric measurements
            for measurement_key, measurement_value in anthropo.items():
                if isinstance(measurement_value, (int, float)) and not np.isnan(measurement_value):
                    features[f'anthropo_{measurement_key}'] = float(measurement_value)
            
            # Derived anthropometric features for accelerometer interpretation
            if 'height' in anthropo:
                height = anthropo['height']
                
                # Scale accelerometer features by anthropometric characteristics
                for axis_col in self.accel_cols:
                    axis = axis_col.replace('[mg]', '').replace('acc_', '')
                    if f'global_{axis}_mean' in features:
                        # Height-normalized accelerometer patterns
                        features[f'height_normalized_{axis}_mean'] = features[f'global_{axis}_mean'] / height
                        features[f'height_normalized_{axis}_std'] = features[f'global_{axis}_std'] / height
                        features[f'height_normalized_{axis}_energy'] = features[f'global_{axis}_energy'] / (height**2)
                
                # Height-normalized magnitude features
                if 'global_magnitude_mean' in features:
                    features['height_normalized_magnitude_mean'] = features['global_magnitude_mean'] / height
                    features['height_normalized_magnitude_std'] = features['global_magnitude_std'] / height
                    features['height_normalized_magnitude_energy'] = features['global_magnitude_energy'] / (height**2)
            
            # Limb-specific feature interpretations
            if 'leg_length' in anthropo and 'arm_length' in anthropo:
                leg_length = anthropo['leg_length']
                arm_length = anthropo['arm_length']
                
                # Limb-normalized accelerometer features (for different body parts)
                for axis_col in self.accel_cols:
                    axis = axis_col.replace('[mg]', '').replace('acc_', '')
                    if f'global_{axis}_mean' in features:
                        # Leg-normalized (for ankle/walking patterns)
                        features[f'leg_normalized_{axis}_mean'] = features[f'global_{axis}_mean'] / leg_length
                        # Arm-normalized (for wrist patterns)
                        features[f'arm_normalized_{axis}_mean'] = features[f'global_{axis}_mean'] / arm_length
            
            # Dominance-based feature interpretations
            if 'dominant_hand' in anthropo:
                features['dominant_hand_encoded'] = anthropo['dominant_hand']
            if 'dominant_foot' in anthropo:
                features['dominant_foot_encoded'] = anthropo['dominant_foot']
            if 'gender_indicator' in anthropo:
                features['gender_encoded'] = anthropo['gender_indicator']
        
        # 6. SENSOR COUNT METADATA (without placement identity)
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
                        # Extract placement-anonymous accelerometer features
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_anonymous_accelerometer_features(df, user_id)
                        
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
            
            # Train VotingClassifier ensemble (same as original Game-2)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
            svm = SVC(random_state=42, class_weight='balanced', probability=True)
            
            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
                voting='soft'
            )
            
            ensemble.fit(X_train_scaled, y_train_encoded)
            
            # Predict
            y_pred = ensemble.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            return {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'n_train_samples': len(X_train_aligned),
                'n_test_samples': len(X_test_aligned),
                'n_features': len(common_cols),
                'has_user_identity': True,
                'placement_anonymous': True
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
        
        stats = {
            'split_ratio': split_name,
            'has_user_identity': True,
            'placement_anonymous': True,
            'game_type': 'Game-2 Accelerometer',
            'n_combinations': len(results),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_min': np.min(accuracies),
            'accuracy_max': np.max(accuracies),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_mean': np.mean(precisions),
            'recall_mean': np.mean(recalls),
            'avg_train_samples': np.mean([r['n_train_samples'] for r in results]),
            'avg_test_samples': np.mean([r['n_test_samples'] for r in results]),
            'avg_features': np.mean([r['n_features'] for r in results])
        }
        
        # Print summary
        print(f"\n📊 PLACEMENT-ANONYMOUS GAME-2 ACCELEROMETER RESULTS (WITH user identity):")
        print(f"   Accuracy:  {stats['accuracy_mean']:.3f} ± {stats['accuracy_std']:.3f}")
        print(f"   F1-Score:  {stats['f1_mean']:.3f} ± {stats['f1_std']:.3f}")
        print(f"   Precision: {stats['precision_mean']:.3f}")
        print(f"   Recall:    {stats['recall_mean']:.3f}")
        print(f"   Features:  {stats['avg_features']:.0f} (placement-anonymous + user identity)")
        
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
        Run complete user identity study with placement-anonymous Game-2 accelerometer
        """
        print("🚀 Starting Game-2 Accelerometer User Identity Study...")
        print("Testing adversarial effectiveness WITH user identity semantic information")
        print("Accelerometer + Binary data available, placement information anonymized")
        print()
        
        all_results = {}
        
        # Test each split ratio with user identity
        for split_name in self.split_ratios.keys():
            print(f"\n🔬 Testing {split_name} WITH user identity semantic information...")
            stats = self.analyze_split_ratio(split_name)
            all_results[split_name] = stats
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, results):
        """Generate a summary report of Game-2 user identity study results"""
        print(f"\n{'='*100}")
        print("🎭 GAME-2 ACCELEROMETER USER IDENTITY STUDY - SUMMARY REPORT")
        print(f"{'='*100}")
        
        print("\n📊 ADVERSARIAL HAR EFFECTIVENESS WITH USER IDENTITY:")
        print("   Accelerometer + Binary data | Placement information ANONYMIZED | User identity AVAILABLE")
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
        print("   • Accelerometer + binary data available (Game-2 constraint)")
        print("   • Results show adversarial effectiveness with user-specific modeling")
        print("   • Cross-sensor and placement-specific features have been omitted")
        print("   • No activity labels used during feature extraction (no cheating)")
        print("   • VotingClassifier ensemble used (same as original Game-2)")
        
        # Save comprehensive report
        report_file = os.path.join(self.base_results_dir, "game2_user_identity_study_report.json")
        with open(report_file, 'w') as f:
            json.dump({
                'study_type': 'Game-2 Accelerometer User Identity Study',
                'semantic_info_removed': ['sensor_placement_identity', 'cross_sensor_coordination', 'placement_specific_features'],
                'semantic_info_available': ['user_identity'],
                'data_available': ['accelerometer', 'binary_decision_tree'],
                'placement_anonymous': True,
                'activity_labels_in_features': False,
                'ensemble_method': 'VotingClassifier',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Complete report saved to: {report_file}")


if __name__ == "__main__":
    analyzer = Game2UserIdentityAnthropometricsAnalyzer()
    results = analyzer.run_user_identity_study()
