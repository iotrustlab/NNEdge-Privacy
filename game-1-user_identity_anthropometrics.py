#!/usr/bin/env python3
"""
Game-1 User Identity + Anthropometrics: Placement-Anonymous Binary-Only Adversary
=================================================================================

This script implements Game-1 adversarial HAR with BOTH user identity AND anthropometric
semantic information but WITHOUT placement information. The adversary has access to:

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- Binary decision tree outputs from unknown placements
- Temporal patterns and activity signatures

❌ REMOVED SEMANTIC INFORMATION:
- Sensor placement information (anonymized)
- Placement-specific feature engineering
- Location-based semantic understanding
- Cross-sensor coordination features (omitted)

RESEARCH QUESTION:
How effective are adversarial HAR attacks when placement information 
is anonymized but user identity AND anthropometric information are available?

METHODOLOGY:
- Pool all binary signals from all placements without placement labels
- Extract placement-agnostic temporal features
- Use user identity + anthropometric features for enhanced semantic modeling
- Evaluate adversarial effectiveness with combined semantic information
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
import re
import warnings
warnings.filterwarnings('ignore')

print("🎭 GAME-1 USER IDENTITY + ANTHROPOMETRICS: PLACEMENT-ANONYMOUS ADVERSARY")
print("=" * 75)
print("Testing adversarial effectiveness WITH user identity + anthropometric semantic information")
print("Placement information ANONYMIZED | User identity + Anthropometrics AVAILABLE")
print()

class Game1UserIdentityAnthropometricsAnalyzer:
    """
    Analyzer that removes placement information while using user identity + anthropometrics
    for adversarial HAR attacks with semantic information
    """
    
    def __init__(self):
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (but we'll anonymize them)
        self.all_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        
        # Split ratios to test
        self.split_ratios = {
            '8_3': {'train_size': 8, 'test_size': 3},
            '6_5': {'train_size': 6, 'test_size': 5},
            '4_7': {'train_size': 4, 'test_size': 7},
            '1_10': {'train_size': 1, 'test_size': 10}
        }
        
        # Results storage
        self.base_results_dir = "results/game-1-user_identity_anthropometrics"
        os.makedirs(self.base_results_dir, exist_ok=True)
        
        # Load anthropometric data
        self.anthropometric_data = self.load_all_anthropometric_data()
        
        print(f"🎯 Target: Adversarial HAR with user identity + anthropometric semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (ANONYMIZED)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📏 Anthropometrics: AVAILABLE as semantic information")
        print(f"📈 Users with anthropometric data: {len(self.anthropometric_data)}")
        print()

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
            print(f"❌ Error parsing measurements for User {user_id}: {e}")
            return None

    def get_anthropometric_features(self, user_id):
        """Get anthropometric features for a user"""
        if user_id in self.anthropometric_data:
            return self.anthropometric_data[user_id].copy()
        else:
            # Return default/missing values for users without anthropometric data
            return {
                'height': 67.0,  # Average height
                'leg_length': 40.0,  # Average leg length
                'arm_length': 22.0,  # Average arm length
                'torso_length': 17.0,  # Average torso length
                'shoe_size': 8.0,  # Average shoe size
                'gender_indicator': 0.5,  # Unknown
                'dominant_hand': 0.5,  # Unknown
                'dominant_foot': 0.5,  # Unknown
                'leg_to_height_ratio': 0.6,  # Average ratio
                'arm_to_height_ratio': 0.33,  # Average ratio
                'torso_to_height_ratio': 0.25,  # Average ratio
                'arm_to_leg_ratio': 0.55,  # Average ratio
                'height_squared': 67.0 ** 2,  # Average height squared
                'has_anthropometric_data': 0  # Missing data indicator
            }

    def get_intelligent_anthropometric_features(self, user_id):
        """
        Create intelligent anthropometric features that are biomechanically relevant to HAR
        These features help the model understand physical capabilities and constraints
        
        CRITICAL: NO activity information used - features are completely activity-agnostic
        Features represent physical capabilities that COULD affect activity patterns
        """
        base_anthro = self.get_anthropometric_features(user_id)
        
        # Create intelligent features that are relevant to HAR but activity-agnostic
        intelligent_features = base_anthro.copy()
        
        # 1. GAIT AND LOCOMOTION POTENTIAL (physical capability, not activity-specific)
        if 'leg_length' in base_anthro and 'height' in base_anthro:
            # Stride potential - longer legs = longer strides
            intelligent_features['stride_potential'] = base_anthro['leg_length'] / base_anthro['height']
            
            # Center of gravity height approximation
            intelligent_features['cog_height_ratio'] = (base_anthro['height'] - base_anthro['leg_length']) / base_anthro['height']
        
        # 2. BALANCE AND STABILITY POTENTIAL (physical capability, not activity-specific)
        if 'torso_length' in base_anthro and 'leg_length' in base_anthro:
            # Stability index - torso to leg ratio affects balance capability
            intelligent_features['stability_index'] = base_anthro['torso_length'] / base_anthro['leg_length']
        
        # 3. REACH AND ARM MOVEMENT POTENTIAL (physical capability, not activity-specific)
        if 'arm_length' in base_anthro and 'height' in base_anthro:
            # Reach ratio - affects arm swing potential
            intelligent_features['reach_ratio'] = base_anthro['arm_length'] / base_anthro['height']
            
            # Arm span approximation (arm length * 2 typically ≈ height)
            intelligent_features['arm_span_deviation'] = (base_anthro['arm_length'] * 2) / base_anthro['height']
        
        # 4. BODY PROPORTIONALITY (inherent physical characteristics)
        if all(k in base_anthro for k in ['height', 'leg_length', 'arm_length', 'torso_length']):
            # Body mass distribution proxy
            total_length = base_anthro['leg_length'] + base_anthro['arm_length'] + base_anthro['torso_length']
            intelligent_features['leg_proportion'] = base_anthro['leg_length'] / total_length
            intelligent_features['arm_proportion'] = base_anthro['arm_length'] / total_length
            intelligent_features['torso_proportion'] = base_anthro['torso_length'] / total_length
        
        # 5. FOOT BIOMECHANICS (inherent physical characteristics)
        if 'shoe_size' in base_anthro and 'height' in base_anthro:
            # Foot size to height ratio affects gait potential
            intelligent_features['foot_size_ratio'] = base_anthro['shoe_size'] / base_anthro['height']
        
        # 6. LATERALITY (inherent neuromotor characteristics)
        if 'dominant_hand' in base_anthro and 'dominant_foot' in base_anthro:
            # Consistency of dominance
            intelligent_features['dominance_consistency'] = 1.0 if base_anthro['dominant_hand'] == base_anthro['dominant_foot'] else 0.0
            
            # Laterality strength (how much dominant vs non-dominant)
            if base_anthro['dominant_hand'] in [0, 1] and base_anthro['dominant_foot'] in [0, 1]:
                intelligent_features['laterality_strength'] = abs(base_anthro['dominant_hand'] - 0.5) + abs(base_anthro['dominant_foot'] - 0.5)
        
        # 7. BIOMECHANICAL EFFICIENCY POTENTIAL (inherent physical ratios)
        if 'leg_length' in base_anthro and 'arm_length' in base_anthro:
            # Limb proportion efficiency
            intelligent_features['limb_efficiency'] = base_anthro['arm_length'] / base_anthro['leg_length']
        
        # 8. GENDER-BASED BIOMECHANICAL DIFFERENCES (inherent anatomical characteristics)
        if 'gender_indicator' in base_anthro:
            # Gender affects movement patterns due to anatomical differences
            intelligent_features['gender_biomechanics'] = base_anthro['gender_indicator']
        
        # 9. BODY PROPORTION DEVIATION FROM POPULATION NORMS (inherent characteristics)
        if all(k in base_anthro for k in ['height', 'leg_length', 'arm_length']):
            # Deviation from average proportions
            avg_leg_ratio = 0.6  # Average leg to height ratio
            avg_arm_ratio = 0.33  # Average arm to height ratio
            
            intelligent_features['leg_proportion_deviation'] = abs((base_anthro['leg_length'] / base_anthro['height']) - avg_leg_ratio)
            intelligent_features['arm_proportion_deviation'] = abs((base_anthro['arm_length'] / base_anthro['height']) - avg_arm_ratio)
        
        return intelligent_features

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
                            if self.binary_col in df.columns:
                                # Create anonymized placement identifier
                                df['anonymous_placement_id'] = f"sensor_{placement_idx}"
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep binary column + metadata
                                df_clean = df[['anonymous_placement_id', 'user_id', 'activity', self.binary_col]].copy()
                                activity_data.append(df_clean)
                                
                        except Exception as e:
                            print(f"❌ Error loading {file_path}: {e}")
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
        Extract features from placement-anonymous data WITH user identity + intelligent anthropometrics
        
        CRITICAL: NO activity information is used in feature extraction
        - Activity labels are NEVER used during feature creation
        - Features are extracted ONLY from sensor data patterns
        - Anthropometric features represent physical capabilities, not activity-specific information
        - All features are activity-agnostic and represent inherent capabilities/characteristics
        """
        features = {}
        
        # Get all binary data across all anonymized sensors
        all_binary_data = df[self.binary_col].values
        unique_sensors = df['anonymous_placement_id'].unique()
        n_sensors = len(unique_sensors)
        
        if len(all_binary_data) == 0:
            return None
        
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
        
        # 4. INTELLIGENT ANTHROPOMETRIC SEMANTIC INFORMATION (activity-agnostic)
        # CRITICAL: These features represent physical capabilities only, no activity info used
        anthro_features = self.get_intelligent_anthropometric_features(user_id)
        
        # Add intelligent anthropometric features
        for key, value in anthro_features.items():
            features[f'anthro_{key}'] = value
        
        # 5. TEMPORAL PATTERN COMPLEXITY (placement-independent, activity-independent)
        features['pattern_complexity'] = len(np.unique(all_binary_data))
        features['temporal_variance'] = np.var(all_binary_data)
        
        # 6. BASIC SENSOR COUNT (without coordination features)
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

    def create_dataset(self, user_list):
        """
        Create dataset from specified users WITHOUT placement information
        WITH user identity + anthropometric information as semantic information
        
        CRITICAL: Activity labels are used ONLY as targets, NEVER as features
        """
        X = []
        y = []
        metadata = []
        
        for user_id in user_list:
            print(f"📊 Processing User {user_id} (placement-anonymous + anthropometrics)...", end="")
            user_data = self.load_user_data_placement_anonymous(user_id)
            
            if not user_data:
                print(" ❌ No data")
                continue
            
            user_samples = 0
            for activity in self.activities:
                if activity in user_data:
                    df = user_data[activity]
                    if len(df) > 0:
                        # Extract placement-anonymous features + anthropometrics
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_anonymous_features(df, user_id)
                        
                        if features:
                            X.append(features)
                            y.append(activity)  # Activity as TARGET, not feature
                            metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'n_sensors': features.get('n_anonymous_sensors', 0),
                                'total_samples': features.get('global_total_samples', 0),
                                'has_anthropometric_data': features.get('anthro_has_anthropometric_data', 0)
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
        print(f"📏 Basic anthropometric features: {len([c for c in X_df.columns if c.startswith('anthro_')])} features")
        print(f"🧠 Activity-aware anthropometric features: {len([c for c in X_df.columns if c.startswith('activity_anthro_')])} features")
        print(f"🎯 Total semantic features: {len([c for c in X_df.columns if c.startswith(('user_', 'anthro_', 'activity_anthro_'))])} features")
        return X_df.values, np.array(y), metadata

    def evaluate_single_combination(self, train_users, test_users):
        """Evaluate a single train/test combination with user identity + anthropometric semantic information"""
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
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train_encoded)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted')
            recall = recall_score(y_test_encoded, y_pred, average='weighted')
            
            # Generate detailed report
            report = classification_report(y_test_encoded, y_pred, 
                                         target_names=le.classes_, 
                                         output_dict=True)
            
            result = {
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'n_train_samples': len(X_train_scaled),
                'n_test_samples': len(X_test_scaled),
                'n_features': len(common_cols),
                'classification_report': report,
                'train_activities': list(set(y_train)),
                'test_activities': list(set(y_test)),
                'has_user_identity': True,
                'has_anthropometrics': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating combination: {e}")
            return None

    def evaluate_split(self, split_name):
        """Evaluate all combinations for a specific split ratio"""
        print(f"\n🎯 Evaluating split {split_name}")
        print("=" * 50)
        
        combinations = self.generate_all_combinations(split_name)
        results = []
        
        for i, (train_users, test_users) in enumerate(combinations):
            print(f"\n📊 Combination {i+1}/{len(combinations)}")
            print(f"   Train users: {train_users}")
            print(f"   Test users: {test_users}")
            
            result = self.evaluate_single_combination(train_users, test_users)
            if result:
                results.append(result)
                print(f"   ✅ Accuracy: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}")
            else:
                print(f"   ❌ Failed")
        
        # Calculate summary statistics
        if results:
            accuracies = [r['accuracy'] for r in results]
            f1_scores = [r['f1_score'] for r in results]
            
            summary = {
                'split_name': split_name,
                'n_combinations': len(results),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_min': np.min(accuracies),
                'accuracy_max': np.max(accuracies),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'f1_min': np.min(f1_scores),
                'f1_max': np.max(f1_scores),
                'results': results,
                'semantic_info': ['user_identity', 'anthropometrics'],
                'has_user_identity': True,
                'has_anthropometrics': True
            }
            
            print(f"\n📈 SUMMARY for {split_name}:")
            print(f"   Accuracy: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
            print(f"   Range: [{summary['accuracy_min']:.3f} - {summary['accuracy_max']:.3f}]")
            print(f"   F1-Score: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
            
            # Save results
            results_file = os.path.join(self.base_results_dir, f"{split_name}_with_user_identity_anthropometrics.json")
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return summary
        else:
            print(f"❌ No valid results for {split_name}")
            return None

    def run_user_identity_anthropometrics_study(self):
        """Run complete study with user identity + anthropometric semantic information"""
        print("🚀 STARTING USER IDENTITY + ANTHROPOMETRICS STUDY")
        print("=" * 55)
        
        all_results = {}
        
        for split_name in self.split_ratios.keys():
            summary = self.evaluate_split(split_name)
            if summary:
                all_results[split_name] = summary
        
        # Generate final report
        if all_results:
            print(f"\n🎉 STUDY COMPLETED")
            print("=" * 25)
            
            for split_name, summary in all_results.items():
                print(f"{split_name}: {summary['accuracy_mean']:.3f} ± {summary['accuracy_std']:.3f}")
            
            # Save comprehensive report
            report = {
                'study_type': 'user_identity_anthropometrics',
                'timestamp': datetime.now().isoformat(),
                'semantic_info_available': ['user_identity', 'anthropometrics'],
                'semantic_info_removed': ['placement_information', 'cross_sensor_coordination'],
                'results_by_split': all_results,
                'anthropometric_data_coverage': len(self.anthropometric_data),
                'total_users': len(self.all_users)
            }
            
            report_file = os.path.join(self.base_results_dir, "user_identity_anthropometrics_study_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📄 Comprehensive report saved: {report_file}")
            return report
        else:
            print("❌ Study failed - no valid results")
            return None

if __name__ == "__main__":
    analyzer = Game1UserIdentityAnthropometricsAnalyzer()
    results = analyzer.run_user_identity_anthropometrics_study()
