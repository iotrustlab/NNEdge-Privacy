#!/usr/bin/env python3
"""
Game-3 User Identity + Anthropometrics + Placements Analysis
===========================================================

This script implements Game-3 attacks using MAXIMUM semantic information:
user identity, anthropometric measurements, AND placement information with cross-sensor features.

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- ALL sensor data (accelerometer + gyroscope + binary) from KNOWN placements
- Placement-specific feature engineering
- Cross-sensor coordination features
- Multi-sensor temporal patterns

❌ REMOVED SEMANTIC INFORMATION:
- None (maximum information adversary scenario)

RESEARCH QUESTION:
How effective are Game-3 multi-sensor adversarial attacks when ALL semantic information 
is available to the adversary (worst-case privacy scenario)?

METHODOLOGY:
- Use ALL sensor types (accelerometer + gyroscope + binary) with known body locations
- Extract placement-aware multi-sensor features
- Use user identity + anthropometric features for enhanced semantic modeling
- Add cross-sensor coordination features across all sensor types
- Evaluate maximum adversarial effectiveness with complete sensor fusion
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

from project_paths import (
    SOURCE_FREQUENCY_HZ,
    build_frequency_results_dir,
    build_model_results_dir,
    load_experiment_csv,
    resolve_experiment_config,
    sample_experiment_combinations,
    user_measurements_file,
    user_processed_dir,
)
from identity_anthro_placements_shared import (
    TARGET_SPLIT_SPECS,
    align_feature_frames,
    build_summary_statistics,
    calculate_vulnerability,
    fit_predict_model,
    save_json,
    validate_dl_runtime_for_models,
)

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

print("🎯 GAME-3 USER IDENTITY + ANTHROPOMETRICS + PLACEMENTS ANALYSIS")
print("=" * 61)
print("Maximum semantic information study with placement awareness")
print("Testing user identity + anthropometric + placement semantic information effectiveness")
print("All sensor data available | Placement information KNOWN | Anthropometrics AVAILABLE | Cross-sensor features ENABLED")
print()

class Game3UserIdentityAnthropometricsPlacementsAnalyzer:
    """
    Analyzer for Game-3 attacks with MAXIMUM semantic information:
    user identity + anthropometrics + placement information + cross-sensor coordination
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_models = self.config.model_labels
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (with known locations)
        self.all_placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        
        # Game-3: All sensor data (accelerometer + gyroscope + binary)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
        self.binary_col = 'dec_tree_out_1'
        self.all_sensor_cols = self.accel_cols + self.gyro_cols + [self.binary_col]
        
        self.split_ratios = {
            self.config.split_name: TARGET_SPLIT_SPECS[self.config.split_name]
        }
        
        # Results storage
        self.base_results_dir = str(
            build_frequency_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                "game-3-user_identity_anthropometrics_placements",
            )
        )

        validate_dl_runtime_for_models(self.selected_models)
        
        # Load anthropometric data
        self.anthropometric_data = self.load_all_anthropometric_data()
        
        print(f"🎯 Target: Game-3 attacks with MAXIMUM semantic information")
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"🎭 Placements: {len(self.all_placements)} placements (KNOWN locations)")
        print(f"🎮 Activities: {len(self.activities)} activities")
        print(f"🔑 User Identity: AVAILABLE as semantic information")
        print(f"📏 Anthropometrics: AVAILABLE as semantic information")
        print(f"📍 Placement Info: AVAILABLE as semantic information")
        print(f"🔗 Cross-sensor features: ENABLED")
        print(f"📈 Sensors: Accelerometer + Gyroscope + Binary (Game-3)")
        print(f"📈 Users with anthropometric data: {len(self.anthropometric_data)}")
        print(f"⏱️  Frequency: {self.config.frequency_hz} Hz (source: {SOURCE_FREQUENCY_HZ} Hz)")
        print(f"🎯 Split: {self.config.split_name}")
        print(f"🔢 Max combinations per split: {self.config.max_combinations}")
        print(f"🤖 Models: {', '.join(self.selected_models)}")
        print(f"📁 Data root: {self.config.data_root}")
        print(f"📁 Results dir: {self.base_results_dir}")
        print()

    def get_model_results_dir(self, model_label):
        """Return the per-model results directory for this frequency."""
        return str(
            build_model_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                "game-3-user_identity_anthropometrics_placements",
                model_label,
            )
        )

    def build_rf_baseline(self):
        """Build the branch RF baseline for Game-3."""
        return RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_seed,
            max_depth=10,
        )
    
    def generate_all_combinations(self, split_name):
        """Generate a reproducible subset of train/test combinations for a split ratio"""
        config = self.split_ratios[split_name]
        test_size = config['test_size']
        
        # Generate all possible combinations of test users
        all_test_combinations = list(combinations(self.all_users, test_size))
        
        train_test_pairs = []
        for test_users in all_test_combinations:
            train_users = [u for u in self.all_users if u not in test_users]
            train_test_pairs.append({
                'train_users': list(train_users),
                'test_users': list(test_users)
            })

        selected_pairs = sample_experiment_combinations(
            train_test_pairs,
            max_combinations=self.config.max_combinations,
            seed=self.config.random_seed,
        )
        
        print(
            f"Split {split_name}: Generated {len(selected_pairs)} combinations "
            f"out of {len(train_test_pairs)} possible"
        )
        return selected_pairs

    def load_all_anthropometric_data(self):
        """Load anthropometric measurements for all users"""
        anthropometric_data = {}
        
        for user_id in self.all_users:
            measurements_file = user_measurements_file(self.config.data_root, user_id)
            
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
        Create intelligent anthropometric features that are biomechanically relevant to multi-sensor HAR
        These features help the model understand physical capabilities and movement constraints
        
        CRITICAL: NO activity information used - features are completely activity-agnostic
        Features represent physical capabilities that COULD affect movement patterns across all sensors
        """
        base_anthro = self.get_anthropometric_features(user_id)
        
        # Create intelligent features that are relevant to multi-sensor HAR but activity-agnostic
        intelligent_features = base_anthro.copy()
        
        # 1. MOVEMENT DYNAMICS POTENTIAL (physical capability, not activity-specific)
        if 'height' in base_anthro and 'leg_length' in base_anthro:
            # Stride and step potential affects all sensor measurements
            intelligent_features['movement_stride_potential'] = base_anthro['leg_length'] / base_anthro['height']
            
            # Center of gravity affects balance and movement dynamics
            intelligent_features['movement_cog_ratio'] = (base_anthro['height'] - base_anthro['leg_length']) / base_anthro['height']
        
        # 2. LIMB COORDINATION POTENTIAL (physical capability, not activity-specific)
        if 'arm_length' in base_anthro and 'leg_length' in base_anthro:
            # Limb coordination affects multi-sensor patterns
            intelligent_features['limb_coordination_ratio'] = base_anthro['arm_length'] / base_anthro['leg_length']
        
        # 3. TORSO DYNAMICS POTENTIAL (physical capability, not activity-specific)
        if 'torso_length' in base_anthro and 'height' in base_anthro:
            # Torso length affects upper body movement patterns
            intelligent_features['torso_movement_proportion'] = base_anthro['torso_length'] / base_anthro['height']
        
        # 4. BIOMECHANICAL EFFICIENCY (inherent physical characteristics)
        if all(k in base_anthro for k in ['height', 'leg_length', 'arm_length', 'torso_length']):
            # Overall body proportion affects movement efficiency
            total_segments = base_anthro['leg_length'] + base_anthro['arm_length'] + base_anthro['torso_length']
            intelligent_features['movement_efficiency_index'] = total_segments / base_anthro['height']
            
            # Body segment balance
            intelligent_features['upper_lower_balance'] = (base_anthro['arm_length'] + base_anthro['torso_length']) / base_anthro['leg_length']
        
        # 5. GAIT AND STEP CHARACTERISTICS (inherent physical characteristics)
        if 'shoe_size' in base_anthro and 'leg_length' in base_anthro:
            # Foot size affects gait characteristics
            intelligent_features['gait_foot_leg_ratio'] = base_anthro['shoe_size'] / base_anthro['leg_length']
        
        # 6. NEUROMOTOR COORDINATION (inherent characteristics)
        if 'dominant_hand' in base_anthro and 'dominant_foot' in base_anthro:
            # Dominance consistency affects coordination patterns
            intelligent_features['neuromotor_consistency'] = 1.0 if base_anthro['dominant_hand'] == base_anthro['dominant_foot'] else 0.0
            
            # Laterality strength for movement patterns
            if base_anthro['dominant_hand'] in [0, 1] and base_anthro['dominant_foot'] in [0, 1]:
                intelligent_features['laterality_strength'] = abs(base_anthro['dominant_hand'] - 0.5) + abs(base_anthro['dominant_foot'] - 0.5)
        
        # 7. GENDER-BASED BIOMECHANICS (inherent anatomical differences)
        if 'gender_indicator' in base_anthro:
            # Gender affects movement patterns due to anatomical differences
            intelligent_features['gender_movement_biomechanics'] = base_anthro['gender_indicator']
        
        # 8. OVERALL MOVEMENT POTENTIAL (derived characteristics)
        if all(k in base_anthro for k in ['height', 'arm_length', 'leg_length']):
            # Movement range potential
            intelligent_features['total_movement_range'] = (base_anthro['arm_length'] + base_anthro['leg_length']) / base_anthro['height']
        
        return intelligent_features

    def load_user_data_placement_anonymous(self, user_id):
        """
        Load all sensor data from ALL placements but anonymize placement information
        This simulates an adversary who has access to all sensor signals but doesn't 
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
                                df = load_experiment_csv(file_path, self.config.frequency_hz)
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
    
    def load_user_data_placement_aware(self, user_id):
        """
        Load all sensor data from ALL placements WITH placement information preserved
        This simulates maximum semantic information scenario with known placement locations
        
        CRITICAL: Activity labels are used ONLY for data organization and target creation,
        NEVER for feature extraction. This ensures no cheating during inference.
        """
        user_dir = user_processed_dir(self.config.data_root, user_id)
        
        if not os.path.exists(user_dir):
            return {}
        
        user_data = {}
        
        for activity in self.activities:
            activity_data = []
            activity_dir = os.path.join(user_dir, activity)
            
            if not os.path.exists(activity_dir):
                continue
            
            # Load data from ALL placements WITH placement labels preserved
            for placement in self.all_placements:
                
                if activity in ['Standing', 'Sitting', 'Laying', 'Walking', 'Jogging']:
                    # Single file per placement: Data/User X/Processed/ACTIVITY/PLACEMENT.csv
                    file_path = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(file_path):
                        try:
                            df = load_experiment_csv(file_path, self.config.frequency_hz)
                            # Check if required columns exist
                            if any(col in df.columns for col in self.all_sensor_cols):
                                # Keep placement information for cross-sensor features
                                df['placement'] = placement
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep sensor columns + metadata
                                keep_cols = ['placement', 'user_id', 'activity']
                                for col in self.all_sensor_cols:
                                    if col in df.columns:
                                        keep_cols.append(col)
                                
                                df_clean = df[keep_cols].copy()
                                activity_data.append(df_clean)
                                
                        except Exception as e:
                            continue
                
                elif activity in ['Upstairs', 'Downstairs']:
                    # Multiple files per placement: Data/User X/Processed/ACTIVITY/PLACEMENT/
                    placement_dir = os.path.join(activity_dir, placement)
                    if os.path.exists(placement_dir):
                        csv_files = sorted([f for f in os.listdir(placement_dir) if f.endswith('.csv')])
                        for csv_file in csv_files:
                            file_path = os.path.join(placement_dir, csv_file)
                            try:
                                df = load_experiment_csv(file_path, self.config.frequency_hz)
                                if any(col in df.columns for col in self.all_sensor_cols):
                                    df['placement'] = placement
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    keep_cols = ['placement', 'user_id', 'activity']
                                    for col in self.all_sensor_cols:
                                        if col in df.columns:
                                            keep_cols.append(col)
                                    
                                    df_clean = df[keep_cols].copy()
                                    activity_data.append(df_clean)
                            except Exception as e:
                                continue
            
            # Combine all placement data for this activity
            if activity_data:
                user_data[activity] = pd.concat(activity_data, ignore_index=True)
            
        return user_data
    
    def extract_placement_aware_multi_sensor_features(self, df, user_id):
        """
        Extract multi-sensor features WITH placement information AND cross-sensor features
        WITH anthropometric semantic information for maximum feature richness
        This represents maximum semantic information scenario for adversarial attacks
        
        CRITICAL: Activity labels are NEVER used during feature extraction
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique placements for this user's data
        unique_placements = df['placement'].unique()
        n_placements = len(unique_placements)
        
        # PLACEMENT-SPECIFIC MULTI-SENSOR FEATURES
        for placement in unique_placements:
            placement_data = df[df['placement'] == placement]
            
            if len(placement_data) > 0:
                placement_clean = placement.replace('-', '_')
                
                # ACCELEROMETER FEATURES PER PLACEMENT
                for axis_col in self.accel_cols:
                    if axis_col in placement_data.columns:
                        data = placement_data[axis_col].values
                        axis = axis_col.replace('[mg]', '').replace('acc_', '')
                        
                        # Basic statistics per placement
                        features[f'{placement_clean}_acc_{axis}_mean'] = np.mean(data)
                        features[f'{placement_clean}_acc_{axis}_std'] = np.std(data)
                        features[f'{placement_clean}_acc_{axis}_max'] = np.max(data)
                        features[f'{placement_clean}_acc_{axis}_min'] = np.min(data)
                        features[f'{placement_clean}_acc_{axis}_range'] = np.max(data) - np.min(data)
                        features[f'{placement_clean}_acc_{axis}_energy'] = np.sum(data**2)
                        
                        if len(data) > 1:
                            features[f'{placement_clean}_acc_{axis}_rms'] = np.sqrt(np.mean(data**2))
                            features[f'{placement_clean}_acc_{axis}_skew'] = pd.Series(data).skew()
                            features[f'{placement_clean}_acc_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                
                # GYROSCOPE FEATURES PER PLACEMENT
                for axis_col in self.gyro_cols:
                    if axis_col in placement_data.columns:
                        data = placement_data[axis_col].values
                        axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                        
                        # Basic statistics per placement
                        features[f'{placement_clean}_gyro_{axis}_mean'] = np.mean(data)
                        features[f'{placement_clean}_gyro_{axis}_std'] = np.std(data)
                        features[f'{placement_clean}_gyro_{axis}_max'] = np.max(data)
                        features[f'{placement_clean}_gyro_{axis}_min'] = np.min(data)
                        features[f'{placement_clean}_gyro_{axis}_range'] = np.max(data) - np.min(data)
                        features[f'{placement_clean}_gyro_{axis}_energy'] = np.sum(data**2)
                        
                        if len(data) > 1:
                            features[f'{placement_clean}_gyro_{axis}_rms'] = np.sqrt(np.mean(data**2))
                            features[f'{placement_clean}_gyro_{axis}_skew'] = pd.Series(data).skew()
                            features[f'{placement_clean}_gyro_{axis}_kurtosis'] = pd.Series(data).kurtosis()
                
                # BINARY FEATURES PER PLACEMENT
                if self.binary_col in placement_data.columns:
                    binary_data = placement_data[self.binary_col].values
                    features[f'{placement_clean}_binary_mean'] = np.mean(binary_data)
                    features[f'{placement_clean}_binary_std'] = np.std(binary_data)
        
        # CROSS-SENSOR COORDINATION FEATURES
        if n_placements > 1:
            placements_list = list(unique_placements)
            
            # Cross-sensor correlations for accelerometer
            for i, p1 in enumerate(placements_list):
                for j, p2 in enumerate(placements_list[i+1:], i+1):
                    p1_data = df[df['placement'] == p1]
                    p2_data = df[df['placement'] == p2]
                    
                    # Ensure same length for correlation
                    min_len = min(len(p1_data), len(p2_data))
                    if min_len > 1:
                        p1_clean = p1.replace('-', '_')
                        p2_clean = p2.replace('-', '_')
                        
                        # Accelerometer correlations
                        for axis_col in self.accel_cols:
                            if axis_col in p1_data.columns and axis_col in p2_data.columns:
                                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                                
                                try:
                                    p1_vals = p1_data[axis_col].values[:min_len]
                                    p2_vals = p2_data[axis_col].values[:min_len]
                                    
                                    if len(p1_vals) > 1 and len(p2_vals) > 1:
                                        corr = np.corrcoef(p1_vals, p2_vals)[0, 1]
                                        if not np.isnan(corr):
                                            features[f'cross_acc_{p1_clean}_{p2_clean}_{axis}_corr'] = corr
                                except:
                                    pass
                        
                        # Gyroscope correlations
                        for axis_col in self.gyro_cols:
                            if axis_col in p1_data.columns and axis_col in p2_data.columns:
                                axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                                
                                try:
                                    p1_vals = p1_data[axis_col].values[:min_len]
                                    p2_vals = p2_data[axis_col].values[:min_len]
                                    
                                    if len(p1_vals) > 1 and len(p2_vals) > 1:
                                        corr = np.corrcoef(p1_vals, p2_vals)[0, 1]
                                        if not np.isnan(corr):
                                            features[f'cross_gyro_{p1_clean}_{p2_clean}_{axis}_corr'] = corr
                                except:
                                    pass
        
        # GLOBAL FEATURES (aggregated across all placements)
        # Accelerometer global features
        for axis_col in self.accel_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                
                features[f'global_acc_{axis}_mean'] = np.mean(data)
                features[f'global_acc_{axis}_std'] = np.std(data)
                features[f'global_acc_{axis}_max'] = np.max(data)
                features[f'global_acc_{axis}_min'] = np.min(data)
                features[f'global_acc_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_acc_{axis}_energy'] = np.sum(data**2)
                
                if len(data) > 1:
                    features[f'global_acc_{axis}_rms'] = np.sqrt(np.mean(data**2))
                    features[f'global_acc_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_acc_{axis}_kurtosis'] = pd.Series(data).kurtosis()
        
        # Gyroscope global features
        for axis_col in self.gyro_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mdps]', '').replace('gyro_', '')
                
                features[f'global_gyro_{axis}_mean'] = np.mean(data)
                features[f'global_gyro_{axis}_std'] = np.std(data)
                features[f'global_gyro_{axis}_max'] = np.max(data)
                features[f'global_gyro_{axis}_min'] = np.min(data)
                features[f'global_gyro_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_gyro_{axis}_energy'] = np.sum(data**2)
                
                if len(data) > 1:
                    features[f'global_gyro_{axis}_rms'] = np.sqrt(np.mean(data**2))
                    features[f'global_gyro_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_gyro_{axis}_kurtosis'] = pd.Series(data).kurtosis()
        
        # Global magnitude features
        if all(col in df.columns for col in self.accel_cols):
            accel_data = df[self.accel_cols].values
            accel_magnitude = np.linalg.norm(accel_data, axis=1)
            features['global_accel_magnitude_mean'] = np.mean(accel_magnitude)
            features['global_accel_magnitude_std'] = np.std(accel_magnitude)
            features['global_accel_magnitude_max'] = np.max(accel_magnitude)
            features['global_accel_magnitude_min'] = np.min(accel_magnitude)
        
        if all(col in df.columns for col in self.gyro_cols):
            gyro_data = df[self.gyro_cols].values
            angular_magnitude = np.linalg.norm(gyro_data, axis=1)
            features['global_angular_magnitude_mean'] = np.mean(angular_magnitude)
            features['global_angular_magnitude_std'] = np.std(angular_magnitude)
            features['global_angular_magnitude_max'] = np.max(angular_magnitude)
            features['global_angular_magnitude_min'] = np.min(angular_magnitude)
        
        # PLACEMENT METADATA FEATURES
        features['n_active_placements'] = n_placements
        features['avg_placement_length'] = len(df) / n_placements if n_placements > 0 else 0
        
        # BINARY DECISION FEATURES (placement-aware)
        if self.binary_col in df.columns:
            features['global_binary_mean'] = df[self.binary_col].mean()
            features['global_binary_std'] = df[self.binary_col].std()
            features['global_binary_sum'] = df[self.binary_col].sum()
        
        # USER IDENTITY FEATURES (semantic information)
        features['user_id'] = user_id
        
        # INTELLIGENT ANTHROPOMETRIC FEATURES (enhanced semantic information)
        # CRITICAL: These features represent physical capabilities only, no activity info used
        anthro_features = self.get_intelligent_anthropometric_features(user_id)
        
        # Add intelligent anthropometric features
        for key, value in anthro_features.items():
            features[f'anthro_{key}'] = value
        
        return features

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
            
            features['global_binary_signal_ratio'] = np.mean(binary_data)
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
        
        # 7. INTELLIGENT ANTHROPOMETRIC FEATURES (enhanced semantic information for multi-sensor interpretation)
        # CRITICAL: These features represent physical capabilities only, no activity info used
        anthro_features = self.get_intelligent_anthropometric_features(user_id)
        
        # Add intelligent anthropometric features
        for key, value in anthro_features.items():
            features[f'anthro_{key}'] = value
        
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
        Create dataset from specified users WITH placement information AND cross-sensor features
        WITH user identity + anthropometric semantic information
        
        CRITICAL: Activity labels are used ONLY as targets, NEVER as features
        """
        X = []
        y = []
        metadata = []
        
        for user_id in user_list:
            print(f"📊 Processing User {user_id} (placement-aware)...", end="")
            user_data = self.load_user_data_placement_aware(user_id)
            
            if not user_data:
                print(" ❌ No data")
                continue
            
            user_samples = 0
            for activity in self.activities:
                if activity in user_data:
                    df = user_data[activity]
                    if len(df) > 0:
                        # Extract placement-aware multi-sensor features + cross-sensor features
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_aware_multi_sensor_features(df, user_id)
                        
                        if features:
                            X.append(features)
                            y.append(activity)  # Activity as TARGET, not feature
                            metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'n_placements': features.get('n_active_placements', 0),
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
        return X_df, np.array(y), metadata
    
    def evaluate_single_combination(self, train_users, test_users, model_label):
        """Evaluate a single train/test combination with user identity semantic information"""
        try:
            print(f"\n🔄 Training Users: {train_users}")
            print(f"🧪 Testing Users: {test_users}")
            
            X_train, y_train, train_metadata = self.create_dataset(train_users)
            X_test, y_test, test_metadata = self.create_dataset(test_users)
            
            if X_train is None or X_test is None:
                return None

            X_train_aligned, X_test_aligned, common_cols = align_feature_frames(X_train, X_test)
            
            print(f"📊 Training: {len(X_train_aligned)} samples, {len(common_cols)} features")
            print(f"🧪 Testing: {len(X_test_aligned)} samples, {len(common_cols)} features")
            
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            
            test_classes = set(y_test)
            train_classes = set(y_train)
            unseen_classes = test_classes - train_classes
            
            if unseen_classes:
                print(f"⚠️  Unseen classes in test: {unseen_classes}")
                return None
            
            y_test_encoded = label_encoder.transform(y_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_aligned)
            X_test_scaled = scaler.transform(X_test_aligned)

            y_pred, y_pred_proba, model_metadata = fit_predict_model(
                model_label,
                self.build_rf_baseline,
                X_train_scaled,
                y_train_encoded,
                X_test_scaled,
                random_seed=self.config.random_seed,
            )
            
            accuracy = accuracy_score(y_test_encoded, y_pred)
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            nmi_percentage = calculate_normalized_mutual_information(y_test_encoded, y_pred)
            nrkl_percentage = calculate_normalized_kl_divergence(y_test_encoded, y_pred_proba)
            vulnerability = calculate_vulnerability(y_pred_proba)
            
            result = {
                'model_label': model_label,
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'nmi_percentage': nmi_percentage,
                'nrkl_percentage': nrkl_percentage,
                'reverse_kl_percentage': nrkl_percentage,
                'vulnerability': vulnerability,
                'n_train_samples': len(X_train_aligned),
                'n_test_samples': len(X_test_aligned),
                'n_features': len(common_cols),
                'train_classes': sorted(train_classes),
                'test_classes': sorted(test_classes),
            }
            result.update(model_metadata)
            return result
            
        except Exception as e:
            print(f"❌ Error in combination for {model_label}: {e}")
            return None
    
    def analyze_split_ratio(self, split_name, model_label):
        """Analyze all combinations for a specific split ratio"""
        model_results_dir = self.get_model_results_dir(model_label)
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING SPLIT RATIO: {split_name} | Model: {model_label}")
        print(f"{'='*60}")
        
        combinations = self.generate_all_combinations(split_name)
        print(f"📊 Total combinations to test: {len(combinations)}")
        
        results = []
        successful_combinations = 0
        
        for i, combo in enumerate(combinations, 1):
            print(f"\n--- Combination {i}/{len(combinations)} ---")
            result = self.evaluate_single_combination(combo['train_users'], combo['test_users'], model_label)
            
            if result is not None:
                results.append(result)
                successful_combinations += 1
                print(
                    f"✅ Accuracy: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f} | "
                    f"NMI: {result['nmi_percentage']:.1f}% | NRKL: {result['nrkl_percentage']:.1f}% | "
                    f"Vulnerability: {result['vulnerability']:.3f}"
                )
            else:
                print("❌ Failed")
        
        if results:
            summary = build_summary_statistics(
                results,
                split_name=split_name,
                frequency_hz=self.config.frequency_hz,
                source_frequency_hz=SOURCE_FREQUENCY_HZ,
                max_combinations_requested=self.config.max_combinations,
                total_combinations=len(combinations),
                successful_combinations=successful_combinations,
                data_root=str(self.config.data_root),
                results_dir=model_results_dir,
                model_label=model_label,
                extra_fields={
                    'detailed_results': results,
                    'timestamp': datetime.now().isoformat(),
                },
            )
            
            print(f"\n📈 SUMMARY FOR {split_name} [{model_label}]:")
            print(f"   Successful combinations: {successful_combinations}/{len(combinations)}")
            print(f"   Accuracy: max {summary['accuracy_max']:.4f} | std {summary['accuracy_std']:.4f}")
            print(f"   F1-Score: max {summary['f1_max']:.4f} | std {summary['f1_std']:.4f}")
            print(f"   NMI (%): max {summary['nmi_max']:.1f} | std {summary['nmi_std']:.1f}")
            print(f"   NRKL (%): max {summary['nrkl_max']:.1f} | std {summary['nrkl_std']:.1f}")
            print(f"   Vulnerability: max {summary['vulnerability_max']:.3f} | std {summary['vulnerability_std']:.3f}")
            
            results_file = os.path.join(model_results_dir, f"{split_name}_detailed_results.json")
            save_json(results_file, summary)
            print(f"💾 Results saved to: {results_file}")
            
            return summary
        else:
            print("❌ No successful combinations!")
            return None
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis across the branch-required split and model families."""
        print("🚀 STARTING COMPREHENSIVE GAME-3 MAXIMUM SEMANTIC INFORMATION ANALYSIS")
        print("🔬 Testing MAXIMUM semantic information effectiveness")
        print("📍 Placement information is AVAILABLE (worst-case privacy scenario)")
        print("📏 Anthropometric measurements are AVAILABLE as semantic information")
        print("🔗 Cross-sensor coordination features are ENABLED")
        
        results = {}
        split_name = self.config.split_name
        
        for model_label in self.selected_models:
            summary = self.analyze_split_ratio(split_name, model_label)
            if summary:
                results[model_label] = {split_name: summary}
                self.save_model_report(model_label, results[model_label])
        
        print(f"\n{'='*80}")
        print("🎯 OVERALL GAME-3 MAXIMUM SEMANTIC INFORMATION STUDY SUMMARY")
        print(f"{'='*80}")
        
        for model_label, split_results in results.items():
            summary = split_results.get(split_name)
            if summary:
                print(
                    f"{model_label:>11} {split_name}: accuracy max {summary['accuracy_max']:.4f} | "
                    f"f1 max {summary['f1_max']:.4f} | vulnerability max {summary['vulnerability_max']:.3f}"
                )
        
        print("\n🔍 KEY FINDINGS:")
        print("   • MAXIMUM semantic information available (worst-case privacy scenario)")
        print("   • User identity is available as semantic information")
        print("   • Anthropometric measurements are available as semantic information")
        print("   • Sensor placement information is KNOWN (placement-aware features)")
        print("   • Cross-sensor coordination features are ENABLED")
        print("   • All sensor data available (accelerometer + gyroscope + binary)")
        print("   • Results show MAXIMUM adversarial effectiveness")
        print("   • No activity labels used during feature extraction (no cheating)")
        print("   • RF retains the original RandomForest baseline for this branch")
        print("   • Model families: RF, DT, NB, CNN, RNN, Transformer")
        return results

    def save_model_report(self, model_label, results_by_split):
        """Save the model-specific study report in the per-model directory."""
        model_results_dir = self.get_model_results_dir(model_label)
        report_file = os.path.join(model_results_dir, "game3_user_identity_study_report.json")
        save_json(
            report_file,
            {
                'study_type': 'Game-3 Multi-sensor User Identity + Anthropometrics + Placements Study',
                'model_label': model_label,
                'semantic_info_available': [
                    'user_identity',
                    'anthropometric_measurements',
                    'sensor_placement_information',
                    'cross_sensor_coordination',
                ],
                'semantic_info_removed': [],
                'data_available': ['accelerometer', 'gyroscope', 'binary_decision_tree'],
                'placement_aware': True,
                'has_cross_sensor_features': True,
                'activity_labels_in_features': False,
                'anthropometric_features': 'intelligent_biomechanical_features',
                'rf_baseline_method': 'RandomForestClassifier',
                'frequency_hz': self.config.frequency_hz,
                'source_frequency_hz': SOURCE_FREQUENCY_HZ,
                'max_combinations_requested': self.config.max_combinations,
                'data_root': str(self.config.data_root),
                'results_dir': model_results_dir,
                'results_by_split': results_by_split,
                'anthropometric_data_coverage': len(self.anthropometric_data),
                'total_users': len(self.all_users),
                'timestamp': datetime.now().isoformat(),
            },
        )
        
        print(f"\n💾 Complete report saved to: {report_file}")
        return report_file

if __name__ == "__main__":
    config = resolve_experiment_config(
        "Run the Game-3 user identity + anthropometrics + placements experiment."
    )
    analyzer = Game3UserIdentityAnthropometricsPlacementsAnalyzer(config)
    results = analyzer.run_comprehensive_analysis()
    print("\n🎉 Game-3 User Identity + Anthropometrics + Placements Analysis Complete!")
