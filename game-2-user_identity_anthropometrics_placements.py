#!/usr/bin/env python3
"""
Game-2 User Identity + Anthropometrics + Placements: Enhanced Accelerometer Adversary
=====================================================================================

This script implements Game-2 accelerometer adversarial HAR with MAXIMUM semantic information:
user identity, anthropometric measurements, AND placement information with cross-sensor features.

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- Intelligent anthropometric features: stride potential, stability index, biomechanical ratios
- Accelerometer data (acc_x, acc_y, acc_z) from KNOWN placements
- Binary decision tree outputs from KNOWN placements
- Placement-specific feature engineering with anthropometric normalization
- Cross-sensor coordination features
- Location-based semantic understanding
- Placement-anthropometric interaction features
- Dominance-placement interaction features

❌ REMOVED SEMANTIC INFORMATION:
- None (maximum information adversary scenario)

RESEARCH QUESTION:
How effective are Game-2 accelerometer adversarial attacks when ALL semantic information 
is available including placement locations, cross-sensor coordination, and intelligent 
anthropometric features?

METHODOLOGY:
- Load accelerometer data from all placements WITH placement labels
- Extract placement-specific accelerometer features with anthropometric normalization
- Extract cross-sensor coordination features
- Use intelligent anthropometric features for enhanced biomechanical context
- Add placement-anthropometric interaction features
- Add dominance-placement interaction features
- Compare maximum adversarial effectiveness with semantic information
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

# Game-2 with user identity + anthropometrics + placements

class Game2UserIdentityAnthropometricsPlacementsAnalyzer:
    """
    Analyzer for Game-2 accelerometer attacks with user identity + anthropometrics + placement information
    Maximum semantic information scenario with cross-sensor coordination features
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_models = self.config.model_labels
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL POSSIBLE PLACEMENTS (with known locations)
        self.all_placements = ['left-ankle', 'right-ankle', 'left-wrist', 'right-wrist', 'right-pocket']
        
        # Game-2 constraint: Accelerometer + Binary only (NO gyroscope)
        self.accel_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
        self.binary_col = 'dec_tree_out_1'
        self.available_cols = self.accel_cols + [self.binary_col]
        
        self.split_ratios = {
            self.config.split_name: TARGET_SPLIT_SPECS[self.config.split_name]
        }
        
        # Results storage
        self.base_results_dir = str(
            build_frequency_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                "game-2-user_identity_anthropometrics_placements",
            )
        )

        validate_dl_runtime_for_models(self.selected_models)
        
        # Load anthropometric data
        self.anthropometric_data = self.load_all_anthropometric_data()
        print(f"📊 Users: {len(self.all_users)} users")
        print(f"📏 Users with anthropometric data: {len(self.anthropometric_data)}")
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
                "game-2-user_identity_anthropometrics_placements",
                model_label,
            )
        )

    def build_rf_baseline(self):
        """Build the branch RF baseline for Game-2 accelerometer."""
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.random_seed,
            class_weight='balanced',
        )
        lr = LogisticRegression(
            random_state=self.config.random_seed,
            class_weight='balanced',
            max_iter=1000,
        )
        svm = SVC(
            random_state=self.config.random_seed,
            class_weight='balanced',
            probability=True,
        )

        return VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('svm', svm)],
            voting='soft',
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
            train_test_pairs.append((list(train_users), list(test_users)))

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
                    else:
                        pass  # Failed to parse
                except Exception as e:
                    pass  # Error loading measurements
            else:
                pass  # No measurements file
        
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
        Create intelligent anthropometric features that are biomechanically relevant to HAR
        These features help the model understand physical capabilities and constraints
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
    
    def load_user_data_placement_aware(self, user_id):
        """
        Load accelerometer + binary data from ALL placements WITH placement information
        This simulates an adversary who has access to accelerometer + binary signals AND 
        knows which signal comes from which body location (worst-case scenario)
        
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
                            if any(col in df.columns for col in self.available_cols):
                                # Keep placement information for cross-sensor features
                                df['placement'] = placement
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Only keep accelerometer + binary columns + metadata
                                keep_cols = ['placement', 'user_id', 'activity']
                                for col in self.available_cols:
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
                        csv_files = glob.glob(os.path.join(placement_dir, "*.csv"))
                        for file_path in csv_files:
                            try:
                                df = load_experiment_csv(file_path, self.config.frequency_hz)
                                if any(col in df.columns for col in self.available_cols):
                                    df['placement'] = placement
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    keep_cols = ['placement', 'user_id', 'activity']
                                    for col in self.available_cols:
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

    def load_user_data_placement_anonymous(self, user_id):
        """
        Load accelerometer + binary data from ALL placements but anonymize placement information
        This simulates an adversary who has access to accelerometer signals but doesn't 
        know which signal comes from which body location
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
            
            # Load data from ALL placements but anonymize them
            for placement_idx, placement in enumerate(self.all_placements):
                
                if activity in ['Standing', 'Sitting', 'Laying', 'Walking', 'Jogging']:
                    # Single file per placement: Data/User X/Processed/ACTIVITY/PLACEMENT.csv
                    file_path = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(file_path):
                        try:
                            df = load_experiment_csv(file_path, self.config.frequency_hz)
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
                                continue
            
            if activity_data:
                # Concatenate all placement data for this activity
                user_data[activity] = pd.concat(activity_data, ignore_index=True)
                # Debug: Show successful data loading
                if activity == 'Walking':  # Only show for one activity
                    print(f"✅ User {user_id} {activity}: {len(user_data[activity])} samples from {len(activity_data)} placements")
            
        return user_data
    
    def extract_placement_aware_accelerometer_features(self, df, user_id):
        """
        Extract accelerometer features WITH placement information AND cross-sensor features
        WITH anthropometric semantic information for maximum feature richness
        This represents maximum semantic information scenario for adversarial attacks
        """
        if len(df) == 0:
            return {}
        
        features = {}
        
        # Get unique placements for this user's data
        unique_placements = df['placement'].unique()
        n_placements = len(unique_placements)
        
        # PLACEMENT-SPECIFIC ACCELEROMETER FEATURES
        for placement in unique_placements:
            placement_data = df[df['placement'] == placement]
            
            if len(placement_data) > 0:
                for axis_col in self.accel_cols:
                    if axis_col in placement_data.columns:
                        data = placement_data[axis_col].values
                        axis = axis_col.replace('[mg]', '').replace('acc_', '')
                        placement_clean = placement.replace('-', '_')
                        
                        # Basic statistics per placement
                        features[f'{placement_clean}_{axis}_mean'] = np.mean(data)
                        features[f'{placement_clean}_{axis}_std'] = np.std(data)
                        features[f'{placement_clean}_{axis}_max'] = np.max(data)
                        features[f'{placement_clean}_{axis}_min'] = np.min(data)
                        features[f'{placement_clean}_{axis}_range'] = np.max(data) - np.min(data)
                        features[f'{placement_clean}_{axis}_energy'] = np.sum(data**2)
                        
                        if len(data) > 1:
                            features[f'{placement_clean}_{axis}_rms'] = np.sqrt(np.mean(data**2))
                            features[f'{placement_clean}_{axis}_skew'] = pd.Series(data).skew()
                            features[f'{placement_clean}_{axis}_kurtosis'] = pd.Series(data).kurtosis()
        
        # CROSS-SENSOR COORDINATION FEATURES
        if n_placements > 1:
            placements_list = list(unique_placements)
            
            # Cross-sensor correlations
            for i, p1 in enumerate(placements_list):
                for j, p2 in enumerate(placements_list[i+1:], i+1):
                    p1_data = df[df['placement'] == p1]
                    p2_data = df[df['placement'] == p2]
                    
                    # Ensure same length for correlation
                    min_len = min(len(p1_data), len(p2_data))
                    if min_len > 1:
                        for axis_col in self.accel_cols:
                            if axis_col in p1_data.columns and axis_col in p2_data.columns:
                                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                                p1_clean = p1.replace('-', '_')
                                p2_clean = p2.replace('-', '_')
                                
                                try:
                                    p1_vals = p1_data[axis_col].values[:min_len]
                                    p2_vals = p2_data[axis_col].values[:min_len]
                                    
                                    if len(p1_vals) > 1 and len(p2_vals) > 1:
                                        corr = np.corrcoef(p1_vals, p2_vals)[0, 1]
                                        if not np.isnan(corr):
                                            features[f'cross_{p1_clean}_{p2_clean}_{axis}_corr'] = corr
                                except:
                                    pass
        
        # GLOBAL FEATURES (aggregated across all placements)
        for axis_col in self.accel_cols:
            if axis_col in df.columns:
                data = df[axis_col].values
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                
                features[f'global_{axis}_mean'] = np.mean(data)
                features[f'global_{axis}_std'] = np.std(data)
                features[f'global_{axis}_max'] = np.max(data)
                features[f'global_{axis}_min'] = np.min(data)
                features[f'global_{axis}_range'] = np.max(data) - np.min(data)
                features[f'global_{axis}_energy'] = np.sum(data**2)
                
                if len(data) > 1:
                    features[f'global_{axis}_rms'] = np.sqrt(np.mean(data**2))
                    features[f'global_{axis}_skew'] = pd.Series(data).skew()
                    features[f'global_{axis}_kurtosis'] = pd.Series(data).kurtosis()
        
        # PLACEMENT METADATA FEATURES
        features['n_active_placements'] = n_placements
        features['avg_placement_length'] = len(df) / n_placements if n_placements > 0 else 0
        
        # BINARY DECISION FEATURES (placement-aware)
        if self.binary_col in df.columns:
            features['global_binary_mean'] = df[self.binary_col].mean()
            features['global_binary_std'] = df[self.binary_col].std()
            features['global_binary_sum'] = df[self.binary_col].sum()
            
            # Per-placement binary features
            for placement in unique_placements:
                placement_data = df[df['placement'] == placement]
                if len(placement_data) > 0 and self.binary_col in placement_data.columns:
                    placement_clean = placement.replace('-', '_')
                    features[f'{placement_clean}_binary_mean'] = placement_data[self.binary_col].mean()
                    features[f'{placement_clean}_binary_std'] = placement_data[self.binary_col].std()
        
        # USER IDENTITY FEATURES (semantic information)
        features['user_id'] = user_id
        
        # INTELLIGENT ANTHROPOMETRIC FEATURES (enhanced semantic information)
        intelligent_anthro_features = self.get_intelligent_anthropometric_features(user_id)
        for key, value in intelligent_anthro_features.items():
            features[f'anthro_{key}'] = value
        
        # PLACEMENT-ANTHROPOMETRIC INTERACTION FEATURES
        # Combine placement information with anthropometric data for enhanced interpretation
        anthro_data = self.get_anthropometric_features(user_id)
        
        if 'height' in anthro_data and n_placements > 0:
            # Height-normalized placement features
            features['height_normalized_placement_count'] = n_placements / anthro_data['height']
            
            # Per-placement anthropometric normalization
            for placement in unique_placements:
                placement_data = df[df['placement'] == placement]
                if len(placement_data) > 0:
                    placement_clean = placement.replace('-', '_')
                    
                    # Height-normalized accelerometer features per placement
                    for axis_col in self.accel_cols:
                        if axis_col in placement_data.columns:
                            axis = axis_col.replace('[mg]', '').replace('acc_', '')
                            placement_mean = placement_data[axis_col].mean()
                            features[f'{placement_clean}_{axis}_height_normalized'] = placement_mean / anthro_data['height']
        
        # LIMB-SPECIFIC PLACEMENT FEATURES (using anthropometric context)
        if 'arm_length' in anthro_data and 'leg_length' in anthro_data:
            arm_placements = [p for p in unique_placements if 'wrist' in p]
            leg_placements = [p for p in unique_placements if 'ankle' in p]
            
            # Arm-specific features normalized by arm length
            for arm_placement in arm_placements:
                placement_data = df[df['placement'] == arm_placement]
                if len(placement_data) > 0:
                    placement_clean = arm_placement.replace('-', '_')
                    for axis_col in self.accel_cols:
                        if axis_col in placement_data.columns:
                            axis = axis_col.replace('[mg]', '').replace('acc_', '')
                            placement_mean = placement_data[axis_col].mean()
                            features[f'{placement_clean}_{axis}_arm_normalized'] = placement_mean / anthro_data['arm_length']
            
            # Leg-specific features normalized by leg length
            for leg_placement in leg_placements:
                placement_data = df[df['placement'] == leg_placement]
                if len(placement_data) > 0:
                    placement_clean = leg_placement.replace('-', '_')
                    for axis_col in self.accel_cols:
                        if axis_col in placement_data.columns:
                            axis = axis_col.replace('[mg]', '').replace('acc_', '')
                            placement_mean = placement_data[axis_col].mean()
                            features[f'{placement_clean}_{axis}_leg_normalized'] = placement_mean / anthro_data['leg_length']
        
        # DOMINANCE-PLACEMENT INTERACTION FEATURES
        if 'dominant_hand' in anthro_data and 'dominant_foot' in anthro_data:
            # Left/right placement features with dominance context
            left_placements = [p for p in unique_placements if 'left' in p]
            right_placements = [p for p in unique_placements if 'right' in p]
            
            if left_placements and right_placements:
                # Dominant vs non-dominant side features
                for axis_col in self.accel_cols:
                    if axis_col in df.columns:
                        axis = axis_col.replace('[mg]', '').replace('acc_', '')
                        
                        # Calculate mean activity for left and right sides
                        left_data = df[df['placement'].isin(left_placements)][axis_col]
                        right_data = df[df['placement'].isin(right_placements)][axis_col]
                        
                        if len(left_data) > 0 and len(right_data) > 0:
                            left_mean = left_data.mean()
                            right_mean = right_data.mean()
                            
                            # Dominant side preference features
                            if anthro_data['dominant_hand'] == 1:  # Right dominant
                                features[f'{axis}_dominant_side_preference'] = right_mean - left_mean
                            elif anthro_data['dominant_hand'] == 0:  # Left dominant
                                features[f'{axis}_dominant_side_preference'] = left_mean - right_mean
                            else:  # Unknown dominance
                                features[f'{axis}_dominant_side_preference'] = 0.0
        
        return features

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
        
        # 4. USER-SPECIFIC PATTERN ENCODING (using user identity semantic info)
        features['user_id'] = user_id  # Use user identity as semantic information
        
        # 5. INTELLIGENT ANTHROPOMETRIC FEATURES (enhanced semantic information)
        intelligent_anthro_features = self.get_intelligent_anthropometric_features(user_id)
        for key, value in intelligent_anthro_features.items():
            features[f'anthro_{key}'] = value
        
        # 6. ENHANCED ANTHROPOMETRIC-ACCELEROMETER INTERACTIONS
        # These features combine anthropometric measurements with accelerometer patterns
        anthro_data = self.get_anthropometric_features(user_id)
        
        if 'height' in anthro_data:
            height = anthro_data['height']
            
            # Height-normalized accelerometer features
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
        
        # 7. LIMB-SPECIFIC FEATURE INTERPRETATIONS
        if 'leg_length' in anthro_data and 'arm_length' in anthro_data:
            leg_length = anthro_data['leg_length']
            arm_length = anthro_data['arm_length']
            
            # Limb-normalized accelerometer features (for different body parts)
            for axis_col in self.accel_cols:
                axis = axis_col.replace('[mg]', '').replace('acc_', '')
                if f'global_{axis}_mean' in features:
                    # Leg-normalized (for ankle/walking patterns)
                    features[f'leg_normalized_{axis}_mean'] = features[f'global_{axis}_mean'] / leg_length
                    # Arm-normalized (for wrist patterns)
                    features[f'arm_normalized_{axis}_mean'] = features[f'global_{axis}_mean'] / arm_length
        
        # 8. BIOMECHANICAL CONTEXT FEATURES
        if 'stride_potential' in intelligent_anthro_features and 'global_magnitude_mean' in features:
            # Stride-normalized movement intensity
            features['stride_normalized_magnitude'] = features['global_magnitude_mean'] / intelligent_anthro_features['stride_potential']
        
        if 'stability_index' in intelligent_anthro_features and 'global_magnitude_std' in features:
            # Stability-normalized movement variability
            features['stability_normalized_variability'] = features['global_magnitude_std'] / intelligent_anthro_features['stability_index']
        
        # 9. DOMINANCE-BASED FEATURE INTERPRETATIONS
        if 'dominant_hand' in anthro_data:
            features['dominant_hand_encoded'] = anthro_data['dominant_hand']
        if 'dominant_foot' in anthro_data:
            features['dominant_foot_encoded'] = anthro_data['dominant_foot']
        if 'gender_indicator' in anthro_data:
            features['gender_encoded'] = anthro_data['gender_indicator']
        
        # 10. SENSOR COUNT METADATA (without placement identity)
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
        Create dataset from specified users WITH placement information AND cross-sensor features
        WITH user identity + anthropometric semantic information
        """
        X = []
        y = []
        metadata = []
        
        for user_id in user_list:
            user_data = self.load_user_data_placement_aware(user_id)
            
            if not user_data:
                continue
            
            user_samples = 0
            for activity in self.activities:
                if activity in user_data:
                    df = user_data[activity]
                    if len(df) > 0:
                        # Extract placement-aware accelerometer features + cross-sensor features
                        features = self.extract_placement_aware_accelerometer_features(df, user_id)
                        
                        if features:
                            X.append(features)
                            y.append(activity)  # Activity as TARGET, not feature
                            metadata.append({
                                'user_id': user_id,
                                'activity': activity,
                                'n_placements': features.get('n_active_placements', 0),
                                'total_samples': features.get('avg_placement_length', 0) * features.get('n_active_placements', 0)
                            })
                            user_samples += 1
        
        if not X:
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
        
        return X_df, np.array(y), metadata
    
    def evaluate_single_combination(self, train_users, test_users, model_label):
        """Evaluate a single train/test combination with user identity semantic information"""
        try:
            # Create datasets
            X_train, y_train, train_meta = self.create_dataset(train_users)
            X_test, y_test, test_meta = self.create_dataset(test_users)
            
            if X_train is None or X_test is None:
                return None
            
            if len(X_train) == 0 or len(X_test) == 0:
                return None

            X_train_aligned, X_test_aligned, common_cols = align_feature_frames(X_train, X_test)
            
            # Encode labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            unseen_classes = set(y_test) - set(y_train)
            if unseen_classes:
                print(f"⚠️  Unseen classes in test: {sorted(unseen_classes)}")
                return None

            y_test_encoded = le.transform(y_test)
            
            # Scale features
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
            
            # Calculate traditional metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # Calculate information-theoretic metrics (class-agnostic)
            nmi_percentage = calculate_normalized_mutual_information(y_test_encoded, y_pred)
            nrkl_percentage = calculate_normalized_kl_divergence(y_test_encoded, y_pred_proba)
            vulnerability = calculate_vulnerability(y_pred_proba)
            
            result = {
                'model_label': model_label,
                'train_users': train_users,
                'test_users': test_users,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'nmi_percentage': nmi_percentage,
                'nrkl_percentage': nrkl_percentage,
                'reverse_kl_percentage': nrkl_percentage,
                'vulnerability': vulnerability,
                'n_train_samples': len(X_train_aligned),
                'n_test_samples': len(X_test_aligned),
                'n_features': len(common_cols),
                'has_user_identity': True,
                'placement_anonymous': False,
            }
            result.update(model_metadata)
            return result
            
        except Exception as e:
            print(f"❌ Error in combination for {model_label}: {e}")
            return None
    
    def analyze_split_ratio(self, split_name, model_label):
        """Analyze all combinations for a specific split ratio with user identity"""
        model_results_dir = self.get_model_results_dir(model_label)
        print(f"\n{'='*80}")
        print(f"🎭 ANALYZING SPLIT RATIO: {split_name} | Model: {model_label} (WITH maximum semantic information)")
        print(f"{'='*80}")
        
        # Generate all combinations
        combinations_list = self.generate_all_combinations(split_name)
        
        # Evaluate each combination
        results = []
        successful_runs = 0
        
        for i, (train_users, test_users) in enumerate(combinations_list):
            if (i + 1) % 10 == 0:
                print(f"🔄 Progress: {i+1}/{len(combinations_list)} combinations...")
            
            result = self.evaluate_single_combination(train_users, test_users, model_label)
            if result:
                results.append(result)
                successful_runs += 1
        
        print(f"✅ Completed: {successful_runs}/{len(combinations_list)} successful runs")
        
        if not results:
            print("❌ No successful runs!")
            return None
        
        stats = build_summary_statistics(
            results,
            split_name=split_name,
            frequency_hz=self.config.frequency_hz,
            source_frequency_hz=SOURCE_FREQUENCY_HZ,
            max_combinations_requested=self.config.max_combinations,
            total_combinations=len(combinations_list),
            successful_combinations=successful_runs,
            data_root=str(self.config.data_root),
            results_dir=model_results_dir,
            model_label=model_label,
            extra_fields={
                'has_user_identity': True,
                'placement_anonymous': False,
                'game_type': 'Game-2 Accelerometer',
                'individual_results': results,
                'timestamp': datetime.now().isoformat(),
            },
        )
        
        # Print summary
        print(f"\n📊 PLACEMENT-AWARE GAME-2 ACCELEROMETER RESULTS [{model_label}] (WITH maximum semantic information):")
        print(f"   Accuracy:  max {stats['accuracy_max']:.3f} | std {stats['accuracy_std']:.3f}")
        print(f"   F1-Score:  max {stats['f1_max']:.3f} | std {stats['f1_std']:.3f}")
        print(f"   Precision: {stats['precision_mean']:.3f}")
        print(f"   Recall:    {stats['recall_mean']:.3f}")
        print(f"   📈 NMI (%):      max {stats['nmi_max']:.1f} | std {stats['nmi_std']:.1f}")
        print(f"   📈 NRKL (%):     max {stats['nrkl_max']:.1f} | std {stats['nrkl_std']:.1f}")
        print(f"   🔓 Vulnerability: max {stats['vulnerability_max']:.3f} | std {stats['vulnerability_std']:.3f}")
        print(f"   Features:  {stats['avg_features']:.0f} (placement-aware + user identity)")
        print(f"   💡 NMI & NRKL are class-agnostic and comparable across datasets")
        
        # Save results
        results_file = os.path.join(model_results_dir, f"{split_name}_with_user_identity.json")
        save_json(results_file, stats)
        
        print(f"💾 Results saved to: {results_file}")
        
        return stats
    
    def run_user_identity_study(self):
        """
        Run complete user identity study with placement-aware Game-2 accelerometer
        """
        print("🚀 Starting Game-2 Accelerometer User Identity Study...")
        print("Testing adversarial effectiveness WITH user identity semantic information")
        print("Accelerometer + Binary data available, placement information AVAILABLE")
        print()
        
        all_results = {}

        split_name = self.config.split_name
        for model_label in self.selected_models:
            print(f"\n🔬 Testing {split_name} WITH user identity semantic information using {model_label}...")
            stats = self.analyze_split_ratio(split_name, model_label)
            if stats:
                all_results[model_label] = {split_name: stats}
                self._generate_summary_report(model_label, all_results[model_label])

        return all_results
    
    def _generate_summary_report(self, model_label, results_by_split):
        """Generate a summary report of Game-2 user identity study results"""
        model_results_dir = self.get_model_results_dir(model_label)
        print(f"\n{'='*100}")
        print(f"🎭 GAME-2 ACCELEROMETER USER IDENTITY + ANTHROPOMETRICS + PLACEMENTS STUDY - SUMMARY REPORT [{model_label}]")
        print(f"{'='*120}")
        
        print("\n📊 ADVERSARIAL HAR EFFECTIVENESS WITH MAXIMUM SEMANTIC INFORMATION:")
        print("   Accelerometer + Binary data | Placement information AVAILABLE | User identity AVAILABLE | Intelligent Anthropometrics AVAILABLE")
        print("   Cross-sensor coordination | Placement-anthropometric interactions | Dominance-placement interactions")
        print()
        
        for split_name, stats in results_by_split.items():
            if stats:
                print(f"🎯 SPLIT RATIO {split_name}:")
                print(f"   📊 Accuracy:  max {stats['accuracy_max']:.3f} | std {stats['accuracy_std']:.3f}")
                print(f"   📊 F1-Score:  max {stats['f1_max']:.3f} | std {stats['f1_std']:.3f}")
                print(f"   📊 Precision: {stats['precision_mean']:.3f}")
                print(f"   📊 Recall:    {stats['recall_mean']:.3f}")
                print(f"   📈 NMI (%):      max {stats['nmi_max']:.1f} | std {stats['nmi_std']:.1f}")
                print(f"   📈 NRKL (%):     max {stats['nrkl_max']:.1f} | std {stats['nrkl_std']:.1f}")
                print(f"   🔓 Vulnerability: max {stats['vulnerability_max']:.3f} | std {stats['vulnerability_std']:.3f}")
                print(f"   �📊 Features:  {stats['avg_features']:.0f}")
                print()
        
        print("🔍 KEY FINDINGS - MAXIMUM SEMANTIC INFORMATION SCENARIO:")
        print("   • Placement information is AVAILABLE (known sensor locations)")
        print("   • User identity is available as semantic information")
        print("   • Intelligent anthropometric features provide biomechanical context")
        print("   • Cross-sensor coordination features capture multi-sensor patterns")
        print("   • Placement-anthropometric interactions for enhanced interpretation")
        print("   • Dominance-placement interactions for laterality understanding")
        print("   • Accelerometer + binary data available (Game-2 constraint)")
        print("   • Results show MAXIMUM adversarial effectiveness (worst-case privacy)")
        print("   • No activity labels used during feature extraction (no cheating)")
        print("   • RF retains the original VotingClassifier baseline for this branch")
        print("   • Model families: RF, DT, NB, CNN, RNN, Transformer")
        print("   📈 NMI & NRKL metrics are class-agnostic and comparable across datasets")
        print("   📈 Higher NMI = better information preservation (0-100%)")
        print("   📈 Higher NRKL = better model calibration (0-100%)")
        
        # Save comprehensive report
        report_file = os.path.join(model_results_dir, "game2_user_identity_anthropometrics_placements_study_report.json")
        save_json(
            report_file,
            {
                'study_type': 'Game-2 Accelerometer User Identity + Anthropometrics + Placements Study',
                'model_label': model_label,
                'semantic_info_removed': [],
                'semantic_info_available': [
                    'user_identity',
                    'anthropometric_measurements',
                    'intelligent_anthropometric_features',
                    'sensor_placement_locations',
                    'cross_sensor_coordination',
                    'placement_specific_features',
                    'placement_anthropometric_interactions',
                    'dominance_placement_interactions',
                ],
                'data_available': ['accelerometer', 'binary_decision_tree'],
                'placement_anonymous': False,
                'activity_labels_in_features': False,
                'rf_baseline_method': 'VotingClassifier',
                'feature_categories': [
                    'placement_specific_accelerometer',
                    'cross_sensor_coordination',
                    'intelligent_anthropometric',
                    'placement_anthropometric_interactions',
                    'dominance_placement_interactions',
                    'user_identity_encoding',
                ],
                'privacy_scenario': 'worst_case_maximum_information',
                'frequency_hz': self.config.frequency_hz,
                'source_frequency_hz': SOURCE_FREQUENCY_HZ,
                'max_combinations_requested': self.config.max_combinations,
                'data_root': str(self.config.data_root),
                'results_dir': model_results_dir,
                'results_by_split': results_by_split,
                'timestamp': datetime.now().isoformat(),
            },
        )
        
        print(f"\n💾 Complete report saved to: {report_file}")


if __name__ == "__main__":
    config = resolve_experiment_config(
        "Run the Game-2 accelerometer user identity + anthropometrics + placements experiment."
    )
    analyzer = Game2UserIdentityAnthropometricsPlacementsAnalyzer(config)
    results = analyzer.run_user_identity_study()
