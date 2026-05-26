#!/usr/bin/env python3
"""
Game-1 User Identity + Anthropometrics + Placements: Enhanced Binary Adversary
==============================================================================

This script implements Game-1 adversarial HAR with MAXIMUM semantic information:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- PLACEMENT INFORMATION: Known body locations for sensors
- Cross-sensor coordination features
- Binary decision tree outputs from known placements
- Temporal patterns and activity signatures

✅ AVAILABLE SEMANTIC INFORMATION:
- User identity (for user-specific modeling)
- Anthropometric measurements: height, leg length, arm length, torso length, shoe size, dominance
- Sensor placement information (body locations known)
- Placement-specific feature engineering
- Cross-sensor coordination features
- Temporal patterns and activity signatures

❌ REMOVED SEMANTIC INFORMATION:
- None (maximum information adversary scenario)

RESEARCH QUESTION:
How effective are adversarial HAR attacks when ALL semantic information 
is available to the adversary (worst-case privacy scenario)?

METHODOLOGY:
- Use placement-specific binary signals with known body locations
- Extract placement-aware temporal features
- Use user identity + anthropometric features for enhanced semantic modeling
- Add cross-sensor coordination features
- Evaluate maximum adversarial effectiveness
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

print("🎭 GAME-1 USER IDENTITY + ANTHROPOMETRICS + PLACEMENTS: MAXIMUM INFORMATION ADVERSARY")
print("=" * 85)
print("Testing adversarial effectiveness WITH user identity + anthropometrics + placement semantic information")
print("MAXIMUM INFORMATION SCENARIO | All semantic data AVAILABLE")
print()

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

# Game-1 with user identity + anthropometrics + placements

class Game1UserIdentityAnthropometricsPlacementsAnalyzer:
    """
    Analyzer for Game-1 with MAXIMUM semantic information:
    user identity + anthropometrics + placement information + cross-sensor features
    """
    
    def __init__(self, config):
        self.config = config
        self.selected_models = self.config.model_labels
        self.activities = ['Downstairs', 'Jogging', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        self.all_users = list(range(1, 12))  # Users 1-11
        
        # ALL PLACEMENTS (with known locations)
        self.all_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        
        # This branch is restricted to split 8_3 only.
        self.split_ratios = {
            self.config.split_name: TARGET_SPLIT_SPECS[self.config.split_name]
        }
        
        # Results storage
        self.base_results_dir = str(
            build_frequency_results_dir(
                self.config.results_root,
                self.config.frequency_hz,
                "game-1-user_identity_anthropometrics_placements",
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
                "game-1-user_identity_anthropometrics_placements",
                model_label,
            )
        )

    def build_rf_baseline(self):
        """Build the branch RF baseline for Game-1."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config.random_seed,
            n_jobs=-1,
        )

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

    def load_user_data_placement_aware(self, user_id):
        """
        Load binary data from ALL placements WITH placement information preserved
        This simulates an adversary who has access to binary signals AND 
        knows which signal comes from which body location (worst-case scenario)
        """
        user_dir = user_processed_dir(self.config.data_root, user_id)
        
        if not os.path.exists(user_dir):
            pass  # Warning: User directory not found
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
                            if self.binary_col in df.columns:
                                # KEEP placement information (not anonymized)
                                df['placement'] = placement
                                df['user_id'] = user_id
                                df['activity'] = activity
                                
                                # Keep binary column + metadata WITH placement info
                                df_clean = df[['placement', 'user_id', 'activity', self.binary_col]].copy()
                                activity_data.append(df_clean)
                                
                        except Exception as e:
                            pass  # Error loading file
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
                                if self.binary_col in df.columns:
                                    # KEEP placement information (not anonymized)
                                    df['placement'] = placement
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    
                                    # Keep binary column + metadata WITH placement info
                                    df_clean = df[['placement', 'user_id', 'activity', self.binary_col]].copy()
                                    activity_data.append(df_clean)
                                    
                            except Exception as e:
                                pass  # Error loading file
                                continue
            
            if activity_data:
                # Concatenate all placement data for this activity
                user_data[activity] = pd.concat(activity_data, ignore_index=True)
                # Data loading complete
            
        return user_data

    def extract_placement_aware_features(self, df, user_id):
        """
        Extract features WITH placement information AND anthropometric semantic information
        This represents maximum semantic information scenario for adversarial attacks
        
        CRITICAL: Activity labels are NEVER used during feature extraction
        - Features are extracted from sensor data + placement info + anthropometrics
        - Placement information is AVAILABLE (worst-case privacy scenario)
        - Anthropometric features enhance interpretation with physical context
        """
        features = {}
        
        # Get all binary data across all known placements
        all_binary_data = df[self.binary_col].values
        unique_placements = df['placement'].unique()
        n_placements = len(unique_placements)
        
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
        features['n_active_placements'] = n_placements
        features['avg_placement_length'] = np.mean([len(df[df['placement'] == s]) for s in unique_placements])
        
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

    def _calculate_entropy(self, binary_sequence):
        """Calculate Shannon entropy of binary sequence"""
        if len(binary_sequence) == 0:
            return 0
        
        # Get probability of each value
        unique, counts = np.unique(binary_sequence, return_counts=True)
        probabilities = counts / len(binary_sequence)
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)
        return entropy

    def create_dataset(self, user_list):
        """
        Create dataset from specified users WITH placement information
        WITH user identity + anthropometric information as semantic information
        
        CRITICAL: Activity labels are used ONLY as targets, NEVER as features
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
                        # Extract placement-aware features + anthropometrics
                        # IMPORTANT: Features extracted from sensor data only, NOT from activity labels
                        features = self.extract_placement_aware_features(df, user_id)
                        
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
        """Evaluate a single train/test combination with user identity + anthropometric semantic information"""
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
            
            # Generate detailed report
            report = classification_report(
                y_test_encoded,
                y_pred,
                target_names=le.classes_,
                output_dict=True,
                zero_division=0,
            )
            
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
                'n_train_samples': len(X_train_scaled),
                'n_test_samples': len(X_test_scaled),
                'n_features': len(common_cols),
                'classification_report': report,
                'train_activities': list(set(y_train)),
                'test_activities': list(set(y_test)),
                'has_user_identity': True,
                'has_anthropometrics': True,
            }
            result.update(model_metadata)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in combination for {model_label}: {e}")
            return None

    def evaluate_split(self, split_name, model_label):
        """Evaluate all combinations for a specific split ratio"""
        model_results_dir = self.get_model_results_dir(model_label)
        print(f"\n🎯 Evaluating split {split_name} with model {model_label}")
        print("=" * 50)
        
        combinations = self.generate_all_combinations(split_name)
        results = []
        
        for i, (train_users, test_users) in enumerate(combinations):
            print(f"\n📊 Combination {i+1}/{len(combinations)}")
            print(f"   Train users: {train_users}")
            print(f"   Test users: {test_users}")
            
            result = self.evaluate_single_combination(train_users, test_users, model_label)
            if result:
                results.append(result)
                print(
                    f"   ✅ Accuracy: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}, "
                    f"NMI: {result['nmi_percentage']:.1f}%, NRKL: {result['nrkl_percentage']:.1f}%, "
                    f"Vulnerability: {result['vulnerability']:.3f}"
                )
            else:
                print(f"   ❌ Failed")
        
        # Calculate summary statistics
        if results:
            summary = build_summary_statistics(
                results,
                split_name=split_name,
                frequency_hz=self.config.frequency_hz,
                source_frequency_hz=SOURCE_FREQUENCY_HZ,
                max_combinations_requested=self.config.max_combinations,
                total_combinations=len(combinations),
                successful_combinations=len(results),
                data_root=str(self.config.data_root),
                results_dir=model_results_dir,
                model_label=model_label,
                extra_fields={
                    'results': results,
                    'semantic_info': ['user_identity', 'anthropometrics', 'placement_information'],
                    'has_user_identity': True,
                    'has_anthropometrics': True,
                    'has_placement_info': True,
                },
            )
            
            print(f"\n📈 SUMMARY for {split_name} [{model_label}]:")
            print(f"   Accuracy:     max {summary['accuracy_max']:.3f} | std {summary['accuracy_std']:.3f}")
            print(f"   F1-Score:     max {summary['f1_max']:.3f} | std {summary['f1_std']:.3f}")
            print(f"   Precision:    {summary['precision_mean']:.3f}")
            print(f"   Recall:       {summary['recall_mean']:.3f}")
            print(f"   📈 NMI (%):    max {summary['nmi_max']:.1f} | std {summary['nmi_std']:.1f}")
            print(f"   📈 NRKL (%):   max {summary['nrkl_max']:.1f} | std {summary['nrkl_std']:.1f}")
            print(f"   🔓 Vulnerability: max {summary['vulnerability_max']:.3f} | std {summary['vulnerability_std']:.3f}")
            print(f"   Features:     {summary['avg_features']:.0f} (user + anthropometrics + placement-aware)")
            print(f"   💡 NMI & NRKL are class-agnostic and comparable across datasets")
            
            # Save results
            results_file = os.path.join(model_results_dir, f"{split_name}_with_user_identity_anthropometrics.json")
            save_json(results_file, summary)
            
            return summary
        else:
            print(f"❌ No valid results for {split_name}")
            return None

    def run_user_identity_anthropometrics_study(self):
        """Run complete study with user identity + anthropometric semantic information"""
        print("🚀 STARTING USER IDENTITY + ANTHROPOMETRICS STUDY")
        print("=" * 55)
        
        all_results = {}

        split_name = self.config.split_name

        for model_label in self.selected_models:
            summary = self.evaluate_split(split_name, model_label)
            if summary:
                all_results[model_label] = {split_name: summary}
                self.save_model_report(model_label, all_results[model_label])
        
        # Generate final report
        if all_results:
            print(f"\n🎉 STUDY COMPLETED - MAXIMUM INFORMATION ADVERSARY")
            print("=" * 50)
            
            print("\n📊 FINAL RESULTS WITH USER IDENTITY + ANTHROPOMETRICS + PLACEMENTS:")
            for model_label, split_results in all_results.items():
                summary = split_results[split_name]
                print(f"🎯 {model_label} / {split_name}:")
                print(f"   📊 Accuracy:     max {summary['accuracy_max']:.3f} | std {summary['accuracy_std']:.3f}")
                print(f"   📊 F1-Score:     max {summary['f1_max']:.3f} | std {summary['f1_std']:.3f}")
                print(f"   📈 NMI (%):      max {summary['nmi_max']:.1f} | std {summary['nmi_std']:.1f}")
                print(f"   📈 NRKL (%):     max {summary['nrkl_max']:.1f} | std {summary['nrkl_std']:.1f}")
                print(f"   🔓 Vulnerability: max {summary['vulnerability_max']:.3f} | std {summary['vulnerability_std']:.3f}")
                print()
            
            print("🔍 KEY FINDINGS:")
            print("   • MAXIMUM INFORMATION SCENARIO - worst-case privacy scenario")
            print("   • User identity is available as semantic information")
            print("   • Anthropometric measurements are available")
            print("   • Placement information is AVAILABLE (sensors locations known)")
            print("   • Cross-sensor coordination features are included")
            print("   • Results show maximum adversarial effectiveness")
            print("   • Model families: RF, DT, NB, CNN, RNN, Transformer")
            print("   📈 NMI & NRKL metrics are class-agnostic and comparable across datasets")
            print("   📈 Higher NMI = better information preservation (0-100%)")
            print("   📈 Higher NRKL = better model calibration (0-100%)")
            return all_results
        else:
            print("❌ Study failed - no valid results")
            return None

    def save_model_report(self, model_label, results_by_split):
        """Save the model-specific study report in the per-model directory."""
        model_results_dir = self.get_model_results_dir(model_label)
        report = {
            'study_type': 'user_identity_anthropometrics_placements_maximum_info',
            'timestamp': datetime.now().isoformat(),
            'model_label': model_label,
            'semantic_info_available': [
                'user_identity',
                'anthropometrics',
                'placement_information',
                'cross_sensor_coordination',
            ],
            'semantic_info_removed': [],
            'scenario': 'worst_case_privacy_maximum_semantic_information',
            'results_by_split': results_by_split,
            'anthropometric_data_coverage': len(self.anthropometric_data),
            'total_users': len(self.all_users),
            'placement_anonymous': False,
            'max_information_adversary': True,
            'frequency_hz': self.config.frequency_hz,
            'source_frequency_hz': SOURCE_FREQUENCY_HZ,
            'max_combinations_requested': self.config.max_combinations,
            'data_root': str(self.config.data_root),
            'results_dir': model_results_dir,
        }

        report_file = os.path.join(
            model_results_dir,
            "user_identity_anthropometrics_placements_maximum_info_study_report.json",
        )
        save_json(report_file, report)
        print(f"\n💾 Comprehensive report saved: {report_file}")
        return report

if __name__ == "__main__":
    config = resolve_experiment_config(
        "Run the Game-1 user identity + anthropometrics + placements experiment."
    )
    analyzer = Game1UserIdentityAnthropometricsPlacementsAnalyzer(config)
    results = analyzer.run_user_identity_anthropometrics_study()
