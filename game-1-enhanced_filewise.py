#!/usr/bin/env python3
"""
Enhanced File-Wise Cross-Sensor Cross-Dataset HAR Study: STM ↔ MotionSense
===========================================================================

This script implements the improved file-wise processing approach that increased
performance from ~16% to 36.8% F1-score in STM-UCI experiments, now applied to
STM-MotionSense cross-sensor cross-dataset analysis.

KEY IMPROVEMENTS:
- File-wise processing (complete activity sessions)
- 29 robust temporal features (vs basic temporal features)
- Enhanced run-length analysis 
- Window-based burst intensity features
- Entropy and information-theoretic features
- Advanced pattern recognition features

EXPERIMENTAL DESIGN:
Case-1: STM (5 placements) → MotionSense (smartphone)
Case-2: MotionSense (smartphone) → STM (5 placements)
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           f1_score, precision_score, recall_score, normalized_mutual_info_score)
from scipy.stats import entropy
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from cross_dataset_utils import get_motionsense_root, get_stm_root
warnings.filterwarnings('ignore')

class EnhancedFileWiseSTMMotionSenseAnalyzer:
    """
    Enhanced File-Wise Cross-Sensor Cross-Dataset HAR Analyzer for STM-MotionSense
    Implements the same improvements that boosted STM-UCI performance from 16% to 36.8% F1
    """
    
    def __init__(self):
        # Dataset paths
        self.stm_path = str(get_stm_root())
        self.motionsense_path = str(get_motionsense_root())
        
        # STM placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Activity mappings
        self.activities = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # Binary column
        self.binary_col = 'dec_tree_out_1'
        
        # Results directory
        self.results_dir = "results/game-1-enhanced_filewise"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("🚀 ENHANCED FILE-WISE STM-MOTIONSENSE CROSS-SENSOR ANALYSIS")
        print("=" * 65)
        print("🔧 Implementation: 29 robust temporal features")
        print("📁 Processing: File-wise (complete activity sessions)")
        print("🎯 Goal: Improve upon 38% accuracy baseline")
        print()
    
    def load_stm_placement_data(self, placement, as_files=True):
        """
        Load STM data for specific placement with file-wise organization
        Handles the correct STM dataset structure:
        - Regular activities: User X/Processed/Activity/placement.csv
        - Stairs activities: User X/Processed/Upstairs/placement/placement_upstairs1.csv, etc.
        """
        print(f"📂 Loading STM {placement} data...")
        
        all_files_data = []
        
        # Get all user directories
        import glob
        user_dirs = glob.glob(os.path.join(self.stm_path, "User*"))
        
        for user_dir in sorted(user_dirs):
            user_id = os.path.basename(user_dir)
            processed_dir = os.path.join(user_dir, "Processed")
            
            if not os.path.exists(processed_dir):
                print(f"   ⚠️ No Processed directory for {user_id}")
                continue
            
            for activity in self.activities:
                activity_dir = os.path.join(processed_dir, activity)
                
                if not os.path.exists(activity_dir):
                    print(f"   ⚠️ No {activity} directory for {user_id}")
                    continue
                
                # Handle different directory structures for stairs vs regular activities
                if activity in ['Upstairs', 'Downstairs']:
                    # Stairs activities: Activity/placement/placement_activity*.csv
                    placement_dir = os.path.join(activity_dir, placement)
                    if os.path.exists(placement_dir):
                        # Get all CSV files for this placement and activity
                        csv_files = glob.glob(os.path.join(placement_dir, f"{placement}_{activity.lower()}*.csv"))
                        
                        for csv_file in csv_files:
                            try:
                                df = pd.read_csv(csv_file)
                                
                                if self.binary_col in df.columns and len(df) > 0:
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    df['placement'] = placement
                                    file_name = os.path.basename(csv_file)
                                    df['file_id'] = f"{user_id}_{activity}_{placement}_{file_name}"
                                    
                                    if as_files:
                                        all_files_data.append(df.copy())
                                    else:
                                        all_files_data.append(df)
                                        
                            except Exception as e:
                                print(f"   ⚠️ Error loading {csv_file}: {e}")
                else:
                    # Regular activities: Activity/placement.csv
                    activity_file = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(activity_file):
                        try:
                            df = pd.read_csv(activity_file)
                            
                            if self.binary_col in df.columns and len(df) > 0:
                                df['user_id'] = user_id
                                df['activity'] = activity
                                df['placement'] = placement
                                df['file_id'] = f"{user_id}_{activity}_{placement}"
                                
                                if as_files:
                                    all_files_data.append(df.copy())
                                else:
                                    all_files_data.append(df)
                                    
                        except Exception as e:
                            print(f"   ⚠️ Error loading {activity_file}: {e}")
        
        if all_files_data:
            if as_files:
                print(f"   ✅ Loaded {len(all_files_data)} complete files")
                return all_files_data  # List of file DataFrames
            else:
                combined_df = pd.concat(all_files_data, ignore_index=True)
                print(f"   ✅ Loaded {len(combined_df)} samples")
                return combined_df
        else:
            print(f"   ❌ No data found for {placement}")
            return [] if as_files else pd.DataFrame()
    
    def load_motionsense_data(self, as_files=True):
        """
        Load MotionSense data with file-wise organization
        Correct structure: subject_X/activity_activityname/data.csv
        """
        print(f"📂 Loading MotionSense data...")
        
        all_files_data = []
        
        # Get all subject directories
        import glob
        subject_dirs = glob.glob(os.path.join(self.motionsense_path, "subject_*"))
        
        for subject_dir in sorted(subject_dirs):
            subject_id = os.path.basename(subject_dir)
            
            for activity in self.activities:
                # MotionSense structure: subject_X/activity_activityname/data.csv
                activity_dir = os.path.join(subject_dir, f"activity_{activity.lower()}")
                activity_file = os.path.join(activity_dir, "data.csv")
                
                if os.path.exists(activity_file):
                    try:
                        df = pd.read_csv(activity_file)
                        
                        # Check if we have the binary column
                        if self.binary_col in df.columns and len(df) > 0:
                            df['user_id'] = subject_id
                            df['activity'] = activity
                            df['placement'] = 'smartphone'
                            df['file_id'] = f"{subject_id}_{activity}_smartphone"
                            
                            if as_files:
                                all_files_data.append(df.copy())
                            else:
                                all_files_data.append(df)
                        else:
                            print(f"   ⚠️ Missing binary column in {activity_file}")
                                
                    except Exception as e:
                        print(f"   ⚠️ Error loading {activity_file}: {e}")
                else:
                    print(f"   ⚠️ File not found: {activity_file}")
        
        if all_files_data:
            if as_files:
                print(f"   ✅ Loaded {len(all_files_data)} complete files")
                return all_files_data
            else:
                combined_df = pd.concat(all_files_data, ignore_index=True)
                print(f"   ✅ Loaded {len(combined_df)} samples")
                return combined_df
        else:
            print(f"   ❌ No MotionSense data found")
            return [] if as_files else pd.DataFrame()
    
    def extract_enhanced_temporal_features(self, binary_sequence):
        """
        Extract 29 robust temporal features - same approach that improved STM-UCI performance
        
        CRITICAL: This function is ACTIVITY-AGNOSTIC
        - Only uses binary sensor data (0s and 1s)
        - No activity labels or activity-specific information used
        - Features work for any unknown activity
        - Suitable for real-world inference scenarios
        """
        if len(binary_sequence) == 0:
            return {}
        
        binary_data = np.array(binary_sequence)
        features = {}
        
        # 1. BASIC ACTIVITY STATISTICS (4 features)
        # Based purely on binary pattern analysis
        features['activity_ratio'] = np.mean(binary_data)
        features['total_samples'] = len(binary_data)
        features['active_samples'] = np.sum(binary_data)
        features['inactive_samples'] = np.sum(1 - binary_data)
        
        # 2. TEMPORAL DYNAMICS (3 features)
        if len(binary_data) > 1:
            transitions = np.sum(np.abs(np.diff(binary_data)))
            features['transitions'] = transitions
            features['transition_rate'] = transitions / len(binary_data)
            features['transition_density'] = transitions / max(1, np.sum(binary_data))
        else:
            features['transitions'] = 0
            features['transition_rate'] = 0
            features['transition_density'] = 0
        
        # 3. RUN-LENGTH ANALYSIS (12 features)
        runs = self._get_run_lengths(binary_data)
        if runs:
            run_features = self._extract_enhanced_run_length_features(runs)
            features.update(run_features)
        
        # 4. WINDOW-BASED FEATURES (4 features)
        window_features = self._extract_window_based_features(binary_data)
        features.update(window_features)
        
        # 5. BURST INTENSITY ANALYSIS (3 features) 
        burst_features = self._extract_burst_intensity_features(binary_data)
        features.update(burst_features)
        
        # 6. ENTROPY AND INFORMATION FEATURES (3 features)
        entropy_features = self._extract_entropy_features(binary_data)
        features.update(entropy_features)
        
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
    
    def _extract_enhanced_run_length_features(self, runs):
        """Extract 12 enhanced run-length features"""
        if not runs:
            return {f'run_feature_{i}': 0 for i in range(12)}
        
        active_runs = [length for val, length in runs if val == 1]
        inactive_runs = [length for val, length in runs if val == 0]
        all_runs = [length for val, length in runs]
        
        features = {}
        
        # Active period statistics (5 features)
        if active_runs:
            features['active_runs_mean'] = np.mean(active_runs)
            features['active_runs_std'] = np.std(active_runs)
            features['active_runs_max'] = np.max(active_runs)
            features['active_runs_count'] = len(active_runs)
            features['active_runs_median'] = np.median(active_runs)
        else:
            features['active_runs_mean'] = 0
            features['active_runs_std'] = 0
            features['active_runs_max'] = 0
            features['active_runs_count'] = 0
            features['active_runs_median'] = 0
        
        # Inactive period statistics (5 features)
        if inactive_runs:
            features['inactive_runs_mean'] = np.mean(inactive_runs)
            features['inactive_runs_std'] = np.std(inactive_runs)
            features['inactive_runs_max'] = np.max(inactive_runs)
            features['inactive_runs_count'] = len(inactive_runs)
            features['inactive_runs_median'] = np.median(inactive_runs)
        else:
            features['inactive_runs_mean'] = 0
            features['inactive_runs_std'] = 0
            features['inactive_runs_max'] = 0
            features['inactive_runs_count'] = 0
            features['inactive_runs_median'] = 0
        
        # Overall pattern statistics (2 features)
        features['total_runs'] = len(runs)
        features['run_length_variance'] = np.var(all_runs)
        
        return features
    
    def _extract_window_based_features(self, binary_data):
        """Extract 4 window-based temporal features"""
        features = {}
        
        if len(binary_data) < 10:
            return {'window_activity_std': 0, 'window_transitions_std': 0, 
                   'window_active_ratio_max': 0, 'window_active_ratio_min': 0}
        
        # Sliding window analysis
        window_size = min(50, len(binary_data) // 4)
        if window_size < 5:
            window_size = len(binary_data)
        
        window_activities = []
        window_transitions = []
        
        for i in range(0, len(binary_data) - window_size + 1, window_size // 2):
            window = binary_data[i:i + window_size]
            window_activities.append(np.mean(window))
            if len(window) > 1:
                window_transitions.append(np.sum(np.abs(np.diff(window))))
            else:
                window_transitions.append(0)
        
        if window_activities:
            features['window_activity_std'] = np.std(window_activities)
            features['window_active_ratio_max'] = np.max(window_activities)
            features['window_active_ratio_min'] = np.min(window_activities)
        else:
            features['window_activity_std'] = 0
            features['window_active_ratio_max'] = 0
            features['window_active_ratio_min'] = 0
        
        if window_transitions:
            features['window_transitions_std'] = np.std(window_transitions)
        else:
            features['window_transitions_std'] = 0
        
        return features
    
    def _extract_burst_intensity_features(self, binary_data):
        """Extract 3 burst intensity features"""
        features = {}
        
        if len(binary_data) == 0:
            return {'burst_intensity_mean': 0, 'burst_frequency': 0, 'burst_duration_ratio': 0}
        
        # Identify bursts (sequences of continuous 1s)
        burst_lengths = []
        burst_gaps = []
        
        runs = self._get_run_lengths(binary_data)
        
        for val, length in runs:
            if val == 1:  # Active burst
                burst_lengths.append(length)
            else:  # Gap between bursts
                burst_gaps.append(length)
        
        if burst_lengths:
            features['burst_intensity_mean'] = np.mean(burst_lengths)
            features['burst_frequency'] = len(burst_lengths) / len(binary_data)
            features['burst_duration_ratio'] = np.sum(burst_lengths) / len(binary_data)
        else:
            features['burst_intensity_mean'] = 0
            features['burst_frequency'] = 0
            features['burst_duration_ratio'] = 0
        
        return features
    
    def _extract_entropy_features(self, binary_data):
        """Extract 3 entropy and information-theoretic features"""
        features = {}
        
        if len(binary_data) == 0:
            return {'shannon_entropy': 0, 'pattern_regularity': 0, 'information_density': 0}
        
        # Shannon entropy
        unique, counts = np.unique(binary_data, return_counts=True)
        if len(counts) > 1:
            features['shannon_entropy'] = entropy(counts, base=2)
        else:
            features['shannon_entropy'] = 0
        
        # Pattern regularity (autocorrelation-based)
        if len(binary_data) > 1:
            autocorr = np.corrcoef(binary_data[:-1], binary_data[1:])[0, 1]
            features['pattern_regularity'] = autocorr if not np.isnan(autocorr) else 0
        else:
            features['pattern_regularity'] = 0
        
        # Information density
        features['information_density'] = np.sum(np.abs(np.diff(binary_data.astype(float)))) / len(binary_data)
        
        return features
    
    def create_file_wise_features(self, files_data, dataset_name, extract_labels=True):
        """
        Create feature matrix from file-wise data (improved approach)
        
        Args:
            files_data: List of file DataFrames
            dataset_name: Name of the dataset  
            extract_labels: Whether to extract true labels (False for pure inference)
        
        Returns:
            X: Feature matrix
            y: Labels (None if extract_labels=False)
            metadata: File metadata
        """
        if len(files_data) == 0:
            return None, None, None
        
        X = []
        y = [] if extract_labels else None
        metadata = []
        
        print(f"🔧 Processing {len(files_data)} files for {dataset_name}...")
        if not extract_labels:
            print(f"   🔒 INFERENCE MODE: Not using true activity labels")
        
        for file_df in files_data:
            if len(file_df) == 0 or self.binary_col not in file_df.columns:
                continue
            
            # ACTIVITY-AGNOSTIC FEATURE EXTRACTION
            # Only uses binary sensor data - no activity information
            binary_sequence = file_df[self.binary_col].values
            features = self.extract_enhanced_temporal_features(binary_sequence)
            
            if features:
                X.append(features)
                
                # Extract labels only for training/evaluation (not for pure inference)
                if extract_labels:
                    y.append(file_df['activity'].iloc[0])
                
                # Metadata for tracking (user_id, placement info)
                file_metadata = {
                    'user_id': file_df['user_id'].iloc[0],
                    'placement': file_df['placement'].iloc[0],
                    'file_id': file_df['file_id'].iloc[0],
                    'n_samples': len(file_df),
                    'dataset': dataset_name
                }
                
                # Include activity in metadata only if extracting labels
                if extract_labels:
                    file_metadata['activity'] = file_df['activity'].iloc[0]
                
                metadata.append(file_metadata)
        
        if not X:
            return None, None, None
        
        # Convert to DataFrame and handle missing features
        X_df = pd.DataFrame(X)
        X_df = X_df.fillna(0)
        
        label_info = f", {len(y)} labels" if extract_labels else ", no labels (inference mode)"
        print(f"📈 {dataset_name}: {len(X_df)} files, {len(X_df.columns)} features{label_info}")
        
        return X_df.values, (np.array(y) if extract_labels else None), metadata
    
    def calculate_information_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate NMI and reverse KL divergence percentage
        """
        # NMI percentage
        nmi_score = normalized_mutual_info_score(y_true, y_pred)
        nmi_percentage = nmi_score * 100
        
        # Reverse KL divergence percentage  
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        
        # True distribution (uniform for balanced test)
        n_classes = len(np.unique(y_true_encoded))
        true_dist = np.ones(n_classes) / n_classes
        
        # Predicted distribution (averaged probabilities)
        pred_dist = np.mean(y_pred_proba, axis=0)
        pred_dist = pred_dist / np.sum(pred_dist)  # Normalize
        
        # Reverse KL: KL(true || pred) 
        reverse_kl = entropy(true_dist, pred_dist)
        reverse_kl_percentage = (1 - reverse_kl / np.log(n_classes)) * 100
        reverse_kl_percentage = max(0, reverse_kl_percentage)  # Ensure non-negative
        
        return nmi_percentage, reverse_kl_percentage
    
    def train_and_evaluate_enhanced(self, X_train, y_train, X_test, y_test, 
                                  train_name, test_name, metadata_train, metadata_test):
        """
        Train and evaluate model with enhanced metrics
        
        IMPORTANT: This function uses true labels for EVALUATION only.
        Feature extraction is activity-agnostic and doesn't use activity labels.
        """
        print(f"🎯 Training: {train_name} → Testing: {test_name}")
        print(f"   📊 Train: {len(X_train)} files, Test: {len(X_test)} files")
        print(f"   🔒 Features extracted WITHOUT using activity information")
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Scale features (activity-agnostic)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest (same parameters as baseline)
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # Predictions (model doesn't see true test labels)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Evaluation metrics (using true labels only for evaluation)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        
        # Information-theoretic metrics
        nmi_percentage, reverse_kl_percentage = self.calculate_information_metrics(
            y_test_encoded, y_pred, y_pred_proba)
        
        # Print results
        print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   🎯 F1-Score: {f1:.3f} ({f1*100:.1f}%)")
        print(f"   🎯 Precision: {precision:.3f}")
        print(f"   🎯 Recall: {recall:.3f}")
        print(f"   📊 NMI: {nmi_percentage:.1f}%")
        print(f"   📊 Reverse KL: {reverse_kl_percentage:.2f}%")
        
        # Detailed results (JSON-serializable)
        results = {
            'train_source': train_name,
            'test_source': test_name,
            'accuracy': accuracy * 100,  # Convert to percentage
            'f1_score': f1 * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'nmi_percentage': nmi_percentage,
            'reverse_kl_percentage': reverse_kl_percentage,
            'train_files': len(X_train),
            'test_files': len(X_test),
            'features_count': X_train.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        return results, model, scaler
    
    def pure_inference_demo(self, trained_model, trained_scaler, test_files, dataset_name):
        """
        Demonstrate pure inference without using true activity labels
        This simulates real-world deployment where activities are unknown
        """
        print(f"\n🔒 PURE INFERENCE DEMO on {dataset_name}")
        print(f"📊 Extracting features WITHOUT using true activity labels...")
        
        # Extract features in inference mode (no labels)
        X_test, _, metadata = self.create_file_wise_features(test_files, dataset_name, extract_labels=False)
        
        if X_test is None:
            print("❌ No test data available for inference")
            return None
        
        # Scale features using trained scaler
        X_test_scaled = trained_scaler.transform(X_test)
        
        # Make predictions
        predictions = trained_model.predict(X_test_scaled)
        prediction_probs = trained_model.predict_proba(X_test_scaled)
        
        # Convert predictions back to activity names
        predicted_activities = [self.activities[pred] for pred in predictions]
        
        # Show inference results
        print(f"🎯 Inference Results:")
        for i, (pred_activity, metadata_item) in enumerate(zip(predicted_activities, metadata)):
            confidence = np.max(prediction_probs[i]) * 100
            print(f"   File {metadata_item['file_id']}: {pred_activity} (confidence: {confidence:.1f}%)")
        
        # Summary of predictions
        unique_preds, counts = np.unique(predicted_activities, return_counts=True)
        print(f"\n📈 Prediction Summary:")
        for activity, count in zip(unique_preds, counts):
            percentage = (count / len(predicted_activities)) * 100
            print(f"   {activity}: {count} files ({percentage:.1f}%)")
        
        return {
            'predictions': predicted_activities,
            'probabilities': prediction_probs,
            'metadata': metadata,
            'n_files': len(predicted_activities)
        }
    
    def run_enhanced_experiments(self):
        """
        Run enhanced cross-sensor cross-dataset experiments
        
        IMPORTANT: Activity labels are used ONLY for:
        1. Training the model (supervised learning)
        2. Evaluating the model (computing accuracy, F1, etc.)
        
        Feature extraction is completely activity-agnostic.
        """
        print("🚀 ENHANCED STM-MOTIONSENSE CROSS-SENSOR EXPERIMENTS")
        print("=" * 60)
        print("🔒 ETHICAL NOTE: Features are extracted without using activity labels")
        print("📊 Labels are used ONLY for training and evaluation")
        
        all_results = []
        
        # Load MotionSense data once
        print("\n📥 Loading datasets...")
        motionsense_files = self.load_motionsense_data(as_files=True)
        
        if not motionsense_files:
            print("❌ No MotionSense data available")
            return
        
        # Extract features WITH labels (for training/evaluation purposes)
        ms_X, ms_y, ms_metadata = self.create_file_wise_features(motionsense_files, 'MotionSense', extract_labels=True)
        
        if ms_X is None:
            print("❌ Failed to create MotionSense features")
            return
        
        # Run experiments for each STM placement
        for placement in self.stm_placements:
            print(f"\n🔄 Processing STM {placement}...")
            
            # Load STM placement data
            stm_files = self.load_stm_placement_data(placement, as_files=True)
            
            if not stm_files:
                print(f"❌ No data for STM {placement}")
                continue
            
            # Extract features WITH labels (for training/evaluation purposes)
            stm_X, stm_y, stm_metadata = self.create_file_wise_features(stm_files, f'STM-{placement}', extract_labels=True)
            
            if stm_X is None:
                print(f"❌ Failed to create features for STM {placement}")
                continue
            
            # Experiment 1: STM → MotionSense
            try:
                results_1, model_1, scaler_1 = self.train_and_evaluate_enhanced(
                    stm_X, stm_y, ms_X, ms_y,
                    f'STM-{placement}', 'MotionSense',
                    stm_metadata, ms_metadata
                )
                results_1['direction'] = 'STM_to_MotionSense'
                results_1['stm_placement'] = placement
                all_results.append(results_1)
                
                # Demonstrate pure inference (no labels used)
                print(f"\n🔒 DEMONSTRATING PURE INFERENCE...")
                inference_demo = self.pure_inference_demo(
                    model_1, scaler_1,
                    motionsense_files[:3],  # Test on 3 files
                    f'MotionSense-Inference'
                )
                
            except Exception as e:
                print(f"❌ STM→MotionSense failed for {placement}: {e}")
            
            # Experiment 2: MotionSense → STM
            try:
                results_2, model_2, scaler_2 = self.train_and_evaluate_enhanced(
                    ms_X, ms_y, stm_X, stm_y,
                    'MotionSense', f'STM-{placement}',
                    ms_metadata, stm_metadata
                )
                results_2['direction'] = 'MotionSense_to_STM'
                results_2['stm_placement'] = placement
                all_results.append(results_2)
            except Exception as e:
                print(f"❌ MotionSense→STM failed for {placement}: {e}")
        
        # Save and analyze results
        self.save_and_analyze_results(all_results)
        
        return all_results
    
    def save_and_analyze_results(self, all_results):
        """
        Save results and create comprehensive analysis
        """
        if not all_results:
            print("❌ No results to analyze")
            return
        
        # Save detailed results
        results_file = os.path.join(self.results_dir, 'enhanced_stm_motionsense_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary analysis
        df_results = pd.DataFrame(all_results)
        
        # Separate by direction
        stm_to_ms = df_results[df_results['direction'] == 'STM_to_MotionSense']
        ms_to_stm = df_results[df_results['direction'] == 'MotionSense_to_STM']
        
        # Find best and worst experiments
        best_exp = df_results.loc[df_results['f1_score'].idxmax()]
        worst_exp = df_results.loc[df_results['f1_score'].idxmin()]
        
        # Create summary
        summary = {
            'total_experiments': len(all_results),
            'mean_accuracy': df_results['accuracy'].mean(),
            'std_accuracy': df_results['accuracy'].std(),
            'mean_f1_score': df_results['f1_score'].mean(),
            'std_f1_score': df_results['f1_score'].std(),
            'mean_nmi_percentage': df_results['nmi_percentage'].mean(),
            'std_nmi_percentage': df_results['nmi_percentage'].std(),
            'mean_reverse_kl_percentage': df_results['reverse_kl_percentage'].mean(),
            'std_reverse_kl_percentage': df_results['reverse_kl_percentage'].std(),
            'best_experiment': {
                'train': best_exp['train_source'],
                'test': best_exp['test_source'],
                'accuracy': best_exp['accuracy'],
                'f1_score': best_exp['f1_score'],
                'nmi_percentage': best_exp['nmi_percentage'],
                'reverse_kl_percentage': best_exp['reverse_kl_percentage']
            },
            'worst_experiment': {
                'train': worst_exp['train_source'],
                'test': worst_exp['test_source'],
                'accuracy': worst_exp['accuracy'],
                'f1_score': worst_exp['f1_score'],
                'nmi_percentage': worst_exp['nmi_percentage'],
                'reverse_kl_percentage': worst_exp['reverse_kl_percentage']
            },
            'stm_to_motionsense': {
                'mean_accuracy': stm_to_ms['accuracy'].mean(),
                'mean_f1_score': stm_to_ms['f1_score'].mean(),
                'mean_nmi_percentage': stm_to_ms['nmi_percentage'].mean(),
                'mean_reverse_kl_percentage': stm_to_ms['reverse_kl_percentage'].mean()
            },
            'motionsense_to_stm': {
                'mean_accuracy': ms_to_stm['accuracy'].mean(),
                'mean_f1_score': ms_to_stm['f1_score'].mean(),
                'mean_nmi_percentage': ms_to_stm['nmi_percentage'].mean(),
                'mean_reverse_kl_percentage': ms_to_stm['reverse_kl_percentage'].mean()
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.results_dir, 'enhanced_stm_motionsense_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print comprehensive summary
        print(f"\n{'='*60}")
        print(f"ENHANCED STM-MOTIONSENSE RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Overall Performance:")
        print(f"  Accuracy: {summary['mean_accuracy']:.1f}% ± {summary['std_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['mean_f1_score']:.1f}% ± {summary['std_f1_score']:.1f}%")
        print(f"  NMI: {summary['mean_nmi_percentage']:.1f}% ± {summary['std_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['mean_reverse_kl_percentage']:.2f}% ± {summary['std_reverse_kl_percentage']:.2f}%")
        
        print(f"\nBest Experiment: {summary['best_experiment']['train']} → {summary['best_experiment']['test']}")
        print(f"  Accuracy: {summary['best_experiment']['accuracy']:.1f}%")
        print(f"  F1-Score: {summary['best_experiment']['f1_score']:.1f}%")
        print(f"  NMI: {summary['best_experiment']['nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['best_experiment']['reverse_kl_percentage']:.2f}%")
        
        print(f"\nSTM → MotionSense Performance:")
        print(f"  Accuracy: {summary['stm_to_motionsense']['mean_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['stm_to_motionsense']['mean_f1_score']:.1f}%")
        print(f"  NMI: {summary['stm_to_motionsense']['mean_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['stm_to_motionsense']['mean_reverse_kl_percentage']:.2f}%")
        
        print(f"\nMotionSense → STM Performance:")
        print(f"  Accuracy: {summary['motionsense_to_stm']['mean_accuracy']:.1f}%")
        print(f"  F1-Score: {summary['motionsense_to_stm']['mean_f1_score']:.1f}%")
        print(f"  NMI: {summary['motionsense_to_stm']['mean_nmi_percentage']:.1f}%")
        print(f"  Reverse KL: {summary['motionsense_to_stm']['mean_reverse_kl_percentage']:.2f}%")
        
        print(f"\n💾 Results saved:")
        print(f"   📁 Detailed: {results_file}")
        print(f"   📁 Summary: {summary_file}")

def main():
    """
    Run enhanced file-wise STM-MotionSense experiments
    """
    print("🚀 ENHANCED FILE-WISE STM-MOTIONSENSE EXPERIMENT")
    print("=" * 55)
    print("🔧 Implementation Features:")
    print("   ✅ File-wise processing (complete activity sessions)")
    print("   ✅ 29 robust temporal features")
    print("   ✅ Enhanced run-length analysis")
    print("   ✅ Window-based burst features")
    print("   ✅ Entropy & information-theoretic metrics")
    print("   ✅ NMI and reverse KL divergence")
    print()
    
    # Create analyzer
    analyzer = EnhancedFileWiseSTMMotionSenseAnalyzer()
    
    # Run enhanced experiments
    results = analyzer.run_enhanced_experiments()
    
    if results:
        print(f"\n🎉 ENHANCED EXPERIMENT COMPLETED!")
        print(f"📊 {len(results)} experiments completed successfully")
        print(f"📁 Results saved in: {analyzer.results_dir}")
        print(f"\n💡 Expected: Significant improvement over 38% accuracy / 30% F1 baseline")
        print(f"🎯 Target: Performance similar to STM-UCI improvement (16% → 36.8% F1)")
    else:
        print(f"\n❌ Experiment failed - check data availability")

if __name__ == "__main__":
    main()
