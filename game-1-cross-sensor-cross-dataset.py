#!/usr/bin/env python3
"""
Game-1 Cross-Sensor Cross-Dataset Adversarial HAR Study
======================================================

This script implements your exact specifications for cross-sensor cross-dataset HAR:

EXPERIMENTAL DESIGN:
Case-1: Train Dataset: STM → Test Dataset: MotionSense
- Train 5 individual models on 5 different STM placements
- Train 1 combined model on all STM sensors together  
- Test each model on MotionSense (smartphone)

Case-2: Train Dataset: MotionSense → Test Dataset: STM
- Train 1 model on MotionSense (smartphone)
- Test on 5 STM placements separately
- Test on combined STM dataset

FEATURES: Only binary temporal patterns (Game-1 focus)
GOAL: Analyze if binary features yield good accuracy for Game-1
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, normalized_mutual_info_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from datetime import datetime
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from cross_sensor_path_utils import get_results_root, get_stm_uc_root, get_uci_root
import warnings
warnings.filterwarnings('ignore')

print("🎭 GAME-1 CROSS-SENSOR CROSS-DATASET ADVERSARIAL HAR STUDY")
print("=" * 65)
print("🎯 Case-1: STM-UC (5 placements + combined) → UCI_HAR")
print("🎯 Case-2: UCI_HAR → STM-UC (5 placements + combined)")  
print("🔑 Features: Binary temporal patterns only (Game-1)")
print("📊 Goal: Analyze binary feature effectiveness for Game-1")
print()

class CrossSensorCrossDatasetAnalyzer:
    """
    Analyzer for testing cross-sensor cross-dataset HAR performance
    using only binary temporal features
    """
    
    def __init__(self, experiment_direction="stm_to_uci"):
        # Activities available in both datasets (excluding Jogging)
        self.activities = ['Downstairs', 'Laying', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        
        # STM sensor placements
        self.stm_placements = ['left-ankle', 'left-wrist', 'right-ankle', 'right-pocket', 'right-wrist']
        
        # Binary column name
        self.binary_col = 'dec_tree_out_1'
        self.stm_root = str(get_stm_uc_root())
        self.uci_root = str(get_uci_root())
        self.results_root = str(get_results_root("UCI-HAR"))
        
        # Experiment direction
        self.experiment_direction = experiment_direction
        
        # Results directory and experiment details based on direction
        if experiment_direction == "stm_to_uci":
            self.results_dir = os.path.join(self.results_root, "STM_HAR_to_UCI_HAR")
            self.train_dataset = "STM-HAR (Multiple Placements)"
            self.test_dataset = "UCI_HAR (Smartphone)"
            self.experiment_title = "Cross-Sensor Transfer: STM-HAR → UCI_HAR"
            self.experiment_description = "Train on STM-HAR placements, test cross-dataset on UCI_HAR"
        elif experiment_direction == "uci_to_stm":
            self.results_dir = self.results_root
            self.train_dataset = "UCI_HAR (Smartphone)" 
            self.test_dataset = "STM-HAR (Individual Placements)"
            self.experiment_title = "Cross-Dataset Transfer: UCI_HAR → STM-HAR Placements"
            self.experiment_description = "Train once on UCI_HAR, test on each STM-HAR placement"
        else:
            raise ValueError("experiment_direction must be 'stm_to_uci' or 'uci_to_stm'")
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Add combined placement for training on all placements
        self.all_placements = self.stm_placements + ['combined']
        
        print(f"🎯 Target: Cross-sensor cross-dataset adversarial HAR")
        print(f"📊 Activities: {len(self.activities)} common activities")
        print(f"🎭 STM Placements: {len(self.stm_placements)} placements")
        print(f"🔑 Features: Binary temporal patterns only (NO user identity)")
        print(f"� Direction: {self.train_dataset} → {self.test_dataset}")
        print(f"📁 Results: {self.results_dir}")
        print()
    
    def load_stm_placement_data(self, placement):
        """
        Load binary data from specific STM-UC placement across all users
        """
        print(f"📊 Loading STM-UC {placement} data...", end="")
        
        all_data = []
        users_loaded = 0
        
        for user_id in range(1, 12):  # Users 1-11
            user_dir = os.path.join(self.stm_root, f"User {user_id}", "Processed")
            
            if not os.path.exists(user_dir):
                continue
            
            user_activities = 0
            for activity in self.activities:
                activity_dir = os.path.join(user_dir, activity)
                
                if not os.path.exists(activity_dir):
                    continue
                
                # Load data from specific placement
                if activity in ['Standing', 'Sitting', 'Walking', 'Laying']:
                    # Single file: STM-UC/User X/Processed/ACTIVITY/PLACEMENT.csv
                    file_path = os.path.join(activity_dir, f"{placement}.csv")
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if self.binary_col in df.columns and len(df) > 0:
                                df['user_id'] = user_id
                                df['activity'] = activity
                                df['placement'] = placement
                                all_data.append(df[[self.binary_col, 'user_id', 'activity', 'placement']])
                                user_activities += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {file_path}: {e}")
                
                elif activity in ['Upstairs', 'Downstairs']:
                    # Multiple files: STM-UC/User X/Processed/ACTIVITY/PLACEMENT/
                    placement_dir = os.path.join(activity_dir, placement)
                    
                    if os.path.exists(placement_dir):
                        csv_files = [f for f in os.listdir(placement_dir) if f.endswith('.csv')]
                        
                        for csv_file in csv_files:
                            file_path = os.path.join(placement_dir, csv_file)
                            try:
                                df = pd.read_csv(file_path)
                                if self.binary_col in df.columns and len(df) > 0:
                                    df['user_id'] = user_id
                                    df['activity'] = activity
                                    df['placement'] = placement
                                    all_data.append(df[[self.binary_col, 'user_id', 'activity', 'placement']])
                                    user_activities += 1
                            except Exception as e:
                                print(f"⚠️ Error loading {file_path}: {e}")
            
            if user_activities > 0:
                users_loaded += 1
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" ✅ {len(combined_df)} samples from {users_loaded} users")
            return combined_df
        else:
            print(f" ❌ No data found")
            return pd.DataFrame()
    
    def load_stm_combined_data(self):
        """
        Load binary data from ALL STM-UC placements combined
        """
        print(f"📊 Loading STM-UC combined (all placements) data...", end="")
        
        all_data = []
        users_loaded = 0
        placement_counts = {placement: 0 for placement in self.stm_placements}
        
        for placement in self.stm_placements:
            placement_df = self.load_stm_placement_data(placement)
            if len(placement_df) > 0:
                placement_counts[placement] = len(placement_df)
                all_data.append(placement_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" ✅ {len(combined_df)} total samples")
            
            # Print breakdown by placement
            print(f"   📍 Placement breakdown:")
            for placement, count in placement_counts.items():
                if count > 0:
                    print(f"      {placement}: {count} samples")
            
            return combined_df
        else:
            print(f" ❌ No data found")
            return pd.DataFrame()
    
    def load_motionsense_data(self):
        """
        Load UCI_HAR data (accelerometer + gyroscope + binary)
        """
        print(f"📱 Loading UCI_HAR data...", end="")
        
        all_data = []
        
        # Load accelerometer data (contains binary field)
        accel_root = self.uci_root
        
        if not os.path.exists(accel_root):
            print(f" ❌ UCI_HAR directory not found: {accel_root}")
            return pd.DataFrame()
        
        activities_loaded = 0
        
        # Activity mapping based on actual directory structure
        activity_mapping = {
            'activity_WALKING_DOWNSTAIRS': 'Downstairs',
            'activity_LAYING': 'Laying', 
            'activity_SITTING': 'Sitting',
            'activity_STANDING': 'Standing',
            'activity_WALKING_UPSTAIRS': 'Upstairs',
            'activity_WALKING': 'Walking'
        }
        
        for subject_id in range(1, 31):  # subjects 1-30
            subject_dir = os.path.join(self.uci_root, f'subject_{subject_id}')
            
            if not os.path.exists(subject_dir):
                continue
                
            for activity_folder in os.listdir(subject_dir):
                activity_path = os.path.join(subject_dir, activity_folder)
                if not os.path.isdir(activity_path):
                    continue
                
                # Map activity folder name to our standard names
                if activity_folder not in activity_mapping:
                    continue
                
                activity_name = activity_mapping[activity_folder]
                if activity_name not in self.activities:
                    continue
                
                # Load data.csv from activity folder
                csv_file = os.path.join(activity_path, 'data.csv')
                
                if os.path.exists(csv_file):
                    try:
                        df = pd.read_csv(csv_file)
                        if self.binary_col in df.columns and len(df) > 0:
                            df['user_id'] = subject_id
                            df['activity'] = activity_name
                            df['placement'] = 'smartphone'  # UCI_HAR is smartphone data
                            
                            all_data.append(df[[self.binary_col, 'user_id', 'activity', 'placement']])
                            
                    except Exception as e:
                        print(f"⚠️ Error loading {csv_file}: {e}")
            
            activities_loaded += 1
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f" ✅ {len(combined_df)} samples, {activities_loaded} activities")
            return combined_df
        else:
            print(f" ❌ No data found")
            return pd.DataFrame()
    
    def extract_binary_temporal_features(self, binary_sequence):
        """
        Extract temporal features from binary sequence (placement-agnostic)
        NO user identity, NO raw sensor data - only temporal patterns
        """
        if len(binary_sequence) == 0:
            return {}
        
        binary_data = np.array(binary_sequence)
        features = {}
        
        # 1. BASIC ACTIVITY STATISTICS
        features['activity_ratio'] = np.mean(binary_data)
        features['total_samples'] = len(binary_data)
        features['active_samples'] = np.sum(binary_data)
        features['inactive_samples'] = np.sum(1 - binary_data)
        
        # 2. TEMPORAL DYNAMICS
        if len(binary_data) > 1:
            # State transitions
            transitions = np.sum(np.abs(np.diff(binary_data)))
            features['transitions'] = transitions
            features['transition_rate'] = transitions / len(binary_data) if len(binary_data) > 0 else 0
            
            # Run-length encoding
            runs = self._get_run_lengths(binary_data)
            if runs:
                features.update(self._extract_run_length_features(runs))
        
        # 3. PATTERN COMPLEXITY
        features['pattern_complexity'] = len(np.unique(binary_data))
        features['temporal_variance'] = np.var(binary_data)
        
        # 4. ADVANCED TEMPORAL FEATURES
        if len(binary_data) > 0:
            features['mean_activity'] = np.mean(binary_data)
            features['std_activity'] = np.std(binary_data)
            features['activity_energy'] = np.sum(binary_data**2)
            
            if len(binary_data) > 1:
                # Differential features
                diff_data = np.diff(binary_data.astype(float))
                features['diff_mean'] = np.mean(diff_data)
                features['diff_std'] = np.std(diff_data)
                features['diff_max'] = np.max(np.abs(diff_data)) if len(diff_data) > 0 else 0
                
                # Zero crossings (periodicity detection)
                mean_val = np.mean(binary_data)
                if mean_val != 0:
                    features['zero_crossings'] = np.sum(np.diff(np.signbit(binary_data - mean_val)))
                else:
                    features['zero_crossings'] = 0
        
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
    
    def _calculate_kl_divergence(self, y_true_encoded, y_pred_proba, label_encoder):
        """
        Calculate normalized KL-Divergence between true and predicted distributions
        Normalizes using a practical upper bound to make it comparable across different numbers of classes
        Returns both normalized KL-Divergence and Reverse KL (1-normalized_KL) for easier interpretation
        """
        try:
            # Convert true labels to probability distribution
            n_classes = len(label_encoder.classes_)
            n_samples = len(y_true_encoded)
            
            # Create true distribution (one-hot encoded)
            true_dist = np.zeros((n_samples, n_classes))
            true_dist[np.arange(n_samples), y_true_encoded] = 1.0
            
            # Ensure predicted probabilities are valid (no zeros for KL calculation)
            pred_dist = np.clip(y_pred_proba, 1e-15, 1.0)
            pred_dist = pred_dist / pred_dist.sum(axis=1, keepdims=True)  # Renormalize
            
            # Calculate average KL-Divergence across all samples
            kl_divs = []
            for i in range(n_samples):
                # KL(P||Q) where P=true, Q=predicted
                kl_div = entropy(true_dist[i], pred_dist[i])
                if not np.isnan(kl_div) and not np.isinf(kl_div):
                    kl_divs.append(kl_div)
            
            if kl_divs:
                avg_kl_divergence = np.mean(kl_divs)
                
                # Use a practical upper bound: KL when predicting completely wrong class with high confidence
                # This is approximately log(n_classes) + log(max_confidence)
                # We'll use a simpler approach: normalize by the 95th percentile of observed KL values
                # or use a fixed upper bound that works well in practice
                practical_upper_bound = np.log(n_classes) * 2.0  # More generous upper bound
                
                # Normalize by practical upper bound
                normalized_kl = avg_kl_divergence / practical_upper_bound if practical_upper_bound > 0 else 0.0
                # Use sigmoid-like transformation to better handle extreme values
                normalized_kl = normalized_kl / (1.0 + normalized_kl)  # Maps [0, inf) to [0, 1)
                
                reverse_kl = 1.0 - normalized_kl  # Reverse for easier interpretation (higher = better)
                return normalized_kl, reverse_kl
            else:
                return 0.0, 1.0
                
        except Exception as e:
            print(f"⚠️ Error calculating KL-Divergence: {e}")
            return 0.0, 1.0

    def _calculate_vulnerability(self, y_pred_proba):
        """Calculate mean max predicted probability across samples."""
        if len(y_pred_proba) == 0:
            return 0.0
        return float(np.mean(np.max(y_pred_proba, axis=1)))
    
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
    
    def _extract_run_length_features(self, runs):
        """Extract statistical features from run lengths"""
        if not runs:
            return {}
        
        active_runs = [length for val, length in runs if val == 1]
        inactive_runs = [length for val, length in runs if val == 0]
        all_runs = [length for val, length in runs]
        
        features = {}
        
        # Active period features (shake periods)
        if active_runs:
            features['active_runs_count'] = len(active_runs)
            features['active_runs_mean'] = np.mean(active_runs)
            features['active_runs_std'] = np.std(active_runs)
            features['active_runs_max'] = np.max(active_runs)
            features['active_runs_min'] = np.min(active_runs)
        else:
            features['active_runs_count'] = 0
            features['active_runs_mean'] = 0
            features['active_runs_std'] = 0
            features['active_runs_max'] = 0
            features['active_runs_min'] = 0
        
        # Inactive period features (static periods)
        if inactive_runs:
            features['inactive_runs_count'] = len(inactive_runs)
            features['inactive_runs_mean'] = np.mean(inactive_runs)
            features['inactive_runs_std'] = np.std(inactive_runs)
            features['inactive_runs_max'] = np.max(inactive_runs)
            features['inactive_runs_min'] = np.min(inactive_runs)
        else:
            features['inactive_runs_count'] = 0
            features['inactive_runs_mean'] = 0
            features['inactive_runs_std'] = 0
            features['inactive_runs_max'] = 0
            features['inactive_runs_min'] = 0
        
        # Overall pattern features
        features['total_runs'] = len(runs)
        features['run_variance'] = np.var(all_runs)
        features['run_mean'] = np.mean(all_runs)
        
        return features
    
    def _get_progress_file(self, placement):
        """Get progress file path for a placement"""
        return os.path.join(self.results_dir, f"progress_{placement}.json")
    
    def _get_results_file(self, placement):
        """Get results file path for a placement"""
        return os.path.join(self.results_dir, f"results_{placement}.json")
    
    def _get_model_file(self, placement):
        """Get model file path for a placement"""
        return os.path.join(self.results_dir, f"model_{placement}.joblib")
    
    def _load_progress(self, placement):
        """Load progress for a specific placement"""
        progress_file = self._get_progress_file(placement)
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {'completed': False, 'timestamp': None}
        return {'completed': False, 'timestamp': None}
    
    def _save_progress(self, placement, completed=True):
        """Save progress for a specific placement"""
        progress_file = self._get_progress_file(placement)
        progress_data = {
            'completed': completed,
            'timestamp': datetime.now().isoformat(),
            'placement': placement
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _is_placement_completed(self, placement):
        """Check if a placement has been completed"""
        progress = self._load_progress(placement)
        return progress.get('completed', False)
    
    def _load_cached_results(self, placement):
        """Load cached results for a placement"""
        results_file = self._get_results_file(placement)
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_placement_results(self, placement, result, save_model=True):
        """Save results for a specific placement with model serialization"""
        # Save JSON results (without non-serializable objects)
        results_file = self._get_results_file(placement)
        serializable_result = {k: v for k, v in result.items() 
                              if k not in ['model', 'scaler', 'label_encoder']}
        
        with open(results_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        # Save model objects separately
        if save_model and 'model' in result:
            model_file = self._get_model_file(placement)
            model_data = {
                'model': result['model'],
                'scaler': result['scaler'],
                'label_encoder': result['label_encoder']
            }
            joblib.dump(model_data, model_file)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.str_, np.unicode_)):
            return str(obj)
        elif hasattr(obj, 'tolist'):  # Handle other numpy types
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalar types
            return obj.item()
        else:
            return obj
        
        # Mark as completed
        self._save_progress(placement, completed=True)
        
        print(f"💾 Results saved for {placement}: {results_file}")
        if save_model:
            print(f"🤖 Model saved for {placement}: {self._get_model_file(placement)}")
    
    def _create_confusion_matrix_plot(self, result, placement):
        """Create and save confusion matrix plot"""
        try:
            cm = np.array(result['confusion_matrix'])
            classes = result['label_encoder_classes']
            title_prefix = (
                f'UCI_HAR → {placement}'
                if self.experiment_direction == "uci_to_stm"
                else f'{placement} → UCI_HAR'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix: {title_prefix}\nAccuracy: {result["accuracy"]:.3f}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            plot_file = os.path.join(self.results_dir, f"confusion_matrix_{placement}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Confusion matrix saved: {plot_file}")
            return plot_file
            
        except Exception as e:
            print(f"⚠️ Error creating confusion matrix for {placement}: {e}")
            return None
    
    def _create_feature_importance_plot(self, result, placement):
        """Create and save feature importance plot"""
        try:
            # Get feature names (simplified)
            n_features = len(result['feature_importance'])
            feature_names = [f"feature_{i}" for i in range(n_features)]
            title_prefix = (
                f'UCI_HAR → {placement}'
                if self.experiment_direction == "uci_to_stm"
                else f'{placement} → UCI_HAR'
            )
            
            # Sort by importance
            importance_data = list(zip(feature_names, result['feature_importance']))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            
            # Plot top 20 features
            top_features = importance_data[:20]
            if top_features:
                names, importances = zip(*top_features)
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(names)), importances)
                plt.yticks(range(len(names)), names)
                plt.xlabel('Feature Importance')
                plt.title(f'Top Feature Importances: {title_prefix}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_file = os.path.join(self.results_dir, f"feature_importance_{placement}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📈 Feature importance plot saved: {plot_file}")
                return plot_file
            
        except Exception as e:
            print(f"⚠️ Error creating feature importance plot for {placement}: {e}")
            return None
    
    def _create_summary_plots(self, all_results, model_comparison):
        """Create comprehensive summary plots"""
        try:
            # Performance comparison plot
            self._create_performance_comparison_plot(model_comparison)
            
            # Side-by-side accuracy bar chart
            self._create_accuracy_bar_chart(model_comparison)
            
            # Information-theoretic metrics plots (NMI and KL-Divergence)
            self._create_information_theoretic_plots(model_comparison)
            
            # Hypothesis test visualization
            self._create_hypothesis_test_plot(model_comparison)
            
            # Detailed metrics heatmap
            self._create_metrics_heatmap(all_results)
            
        except Exception as e:
            print(f"⚠️ Error creating summary plots: {e}")
    
    def _create_performance_comparison_plot(self, model_comparison):
        """Create performance comparison plot"""
        try:
            if not model_comparison:
                return
            
            df = pd.DataFrame(model_comparison)
            placement_axis_label = "STM-HAR Test Placement" if self.experiment_direction == "uci_to_stm" else "STM-HAR Training Placement"
            performance_title = f'{self.train_dataset} → {self.test_dataset}'
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy comparison
            placements = df['placement'].values
            accuracies = df['accuracy'].values
            colors = []
            for p in placements:
                if p == 'combined':
                    colors.append('green')
                elif p == 'right-pocket':
                    colors.append('red')
                else:
                    colors.append('blue')
            
            ax1.bar(range(len(placements)), accuracies, color=colors, alpha=0.7)
            ax1.set_xlabel(placement_axis_label)
            ax1.set_ylabel(f'Accuracy on {self.test_dataset}')
            ax1.set_title(f'{performance_title}\n(Red = Same, Blue = Cross, Green = Combined)')
            ax1.set_xticks(range(len(placements)))
            ax1.set_xticklabels(placements, rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add accuracy values on top of bars
            for i, acc in enumerate(accuracies):
                ax1.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            f1_scores = df['f1_score'].values
            ax2.bar(range(len(placements)), f1_scores, color=colors, alpha=0.7)
            ax2.set_xlabel(placement_axis_label)
            ax2.set_ylabel(f'F1-Score on {self.test_dataset}')
            ax2.set_title('F1-Score Comparison\n(Red = Same, Blue = Cross, Green = Combined)')
            ax2.set_xticks(range(len(placements)))
            ax2.set_xticklabels(placements, rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add F1 values on top of bars
            for i, f1 in enumerate(f1_scores):
                ax2.text(i, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "performance_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Performance comparison plot saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating performance comparison plot: {e}")
    
    def _create_hypothesis_test_plot(self, model_comparison):
        """Create hypothesis test visualization"""
        try:
            if not model_comparison:
                return
            
            df = pd.DataFrame(model_comparison)
            
            # Group by sensor type
            same_sensor = df[df['is_same_sensor'] == True]
            cross_sensor = df[df['is_same_sensor'] == False]
            
            if len(same_sensor) == 0 or len(cross_sensor) == 0:
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Box plot comparison
            data_to_plot = [same_sensor['accuracy'].values, cross_sensor['accuracy'].values]
            labels = ['Same-sensor\n(right-pocket)', 'Cross-sensor\n(others)']
            
            box_plot = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('red')
            box_plot['boxes'][0].set_alpha(0.7)
            box_plot['boxes'][1].set_facecolor('blue') 
            box_plot['boxes'][1].set_alpha(0.7)
            
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Hypothesis Test: Same-sensor vs Cross-sensor')
            ax1.grid(True, alpha=0.3)
            
            # Add statistical annotations
            same_mean = same_sensor['accuracy'].mean()
            cross_mean = cross_sensor['accuracy'].mean()
            diff = same_mean - cross_mean
            
            ax1.text(0.5, 0.95, f'Same-sensor mean: {same_mean:.3f}', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            ax1.text(0.5, 0.88, f'Cross-sensor mean: {cross_mean:.3f}',
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
            ax1.text(0.5, 0.81, f'Difference: {diff:.3f}',
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='gray', alpha=0.3))
            
            # Hypothesis conclusion
            if abs(diff) < 0.1:  # Within 10%
                conclusion = "✅ HYPOTHESIS SUPPORTED"
                color = 'green'
            else:
                conclusion = "❌ HYPOTHESIS REJECTED"
                color = 'red'
            
            ax1.text(0.5, 0.74, conclusion,
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
                    fontweight='bold')
            
            # Scatter plot
            ax2.scatter(same_sensor.index, same_sensor['accuracy'], 
                       color='red', alpha=0.7, s=100, label='Same-sensor')
            ax2.scatter(cross_sensor.index, cross_sensor['accuracy'], 
                       color='blue', alpha=0.7, s=100, label='Cross-sensor')
            
            ax2.axhline(y=same_mean, color='red', linestyle='--', alpha=0.7, label=f'Same-sensor mean: {same_mean:.3f}')
            ax2.axhline(y=cross_mean, color='blue', linestyle='--', alpha=0.7, label=f'Cross-sensor mean: {cross_mean:.3f}')
            
            ax2.set_xlabel('Experiment Index')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Individual Results: Same-sensor vs Cross-sensor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "hypothesis_test.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔬 Hypothesis test plot saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating hypothesis test plot: {e}")
    
    def _create_metrics_heatmap(self, all_results):
        """Create metrics heatmap for all placements"""
        try:
            if not all_results:
                return
            
            # Extract metrics
            placements = list(all_results.keys())
            metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'nmi_percentage', 'vulnerability']
            
            data = []
            for placement in placements:
                if placement in all_results:
                    result = all_results[placement]
                    row = [result.get(metric, 0) for metric in metrics]
                    data.append(row)
            
            if not data:
                return
            
            df_metrics = pd.DataFrame(data, index=placements, columns=metrics)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       center=0.5, square=True, linewidths=0.5)
            plt.title(f'Performance Metrics Heatmap: {self.train_dataset} → {self.test_dataset}')
            plt.xlabel('Metrics')
            plt.ylabel('Placement')
            plt.tight_layout()
            
            plot_file = os.path.join(self.results_dir, "metrics_heatmap.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"🔥 Metrics heatmap saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating metrics heatmap: {e}")
    
    def _create_accuracy_bar_chart(self, model_comparison):
        """Create a focused side-by-side accuracy comparison bar chart"""
        try:
            if not model_comparison:
                return
            
            df = pd.DataFrame(model_comparison)
            configuration_label = "Testing Configuration" if self.experiment_direction == "uci_to_stm" else "Training Configuration"
            
            # Sort by accuracy for better visualization
            df = df.sort_values('accuracy', ascending=True)
            
            plt.figure(figsize=(14, 8))
            
            # Define colors based on placement type
            colors = []
            for _, row in df.iterrows():
                if row['placement'] == 'combined':
                    colors.append('green')  # Combined = green
                elif row['is_same_sensor']:
                    colors.append('red')    # Same-sensor = red
                else:
                    colors.append('blue')   # Cross-sensor = blue
            
            # Create horizontal bar chart for better readability
            bars = plt.barh(range(len(df)), df['accuracy'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Customize the plot
            plt.xlabel('Accuracy', fontsize=14, fontweight='bold')
            plt.ylabel(configuration_label, fontsize=14, fontweight='bold')
            plt.title('Model Accuracy Comparison: Cross-Sensor Cross-Dataset HAR\\n'
                     f'Training: {self.train_dataset} → Testing: {self.test_dataset}',
                     fontsize=16, fontweight='bold', pad=20)
            
            # Set y-axis labels
            labels = []
            for _, row in df.iterrows():
                if row['placement'] == 'combined':
                    labels.append('ALL PLACEMENTS\\n(Combined)')
                else:
                    labels.append(f"{row['placement'].upper()}\\n({row['placement']})")
            
            plt.yticks(range(len(df)), labels, fontsize=11)
            plt.xticks(fontsize=11)
            
            # Add grid for better readability
            plt.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add accuracy values on the bars
            for i, (bar, acc) in enumerate(zip(bars, df['accuracy'])):
                plt.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontweight='bold', fontsize=10)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='Combined (All Placements)'),
                Patch(facecolor='red', alpha=0.8, label='Same-Sensor (right-pocket)'),
                Patch(facecolor='blue', alpha=0.8, label='Cross-Sensor (other placements)')
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
            
            # Set x-axis limits for better view
            plt.xlim(0, max(df['accuracy']) * 1.15)
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "accuracy_bar_chart.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Accuracy bar chart saved: {plot_file}")
            
        except Exception as e:
            print(f"⚠️ Error creating accuracy bar chart: {e}")
    
    def _create_information_theoretic_plots(self, model_comparison):
        """Create plots for information-theoretic metrics (NMI and KL-Divergence)"""
        try:
            if not model_comparison:
                return
            
            df = pd.DataFrame(model_comparison)
            configuration_label = "Testing Configuration" if self.experiment_direction == "uci_to_stm" else "Training Configuration"
            
            # Sort by accuracy for consistent ordering
            df = df.sort_values('accuracy', ascending=True)
            
            # Create subplot figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Information-Theoretic Metrics: Cross-Sensor Cross-Dataset HAR\\n'
                        'Privacy and Information Leakage Analysis', fontsize=16, fontweight='bold', y=0.98)
            
            # Define colors based on placement type
            colors = []
            for _, row in df.iterrows():
                if row['placement'] == 'combined':
                    colors.append('green')  # Combined = green
                elif row['is_same_sensor']:
                    colors.append('red')    # Same-sensor = red
                else:
                    colors.append('blue')   # Cross-sensor = blue
            
            # 1. NMI Percentage Bar Chart
            bars1 = ax1.barh(range(len(df)), df['nmi_percentage'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax1.set_xlabel('Normalized Mutual Information (%)', fontweight='bold')
            ax1.set_ylabel(configuration_label, fontweight='bold')
            ax1.set_title('NMI: Information Preservation\\n(Higher = Better Information Retention)', fontweight='bold', pad=15)
            ax1.set_yticks(range(len(df)))
            ax1.set_yticklabels([f"{row['placement'].upper()}" for _, row in df.iterrows()])
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            ax1.set_xlim(0, max(df['nmi_percentage']) * 1.1)
            
            # Add NMI values on bars
            for i, (bar, nmi) in enumerate(zip(bars1, df['nmi_percentage'])):
                ax1.text(nmi + 0.5, i, f'{nmi:.1f}%', va='center', fontweight='bold', fontsize=9)
            
            # 2. Reverse KL Percentage Bar Chart (easier interpretation)
            bars2 = ax2.barh(range(len(df)), df['reverse_kl_percentage'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_xlabel('Reverse KL-Divergence (%)', fontweight='bold')
            ax2.set_ylabel(configuration_label, fontweight='bold')
            ax2.set_title('Reverse KL: Model Calibration\\n(Higher = Better Calibration, Less Privacy Risk)', fontweight='bold', pad=15)
            ax2.set_yticks(range(len(df)))
            ax2.set_yticklabels([f"{row['placement'].upper()}" for _, row in df.iterrows()])
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            ax2.set_xlim(0, max(df['reverse_kl_percentage']) * 1.1)
            
            # Add Reverse KL values on bars
            for i, (bar, rkl) in enumerate(zip(bars2, df['reverse_kl_percentage'])):
                ax2.text(rkl + 0.5, i, f'{rkl:.1f}%', va='center', fontweight='bold', fontsize=9)
            
            # 3. NMI vs Accuracy Scatter Plot
            scatter1 = ax3.scatter(df['accuracy'], df['nmi_percentage'], c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
            ax3.set_xlabel('Accuracy', fontweight='bold')
            ax3.set_ylabel('NMI (%)', fontweight='bold')
            ax3.set_title('NMI vs Accuracy\\nInformation-Performance Trade-off', fontweight='bold', pad=15)
            ax3.grid(True, alpha=0.3)
            
            # Add placement labels to scatter points
            for i, row in df.iterrows():
                ax3.annotate(row['placement'][:3].upper(), (row['accuracy'], row['nmi_percentage']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
            
            # 4. Reverse KL vs Accuracy Scatter Plot
            scatter2 = ax4.scatter(df['accuracy'], df['reverse_kl_percentage'], c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
            ax4.set_xlabel('Accuracy', fontweight='bold')
            ax4.set_ylabel('Reverse KL (%)', fontweight='bold')
            ax4.set_title('Reverse KL vs Accuracy\\nCalibration-Performance Trade-off', fontweight='bold', pad=15)
            ax4.grid(True, alpha=0.3)
            
            # Add placement labels to scatter points
            for i, row in df.iterrows():
                ax4.annotate(row['placement'][:3].upper(), (row['accuracy'], row['reverse_kl_percentage']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
            
            # Add shared legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='Combined (All Placements)'),
                Patch(facecolor='red', alpha=0.8, label='Same-Sensor (right-pocket)'),
                Patch(facecolor='blue', alpha=0.8, label='Cross-Sensor (other placements)')
            ]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.1)  # Make room for legend
            
            plot_file = os.path.join(self.results_dir, "information_theoretic_metrics.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Information-theoretic metrics plot saved: {plot_file}")
            
            # Print summary statistics
            print(f"\\n📈 INFORMATION-THEORETIC SUMMARY:")
            print(f"   🔍 Average NMI: {df['nmi_percentage'].mean():.1f}% ± {df['nmi_percentage'].std():.1f}%")
            print(f"   🎯 Average Reverse KL: {df['reverse_kl_percentage'].mean():.1f}% ± {df['reverse_kl_percentage'].std():.1f}%")
            print(f"   📊 Best NMI: {df.loc[df['nmi_percentage'].idxmax(), 'placement']} ({df['nmi_percentage'].max():.1f}%)")
            print(f"   📊 Best Reverse KL: {df.loc[df['reverse_kl_percentage'].idxmax(), 'placement']} ({df['reverse_kl_percentage'].max():.1f}%)")
            
        except Exception as e:
            print(f"⚠️ Error creating information-theoretic plots: {e}")
    
    def _save_final_results(self, all_results, model_comparison):
        """Save final comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        final_results = {
            'experiment_name': 'Cross-Sensor Cross-Dataset Adversarial HAR',
            'timestamp': timestamp,
            'hypothesis': 'Sensor placement does not significantly matter for binary-based cross-dataset HAR',
            'model': 'RandomForest',
            'train_dataset': self.train_dataset,
            'test_dataset': self.test_dataset,
            'feature_type': 'Binary temporal patterns only (no user identity)',
            'stm_placements': self.stm_placements,
            'activities': self.activities,
            'individual_results': all_results,
            'model_comparison': model_comparison,
            'summary_statistics': self._calculate_summary_statistics(model_comparison)
        }
        
        # Save comprehensive JSON
        final_file = os.path.join(self.results_dir, f"final_results_{timestamp}.json")
        with open(final_file, 'w') as f:
            json.dump(self._make_json_serializable(final_results), f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = os.path.join(self.results_dir, f"model_comparison_{timestamp}.csv")
        pd.DataFrame(model_comparison).to_csv(csv_file, index=False)
        
        # Save summary report
        self._save_summary_report(final_results, timestamp)
        
        print(f"\n💾 FINAL RESULTS SAVED:")
        print(f"   📄 Comprehensive JSON: {final_file}")
        print(f"   📊 CSV comparison: {csv_file}")
    
    def _calculate_summary_statistics(self, model_comparison):
        """Calculate summary statistics for hypothesis testing"""
        if not model_comparison:
            return {}
        
        df = pd.DataFrame(model_comparison)
        
        same_sensor = df[df['is_same_sensor'] == True]
        cross_sensor = df[df['is_same_sensor'] == False]
        
        summary = {
            'overall': {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'mean_f1': df['f1_score'].mean(),
                'std_f1': df['f1_score'].std(),
                'mean_nmi': df['nmi_percentage'].mean(),
                'std_nmi': df['nmi_percentage'].std(),
                'mean_reverse_kl': df['reverse_kl_percentage'].mean(),
                'std_reverse_kl': df['reverse_kl_percentage'].std(),
                'mean_vulnerability': df['vulnerability'].mean(),
                'std_vulnerability': df['vulnerability'].std()
            }
        }
        
        if len(same_sensor) > 0:
            summary['same_sensor'] = {
                'count': len(same_sensor),
                'mean_accuracy': same_sensor['accuracy'].mean(),
                'std_accuracy': same_sensor['accuracy'].std(),
                'mean_f1': same_sensor['f1_score'].mean(),
                'std_f1': same_sensor['f1_score'].std(),
                'mean_nmi': same_sensor['nmi_percentage'].mean(),
                'std_nmi': same_sensor['nmi_percentage'].std(),
                'mean_reverse_kl': same_sensor['reverse_kl_percentage'].mean(),
                'std_reverse_kl': same_sensor['reverse_kl_percentage'].std(),
                'mean_vulnerability': same_sensor['vulnerability'].mean(),
                'std_vulnerability': same_sensor['vulnerability'].std()
            }
        
        if len(cross_sensor) > 0:
            summary['cross_sensor'] = {
                'count': len(cross_sensor),
                'mean_accuracy': cross_sensor['accuracy'].mean(),
                'std_accuracy': cross_sensor['accuracy'].std(),
                'mean_f1': cross_sensor['f1_score'].mean(),
                'std_f1': cross_sensor['f1_score'].std(),
                'mean_nmi': cross_sensor['nmi_percentage'].mean(),
                'std_nmi': cross_sensor['nmi_percentage'].std(),
                'mean_reverse_kl': cross_sensor['reverse_kl_percentage'].mean(),
                'std_reverse_kl': cross_sensor['reverse_kl_percentage'].std(),
                'mean_vulnerability': cross_sensor['vulnerability'].mean(),
                'std_vulnerability': cross_sensor['vulnerability'].std()
            }
        
        if len(same_sensor) > 0 and len(cross_sensor) > 0:
            acc_diff = same_sensor['accuracy'].mean() - cross_sensor['accuracy'].mean()
            f1_diff = same_sensor['f1_score'].mean() - cross_sensor['f1_score'].mean()
            nmi_diff = same_sensor['nmi_percentage'].mean() - cross_sensor['nmi_percentage'].mean()
            vulnerability_diff = same_sensor['vulnerability'].mean() - cross_sensor['vulnerability'].mean()
            
            summary['hypothesis_test'] = {
                'accuracy_difference': acc_diff,
                'f1_difference': f1_diff,
                'nmi_difference': nmi_diff,
                'vulnerability_difference': vulnerability_diff,
                'hypothesis_supported': abs(acc_diff) < 0.1,  # Within 10%
                'performance_gap': abs(acc_diff),
                'nmi_gap': abs(nmi_diff)
            }
        
        return summary
    
    def _save_summary_report(self, final_results, timestamp):
        """Save human-readable summary report"""
        report_file = os.path.join(self.results_dir, f"summary_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CROSS-SENSOR CROSS-DATASET ADVERSARIAL HAR STUDY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Hypothesis: {final_results['hypothesis']}\n")
            f.write(f"Model: {final_results['model']}\n")
            f.write(f"Train Dataset: {final_results['train_dataset']}\n")
            f.write(f"Test Dataset: {final_results['test_dataset']}\n")
            f.write(f"Features: {final_results['feature_type']}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            summary = final_results.get('summary_statistics', {})
            if 'overall' in summary:
                f.write(f"Overall Mean Accuracy: {summary['overall']['mean_accuracy']:.3f} ± {summary['overall']['std_accuracy']:.3f}\n")
                f.write(f"Overall Mean F1-Score: {summary['overall']['mean_f1']:.3f} ± {summary['overall']['std_f1']:.3f}\n")
                f.write(f"Overall Mean Vulnerability: {summary['overall']['mean_vulnerability']:.3f} ± {summary['overall']['std_vulnerability']:.3f}\n\n")
            
            # Hypothesis test results
            if 'hypothesis_test' in summary:
                ht = summary['hypothesis_test']
                f.write("HYPOTHESIS TEST RESULTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Same-sensor accuracy: {summary['same_sensor']['mean_accuracy']:.3f}\n")
                f.write(f"Cross-sensor accuracy: {summary['cross_sensor']['mean_accuracy']:.3f}\n")
                f.write(f"Performance difference: {ht['accuracy_difference']:.3f}\n")
                f.write(f"Vulnerability difference: {ht['vulnerability_difference']:.3f}\n")
                f.write(f"Hypothesis supported: {'YES' if ht['hypothesis_supported'] else 'NO'}\n\n")
            
            # Individual results
            f.write("INDIVIDUAL PLACEMENT RESULTS\n")
            f.write("-" * 40 + "\n")
            for comp in final_results['model_comparison']:
                sensor_type = "SAME-SENSOR" if comp['is_same_sensor'] else "CROSS-SENSOR"
                f.write(
                    f"{comp['placement']:15} ({sensor_type:12}): "
                    f"Acc={comp['accuracy']:.3f}, F1={comp['f1_score']:.3f}, Vulnerability={comp['vulnerability']:.3f}\n"
                )
        
        print(f"   📝 Summary report: {report_file}")
        return report_file
    
    def create_dataset_from_dataframe(self, df, label='unknown'):
        """
        Create feature matrix from dataframe grouped by user and activity
        """
        if len(df) == 0:
            return None, None, None
        
        X = []
        y = []
        metadata = []
        
        # Group by user and activity to create samples
        grouped = df.groupby(['user_id', 'activity'])
        
        for (user_id, activity), group in grouped:
            if len(group) == 0:
                continue
            
            # Extract binary temporal features from this user-activity combination
            binary_sequence = group[self.binary_col].values
            features = self.extract_binary_temporal_features(binary_sequence)
            
            if features:
                X.append(features)
                y.append(activity)
                metadata.append({
                    'user_id': user_id,
                    'activity': activity,
                    'n_samples': len(group),
                    'dataset': label
                })
        
        if not X:
            return None, None, None
        
        # Convert to DataFrame and handle missing features
        X_df = pd.DataFrame(X)
        X_df = X_df.fillna(0)
        
        print(f"📈 {label} dataset: {len(X_df)} samples, {len(X_df.columns)} features")
        return X_df.values, np.array(y), metadata
    
    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, train_placement):
        """
        Train and evaluate Random Forest model with comprehensive metrics and saving
        """
        try:
            # Encode labels
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model (same as original game-1-cross-sensor)
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Use all available cores
            )
            model.fit(X_train_scaled, y_train_encoded)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_encoded, y_pred)
            f1 = f1_score(y_test_encoded, y_pred, average='weighted')
            precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
            
            # Calculate Normalized Mutual Information (as percentage)
            nmi = normalized_mutual_info_score(y_test_encoded, y_pred, average_method='arithmetic')
            nmi_percentage = nmi * 100  # Convert to percentage for easier interpretation
            
            # Calculate KL-Divergence and Reverse KL
            kl_divergence, reverse_kl = self._calculate_kl_divergence(y_test_encoded, y_pred_proba, le)
            reverse_kl_percentage = reverse_kl * 100  # Convert to percentage
            vulnerability = self._calculate_vulnerability(y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test_encoded, y_pred)
            
            return {
                'train_placement': train_placement,
                'model_type': 'RandomForest',
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'nmi': nmi,
                'nmi_percentage': nmi_percentage,
                'kl_divergence': kl_divergence,
                'reverse_kl': reverse_kl,
                'reverse_kl_percentage': reverse_kl_percentage,
                'vulnerability': vulnerability,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'class_report': class_report,
                'confusion_matrix': cm.tolist(),
                'y_true': y_test_encoded.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist(),
                'feature_importance': model.feature_importances_.tolist(),
                'label_encoder_classes': le.classes_.tolist(),
                'model': model,
                'scaler': scaler,
                'label_encoder': le
            }
            
        except Exception as e:
            print(f"❌ Error training model for {train_placement}: {e}")
            return None
    
    def run_full_experiment(self):
        """
        Run the complete cross-sensor cross-dataset experiment with checkpointing
        Supports both STM→MotionSense and MotionSense→STM directions
        """
        print(f"🚀 Starting Cross-Sensor Cross-Dataset Experiment: {self.experiment_direction}")
        print(f"📋 {self.experiment_title}")
        print("=" * 75)
        print("✅ GRANULAR CHECKPOINTING: Resumes from exact placement")
        print("✅ COMPREHENSIVE RESULTS: Confusion matrices, feature importance, all metrics")
        print("✅ ROBUST SAVING: JSON + Model serialization for maximum recoverability")
        print("🌲 Model: Random Forest (same as game-1-cross-sensor)")
        print(f"🔄 Direction: {self.train_dataset} → {self.test_dataset}")
        print(f"💡 Goal: {self.experiment_description}")
        print()
        
        if self.experiment_direction == "stm_to_uci":
            return self._run_stm_to_uci()
        elif self.experiment_direction == "uci_to_stm":
            return self._run_uci_to_stm()
        else:
            raise ValueError("Invalid experiment direction")
    
    def _run_stm_to_uci(self):
        """
        Run STM-UC → UCI_HAR Cross-Sensor Cross-Dataset Transfer Experiment
        
        Methodology:
        1. Train on EACH STM-UC placement individually (left-ankle, left-wrist, right-ankle, right-pocket, right-wrist)
        2. Train on COMBINED STM-UC dataset (all placements together)
        3. Test ALL models on UCI_HAR dataset (smartphone placement)
        
        Goal: Evaluate how well STM-UC placement-specific models transfer to UCI_HAR
        """
        print("\n" + "🔄" * 25)
        print("📍➡️📱 STM-UC PLACEMENTS → UCI_HAR TRANSFER")
        print("🔄" * 25)
        print("📋 EXPERIMENT: Train on each STM-UC placement, test on UCI_HAR")
        print("🎯 GOAL: Evaluate cross-sensor cross-dataset transfer performance")
        print("🏋️ TRAINING: STM-UC individual placements + combined (11 users)")
        print("🧪 TESTING: UCI_HAR (smartphone, 30 users)")
        print("="*70)
        
        # Load UCI_HAR test data once
        print("\n📱 LOADING TEST DATA (UCI_HAR)")
        uci_df = self.load_motionsense_data()
        if len(uci_df) == 0:
            print("❌ Failed to load UCI_HAR data!")
            return None
        
        X_test, y_test, test_meta = self.create_dataset_from_dataframe(uci_df, 'UCI_HAR')
        if X_test is None:
            print("❌ Failed to create UCI_HAR dataset!")
            return None
        
        print(f"✅ UCI_HAR test set ready: {len(X_test)} samples")
        
        # Results storage
        all_results = {}
        model_comparison = []
        completed_placements = []
        
        # Check what's already completed
        print("\n🔍 CHECKING EXISTING PROGRESS...")
        for placement in self.all_placements:
            if self._is_placement_completed(placement):
                print(f"✅ {placement}: Already completed")
                cached_result = self._load_cached_results(placement)
                if cached_result:
                    # Ensure cached results are JSON serializable
                    all_results[placement] = self._make_json_serializable(cached_result)
                    model_comparison.append({
                        'placement': placement,
                        'model': 'RandomForest',
                        'accuracy': cached_result['accuracy'],
                        'f1_score': cached_result['f1_score'],
                        'precision': cached_result['precision'],
                        'recall': cached_result['recall'],
                        'vulnerability': cached_result.get('vulnerability'),
                        'is_same_sensor': placement == 'right-pocket',
                        'is_combined': placement == 'combined'
                    })
                    completed_placements.append(placement)
            else:
                print(f"⏳ {placement}: Not completed")
        
        if len(completed_placements) == len(self.all_placements):
            print("🎉 ALL PLACEMENTS ALREADY COMPLETED!")
            self.analyze_results(all_results, model_comparison)
            self._create_summary_plots(all_results, model_comparison)
            return all_results, model_comparison
        
        # Train and test models for remaining placements
        print(f"\n🎭 TRAINING REMAINING PLACEMENT-SPECIFIC MODELS")
        print("-" * 50)
        
        for placement in self.all_placements:
            if placement in completed_placements:
                print(f"⏭️  Skipping {placement} (already completed)")
                continue
                
            print(f"\n📍 PLACEMENT: {placement}")
            
            try:
                # Load STM-UC training data for this placement
                if placement == 'combined':
                    stm_df = self.load_stm_combined_data()
                else:
                    stm_df = self.load_stm_placement_data(placement)
                    
                if len(stm_df) == 0:
                    print(f"⚠️ No data found for {placement}, skipping...")
                    continue
                
                X_train, y_train, train_meta = self.create_dataset_from_dataframe(stm_df, f'STM_UC_{placement}')
                if X_train is None:
                    print(f"⚠️ Failed to create dataset for {placement}, skipping...")
                    continue
                
                # Ensure feature alignment between train and test
                train_df = pd.DataFrame(X_train)
                test_df = pd.DataFrame(X_test)
                
                # Get all features and align
                all_features = set(train_df.columns) | set(test_df.columns)
                for col in all_features:
                    if col not in train_df.columns:
                        train_df[col] = 0
                    if col not in test_df.columns:
                        test_df[col] = 0
                
                # Reorder columns
                common_cols = sorted(all_features)
                X_train_aligned = train_df[common_cols].values
                X_test_aligned = test_df[common_cols].values
                
                print(f"  📊 Training: {len(X_train_aligned)} samples, {X_train_aligned.shape[1]} features")
                print(f"  📊 Testing: {len(X_test_aligned)} samples")
                print(f"  🤖 Training Random Forest model...", end="")
                
                # Train model
                result = self.train_and_evaluate_model(
                    X_train_aligned, y_train, X_test_aligned, y_test, placement
                )
                
                if result:
                    print(
                        f" ✅ Acc: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}, "
                        f"NMI: {result['nmi_percentage']:.1f}%, RKL: {result['reverse_kl_percentage']:.1f}%, "
                        f"Vuln: {result['vulnerability']:.3f}"
                    )
                    
                    # Save results immediately with checkpointing
                    self._save_placement_results(placement, result, save_model=True)
                    
                    # Create plots
                    self._create_confusion_matrix_plot(result, placement)
                    self._create_feature_importance_plot(result, placement)
                    
                    # Add to comparison
                    all_results[placement] = self._make_json_serializable({k: v for k, v in result.items() 
                                            if k not in ['model', 'scaler', 'label_encoder']})
                    model_comparison.append({
                        'placement': placement,
                        'model': 'RandomForest',
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score'],
                        'precision': result['precision'],
                        'recall': result['recall'],
                        'nmi': result['nmi'],
                        'nmi_percentage': result['nmi_percentage'],
                        'kl_divergence': result['kl_divergence'],
                        'reverse_kl': result['reverse_kl'],
                        'reverse_kl_percentage': result['reverse_kl_percentage'],
                        'vulnerability': result['vulnerability'],
                        'is_same_sensor': placement == 'right-pocket',
                        'is_combined': placement == 'combined'
                    })
                    
                    print(f"💾 {placement} results saved and checkpointed!")
                    
                else:
                    print(f" ❌ Failed")
                    
            except Exception as e:
                print(f"❌ Error processing {placement}: {e}")
                continue
        
        # Analyze results
        self.analyze_results(all_results, model_comparison)
        
        # Create summary plots and final results
        self._create_summary_plots(all_results, model_comparison)
        
        # Save final comprehensive results
        self._save_final_results(all_results, model_comparison)
        
        return all_results, model_comparison
    
    def _run_uci_to_stm(self):
        """
        Run UCI_HAR → STM-UC Cross-Dataset Transfer Experiment
        
        Methodology:
        1. Train ONCE on UCI_HAR dataset (smartphone placement)
        2. Test on EACH individual STM-UC placement (left-ankle, left-wrist, right-ankle, right-pocket, right-wrist)
        3. Test on COMBINED STM-UC dataset (all placements together)
        
        Goal: Evaluate how well UCI_HAR-trained models transfer to different STM-UC sensor placements
        """
        print("\n" + "🔄" * 25)
        print("📱➡️📍 UCI_HAR → STM-UC PLACEMENT TRANSFER")
        print("🔄" * 25)
        print("📋 EXPERIMENT: Train once on UCI_HAR, test on each STM-UC placement")
        print("🎯 GOAL: Evaluate cross-dataset transfer to different sensor placements")
        print("🏋️ TRAINING: UCI_HAR (smartphone, 30 users)")
        print("🧪 TESTING: STM-UC individual placements + combined")
        print("="*70)
        
        # Load UCI_HAR training data
        print("\n📱 LOADING TRAINING DATA (UCI_HAR)")
        uci_df = self.load_motionsense_data()
        if len(uci_df) == 0:
            print("❌ Failed to load UCI_HAR data!")
            return None
        
        X_train, y_train, train_meta = self.create_dataset_from_dataframe(uci_df, 'UCI_HAR')
        if X_train is None:
            print("❌ Failed to create UCI_HAR dataset!")
            return None
        
        print(f"✅ UCI_HAR training set ready: {len(X_train)} samples")
        
        # Results storage
        all_results = {}
        model_comparison = []
        completed_placements = []
        
        # Check what's already completed
        print("\n🔍 CHECKING EXISTING PROGRESS...")
        for placement in self.all_placements:
            if self._is_placement_completed(placement):
                print(f"✅ {placement}: Already completed")
                cached_result = self._load_cached_results(placement)
                if cached_result:
                    # Ensure cached results are JSON serializable
                    all_results[placement] = self._make_json_serializable(cached_result)
                    model_comparison.append({
                        'placement': placement,
                        'model': 'RandomForest',
                        'accuracy': cached_result['accuracy'],
                        'f1_score': cached_result['f1_score'],
                        'precision': cached_result['precision'],
                        'recall': cached_result['recall'],
                        'vulnerability': cached_result.get('vulnerability'),
                        'is_same_sensor': placement == 'right-pocket',
                        'is_combined': placement == 'combined'
                    })
                    completed_placements.append(placement)
            else:
                print(f"⏳ {placement}: Not completed")
        
        if len(completed_placements) == len(self.all_placements):
            print("🎉 ALL PLACEMENTS ALREADY COMPLETED!")
            self.analyze_results(all_results, model_comparison)
            self._create_summary_plots(all_results, model_comparison)
            return all_results, model_comparison
        
        # Train once on UCI_HAR, then test on each STM-UC placement
        print(f"\n🎭 TRAINING ONCE ON UCI_HAR, TESTING ON ALL STM-UC PLACEMENTS")
        print("-" * 60)
        
        # Train the model once on UCI_HAR
        print(f"\n🤖 Training Single Random Forest Model on UCI_HAR...")
        
        # Encode labels for training
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train_encoded)
        print(f"✅ Model trained on {len(X_train_scaled)} UCI_HAR samples")
        
        # Now test on each STM-UC placement
        for placement in self.all_placements:
            if placement in completed_placements:
                print(f"⏭️  Skipping {placement} (already completed)")
                continue
                
            print(f"\n📍 TESTING ON PLACEMENT: {placement}")
            
            try:
                # Load STM-UC test data for this placement
                if placement == 'combined':
                    stm_df = self.load_stm_combined_data()
                else:
                    stm_df = self.load_stm_placement_data(placement)
                    
                if len(stm_df) == 0:
                    print(f"⚠️ No data found for {placement}, skipping...")
                    continue
                
                X_test, y_test, test_meta = self.create_dataset_from_dataframe(stm_df, f'STM_UC_{placement}')
                if X_test is None:
                    print(f"⚠️ Failed to create dataset for {placement}, skipping...")
                    continue
                
                # Ensure feature alignment between train and test
                train_df = pd.DataFrame(X_train)
                test_df = pd.DataFrame(X_test)
                
                # Get all features and align
                all_features = set(train_df.columns) | set(test_df.columns)
                for col in all_features:
                    if col not in train_df.columns:
                        train_df[col] = 0
                    if col not in test_df.columns:
                        test_df[col] = 0
                
                # Reorder columns
                common_cols = sorted(all_features)
                X_test_aligned = test_df[common_cols].values
                
                # Scale test features using the same scaler
                X_test_scaled = scaler.transform(X_test_aligned)
                
                print(f"  📊 Testing: {len(X_test_scaled)} samples")
                print(f"  🔍 Evaluating on {placement}...", end="")
                
                # Predict using the pre-trained model
                y_test_encoded = le.transform(y_test)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, normalized_mutual_info_score
                accuracy = accuracy_score(y_test_encoded, y_pred)
                f1 = f1_score(y_test_encoded, y_pred, average='weighted')
                precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                
                # Calculate Normalized Mutual Information (as percentage)
                nmi = normalized_mutual_info_score(y_test_encoded, y_pred, average_method='arithmetic')
                nmi_percentage = nmi * 100  # Convert to percentage for easier interpretation
                
                # Calculate KL-Divergence and Reverse KL
                kl_divergence, reverse_kl = self._calculate_kl_divergence(y_test_encoded, y_pred_proba, le)
                reverse_kl_percentage = reverse_kl * 100  # Convert to percentage
                vulnerability = self._calculate_vulnerability(y_pred_proba)
                
                # Classification report
                class_report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
                
                # Confusion matrix
                cm = confusion_matrix(y_test_encoded, y_pred)
                
                # Create result dictionary
                result = {
                    'placement': placement,
                    'model_type': 'RandomForest',
                    'train_dataset': 'UCI_HAR',
                    'test_dataset': f'STM_HAR_{placement}',
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'nmi': nmi,
                    'nmi_percentage': nmi_percentage,
                    'kl_divergence': kl_divergence,
                    'reverse_kl': reverse_kl,
                    'reverse_kl_percentage': reverse_kl_percentage,
                    'vulnerability': vulnerability,
                    'n_train_samples': len(X_train_scaled),
                    'n_test_samples': len(X_test_scaled),
                    'n_features': X_train_scaled.shape[1],
                    'class_report': class_report,
                    'confusion_matrix': cm.tolist(),
                    'y_true': y_test_encoded.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist(),
                    'feature_importance': model.feature_importances_.tolist(),
                    'label_encoder_classes': le.classes_.tolist(),
                    'model': model,
                    'scaler': scaler,
                    'label_encoder': le
                }
                
                print(
                    f" ✅ Acc: {result['accuracy']:.3f}, F1: {result['f1_score']:.3f}, "
                    f"NMI: {result['nmi_percentage']:.1f}%, RKL: {result['reverse_kl_percentage']:.1f}%, "
                    f"Vuln: {result['vulnerability']:.3f}"
                )
                
                # Save results immediately with checkpointing
                self._save_placement_results(placement, result, save_model=True)
                
                # Create plots
                self._create_confusion_matrix_plot(result, placement)
                self._create_feature_importance_plot(result, placement)
                
                # Add to comparison
                all_results[placement] = self._make_json_serializable({k: v for k, v in result.items() 
                                        if k not in ['model', 'scaler', 'label_encoder']})
                model_comparison.append({
                    'placement': placement,
                    'model': 'RandomForest',
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'nmi': result['nmi'],
                    'nmi_percentage': result['nmi_percentage'],
                    'kl_divergence': result['kl_divergence'],
                    'reverse_kl': result['reverse_kl'],
                    'reverse_kl_percentage': result['reverse_kl_percentage'],
                    'vulnerability': result['vulnerability'],
                    'is_same_sensor': placement == 'right-pocket',
                    'is_combined': placement == 'combined'
                })
                
                print(f"💾 {placement} results saved and checkpointed!")
                
            except Exception as e:
                print(f"❌ Error processing {placement}: {e}")
                continue
        
        # Analyze results
        self.analyze_results(all_results, model_comparison)
        
        # Create summary plots and final results
        self._create_summary_plots(all_results, model_comparison)
        
        # Save final comprehensive results
        self._save_final_results(all_results, model_comparison)
        
        return all_results, model_comparison
    
    def analyze_results(self, all_results, model_comparison):
        """
        Analyze and display results focusing on the hypothesis
        """
        print("\n" + "=" * 70)
        print("🔬 HYPOTHESIS ANALYSIS: Cross-Dataset Performance Analysis")
        print("=" * 70)
        
        if not model_comparison:
            print("❌ No results to analyze!")
            return
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(model_comparison)
        
        print(f"\n📊 EXPERIMENT RESULTS ({self.experiment_direction.upper()}):")
        print("-" * 50)
        
        if self.experiment_direction == "stm_to_uci":
            # Same-sensor performance (right-pocket)
            same_sensor = results_df[results_df['is_same_sensor'] == True]
            if len(same_sensor) > 0:
                same_acc = same_sensor['accuracy'].iloc[0]
                same_f1 = same_sensor['f1_score'].iloc[0]
                same_prec = same_sensor['precision'].iloc[0]
                same_rec = same_sensor['recall'].iloc[0]
                print(f"📱 Same-sensor (right-pocket → UCI_HAR):")
                print(f"    Accuracy:  {same_acc:.3f}")
                print(f"    F1-Score:  {same_f1:.3f}")
                print(f"    Precision: {same_prec:.3f}")
                print(f"    Recall:    {same_rec:.3f}")
                print(f"    NMI:       {same_sensor['nmi_percentage'].iloc[0]:.1f}%")
                print(f"    Reverse KL: {same_sensor['reverse_kl_percentage'].iloc[0]:.1f}%")
            
            # Cross-sensor performance (other placements)
            cross_sensor = results_df[results_df['is_same_sensor'] == False]
            if len(cross_sensor) > 0:
                cross_acc_mean = cross_sensor['accuracy'].mean()
                cross_acc_std = cross_sensor['accuracy'].std()
                cross_f1_mean = cross_sensor['f1_score'].mean()
                cross_f1_std = cross_sensor['f1_score'].std()
                cross_prec_mean = cross_sensor['precision'].mean()
                cross_rec_mean = cross_sensor['recall'].mean()
                cross_nmi_mean = cross_sensor['nmi_percentage'].mean()
                cross_nmi_std = cross_sensor['nmi_percentage'].std()
                cross_rkl_mean = cross_sensor['reverse_kl_percentage'].mean()
                cross_rkl_std = cross_sensor['reverse_kl_percentage'].std()
                
                print(f"\n🔄 Cross-sensor (others → UCI_HAR):")
                print(f"    Accuracy:   {cross_acc_mean:.3f} ± {cross_acc_std:.3f}")
                print(f"    F1-Score:   {cross_f1_mean:.3f} ± {cross_f1_std:.3f}")
                print(f"    Precision:  {cross_prec_mean:.3f}")
                print(f"    Recall:     {cross_rec_mean:.3f}")
                print(f"    NMI:        {cross_nmi_mean:.1f}% ± {cross_nmi_std:.1f}%")
                print(f"    Reverse KL: {cross_rkl_mean:.1f}% ± {cross_rkl_std:.1f}%")
                
                # Individual cross-sensor results
                print(f"\n    Individual cross-sensor results:")
                for _, row in cross_sensor.iterrows():
                    print(f"      {row['placement']:15}: Acc={row['accuracy']:.3f}, F1={row['f1_score']:.3f}, NMI={row['nmi_percentage']:.1f}%, RKL={row['reverse_kl_percentage']:.1f}%")
        
        elif self.experiment_direction == "uci_to_stm":
            # All results are cross-dataset (UCI_HAR → STM-UC placements)
            print(f"📱 Cross-dataset (UCI_HAR → STM-HAR placements):")
            
            # Overall performance statistics
            all_acc_mean = results_df['accuracy'].mean()
            all_acc_std = results_df['accuracy'].std()
            all_f1_mean = results_df['f1_score'].mean()
            all_f1_std = results_df['f1_score'].std()
            all_prec_mean = results_df['precision'].mean()
            all_rec_mean = results_df['recall'].mean()
            all_nmi_mean = results_df['nmi_percentage'].mean()
            all_nmi_std = results_df['nmi_percentage'].std()
            all_rkl_mean = results_df['reverse_kl_percentage'].mean()
            all_rkl_std = results_df['reverse_kl_percentage'].std()
            all_vulnerability_mean = results_df['vulnerability'].mean()
            all_vulnerability_std = results_df['vulnerability'].std()
            
            print(f"    Overall Accuracy:   {all_acc_mean:.3f} ± {all_acc_std:.3f}")
            print(f"    Overall F1-Score:   {all_f1_mean:.3f} ± {all_f1_std:.3f}")
            print(f"    Overall Precision:  {all_prec_mean:.3f}")
            print(f"    Overall Recall:     {all_rec_mean:.3f}")
            print(f"    Overall NMI:        {all_nmi_mean:.1f}% ± {all_nmi_std:.1f}%")
            print(f"    Overall Reverse KL: {all_rkl_mean:.1f}% ± {all_rkl_std:.1f}%")
            print(f"    Overall Vulnerability: {all_vulnerability_mean:.3f} ± {all_vulnerability_std:.3f}")
            
            # Individual placement results
            print(f"\n    Individual placement results:")
            for _, row in results_df.iterrows():
                print(
                    f"      {row['placement']:15}: Acc={row['accuracy']:.3f}, F1={row['f1_score']:.3f}, "
                    f"NMI={row['nmi_percentage']:.1f}%, RKL={row['reverse_kl_percentage']:.1f}%, "
                    f"Vuln={row['vulnerability']:.3f}"
                )
            
            # Best and worst placements
            best_placement = results_df.loc[results_df['accuracy'].idxmax()]
            worst_placement = results_df.loc[results_df['accuracy'].idxmin()]
            
            print(f"\n    Best placement:  {best_placement['placement']} (Acc={best_placement['accuracy']:.3f})")
            print(f"    Worst placement: {worst_placement['placement']} (Acc={worst_placement['accuracy']:.3f})")
            print(f"    Placement range: {best_placement['accuracy'] - worst_placement['accuracy']:.3f}")
        
        # Analysis continues for both directions
        if self.experiment_direction == "stm_to_uci" and len(same_sensor) > 0 and len(cross_sensor) > 0:
            acc_diff = same_acc - cross_acc_mean
            f1_diff = same_f1 - cross_f1_mean
            print(f"\n📈 PERFORMANCE COMPARISON:")
            print(f"    Accuracy difference:  {acc_diff:+.3f}")
            print(f"    F1-Score difference:  {f1_diff:+.3f}")
            
            # Hypothesis conclusion
            if abs(acc_diff) < 0.1:  # Within 10%
                print(f"    🎉 HYPOTHESIS SUPPORTED: Difference < 10%!")
                print(f"       Cross-sensor transfer works well!")
            elif abs(acc_diff) < 0.15:  # Within 15%
                print(f"    🤔 HYPOTHESIS PARTIALLY SUPPORTED: Small difference")
                print(f"       Some sensor placement effect exists")
            else:
                print(f"    ❌ HYPOTHESIS REJECTED: Significant difference > 15%")
                print(f"       Sensor placement matters considerably")
        
        # Best overall models
        print(f"\n🏆 BEST RESULTS:")
        print("-" * 25)
        best_by_acc = results_df.loc[results_df['accuracy'].idxmax()]
        best_by_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        
        print(f"🎯 Best Accuracy:  {best_by_acc['placement']} = {best_by_acc['accuracy']:.3f}")
        print(f"🎯 Best F1-Score:  {best_by_f1['placement']} = {best_by_f1['f1_score']:.3f}")
        
        # Best NMI and Reverse KL
        best_by_nmi = results_df.loc[results_df['nmi_percentage'].idxmax()]
        best_by_rkl = results_df.loc[results_df['reverse_kl_percentage'].idxmax()]
        best_by_vulnerability = results_df.loc[results_df['vulnerability'].idxmax()]
        print(f"🎯 Best NMI:       {best_by_nmi['placement']} = {best_by_nmi['nmi_percentage']:.1f}%")
        print(f"🎯 Best Reverse KL: {best_by_rkl['placement']} = {best_by_rkl['reverse_kl_percentage']:.1f}%")
        print(f"🎯 Best Vulnerability: {best_by_vulnerability['placement']} = {best_by_vulnerability['vulnerability']:.3f}")
        
        # Overall hypothesis conclusion
        print(f"\n🧪 OVERALL HYPOTHESIS CONCLUSION:")
        print("-" * 40)
        
        if self.experiment_direction == "stm_to_uci":
            # STM-UC to UCI_HAR analysis
            if len(same_sensor) > 0 and len(cross_sensor) > 0:
                same_avg_acc = same_sensor['accuracy'].mean()
                cross_avg_acc = cross_sensor['accuracy'].mean()
                overall_diff = same_avg_acc - cross_avg_acc
                
                print(f"📊 Same-sensor average accuracy:  {same_avg_acc:.3f}")
                print(f"📊 Cross-sensor average accuracy: {cross_avg_acc:.3f}")
                print(f"📊 Overall difference:            {overall_diff:+.3f}")
                
                if abs(overall_diff) < 0.05:  # Within 5%
                    print(f"🎉 STRONG SUPPORT: Binary features are highly transferable!")
                    print(f"   Sensor placement has minimal impact on performance.")
                elif abs(overall_diff) < 0.1:  # Within 10%
                    print(f"✅ HYPOTHESIS SUPPORTED: Cross-sensor transfer works!")
                    print(f"   Binary temporal patterns are largely placement-independent.")
                elif abs(overall_diff) < 0.15:  # Within 15%
                    print(f"🤔 PARTIAL SUPPORT: Some placement effect exists.")
                    print(f"   Binary features still provide reasonable transferability.")
                else:
                    print(f"❌ HYPOTHESIS NOT SUPPORTED: Significant placement effect.")
                    print(f"   Sensor location affects binary-based activity recognition.")
        
        elif self.experiment_direction == "uci_to_stm":
            # UCI_HAR to STM-UC analysis
            all_acc_mean = results_df['accuracy'].mean()
            all_acc_std = results_df['accuracy'].std()
            best_acc = results_df['accuracy'].max()
            worst_acc = results_df['accuracy'].min()
            acc_range = best_acc - worst_acc
            
            print(f"📊 Cross-dataset performance (UCI_HAR→STM-HAR):")
            print(f"   Average accuracy: {all_acc_mean:.3f} ± {all_acc_std:.3f}")
            print(f"   Best accuracy:    {best_acc:.3f}")
            print(f"   Worst accuracy:   {worst_acc:.3f}")
            print(f"   Accuracy range:   {acc_range:.3f}")
            
            if acc_range < 0.05:  # Within 5%
                print(f"🎉 STRONG SUPPORT: Very consistent across STM placements!")
                print(f"   Binary features are highly placement-independent.")
            elif acc_range < 0.1:  # Within 10%
                print(f"✅ HYPOTHESIS SUPPORTED: Good consistency across placements!")
                print(f"   Binary temporal patterns transfer well across placements.")
            elif acc_range < 0.15:  # Within 15%
                print(f"🤔 PARTIAL SUPPORT: Some placement variation exists.")
                print(f"   Binary features show moderate transferability.")
            else:
                print(f"❌ HYPOTHESIS NOT SUPPORTED: Significant placement variation.")
                print(f"   Sensor location significantly affects cross-dataset transfer.")
        
        # Common implications
        print(f"\n💡 IMPLICATIONS ({self.experiment_direction.upper()}):")
        if self.experiment_direction == "stm_to_uci":
            if 'overall_diff' in locals() and abs(overall_diff) < 0.1:
                print(f"   • Binary temporal features show good cross-sensor transferability")
                print(f"   • Cross-dataset HAR benefits from placement-independent features")
                print(f"   • Your hypothesis is supported by the experimental evidence")
            else:
                print(f"   • Binary temporal features show limited cross-sensor transferability")
                print(f"   • Cross-dataset HAR is affected by placement differences")
                print(f"   • Your hypothesis needs refinement")
        else:  # uci_to_stm
            if 'acc_range' in locals() and acc_range < 0.1:
                print(f"   • Binary features transfer consistently across STM-HAR placements")
                print(f"   • UCI_HAR→STM-HAR transfer is relatively placement-independent")
                print(f"   • Cross-dataset approach shows promise for deployment")
            else:
                print(f"   • Binary features show placement-dependent transfer to STM-HAR")
                print(f"   • UCI_HAR→STM-HAR transfer varies by target placement")
                print(f"   • Placement-specific optimization may be needed")
    
    def _save_summary_report(self, final_results, timestamp):
        """Save human-readable summary report"""
        report_file = os.path.join(self.results_dir, f"summary_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CROSS-SENSOR CROSS-DATASET ADVERSARIAL HAR STUDY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Hypothesis: {final_results['hypothesis']}\n")
            f.write(f"Model: {final_results['model']}\n")
            f.write(f"Train Dataset: {final_results['train_dataset']}\n")
            f.write(f"Test Dataset: {final_results['test_dataset']}\n")
            f.write(f"Features: {final_results['feature_type']}\n\n")
            
            # Results summary
            f.write("RESULTS SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            summary = final_results.get('summary_statistics', {})
            if 'overall' in summary:
                f.write(f"Overall Mean Accuracy: {summary['overall']['mean_accuracy']:.3f} ± {summary['overall']['std_accuracy']:.3f}\n")
                f.write(f"Overall Mean F1-Score: {summary['overall']['mean_f1']:.3f} ± {summary['overall']['std_f1']:.3f}\n")
                f.write(f"Overall Mean Vulnerability: {summary['overall']['mean_vulnerability']:.3f} ± {summary['overall']['std_vulnerability']:.3f}\n\n")
            
            # Hypothesis test results
            if 'hypothesis_test' in summary:
                ht = summary['hypothesis_test']
                f.write("HYPOTHESIS TEST RESULTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Same-sensor accuracy: {summary['same_sensor']['mean_accuracy']:.3f}\n")
                f.write(f"Cross-sensor accuracy: {summary['cross_sensor']['mean_accuracy']:.3f}\n")
                f.write(f"Performance difference: {ht['accuracy_difference']:.3f}\n")
                f.write(f"Vulnerability difference: {ht['vulnerability_difference']:.3f}\n")
                f.write(f"Hypothesis supported: {'YES' if ht['hypothesis_supported'] else 'NO'}\n\n")
            
            # Individual results
            f.write("INDIVIDUAL PLACEMENT RESULTS\n")
            f.write("-" * 40 + "\n")
            for comp in final_results['model_comparison']:
                sensor_type = "SAME-SENSOR" if comp['is_same_sensor'] else "CROSS-SENSOR"
                f.write(
                    f"{comp['placement']:15} ({sensor_type:12}): "
                    f"Acc={comp['accuracy']:.3f}, F1={comp['f1_score']:.3f}, Vulnerability={comp['vulnerability']:.3f}\n"
                )
        
        print(f"   📝 Summary report: {report_file}")
        return report_file

def main():
    """
    Main execution function for the branch-specific UCI_HAR -> STM-HAR run
    """
    print("🚀 ONE-WAY CROSS-SENSOR CROSS-DATASET HAR EXPERIMENT")
    print("=" * 65)
    print("🔬 Running the branch-required direction only: UCI_HAR → STM-HAR")
    print("✅ Features: Only binary temporal patterns (same as game-1-cross-sensor)")
    print("🌲 Model: Random Forest (same as game-1-cross-sensor)")
    print("💾 Results: Results/UCI-HAR")
    print()
    
    print("\\n" + "🔄" * 20 + " UCI_HAR → STM-HAR " + "🔄" * 20)
    analyzer_uci_to_stm = CrossSensorCrossDatasetAnalyzer(experiment_direction="uci_to_stm")
    analyzer_uci_to_stm.run_full_experiment()
    
    print("\\n🎉 ONE-WAY EXPERIMENT COMPLETED!")
    print(f"   📁 UCI_HAR → STM-HAR: {analyzer_uci_to_stm.results_dir}")

def create_bidirectional_comparison(results_stm_to_uci, results_uci_to_stm):
    """Create comparison plots and analysis for both directions"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create comparison directory
    comparison_dir = "results/game-1-stm-uci/bidirectional_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        if results_stm_to_uci and results_uci_to_stm:
            all_results_1, model_comparison_1 = results_stm_to_uci
            all_results_2, model_comparison_2 = results_uci_to_stm
            
            # Create comparison DataFrame
            df1 = pd.DataFrame(model_comparison_1)
            df1['direction'] = 'STM-UC → UCI_HAR'
            
            df2 = pd.DataFrame(model_comparison_2)
            df2['direction'] = 'UCI_HAR → STM-UC'
            
            combined_df = pd.concat([df1, df2], ignore_index=True)
            
            # Create bidirectional comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Bidirectional Cross-Dataset HAR Performance Comparison', fontsize=16)
            
            # 1. Accuracy comparison by placement
            pivot_acc = combined_df.pivot(index='placement', columns='direction', values='accuracy')
            pivot_acc.plot(kind='bar', ax=axes[0,0], width=0.8)
            axes[0,0].set_title('Accuracy by Placement and Direction')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. F1-Score comparison
            pivot_f1 = combined_df.pivot(index='placement', columns='direction', values='f1_score')
            pivot_f1.plot(kind='bar', ax=axes[0,1], width=0.8)
            axes[0,1].set_title('F1-Score by Placement and Direction')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # 3. Direction comparison (boxplot)
            sns.boxplot(data=combined_df, x='direction', y='accuracy', ax=axes[1,0])
            axes[1,0].set_title('Accuracy Distribution by Direction')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].grid(True, alpha=0.3)
            
            # 4. Placement comparison (heatmap)
            heatmap_data = combined_df.pivot_table(
                values='accuracy', 
                index='placement', 
                columns='direction'
            )
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[1,1])
            axes[1,1].set_title('Accuracy Heatmap: Placement vs Direction')
            
            plt.tight_layout()
            plot_file = os.path.join(comparison_dir, "bidirectional_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save combined results
            combined_df.to_csv(os.path.join(comparison_dir, "bidirectional_results.csv"), index=False)
            
            # Calculate summary statistics
            summary_stats = {}
            for direction in ['STM → MotionSense', 'MotionSense → STM']:
                direction_data = combined_df[combined_df['direction'] == direction]
                summary_stats[direction] = {
                    'mean_accuracy': direction_data['accuracy'].mean(),
                    'std_accuracy': direction_data['accuracy'].std(),
                    'mean_f1': direction_data['f1_score'].mean(),
                    'std_f1': direction_data['f1_score'].std(),
                    'mean_nmi': direction_data['nmi_percentage'].mean(),
                    'std_nmi': direction_data['nmi_percentage'].std(),
                    'mean_reverse_kl': direction_data['reverse_kl_percentage'].mean(),
                    'std_reverse_kl': direction_data['reverse_kl_percentage'].std(),
                    'best_placement': direction_data.loc[direction_data['accuracy'].idxmax(), 'placement'],
                    'best_accuracy': direction_data['accuracy'].max()
                }
            
            # Save summary
            import json
            with open(os.path.join(comparison_dir, "bidirectional_summary.json"), 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            print(f"\\n📊 BIDIRECTIONAL COMPARISON SUMMARY:")
            for direction, stats in summary_stats.items():
                print(f"   {direction}:")
                print(f"     Mean Accuracy: {stats['mean_accuracy']:.3f} ± {stats['std_accuracy']:.3f}")
                print(f"     Mean F1-Score: {stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}")
                print(f"     Mean NMI: {stats['mean_nmi']:.3f}% ± {stats['std_nmi']:.3f}%")
                print(f"     Mean Reverse KL: {stats['mean_reverse_kl']:.3f}% ± {stats['std_reverse_kl']:.3f}%")
                print(f"     Best Placement: {stats['best_placement']} ({stats['best_accuracy']:.3f})")
            
            print(f"\\n💾 Bidirectional comparison saved: {plot_file}")
            
    except Exception as e:
        print(f"⚠️ Error creating bidirectional comparison: {e}")

if __name__ == "__main__":
    main()
