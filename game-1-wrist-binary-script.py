#!/usr/bin/env python3
"""
🎯 GAME-1 WRIST BINARY HAR ATTACK SCRIPT
=======================================

Subject-independent evaluation of binary features for wrist HAR attack:
- 21 wrist actions from UTD-MHAD
- Binary decision tree features only
- 6-2 and 4-4 subject splits (like successful thigh Game-1)
- Comprehensive metrics: Accuracy, F1, NMI, NRKL
- No data leakage - complete subject separation
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
import json
from datetime import datetime
from itertools import combinations
from scipy import stats
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial.distance import jensenshannon
from utd_vulnerability_utils import calculate_vulnerability, save_max_vulnerability_from_values
warnings.filterwarnings('ignore')

class Game1WristBinaryHAR:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # 21 wrist actions 
        self.wrist_actions = {
            1: "Right arm swipe to left",
            2: "Right arm swipe to right", 
            3: "Right hand wave",
            4: "Two hand front clap",
            5: "Right arm throw",
            6: "Cross arms in chest",
            7: "Basketball shoot",
            8: "Right hand draw x",
            9: "Right hand draw circle (clockwise)",
            10: "Right hand draw circle (counter clockwise)",
            11: "Draw triangle",
            12: "Bowling (right hand)",
            13: "Front boxing",
            14: "Baseball swing from right",
            15: "Tennis right hand forehand swing",
            16: "Arm curl (two arms)",
            17: "Tennis serve",
            18: "Two hand push",
            19: "Right hand knock on door",
            20: "Right hand catch an object",
            21: "Right hand pick up and throw"
        }
        
        # Map to 0-20 for 21-class classification
        self.action_mapping = {i: i-1 for i in range(1, 22)}
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        
        # Expected subjects
        self.subjects = list(range(1, 9))  # Subjects 1-8
        
        print(f"🎯 GAME-1 WRIST BINARY HAR ATTACK INITIALIZED")
        print(f"📊 21 wrist actions, 8 subjects, binary features only")
        
    def load_wrist_data_by_subject_action(self):
        """Load wrist data organized by subject and action"""
        print(f"\n📂 LOADING WRIST DATA BY SUBJECT-ACTION...")
        
        # Structure: data[subject][action] = [list of DataFrames]
        data = defaultdict(lambda: defaultdict(list))
        file_counts = defaultdict(lambda: defaultdict(int))
        
        # Get all CSV files from right-wrist directories
        csv_pattern = os.path.join(self.data_dir, "**", "right-wrist", "Action_*", "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        for file_path in csv_files:
            try:
                # Extract subject and action from path
                path_parts = file_path.split(os.sep)
                
                # Find subject directory (Subject1, Subject2, etc.)
                subject = None
                for part in path_parts:
                    if part.startswith('Subject'):
                        subject = int(part.replace('Subject', ''))
                        break
                
                # Find action directory (Action_1, Action_2, etc.)
                action = None
                for part in path_parts:
                    if part.startswith('Action_'):
                        action = int(part.split('_')[1])
                        break
                
                if subject and action and action in self.wrist_actions:
                    # Load the CSV file
                    df = pd.read_csv(file_path)
                    if 'dec_tree_out_1' in df.columns:
                        data[subject][action].append(df)
                        file_counts[subject][action] += 1
                        
            except Exception as e:
                continue
        
        print(f"📊 DATA SUMMARY BY SUBJECT:")
        total_files = 0
        for subject in sorted(data.keys()):
            subject_total = sum(file_counts[subject].values())
            total_files += subject_total
            print(f"   Subject {subject}: {subject_total} files across {len(file_counts[subject])} actions")
        
        print(f"📈 Total files loaded: {total_files}")
        return data
    
    def extract_comprehensive_binary_features(self, data):
        """Extract comprehensive binary features (from analysis results)"""
        features = {}
        
        if 'dec_tree_out_1' not in data.columns:
            return {}
            
        binary_data = data['dec_tree_out_1'].values
        n = len(binary_data)
        
        if n == 0:
            return {}
        
        # 1. Basic Statistics (top performing from analysis)
        features['binary_ratio'] = np.mean(binary_data)
        features['binary_variance'] = np.var(binary_data.astype(float))
        features['binary_std'] = np.std(binary_data.astype(float))
        features['sequence_length'] = n
        
        # 2. Temporal Segmentation (20 segments - top performer)
        num_segments = 20
        segment_size = max(1, n // num_segments)
        temporal_ratios = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, n)
            if start_idx < n:
                segment = binary_data[start_idx:end_idx]
                ratio = np.mean(segment) if len(segment) > 0 else 0
                temporal_ratios.append(ratio)
                features[f'temporal_seg_{i:02d}'] = ratio
            else:
                temporal_ratios.append(0)
                features[f'temporal_seg_{i:02d}'] = 0
        
        # 3. Progression Analysis
        if len(temporal_ratios) > 2:
            # Linear trend
            x = np.arange(len(temporal_ratios))
            coeffs = np.polyfit(x, temporal_ratios, 1)
            features['progression_slope'] = coeffs[0]
            features['progression_intercept'] = coeffs[1]
            
            # Quadratic trend
            try:
                quad_coeffs = np.polyfit(x, temporal_ratios, 2)
                features['progression_curvature'] = quad_coeffs[0]
            except:
                features['progression_curvature'] = 0
        else:
            features['progression_slope'] = 0
            features['progression_intercept'] = features['binary_ratio']
            features['progression_curvature'] = 0
        
        features['progression_start'] = temporal_ratios[0] if temporal_ratios else 0
        features['progression_end'] = temporal_ratios[-1] if temporal_ratios else 0
        features['progression_peak'] = max(temporal_ratios) if temporal_ratios else 0
        features['progression_valley'] = min(temporal_ratios) if temporal_ratios else 0
        features['progression_range'] = features['progression_peak'] - features['progression_valley']
        
        # 4. Temporal Pattern Analysis
        if len(temporal_ratios) > 1:
            features['temporal_variance'] = np.var(temporal_ratios)
            features['temporal_std'] = np.std(temporal_ratios)
            features['temporal_skewness'] = skew(temporal_ratios)
            features['temporal_kurtosis'] = kurtosis(temporal_ratios)
        else:
            features['temporal_variance'] = 0
            features['temporal_std'] = 0
            features['temporal_skewness'] = 0
            features['temporal_kurtosis'] = 0
        
        # 5. Transition Analysis (critical for wrist actions)
        if n > 1:
            transitions = np.diff(binary_data.astype(int))
            features['total_transitions'] = np.sum(np.abs(transitions))
            features['transition_rate'] = features['total_transitions'] / n
            features['rise_transitions'] = np.sum(transitions == 1)
            features['fall_transitions'] = np.sum(transitions == -1)
            features['rise_rate'] = features['rise_transitions'] / n
            features['fall_rate'] = features['fall_transitions'] / n
            
            # Transition gaps (top performing feature)
            if features['total_transitions'] > 0:
                transition_indices = np.where(np.abs(transitions) == 1)[0]
                if len(transition_indices) > 1:
                    transition_gaps = np.diff(transition_indices)
                    features['avg_transition_gap'] = np.mean(transition_gaps)
                    features['std_transition_gap'] = np.std(transition_gaps)
                    features['max_transition_gap'] = np.max(transition_gaps)  # Top feature!
                    features['min_transition_gap'] = np.min(transition_gaps)
                else:
                    features['avg_transition_gap'] = n
                    features['std_transition_gap'] = 0
                    features['max_transition_gap'] = n
                    features['min_transition_gap'] = n
            else:
                features['avg_transition_gap'] = n
                features['std_transition_gap'] = 0
                features['max_transition_gap'] = n
                features['min_transition_gap'] = n
        else:
            for key in ['total_transitions', 'transition_rate', 'rise_transitions', 'fall_transitions',
                       'rise_rate', 'fall_rate', 'avg_transition_gap', 'std_transition_gap',
                       'max_transition_gap', 'min_transition_gap']:
                features[key] = 0
        
        # 6. Burst Analysis (top performing features)
        active_bursts, inactive_bursts = self._analyze_bursts(binary_data)
        
        if active_bursts:
            features['num_active_bursts'] = len(active_bursts)
            features['avg_active_burst'] = np.mean(active_bursts)
            features['std_active_burst'] = np.std(active_bursts)  # Top feature!
            features['max_active_burst'] = np.max(active_bursts)  # Top feature!
            features['min_active_burst'] = np.min(active_bursts)
            features['total_active_samples'] = np.sum(active_bursts)  # #1 feature!
        else:
            for key in ['num_active_bursts', 'avg_active_burst', 'std_active_burst',
                       'max_active_burst', 'min_active_burst', 'total_active_samples']:
                features[key] = 0
        
        if inactive_bursts:
            features['num_inactive_bursts'] = len(inactive_bursts)
            features['avg_inactive_burst'] = np.mean(inactive_bursts)
            features['std_inactive_burst'] = np.std(inactive_bursts)
            features['max_inactive_burst'] = np.max(inactive_bursts)
            features['min_inactive_burst'] = np.min(inactive_bursts)
            features['total_inactive_samples'] = np.sum(inactive_bursts)
        else:
            for key in ['num_inactive_bursts', 'avg_inactive_burst', 'std_inactive_burst',
                       'max_inactive_burst', 'min_inactive_burst', 'total_inactive_samples']:
                features[key] = 0
        
        # 7. Positional Analysis
        thirds = [n//3, 2*n//3]
        if n > 3:
            features['start_third_ratio'] = np.mean(binary_data[:thirds[0]])
            features['middle_third_ratio'] = np.mean(binary_data[thirds[0]:thirds[1]])  # Top feature!
            features['end_third_ratio'] = np.mean(binary_data[thirds[1]:])
            features['start_end_diff'] = features['end_third_ratio'] - features['start_third_ratio']
        else:
            features['start_third_ratio'] = features['binary_ratio']
            features['middle_third_ratio'] = features['binary_ratio']
            features['end_third_ratio'] = features['binary_ratio']
            features['start_end_diff'] = 0
        
        # 8. Information Theory Features
        # Binary entropy
        p1 = features['binary_ratio']
        p0 = 1 - p1
        if p0 > 0 and p1 > 0:
            features['binary_entropy'] = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        else:
            features['binary_entropy'] = 0
        
        # Complexity measures
        features['binary_complexity'] = features['total_transitions'] / max(1, np.sum(binary_data))
        
        return features
    
    def _analyze_bursts(self, binary_data):
        """Analyze consecutive active/inactive periods"""
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
    
    def create_subject_action_samples(self, data):
        """Create one sample per subject per action by aggregating sequences"""
        print(f"\n🔗 CREATING SUBJECT-ACTION SAMPLES...")
        
        samples = []
        labels = []
        sample_info = []
        
        for subject in sorted(data.keys()):
            for action in sorted(data[subject].keys()):
                sequences = data[subject][action]
                if sequences:
                    # Aggregate all sequences for this subject-action pair
                    aggregated_data = pd.concat(sequences, ignore_index=True)
                    
                    # Extract features
                    features = self.extract_comprehensive_binary_features(aggregated_data)
                    
                    if features:
                        samples.append(features)
                        labels.append(self.action_mapping[action])
                        sample_info.append({
                            'subject': subject,
                            'action': action,
                            'action_name': self.wrist_actions[action],
                            'num_sequences': len(sequences),
                            'total_samples': len(aggregated_data)
                        })
        
        print(f"✅ Created {len(samples)} subject-action samples")
        print(f"📊 Subjects: {len(set(info['subject'] for info in sample_info))}")
        print(f"📊 Actions: {len(set(info['action'] for info in sample_info))}")
        
        return samples, labels, sample_info
    
    def calculate_nmi(self, y_true, y_pred):
        """Calculate Normalized Mutual Information"""
        return normalized_mutual_info_score(y_true, y_pred)
    
    def calculate_nrkl(self, y_true, y_pred_proba, num_classes=21):
        """Calculate Normalized Reverse KL Divergence (NRKL)"""
        # Ensure predictions are properly normalized
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1.0)
        
        # Calculate negative log-likelihood (cross-entropy) for each sample
        # This is equivalent to KL(true_onehot || predicted) since true is one-hot
        nll_values = []
        for i, true_label in enumerate(y_true):
            # Negative log-likelihood of the true class
            nll = -np.log(y_pred_proba[i, true_label])
            nll_values.append(nll)
        
        # Average negative log-likelihood
        avg_nll = np.mean(nll_values)
        
        # Theoretical maximum NLL (uniform random prediction)
        # When predicting uniformly: p = 1/num_classes, so max_nll = -log(1/num_classes) = log(num_classes)
        max_nll = np.log(num_classes)
        
        # Theoretical minimum NLL (perfect prediction)
        # When predicting correctly with probability 1: min_nll = -log(1) = 0
        min_nll = 0.0
        
        # Calculate normalized NLL following the specification:
        # KL_normalized = min(1.0, KL_avg / KL_max)
        # This ensures NRKL stays in [0%, 100%] range
        if max_nll > min_nll:
            normalized_nll = min(1.0, (avg_nll - min_nll) / (max_nll - min_nll))
        else:
            normalized_nll = 0.0
        
        # Convert to percentage where higher is better (reverse the scale)
        # 100% = perfect prediction, 0% = random prediction, never goes below 0%
        nrkl_percentage = (1.0 - normalized_nll) * 100.0
        
        return nrkl_percentage
    
    def evaluate_train_test_split(self, samples, labels, sample_info, train_subjects, test_subjects):
        """Evaluate one train/test split"""
        # Separate training and test data based on subjects
        train_samples, train_labels, train_info = [], [], []
        test_samples, test_labels, test_info = [], [], []
        
        for i, info in enumerate(sample_info):
            if info['subject'] in train_subjects:
                train_samples.append(samples[i])
                train_labels.append(labels[i])
                train_info.append(info)
            elif info['subject'] in test_subjects:
                test_samples.append(samples[i])
                test_labels.append(labels[i])
                test_info.append(info)
        
        if not train_samples or not test_samples:
            return None
        
        # Convert to DataFrames
        train_df = pd.DataFrame(train_samples).fillna(0)
        test_df = pd.DataFrame(test_samples).fillna(0)
        
        # Ensure same columns
        all_columns = set(train_df.columns) | set(test_df.columns)
        for col in all_columns:
            if col not in train_df.columns:
                train_df[col] = 0
            if col not in test_df.columns:
                test_df[col] = 0
        
        train_df = train_df[sorted(all_columns)]
        test_df = test_df[sorted(all_columns)]
        
        # Use top features from comprehensive analysis
        top_features = [
            'total_active_samples', 'max_active_burst', 'sequence_length', 'max_transition_gap',
            'std_active_burst', 'middle_third_ratio', 'temporal_seg_09', 'std_transition_gap',
            'temporal_seg_08', 'progression_intercept', 'binary_complexity', 'binary_ratio',
            'binary_std', 'temporal_seg_07', 'temporal_skewness', 'total_inactive_samples',
            'binary_entropy', 'avg_active_burst', 'temporal_seg_10', 'binary_variance',
            'temporal_variance', 'rise_rate', 'fall_rate', 'transition_rate', 'avg_transition_gap'
        ]
        
        # Select available top features
        available_features = [f for f in top_features if f in train_df.columns]
        if len(available_features) < 10:
            # Fallback to all available features
            available_features = list(train_df.columns)
        
        X_train = train_df[available_features]
        X_test = test_df[available_features]
        
        # Train classifier (ExtraTreesClassifier was best from analysis)
        clf = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        )
        
        clf.fit(X_train, train_labels)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average='weighted')
        nmi = self.calculate_nmi(test_labels, y_pred)
        nmi_percentage = nmi * 100.0
        nrkl_percentage = self.calculate_nrkl(test_labels, y_pred_proba)
        vulnerability = calculate_vulnerability(y_pred_proba)
        
        return {
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_actions': len(set(info['action'] for info in train_info)),
            'test_actions': len(set(info['action'] for info in test_info)),
            'accuracy': accuracy,
            'f1_score': f1,
            'nmi_percentage': nmi_percentage,
            'nrkl_percentage': nrkl_percentage,
            'vulnerability': vulnerability,
            'features_used': len(available_features)
        }
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with all train/test splits"""
        print(f"🚀 STARTING GAME-1 WRIST BINARY HAR EVALUATION")
        print(f"="*60)
        
        # Load data
        data = self.load_wrist_data_by_subject_action()
        
        # Create subject-action samples
        samples, labels, sample_info = self.create_subject_action_samples(data)
        
        if not samples:
            print("❌ No valid samples created")
            return
        
        print(f"\n🎯 EVALUATING ALL TRAIN/TEST SPLITS...")
        
        results = []
        
        # Generate all 4-4 splits (4 train, 4 test)
        print(f"\n📊 EVALUATING 4-4 SPLITS...")
        for train_subjects in combinations(self.subjects, 4):
            test_subjects = [s for s in self.subjects if s not in train_subjects]
            result = self.evaluate_train_test_split(samples, labels, sample_info, 
                                                  list(train_subjects), test_subjects)
            if result:
                result['split_type'] = '4-4'
                results.append(result)
        
        # Generate all 6-2 splits (6 train, 2 test)
        print(f"\n📊 EVALUATING 6-2 SPLITS...")
        for test_subjects in combinations(self.subjects, 2):
            train_subjects = [s for s in self.subjects if s not in test_subjects]
            result = self.evaluate_train_test_split(samples, labels, sample_info,
                                                  train_subjects, list(test_subjects))
            if result:
                result['split_type'] = '6-2'
                results.append(result)
        
        # Analyze results
        self.analyze_results(results)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results"""
        if not results:
            print("❌ No results to analyze")
            return
        
        # Separate by split type
        results_4_4 = [r for r in results if r['split_type'] == '4-4']
        results_6_2 = [r for r in results if r['split_type'] == '6-2']
        
        print(f"\n📊 GAME-1 WRIST BINARY HAR COMPREHENSIVE RESULTS")
        print(f"=" * 70)
        
        for split_type, split_results in [('4-4', results_4_4), ('6-2', results_6_2)]:
            if not split_results:
                continue
                
            print(f"\n🎯 {split_type} TRAIN-TEST SPLITS ({len(split_results)} combinations)")
            print(f"-" * 50)
            
            # Extract metrics
            accuracies = [r['accuracy'] for r in split_results]
            f1_scores = [r['f1_score'] for r in split_results]
            nmi_percentages = [r['nmi_percentage'] for r in split_results]
            nrkl_percentages = [r['nrkl_percentage'] for r in split_results]
            
            # Calculate statistics
            metrics = {
                'Accuracy': accuracies,
                'F1-Score': f1_scores,
                'NMI (%)': nmi_percentages,
                'NRKL (%)': nrkl_percentages
            }
            
            for metric_name, values in metrics.items():
                mean_val = np.mean(values)
                median_val = np.median(values)
                min_val = np.min(values)
                max_val = np.max(values)
                std_val = np.std(values)
                
                print(f"{metric_name:10s}: Mean={mean_val:.4f}, Median={median_val:.4f}, "
                      f"Min={min_val:.4f}, Max={max_val:.4f}, Std={std_val:.4f}")
            
            # Show best and worst cases
            best_accuracy_idx = np.argmax(accuracies)
            worst_accuracy_idx = np.argmin(accuracies)
            
            print(f"\n🏆 BEST ACCURACY SPLIT:")
            best = split_results[best_accuracy_idx]
            print(f"   Train subjects: {best['train_subjects']}")
            print(f"   Test subjects: {best['test_subjects']}")
            print(f"   Accuracy: {best['accuracy']:.4f}, F1: {best['f1_score']:.4f}")
            print(f"   NMI: {best['nmi_percentage']:.2f}%, NRKL: {best['nrkl_percentage']:.2f}%")
            
            print(f"\n💥 WORST ACCURACY SPLIT:")
            worst = split_results[worst_accuracy_idx]
            print(f"   Train subjects: {worst['train_subjects']}")
            print(f"   Test subjects: {worst['test_subjects']}")
            print(f"   Accuracy: {worst['accuracy']:.4f}, F1: {worst['f1_score']:.4f}")
            print(f"   NMI: {worst['nmi_percentage']:.2f}%, NRKL: {worst['nrkl_percentage']:.2f}%")
        
        # Overall summary
        print(f"\n🎯 GAME-1 WRIST BINARY HAR ATTACK FINAL SUMMARY:")
        print(f"="*60)
        
        # Calculate overall metrics across all splits
        all_accuracies = []
        all_f1s = []
        all_nmis = []
        all_nrkls = []
        
        for split_name, split_results in [('4-4', results_4_4), ('6-2', results_6_2)]:
            if split_results:
                print(f"\n📊 {split_name.upper()} SPLIT COMPREHENSIVE STATISTICS:")
                print(f"-" * 70)
                accuracies = [r['accuracy'] for r in split_results]
                f1_scores = [r['f1_score'] for r in split_results]
                nmi_percentages = [r['nmi_percentage'] for r in split_results]
                nrkl_percentages = [r['nrkl_percentage'] for r in split_results]
                
                # Display comprehensive statistics table
                print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
                print(f"{'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
                
                metrics_data = {
                    'Accuracy': accuracies,
                    'F1-Score': f1_scores,
                    'NMI (%)': nmi_percentages,
                    'NRKL (%)': nrkl_percentages
                }
                
                for metric_name, values in metrics_data.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    median_val = np.median(values)
                    
                    if metric_name in ['NMI (%)', 'NRKL (%)']:
                        print(f"{metric_name:<12} {mean_val:<7.1f} {std_val:<7.1f} {min_val:<7.1f} {max_val:<7.1f} {median_val:<7.1f}")
                    else:
                        print(f"{metric_name:<12} {mean_val:<7.3f} {std_val:<7.3f} {min_val:<7.3f} {max_val:<7.3f} {median_val:<7.3f}")
                
                all_accuracies.extend(accuracies)
                all_f1s.extend(f1_scores)
                all_nmis.extend(nmi_percentages)
                all_nrkls.extend(nrkl_percentages)
        
        if all_accuracies:
            overall_accuracy = np.mean(all_accuracies)
            overall_f1 = np.mean(all_f1s)
            overall_nmi = np.mean(all_nmis)
            overall_nrkl = np.mean(all_nrkls)
            
            # Display overall summary table
            print(f"\n🏆 OVERALL COMPREHENSIVE PERFORMANCE SUMMARY:")
            print(f"="*70)
            print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
            print(f"{'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
            
            overall_metrics = {
                'Accuracy': all_accuracies,
                'F1-Score': all_f1s,
                'NMI (%)': all_nmis,
                'NRKL (%)': all_nrkls
            }
            
            for metric_name, values in overall_metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                median_val = np.median(values)
                
                if metric_name in ['NMI (%)', 'NRKL (%)']:
                    print(f"{metric_name:<12} {mean_val:<7.1f} {std_val:<7.1f} {min_val:<7.1f} {max_val:<7.1f} {median_val:<7.1f}")
                else:
                    print(f"{metric_name:<12} {mean_val:<7.3f} {std_val:<7.3f} {min_val:<7.3f} {max_val:<7.3f} {median_val:<7.3f}")
            
            print(f"\n📊 Performance vs Baseline:")
            print(f"   Overall Mean Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
            print(f"   🎲 Random baseline:    {1/21:.3f} ({100/21:.1f}%)")
            print(f"   🚀 Improvement:        {overall_accuracy/(1/21):.1f}x vs random")
            
            print(f"\n📈 COMPARISON WITH PREVIOUS RESULTS:")
            print(f"   Comprehensive Analysis CV: 28.9% accuracy (6.1x random)")
            print(f"   Subject-Independent:        {overall_accuracy*100:.1f}% accuracy ({overall_accuracy/(1/21):.1f}x random)")
            
            if overall_accuracy > 0.289:
                print(f"   ✅ Subject-independent performance maintained!")
            elif overall_accuracy > 0.2:
                print(f"   📈 Good subject-independent performance")
            else:
                print(f"   ⚠️ Performance degraded in subject-independent setting")
        
        # Save comprehensive results
        self.save_comprehensive_results(results)

    def save_comprehensive_results(self, results):
        """Save only the maximum vulnerability summary."""
        output_file = save_max_vulnerability_from_values(
            __file__,
            [result.get('vulnerability') for result in results],
        )
        print(f"\n💾 Game-1 wrist binary max vulnerability saved to {output_file}")

def main():
    print("🎯 GAME-1 WRIST BINARY HAR ATTACK")
    print("="*50)
    
    # Set data directory
    data_dir = "UTD-MHAD-Reorganized"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        return
    
    # Initialize Game-1 wrist binary attack system
    game1_wrist_system = Game1WristBinaryHAR(data_dir)
    
    # Run comprehensive evaluation
    results = game1_wrist_system.run_comprehensive_evaluation()
    
    print(f"\n🎊 GAME-1 WRIST BINARY HAR ATTACK COMPLETED!")
    print(f"💡 Key findings:")
    print(f"   • Binary features only for 21 wrist actions")
    print(f"   • Subject-independent evaluation (no data leakage)")
    print(f"   • Comprehensive metrics: Accuracy, F1, NMI, NRKL")
    print(f"   • Ready for comparison with other Game-1 attacks")

if __name__ == "__main__":
    main()
