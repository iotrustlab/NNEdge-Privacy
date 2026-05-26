#!/usr/bin/env python3
"""
🎯 ROBUST THIGH-TARGETED ADVERSARIAL ATTACK EVALUATION
=====================================================

Comprehensive evaluation with multiple random splits and information-theoretic metrics:
- 4/5 different train/test combinations 
- Mean, median, min, max statistics
- Information-theoretic metrics: NMI, Normalized KL divergence
- No data leakage verification
"""

import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools
import warnings
from scipy.stats import entropy
from sklearn.preprocessing import LabelBinarizer
from utd_vulnerability_utils import calculate_vulnerability, save_max_vulnerability_from_values
warnings.filterwarnings('ignore')

class RobustThighAttackEvaluation:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.thigh_actions = {
            22: "Jog in place",
            23: "Walking in place", 
            24: "Sit to stand",
            25: "Stand to sit",
            26: "Forward lunge",
            27: "Squat"
        }
        
        # Map to 0-5 for 6-class classification
        self.action_mapping = {22: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5}
        self.reverse_mapping = {v: k for k, v in self.action_mapping.items()}
        
        print(f"🎯 ROBUST THIGH ATTACK EVALUATION INITIALIZED")
        print(f"📊 6 thigh actions with comprehensive metrics")
        
    def load_thigh_data(self):
        """Load only thigh placement data for target actions"""
        print(f"\n📂 LOADING THIGH DATA...")
        
        all_files = []
        action_counts = defaultdict(int)
        
        # Get all CSV files from right-thigh directories
        csv_pattern = os.path.join(self.data_dir, "**", "right-thigh", "Action_*", "*.csv")
        csv_files = glob.glob(csv_pattern, recursive=True)
        
        for file_path in csv_files:
            try:
                # Extract action from path: .../right-thigh/Action_22/sequence_1.csv
                path_parts = file_path.split(os.sep)
                action_dir = None
                for part in path_parts:
                    if part.startswith('Action_'):
                        action_dir = part
                        break
                
                if action_dir:
                    action_num = int(action_dir.split('_')[1])  # Extract number from Action_22
                    
                    # Check if this is a target thigh action (22-27)
                    if action_num in self.thigh_actions:
                        all_files.append(file_path)
                        action_counts[action_num] += 1
                            
            except (ValueError, IndexError) as e:
                continue
        
        print(f"📊 THIGH DATA SUMMARY:")
        total_files = 0
        for action_num in sorted(self.thigh_actions.keys()):
            count = action_counts[action_num]
            total_files += count
            print(f"   Action {action_num} ({self.thigh_actions[action_num]}): {count} files")
        
        print(f"📈 Total thigh files: {total_files}")
        return all_files
    
    def extract_temporal_signature_features(self, data, num_segments=10):
        """Extract temporal signature features (same as original)"""
        features = {}
        
        if 'dec_tree_out_1' not in data.columns:
            return {}
            
        binary_data = data['dec_tree_out_1'].values
        n = len(binary_data)
        
        if n == 0:
            return {}
        
        # 1. TEMPORAL SIGNATURE (10-segment progression)
        segment_size = max(1, n // num_segments)
        temporal_signature = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, n)
            if start_idx < n:
                segment = binary_data[start_idx:end_idx]
                activity_ratio = np.mean(segment) if len(segment) > 0 else 0
                temporal_signature.append(activity_ratio)
            else:
                temporal_signature.append(0)
        
        # Add temporal signature features
        for i, ratio in enumerate(temporal_signature):
            features[f'temporal_seg_{i:02d}'] = ratio
            
        # 2. PROGRESSION ANALYSIS
        features['progression_trend'] = np.polyfit(range(len(temporal_signature)), temporal_signature, 1)[0]
        features['progression_start'] = temporal_signature[0] if temporal_signature else 0
        features['progression_end'] = temporal_signature[-1] if temporal_signature else 0
        features['progression_peak'] = max(temporal_signature) if temporal_signature else 0
        features['progression_valley'] = min(temporal_signature) if temporal_signature else 0
        
        # 3. TRANSITION ANALYSIS
        transitions = np.diff(binary_data.astype(int))
        features['total_transitions'] = np.sum(np.abs(transitions))
        features['transition_rate'] = features['total_transitions'] / n if n > 1 else 0
        
        # Rise and fall transitions
        features['rise_transitions'] = np.sum(transitions == 1)
        features['fall_transitions'] = np.sum(transitions == -1)
        features['rise_rate'] = features['rise_transitions'] / n if n > 1 else 0
        features['fall_rate'] = features['fall_transitions'] / n if n > 1 else 0
        
        # 4. PATTERN CONSISTENCY 
        features['pattern_variance'] = np.var(temporal_signature)
        features['pattern_std'] = np.std(temporal_signature)
        features['pattern_range'] = features['progression_peak'] - features['progression_valley']
        
        # 5. BASIC STATISTICS (enhanced)
        features['overall_activity_ratio'] = np.mean(binary_data)
        features['activity_variance'] = np.var(binary_data.astype(float))
        features['active_sequences'] = self._count_active_sequences(binary_data)
        features['inactive_sequences'] = self._count_inactive_sequences(binary_data)
        
        # 6. START/END STATE PATTERNS
        start_window = min(n//10, 50)  # First 10% or 50 frames
        end_window = min(n//10, 50)    # Last 10% or 50 frames
        
        if n >= start_window:
            features['start_activity'] = np.mean(binary_data[:start_window])
            features['end_activity'] = np.mean(binary_data[-end_window:])
            features['start_end_diff'] = features['end_activity'] - features['start_activity']
        
        return features
    
    def _count_active_sequences(self, binary_data):
        """Count continuous sequences of active (1) states"""
        sequences = 0
        in_sequence = False
        for val in binary_data:
            if val == 1 and not in_sequence:
                sequences += 1
                in_sequence = True
            elif val == 0:
                in_sequence = False
        return sequences
    
    def _count_inactive_sequences(self, binary_data):
        """Count continuous sequences of inactive (0) states"""
        sequences = 0
        in_sequence = False
        for val in binary_data:
            if val == 0 and not in_sequence:
                sequences += 1
                in_sequence = True
            elif val == 1:
                in_sequence = False
        return sequences
    
    def process_files_for_features(self, file_list):
        """Process all files and extract features"""
        print(f"\n🔄 EXTRACTING FEATURES FROM {len(file_list)} FILES...")
        
        features_list = []
        labels = []
        file_info = []
        processed = 0
        
        for file_path in file_list:
            try:
                # Load data
                data = pd.read_csv(file_path)
                
                # Extract action number from path: .../right-thigh/Action_22/sequence_1.csv
                path_parts = file_path.split(os.sep)
                action_dir = None
                for part in path_parts:
                    if part.startswith('Action_'):
                        action_dir = part
                        break
                
                if not action_dir:
                    continue
                    
                action_num = int(action_dir.split('_')[1])  # Extract from Action_22
                
                # Skip if not target action
                if action_num not in self.thigh_actions:
                    continue
                
                # Extract features
                features = self.extract_temporal_signature_features(data)
                
                if features:  # Only add if features were extracted
                    features_list.append(features)
                    labels.append(self.action_mapping[action_num])  # Map to 0-5
                    
                    # Get filename for file info
                    filename = os.path.basename(file_path)
                    
                    file_info.append({
                        'file': filename,
                        'action': action_num,
                        'action_name': self.thigh_actions[action_num],
                        'path': file_path
                    })
                    processed += 1
                    
                    if processed % 20 == 0:
                        print(f"   Processed {processed} files...")
                        
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                continue
        
        print(f"✅ Successfully processed {processed} files")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        labels = np.array(labels)
        
        print(f"📊 Feature matrix shape: {features_df.shape}")
        print(f"🏷️ Labels shape: {labels.shape}")
        
        return features_df, labels, file_info
    
    def calculate_information_theoretic_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate NMI and Normalized KL divergence"""
        
        # 1. NORMALIZED MUTUAL INFORMATION (NMI)
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(y_true, y_pred)
        nmi_percentage = nmi * 100
        
        # 2. NORMALIZED KL DIVERGENCE
        # Convert true labels to one-hot
        lb = LabelBinarizer()
        y_true_onehot = lb.fit_transform(y_true)
        if y_true_onehot.shape[1] == 1:  # Binary case
            y_true_onehot = np.hstack([1-y_true_onehot, y_true_onehot])
        
        # Calculate KL divergence for each sample
        kl_divergences = []
        for i in range(len(y_true)):
            true_dist = y_true_onehot[i]
            pred_dist = y_pred_proba[i]
            
            # Add small epsilon to avoid log(0)
            pred_dist = np.clip(pred_dist, 1e-15, 1.0)
            
            # Calculate KL divergence: KL(true||pred)
            kl = np.sum(true_dist * np.log(true_dist / pred_dist + 1e-15))
            kl_divergences.append(kl)
        
        avg_kl = np.mean(kl_divergences)
        
        # Normalize KL by theoretical maximum
        num_classes = len(np.unique(y_true))
        kl_max = np.log(num_classes)  # Maximum KL for uniform prediction
        normalized_kl = min(1.0, avg_kl / kl_max)
        
        # Reverse KL percentage (higher = better)
        reverse_kl_percentage = (1 - normalized_kl) * 100
        
        return {
            'nmi': nmi,
            'nmi_percentage': nmi_percentage,
            'avg_kl': avg_kl,
            'normalized_kl': normalized_kl,
            'reverse_kl_percentage': reverse_kl_percentage
        }
    
    def generate_subject_splits(self, num_subjects=8):
        """Generate different train/test subject combinations for both 6:2 and 4:4 splits"""
        all_subjects = list(range(1, num_subjects + 1))
        
        # 6:2 splits (6 train, 2 test) - 5 combinations
        splits_6_2 = [
            ([1, 2, 3, 4, 5, 6], [7, 8]),      # Original
            ([1, 2, 3, 4, 7, 8], [5, 6]),      # Test: 5,6
            ([1, 2, 5, 6, 7, 8], [3, 4]),      # Test: 3,4
            ([3, 4, 5, 6, 7, 8], [1, 2]),      # Test: 1,2
            ([2, 3, 4, 5, 6, 7], [1, 8]),      # Test: 1,8
        ]
        
        # 4:4 splits (4 train, 4 test) - 5 combinations
        splits_4_4 = [
            ([1, 2, 3, 4], [5, 6, 7, 8]),      # First half vs second half
            ([1, 2, 5, 6], [3, 4, 7, 8]),      # Mixed combination 1
            ([1, 3, 5, 7], [2, 4, 6, 8]),      # Odd vs even
            ([2, 3, 6, 7], [1, 4, 5, 8]),      # Mixed combination 2
            ([1, 4, 6, 7], [2, 3, 5, 8]),      # Mixed combination 3
        ]
        
        all_splits = {
            '6:2': splits_6_2,
            '4:4': splits_4_4
        }
        
        print(f"📊 Generated splits:")
        for split_type, splits in all_splits.items():
            print(f"   {split_type} splits: {len(splits)} combinations")
            for i, (train, test) in enumerate(splits):
                print(f"      {split_type}-{i+1}: Train={train}, Test={test}")
        
        return all_splits
    
    def train_and_evaluate_single_split(self, features_df, labels, file_info, train_subjects, test_subjects, split_id):
        """Train and evaluate on a single train/test split"""
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Split data by subjects to avoid data leakage
        subject_splits = defaultdict(list)
        for i, info in enumerate(file_info):
            file_path = info['path']
            # Extract subject from path: .../Subject1/right-thigh/Action_22/...
            path_parts = file_path.split(os.sep)
            subject_dir = None
            for part in path_parts:
                if part.startswith('Subject'):
                    subject_dir = part
                    break
            
            if subject_dir:
                subject = int(subject_dir.replace('Subject', ''))  # Extract from Subject1
                subject_splits[subject].append(i)
        
        # Create train/test indices
        train_indices = []
        test_indices = []
        
        for subject in train_subjects:
            if subject in subject_splits:
                train_indices.extend(subject_splits[subject])
        
        for subject in test_subjects:
            if subject in subject_splits:
                test_indices.extend(subject_splits[subject])
        
        if len(train_indices) == 0 or len(test_indices) == 0:
            print(f"❌ Split {split_id}: Insufficient data")
            return None
        
        # Split features and labels
        X_train = features_df.iloc[train_indices]
        y_train = labels[train_indices]
        X_test = features_df.iloc[test_indices]
        y_test = labels[test_indices]
        
        # Train Random Forest classifier
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        
        # Calculate standard metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Calculate information-theoretic metrics
        info_metrics = self.calculate_information_theoretic_metrics(y_test, y_pred, y_pred_proba)
        vulnerability = calculate_vulnerability(y_pred_proba)
        
        print(f"📊 Split {split_id}: Acc={accuracy:.3f}, F1={f1:.3f}, NMI={info_metrics['nmi_percentage']:.1f}%, RKL={info_metrics['reverse_kl_percentage']:.1f}%")
        
        return {
            'split_id': split_id,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'n_train': len(train_indices),
            'n_test': len(test_indices),
            'accuracy': accuracy,
            'f1_score': f1,
            'nmi': info_metrics['nmi'],
            'nmi_percentage': info_metrics['nmi_percentage'],
            'avg_kl': info_metrics['avg_kl'],
            'normalized_kl': info_metrics['normalized_kl'],
            'reverse_kl_percentage': info_metrics['reverse_kl_percentage'],
            'vulnerability': vulnerability,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def robust_evaluation(self, features_df, labels, file_info):
        """Perform robust evaluation across multiple splits"""
        print(f"\n🎯 ROBUST EVALUATION ACROSS MULTIPLE SPLITS")
        print(f"=" * 60)
        
        # Generate different splits (both 6:2 and 4:4)
        all_splits = self.generate_subject_splits(num_subjects=8)
        
        # Store results by split type
        all_results = {}
        
        for split_type, splits in all_splits.items():
            print(f"\n🔄 Evaluating {split_type} splits...")
            results = []
            
            for i, (train_subjects, test_subjects) in enumerate(splits, 1):
                split_id = f"{split_type}-{i}"
                result = self.train_and_evaluate_single_split(
                    features_df, labels, file_info, train_subjects, test_subjects, split_id
                )
                if result is not None:
                    result['split_type'] = split_type
                    results.append(result)
            
            all_results[split_type] = results
        
        # Calculate statistics for each split type
        combined_stats = {}
        
        for split_type, results in all_results.items():
            if not results:
                continue
                
            print(f"\n📊 {split_type.upper()} SPLIT STATISTICS:")
            print(f"=" * 40)
            
            metrics = ['accuracy', 'f1_score', 'nmi_percentage', 'reverse_kl_percentage']
            stats = {}
            
            for metric in metrics:
                values = [r[metric] for r in results]
                stats[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
                
                print(f"\n🎯 {metric.upper().replace('_', ' ')}:")
                print(f"   Mean:   {stats[metric]['mean']:.3f}")
                print(f"   Median: {stats[metric]['median']:.3f}")
                print(f"   Std:    {stats[metric]['std']:.3f}")
                print(f"   Min:    {stats[metric]['min']:.3f}")
                print(f"   Max:    {stats[metric]['max']:.3f}")
            
            combined_stats[split_type] = stats
            
            # Save detailed results for this split type
            self.save_detailed_results(results, stats, split_type)
        
        # Overall comparison
        print(f"\n📋 COMPARISON ACROSS SPLIT TYPES:")
        print(f"=" * 50)
        print(f"{'Split Type':<10} {'Accuracy':<12} {'F1-Score':<12} {'NMI %':<10} {'RKL %':<10}")
        print(f"=" * 50)
        
        for split_type in ['6:2', '4:4']:
            if split_type in combined_stats:
                stats = combined_stats[split_type]
                acc_mean = stats['accuracy']['mean']
                f1_mean = stats['f1_score']['mean']
                nmi_mean = stats['nmi_percentage']['mean']
                rkl_mean = stats['reverse_kl_percentage']['mean']
                print(f"{split_type:<10} {acc_mean:<12.3f} {f1_mean:<12.3f} {nmi_mean:<10.1f} {rkl_mean:<10.1f}")
        
        # Save combined summary
        self.save_combined_summary(combined_stats, all_results)
        
        return {
            'all_results': all_results,
            'statistics': combined_stats
        }
    
    def save_detailed_results(self, results, stats, split_type):
        """Skip detailed artifact generation for vulnerability-only runs."""
        return None
    
    def save_combined_summary(self, combined_stats, all_results):
        """Save only the maximum vulnerability summary."""
        vulnerabilities = [
            result.get('vulnerability')
            for results in all_results.values()
            for result in results
        ]
        output_file = save_max_vulnerability_from_values(__file__, vulnerabilities)
        print(f"💾 Saved max vulnerability summary: {output_file}")

def main():
    print("🎯 ROBUST THIGH-TARGETED ADVERSARIAL ATTACK EVALUATION")
    print("=" * 65)
    
    # Set data directory
    data_dir = "UTD-MHAD-Reorganized"
    
    # Initialize evaluation system
    evaluator = RobustThighAttackEvaluation(data_dir)
    
    # Load thigh data
    thigh_files = evaluator.load_thigh_data()
    
    if not thigh_files:
        print("❌ No thigh data files found!")
        return
    
    # Extract features
    features_df, labels, file_info = evaluator.process_files_for_features(thigh_files)
    
    if len(features_df) == 0:
        print("❌ No features extracted!")
        return
    
    # Perform robust evaluation
    evaluation_results = evaluator.robust_evaluation(features_df, labels, file_info)
    
    if evaluation_results:
        print(f"\n🎊 ROBUST EVALUATION COMPLETED!")
        print("📁 Results saved to: Results/")
        print("💡 Key findings:")
        
        # Show summary for both split types
        for split_type in ['6:2', '4:4']:
            if split_type in evaluation_results['statistics']:
                stats = evaluation_results['statistics'][split_type]
                print(f"\n   📊 {split_type} SPLIT SUMMARY:")
                print(f"      • Mean Accuracy: {stats['accuracy']['mean']:.3f} ± {stats['accuracy']['std']:.3f}")
                print(f"      • Mean F1-Score: {stats['f1_score']['mean']:.3f} ± {stats['f1_score']['std']:.3f}")
                print(f"      • Mean NMI: {stats['nmi_percentage']['mean']:.1f}% ± {stats['nmi_percentage']['std']:.1f}%")
                print(f"      • Mean Reverse KL: {stats['reverse_kl_percentage']['mean']:.1f}% ± {stats['reverse_kl_percentage']['std']:.1f}%")
        
        print(f"\n   📂 File created:")
        print(f"      • {Path(__file__).stem}_max_vulnerability.json")

if __name__ == "__main__":
    main()
