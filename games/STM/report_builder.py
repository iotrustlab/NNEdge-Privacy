#!/usr/bin/env python3
"""
Comprehensive report builder for STM experiment results.
Generates a markdown report with summaries, links to plots, and analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def load_experiment_results(results_csv_path: str) -> pd.DataFrame:
    """Load and preprocess experiment results."""
    df = pd.read_csv(results_csv_path)
    
    # Filter out failed experiments
    df = df[df['error'].isna()]
    
    # Convert numeric columns
    numeric_cols = ['accuracy', 'macro_f1', 'loss', 'train_time_s', 'eval_time_s']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def create_model_names(df: pd.DataFrame) -> pd.DataFrame:
    """Create readable model names."""
    df = df.copy()
    df['model_name'] = df.apply(lambda row: 
        f"{row['mode'].title()}-{row['fusion'] if row['fusion'] != 'NA' else row['placement']}", axis=1)
    return df


def generate_experiment_summary(df: pd.DataFrame) -> str:
    """Generate experiment summary section."""
    
    total_experiments = len(df)
    successful_experiments = len(df[df['error'].isna()])
    failed_experiments = total_experiments - successful_experiments
    
    # Get unique configurations
    unique_games = df['game'].unique()
    unique_modes = df['mode'].unique()
    unique_classes = sorted(df['num_classes'].unique())
    unique_seeds = sorted(df['seed'].unique())
    
    # Get date range
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
        start_date = timestamps.min().strftime('%Y-%m-%d %H:%M:%S')
        end_date = timestamps.max().strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_date = "Unknown"
        end_date = "Unknown"
    
    summary = f"""
## 📊 Experiment Summary

### 🎯 Overview
- **Total Experiments**: {total_experiments}
- **Successful**: {successful_experiments}
- **Failed**: {failed_experiments}
- **Success Rate**: {(successful_experiments/total_experiments*100):.1f}%

### ⚙️ Configuration
- **Games**: {', '.join(unique_games)}
- **Modes**: {', '.join(unique_modes)}
- **Class Counts**: {', '.join(map(str, unique_classes))}
- **Random Seeds**: {', '.join(map(str, unique_seeds))}

### 📅 Timeline
- **Start**: {start_date}
- **End**: {end_date}
"""
    
    return summary


def generate_performance_analysis(df: pd.DataFrame) -> str:
    """Generate performance analysis section."""
    
    df = create_model_names(df)
    
    # Calculate summary statistics
    summary_stats = df.groupby('model_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'macro_f1': ['mean', 'std'],
        'train_time_s': ['mean', 'std'],
        'num_classes': 'count'
    }).round(4)
    
    # Flatten column names
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    
    # Sort by mean accuracy
    summary_stats = summary_stats.sort_values('accuracy_mean', ascending=False)
    
    # Create performance table
    performance_table = "| Model | Mean Accuracy | Std Accuracy | Mean F1 | Mean Train Time (s) |\n"
    performance_table += "|-------|---------------|--------------|---------|-------------------|\n"
    
    for model_name, row in summary_stats.iterrows():
        performance_table += f"| {model_name} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} | {row['train_time_s_mean']:.1f} |\n"
    
    # Find best and worst models
    best_model = summary_stats.index[0]
    best_accuracy = summary_stats.loc[best_model, 'accuracy_mean']
    worst_model = summary_stats.index[-1]
    worst_accuracy = summary_stats.loc[worst_model, 'accuracy_mean']
    
    analysis = f"""
## 🏆 Performance Analysis

### 📈 Model Rankings
{performance_table}

### 🎯 Key Findings
- **Best Model**: {best_model} (Accuracy: {best_accuracy:.4f})
- **Worst Model**: {worst_model} (Accuracy: {worst_accuracy:.4f})
- **Performance Range**: {best_accuracy - worst_accuracy:.4f}

### 📊 Performance Distribution
"""
    
    # Add performance distribution analysis
    if len(df) > 1:
        accuracy_range = df['accuracy'].max() - df['accuracy'].min()
        accuracy_std = df['accuracy'].std()
        analysis += f"""
- **Accuracy Range**: {accuracy_range:.4f}
- **Standard Deviation**: {accuracy_std:.4f}
- **Coefficient of Variation**: {(accuracy_std/df['accuracy'].mean()*100):.1f}%
"""
    
    return analysis


def generate_class_analysis(df: pd.DataFrame) -> str:
    """Generate analysis by number of classes."""
    
    if 'num_classes' not in df.columns:
        return ""
    
    class_analysis = df.groupby('num_classes').agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std']
    }).round(4)
    
    class_analysis.columns = ['_'.join(col).strip() for col in class_analysis.columns]
    
    # Create class performance table
    class_table = "| Classes | Mean Accuracy | Std Accuracy | Mean F1 | Experiments |\n"
    class_table += "|---------|---------------|--------------|---------|-------------|\n"
    
    for num_classes, row in class_analysis.iterrows():
        class_table += f"| {num_classes} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} | {int(row['accuracy_count'])} |\n"
    
    analysis = f"""
## 🎯 Class Count Analysis

### 📊 Performance by Number of Classes
{class_table}

### 📈 Trends
"""
    
    # Add trend analysis
    if len(class_analysis) > 1:
        class_accuracies = class_analysis['accuracy_mean'].sort_index()
        if len(class_accuracies) > 1:
            # Check if accuracy decreases with more classes
            if class_accuracies.iloc[-1] < class_accuracies.iloc[0]:
                analysis += "- **Difficulty Trend**: Accuracy decreases with more classes (expected behavior)\n"
            else:
                analysis += "- **Difficulty Trend**: Accuracy increases with more classes (unexpected behavior)\n"
            
            # Calculate correlation
            correlation = np.corrcoef(class_accuracies.index, class_accuracies.values)[0, 1]
            analysis += f"- **Correlation**: {correlation:.3f} between class count and accuracy\n"
    
    return analysis


def generate_mode_comparison(df: pd.DataFrame) -> str:
    """Generate comparison between single and multi-sensor modes."""
    
    if 'mode' not in df.columns or len(df['mode'].unique()) < 2:
        return ""
    
    mode_analysis = df.groupby('mode').agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'train_time_s': ['mean', 'std']
    }).round(4)
    
    mode_analysis.columns = ['_'.join(col).strip() for col in mode_analysis.columns]
    
    # Create mode comparison table
    mode_table = "| Mode | Mean Accuracy | Std Accuracy | Mean F1 | Mean Train Time (s) | Experiments |\n"
    mode_table += "|------|---------------|--------------|---------|-------------------|-------------|\n"
    
    for mode, row in mode_analysis.iterrows():
        mode_table += f"| {mode.title()} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | {row['macro_f1_mean']:.4f} | {row['train_time_s_mean']:.1f} | {int(row['accuracy_count'])} |\n"
    
    # Compare modes
    modes = list(mode_analysis.index)
    if len(modes) >= 2:
        mode1, mode2 = modes[0], modes[1]
        acc_diff = mode_analysis.loc[mode1, 'accuracy_mean'] - mode_analysis.loc[mode2, 'accuracy_mean']
        time_diff = mode_analysis.loc[mode1, 'train_time_s_mean'] - mode_analysis.loc[mode2, 'train_time_s_mean']
        
        comparison = f"""
## 🔄 Mode Comparison

### 📊 Performance by Mode
{mode_table}

### ⚖️ Comparison Analysis
- **Accuracy Difference**: {mode1.title()} vs {mode2.title()}: {acc_diff:+.4f}
- **Training Time Difference**: {mode1.title()} vs {mode2.title()}: {time_diff:+.1f}s
"""
        
        if acc_diff > 0:
            comparison += f"- **Winner**: {mode1.title()} mode performs better\n"
        elif acc_diff < 0:
            comparison += f"- **Winner**: {mode2.title()} mode performs better\n"
        else:
            comparison += "- **Winner**: Both modes perform similarly\n"
        
        if time_diff > 0:
            comparison += f"- **Efficiency**: {mode2.title()} mode is faster\n"
        elif time_diff < 0:
            comparison += f"- **Efficiency**: {mode1.title()} mode is faster\n"
        else:
            comparison += "- **Efficiency**: Both modes have similar training times\n"
        
        return comparison
    
    return ""


def generate_plots_section(plots_dir: str) -> str:
    """Generate section with links to plots."""
    
    if not os.path.exists(plots_dir):
        return ""
    
    # Get list of plot files
    plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    
    if not plot_files:
        return ""
    
    plots_section = """
## 📊 Visualizations

The following plots provide detailed analysis of the experiment results:

"""
    
    # Group plots by type
    plot_groups = {
        'Performance': ['accuracy_vs_classes', 'f1_vs_classes', 'model_comparison'],
        'Efficiency': ['training_time_comparison'],
        'Analysis': ['accuracy_heatmap', 'seed_variability', 'seed_variability_summary']
    }
    
    for group_name, plot_types in plot_groups.items():
        plots_section += f"### {group_name} Plots\n\n"
        
        for plot_type in plot_types:
            matching_files = [f for f in plot_files if plot_type in f]
            for plot_file in matching_files:
                plot_name = plot_file.replace('.png', '').replace('_', ' ').title()
                plot_path = os.path.join(plots_dir, plot_file)
                plots_section += f"- **{plot_name}**: ![{plot_name}]({plot_path})\n\n"
    
    return plots_section


def generate_privacy_analysis(df: pd.DataFrame) -> str:
    """Generate privacy analysis section."""
    
    privacy_analysis = []
    
    # Overall privacy statistics
    accuracy_range = f"{df['accuracy'].min():.1%} - {df['accuracy'].max():.1%}"
    avg_accuracy = f"{df['accuracy'].mean():.1%}"
    accuracy_f1_gap = df['accuracy'].mean() - df['macro_f1'].mean()
    class_imbalance = 'HIGH' if accuracy_f1_gap > 0.1 else 'MODERATE'
    
    privacy_analysis.append(f"**Cross-user Accuracy Range**: {accuracy_range}")
    privacy_analysis.append(f"**Average Cross-user Accuracy**: {avg_accuracy}")
    privacy_analysis.append(f"**Accuracy vs Macro-F1 Gap**: {accuracy_f1_gap:.3f}")
    privacy_analysis.append(f"**Class Imbalance Severity**: {class_imbalance}")
    
    # Class-wise privacy analysis
    if 'num_classes' in df.columns:
        privacy_analysis.append("\n### Class-wise Privacy Analysis")
        class_stats = df.groupby('num_classes')['accuracy'].agg(['mean', 'std', 'count'])
        for classes, stats in class_stats.iterrows():
            privacy_analysis.append(f"- **{classes} classes**: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Model-wise privacy performance
    if 'model_type' in df.columns:
        privacy_analysis.append("\n### Model-wise Privacy Performance")
        model_stats = df.groupby('model_type')['accuracy'].agg(['mean', 'std', 'count'])
        for model, stats in model_stats.iterrows():
            privacy_analysis.append(f"- **{model}**: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Sensor placement privacy analysis
    if 'placement' in df.columns:
        single_sensor = df[df['placement'] != 'all']
        if len(single_sensor) > 0:
            privacy_analysis.append("\n### Sensor Placement Privacy Analysis")
            sensor_stats = single_sensor.groupby('placement')['accuracy'].agg(['mean', 'std', 'count'])
            for sensor, stats in sensor_stats.iterrows():
                privacy_analysis.append(f"- **{sensor}**: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Feature set privacy impact
    if 'featureset' in df.columns:
        privacy_analysis.append("\n### Feature Set Privacy Impact")
        feature_stats = df.groupby('featureset')['accuracy'].agg(['mean', 'std', 'count'])
        for featureset, stats in feature_stats.iterrows():
            privacy_analysis.append(f"- **{featureset}**: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
    
    # Privacy recommendations
    privacy_analysis.append("\n### Privacy Recommendations")
    if 'num_classes' in df.columns:
        best_3class = df[df['num_classes'] == 3]['accuracy'].max() if 3 in df['num_classes'].values else 0
        best_7class = df[df['num_classes'] == 7]['accuracy'].max() if 7 in df['num_classes'].values else 0
        if best_3class > 0 and best_7class > 0:
            privacy_degradation = best_3class - best_7class
            privacy_analysis.append(f"- **Best 3-class accuracy**: {best_3class:.3f}")
            privacy_analysis.append(f"- **Best 7-class accuracy**: {best_7class:.3f}")
            privacy_analysis.append(f"- **Privacy degradation**: {privacy_degradation:.3f}")
            
            if privacy_degradation < 0.1:
                privacy_analysis.append("- **RECOMMENDATION**: Low privacy impact - 7 classes acceptable")
            elif privacy_degradation < 0.2:
                privacy_analysis.append("- **RECOMMENDATION**: Moderate privacy impact - consider 5 classes")
            else:
                privacy_analysis.append("- **RECOMMENDATION**: High privacy impact - limit to 3-4 classes")
    
    return f"""## 🔒 Privacy Analysis

{chr(10).join(privacy_analysis)}"""


def generate_recommendations(df: pd.DataFrame) -> str:
    """Generate recommendations based on results."""
    
    df = create_model_names(df)
    
    recommendations = """
## 💡 Recommendations

### 🎯 Model Selection
"""
    
    # Best model recommendation
    best_model = df.groupby('model_name')['accuracy'].mean().idxmax()
    best_accuracy = df.groupby('model_name')['accuracy'].mean().max()
    recommendations += f"- **Best Overall**: {best_model} (Accuracy: {best_accuracy:.4f})\n"
    
    # Fastest model
    fastest_model = df.groupby('model_name')['train_time_s'].mean().idxmin()
    fastest_time = df.groupby('model_name')['train_time_s'].mean().min()
    recommendations += f"- **Fastest Training**: {fastest_model} ({fastest_time:.1f}s)\n"
    
    # Best accuracy/time trade-off
    df['efficiency'] = df['accuracy'] / df['train_time_s']
    most_efficient = df.groupby('model_name')['efficiency'].mean().idxmax()
    efficiency_score = df.groupby('model_name')['efficiency'].mean().max()
    recommendations += f"- **Most Efficient**: {most_efficient} (Efficiency: {efficiency_score:.6f})\n"
    
    recommendations += """
### 🔧 Future Improvements
- **Data Augmentation**: Consider augmenting training data to improve generalization
- **Hyperparameter Tuning**: Perform systematic hyperparameter optimization
- **Ensemble Methods**: Combine multiple models for improved performance
- **Feature Engineering**: Explore additional features or feature selection
- **Cross-Validation**: Use k-fold cross-validation for more robust evaluation

### 📈 Next Steps
- **Scale Up**: Run experiments with more seeds for statistical significance
- **Extend Classes**: Test with different class combinations
- **Compare Baselines**: Add traditional ML baselines for comparison
- **Error Analysis**: Analyze confusion matrices for insights
"""
    
    return recommendations


def generate_report(
    results_csv_path: str,
    plots_dir: str,
    output_path: str,
    title: str = "STM Experiment Report"
) -> str:
    """Generate comprehensive markdown report."""
    
    print(f"📊 Loading results from: {results_csv_path}")
    df = load_experiment_results(results_csv_path)
    
    if df.empty:
        print("❌ No valid results found")
        return ""
    
    print(f"📊 Loaded {len(df)} successful experiments")
    
    # Generate report sections
    report = f"""# {title}

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

{generate_experiment_summary(df)}

{generate_performance_analysis(df)}

{generate_class_analysis(df)}

{generate_mode_comparison(df)}

{generate_privacy_analysis(df)}

{generate_plots_section(plots_dir)}

{generate_recommendations(df)}

## 📁 Files

### Data Files
- **Results CSV**: `{results_csv_path}`
- **Per-Class Metrics**: `{os.path.join(os.path.dirname(results_csv_path), 'per_class_metrics.csv')}`
- **Summary Statistics**: `{os.path.join(plots_dir, 'summary_statistics.csv')}`

### Confusion Matrices
- **Location**: `{os.path.join(os.path.dirname(results_csv_path), 'confusions')}`
- **Format**: CSV and PNG files for each experiment

### Plots
- **Location**: `{plots_dir}`
- **Types**: Performance, efficiency, and analysis visualizations

---

*Report generated automatically by STM Experiment Framework*
"""
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {output_path}")
    
    return report


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Generate comprehensive report from STM experiment results")
    parser.add_argument("--results", required=True, help="Path to results.csv file")
    parser.add_argument("--plots", default="results/experiments/plots", help="Directory containing plots")
    parser.add_argument("--output", default="results/experiments/REPORT.md", help="Output path for report")
    parser.add_argument("--title", default="STM Experiment Report", help="Report title")
    
    args = parser.parse_args()
    
    # Generate report
    generate_report(
        results_csv_path=args.results,
        plots_dir=args.plots,
        output_path=args.output,
        title=args.title
    )


if __name__ == "__main__":
    main()
