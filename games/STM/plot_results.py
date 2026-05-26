#!/usr/bin/env python3
"""
Comprehensive plotting module for STM experiment results.
Generates various visualizations from experiment data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Set style
plt.style.use('default')
sns.set_palette("husl")


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


def plot_privacy_analysis(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (16, 12)
):
    """Plot comprehensive privacy analysis including class-wise performance and privacy-utility trade-offs."""
    
    # Create figure with subplots for privacy analysis
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Privacy Analysis: Class-wise Performance and Privacy-Utility Trade-offs', fontsize=16, fontweight='bold')
    
    # 1. Accuracy vs Number of Classes (Privacy-Utility Trade-off)
    ax1 = axes[0, 0]
    if 'num_classes' in df.columns:
        class_accuracy = df.groupby('num_classes')['accuracy'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(class_accuracy['num_classes'], class_accuracy['mean'], 
                    yerr=class_accuracy['std'], marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Number of Classes')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Privacy-Utility Trade-off: Accuracy vs Classes')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(class_accuracy['num_classes'])
    else:
        ax1.text(0.5, 0.5, 'No num_classes data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Privacy-Utility Trade-off (No Data)')
    
    # 2. Model-wise Privacy Performance
    ax2 = axes[0, 1]
    if 'model_type' in df.columns:
        model_stats = df.groupby('model_type')['accuracy'].agg(['mean', 'std']).reset_index()
        model_stats = model_stats.sort_values('mean', ascending=False)
        bars = ax2.bar(range(len(model_stats)), model_stats['mean'], 
                      yerr=model_stats['std'], capsize=5, alpha=0.7)
        ax2.set_xlabel('Model Type')
        ax2.set_ylabel('Average Accuracy')
        ax2.set_title('Model-wise Privacy Performance')
        ax2.set_xticks(range(len(model_stats)))
        ax2.set_xticklabels(model_stats['model_type'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No model_type data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Model-wise Privacy Performance (No Data)')
    
    # 3. Fusion Strategy Privacy Analysis
    ax3 = axes[0, 2]
    if 'fusion' in df.columns and 'num_classes' in df.columns:
        fusion_data = df[df['fusion'] != 'NA']
        if len(fusion_data) > 0:
            for fusion in fusion_data['fusion'].unique():
                fusion_subset = fusion_data[fusion_data['fusion'] == fusion]
                class_accuracy = fusion_subset.groupby('num_classes')['accuracy'].mean()
                ax3.plot(class_accuracy.index, class_accuracy.values, marker='o', label=fusion)
            ax3.set_xlabel('Number of Classes')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Fusion Strategy Privacy Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No fusion data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Fusion Strategy Privacy Analysis (No Data)')
    else:
        ax3.text(0.5, 0.5, 'No fusion/num_classes data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Fusion Strategy Privacy Analysis (No Data)')
    
    # 4. Sensor Placement Privacy Analysis
    ax4 = axes[1, 0]
    if 'placement' in df.columns and 'num_classes' in df.columns:
        single_sensor = df[df['placement'] != 'all']
        if len(single_sensor) > 0:
            sensor_stats = single_sensor.groupby('placement')['accuracy'].agg(['mean', 'std']).reset_index()
            sensor_stats = sensor_stats.sort_values('mean', ascending=False)
            bars = ax4.bar(range(len(sensor_stats)), sensor_stats['mean'], 
                          yerr=sensor_stats['std'], capsize=5, alpha=0.7)
            ax4.set_xlabel('Sensor Placement')
            ax4.set_ylabel('Average Accuracy')
            ax4.set_title('Sensor Placement Privacy Analysis')
            ax4.set_xticks(range(len(sensor_stats)))
            ax4.set_xticklabels(sensor_stats['placement'], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No single-sensor data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Sensor Placement Privacy Analysis (No Data)')
    else:
        ax4.text(0.5, 0.5, 'No placement/num_classes data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Sensor Placement Privacy Analysis (No Data)')
    
    # 5. Feature Set Privacy Impact
    ax5 = axes[1, 1]
    if 'featureset' in df.columns and 'num_classes' in df.columns:
        feature_stats = df.groupby('featureset')['accuracy'].agg(['mean', 'std']).reset_index()
        feature_stats = feature_stats.sort_values('mean', ascending=False)
        bars = ax5.bar(range(len(feature_stats)), feature_stats['mean'], 
                      yerr=feature_stats['std'], capsize=5, alpha=0.7)
        ax5.set_xlabel('Feature Set')
        ax5.set_ylabel('Average Accuracy')
        ax5.set_title('Feature Set Privacy Impact')
        ax5.set_xticks(range(len(feature_stats)))
        ax5.set_xticklabels(feature_stats['featureset'], rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No featureset data available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Feature Set Privacy Impact (No Data)')
    
    # 6. Privacy Degradation Analysis
    ax6 = axes[1, 2]
    if 'num_classes' in df.columns:
        # Calculate privacy degradation (accuracy drop from 3 to 7 classes)
        class_means = df.groupby('num_classes')['accuracy'].mean()
        if 3 in class_means.index and 7 in class_means.index:
            degradation = class_means[3] - class_means[7]
            ax6.bar(['3 Classes', '7 Classes'], [class_means[3], class_means[7]], 
                   color=['green', 'red'], alpha=0.7)
            ax6.set_ylabel('Accuracy')
            ax6.set_title(f'Privacy Degradation: {degradation:.3f}')
            ax6.text(0.5, max(class_means[3], class_means[7]) + 0.02, 
                    f'Degradation: {degradation:.3f}', ha='center', va='bottom', fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Insufficient class data for degradation analysis', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Privacy Degradation Analysis (No Data)')
    else:
        ax6.text(0.5, 0.5, 'No num_classes data available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Privacy Degradation Analysis (No Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'privacy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved privacy analysis plot: {os.path.join(output_dir, 'privacy_analysis.png')}")


def plot_accuracy_vs_classes(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot accuracy vs number of classes for different models."""
    
    plt.figure(figsize=figsize)
    
    # Group by mode and fusion/placement
    if 'multi' in df['mode'].values:
        # Multi-sensor experiments
        for fusion in df['fusion'].unique():
            if fusion == 'NA':
                continue
            fusion_data = df[(df['fusion'] == fusion) & (df['mode'] == 'multi')]
            if not fusion_data.empty:
                plt.plot(fusion_data['num_classes'], fusion_data['accuracy'], 
                        marker='o', linewidth=2, markersize=8, label=f'Multi-{fusion}')
    
    if 'single' in df['mode'].values:
        # Single-sensor experiments
        for placement in df['placement'].unique():
            if placement == 'NA':
                continue
            placement_data = df[(df['placement'] == placement) & (df['mode'] == 'single')]
            if not placement_data.empty:
                plt.plot(placement_data['num_classes'], placement_data['accuracy'], 
                        marker='s', linewidth=2, markersize=8, label=f'Single-{placement}')
    
    plt.xlabel('Number of Classes', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy vs Number of Classes', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'accuracy_vs_classes.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Accuracy vs Classes plot saved: {plot_path}")


def plot_f1_vs_classes(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot F1 score vs number of classes for different models."""
    
    plt.figure(figsize=figsize)
    
    # Group by mode and fusion/placement
    if 'multi' in df['mode'].values:
        # Multi-sensor experiments
        for fusion in df['fusion'].unique():
            if fusion == 'NA':
                continue
            fusion_data = df[(df['fusion'] == fusion) & (df['mode'] == 'multi')]
            if not fusion_data.empty:
                plt.plot(fusion_data['num_classes'], fusion_data['macro_f1'], 
                        marker='o', linewidth=2, markersize=8, label=f'Multi-{fusion}')
    
    if 'single' in df['mode'].values:
        # Single-sensor experiments
        for placement in df['placement'].unique():
            if placement == 'NA':
                continue
            placement_data = df[(df['placement'] == placement) & (df['mode'] == 'single')]
            if not placement_data.empty:
                plt.plot(placement_data['num_classes'], placement_data['macro_f1'], 
                        marker='s', linewidth=2, markersize=8, label=f'Single-{placement}')
    
    plt.xlabel('Number of Classes', fontsize=14)
    plt.ylabel('Macro F1 Score', fontsize=14)
    plt.title('F1 Score vs Number of Classes', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'f1_vs_classes.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 F1 vs Classes plot saved: {plot_path}")


def plot_model_comparison(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """Plot model comparison bar chart."""
    
    plt.figure(figsize=figsize)
    
    # Create model names
    df['model_name'] = df.apply(lambda row: 
        f"{row['mode'].title()}-{row['fusion'] if row['fusion'] != 'NA' else row['placement']}", axis=1)
    
    # Group by model and get mean accuracy
    model_accuracy = df.groupby('model_name')['accuracy'].mean().sort_values(ascending=False)
    
    # Create bar plot
    bars = plt.bar(range(len(model_accuracy)), model_accuracy.values, 
                   color=sns.color_palette("husl", len(model_accuracy)))
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(model_accuracy)), model_accuracy.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, model_accuracy.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison plot saved: {plot_path}")


def plot_training_time_comparison(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot training time comparison."""
    
    plt.figure(figsize=figsize)
    
    # Create model names
    df['model_name'] = df.apply(lambda row: 
        f"{row['mode'].title()}-{row['fusion'] if row['fusion'] != 'NA' else row['placement']}", axis=1)
    
    # Group by model and get mean training time
    model_time = df.groupby('model_name')['train_time_s'].mean().sort_values(ascending=False)
    
    # Create bar plot
    bars = plt.bar(range(len(model_time)), model_time.values, 
                   color=sns.color_palette("viridis", len(model_time)))
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Mean Training Time (seconds)', fontsize=14)
    plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(model_time)), model_time.index, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, model_time.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_time.values) * 0.01,
                f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training time comparison plot saved: {plot_path}")


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (10, 8)
):
    """Plot accuracy heatmap for different configurations."""
    
    plt.figure(figsize=figsize)
    
    # Create pivot table for heatmap
    if 'multi' in df['mode'].values:
        # Multi-sensor heatmap
        pivot_data = df[df['mode'] == 'multi'].pivot_table(
            values='accuracy', 
            index='fusion', 
            columns='num_classes', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Accuracy'})
        plt.title('Multi-Sensor Accuracy Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Classes', fontsize=14)
        plt.ylabel('Fusion Method', fontsize=14)
    
    elif 'single' in df['mode'].values:
        # Single-sensor heatmap
        pivot_data = df[df['mode'] == 'single'].pivot_table(
            values='accuracy', 
            index='placement', 
            columns='num_classes', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Accuracy'})
        plt.title('Single-Sensor Accuracy Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Classes', fontsize=14)
        plt.ylabel('Sensor Placement', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'accuracy_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Accuracy heatmap saved: {plot_path}")


def plot_seed_variability(
    df: pd.DataFrame,
    output_dir: str,
    figsize: Tuple[int, int] = (14, 8)
):
    """Plot accuracy variability across seeds."""
    
    if 'seed' not in df.columns or len(df['seed'].unique()) < 2:
        print("⚠️ Not enough seeds for variability analysis")
        return
    
    plt.figure(figsize=figsize)
    
    # Create model names
    df['model_name'] = df.apply(lambda row: 
        f"{row['mode'].title()}-{row['fusion'] if row['fusion'] != 'NA' else row['placement']}", axis=1)
    
    # Get unique models and classes
    models = df['model_name'].unique()
    classes = sorted(df['num_classes'].unique())
    
    # Create subplots
    n_models = len(models)
    n_classes = len(classes)
    
    if n_models * n_classes > 12:  # Too many subplots
        print("⚠️ Too many combinations for subplot layout, creating summary plot")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # Calculate coefficient of variation for each model
        cv_data = []
        for model in models:
            model_data = df[df['model_name'] == model]
            if len(model_data) > 1:
                cv = model_data['accuracy'].std() / model_data['accuracy'].mean()
                cv_data.append({'model': model, 'cv': cv, 'mean_acc': model_data['accuracy'].mean()})
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            cv_df = cv_df.sort_values('cv', ascending=False)
            
            bars = plt.bar(range(len(cv_df)), cv_df['cv'], 
                          color=sns.color_palette("viridis", len(cv_df)))
            
            plt.xlabel('Model', fontsize=14)
            plt.ylabel('Coefficient of Variation', fontsize=14)
            plt.title('Model Variability Across Seeds', fontsize=16, fontweight='bold')
            plt.xticks(range(len(cv_df)), cv_df['model'], rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add mean accuracy as text
            for i, (bar, row) in enumerate(zip(bars, cv_df.itertuples())):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cv_df['cv']) * 0.01,
                        f'μ={row.mean_acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, 'seed_variability_summary.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Seed variability summary saved: {plot_path}")
    
    else:
        # Create subplots
        fig, axes = plt.subplots(n_models, n_classes, figsize=(4*n_classes, 4*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_classes == 1:
            axes = axes.reshape(-1, 1)
        
        for i, model in enumerate(models):
            for j, num_class in enumerate(classes):
                model_class_data = df[(df['model_name'] == model) & (df['num_classes'] == num_class)]
                
                if not model_class_data.empty:
                    axes[i, j].boxplot(model_class_data['accuracy'])
                    axes[i, j].set_title(f'{model}\n{num_class} classes')
                    axes[i, j].set_ylabel('Accuracy')
                    axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'seed_variability.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Seed variability plot saved: {plot_path}")


def create_summary_table(
    df: pd.DataFrame,
    output_dir: str
):
    """Create a summary table of results."""
    
    # Create model names
    df['model_name'] = df.apply(lambda row: 
        f"{row['mode'].title()}-{row['fusion'] if row['fusion'] != 'NA' else row['placement']}", axis=1)
    
    # Group by model and calculate statistics
    summary = df.groupby('model_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'macro_f1': ['mean', 'std'],
        'train_time_s': ['mean', 'std'],
        'num_classes': 'count'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Save to CSV
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_path)
    
    print(f"📊 Summary statistics saved: {summary_path}")
    
    # Print summary
    print("\n📋 Summary Statistics:")
    print("=" * 80)
    print(summary.to_string())
    
    return summary


def generate_all_plots(
    results_csv_path: str,
    output_dir: str,
    figsize: Tuple[int, int] = (12, 8),
    include_privacy_analysis: bool = False
):
    """Generate all plots from experiment results."""
    
    print(f"🔍 Loading results from: {results_csv_path}")
    df = load_experiment_results(results_csv_path)
    
    if df.empty:
        print("❌ No valid results found")
        return
    
    print(f"📊 Loaded {len(df)} successful experiments")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    print("\n🎨 Generating plots...")
    
    try:
        plot_accuracy_vs_classes(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating accuracy vs classes plot: {e}")
    
    try:
        plot_f1_vs_classes(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating F1 vs classes plot: {e}")
    
    try:
        plot_model_comparison(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating model comparison plot: {e}")
    
    try:
        plot_training_time_comparison(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating training time comparison plot: {e}")
    
    try:
        plot_accuracy_heatmap(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating accuracy heatmap: {e}")
    
    try:
        plot_seed_variability(df, output_dir, figsize)
    except Exception as e:
        print(f"⚠️ Error generating seed variability plot: {e}")
    
    # Generate privacy analysis if requested
    if include_privacy_analysis:
        try:
            plot_privacy_analysis(df, output_dir, (16, 12))
        except Exception as e:
            print(f"⚠️ Error generating privacy analysis plot: {e}")
    
    try:
        create_summary_table(df, output_dir)
    except Exception as e:
        print(f"⚠️ Error generating summary table: {e}")
    
    print(f"\n✅ All plots generated in: {output_dir}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Generate plots from STM experiment results")
    parser.add_argument("--results", required=True, help="Path to results.csv file")
    parser.add_argument("--outdir", default="results/experiments/plots", help="Output directory for plots")
    parser.add_argument("--figsize", nargs=2, type=int, default=[12, 8], help="Figure size (width height)")
    parser.add_argument("--privacy_analysis", action="store_true", help="Generate comprehensive privacy analysis plots")
    
    args = parser.parse_args()
    
    # Generate all plots
    generate_all_plots(
        results_csv_path=args.results,
        output_dir=args.outdir,
        figsize=tuple(args.figsize),
        include_privacy_analysis=args.privacy_analysis
    )


if __name__ == "__main__":
    main()
