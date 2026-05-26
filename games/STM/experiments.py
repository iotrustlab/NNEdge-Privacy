#!/usr/bin/env python3
"""
Comprehensive experiment runner for STM games.
Runs a full grid of experiments and produces tidy CSVs, plots, and reports.
"""

import os
import sys
import time
import json
import argparse
import multiprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Optimize TensorFlow for Metal GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        print(f"🖥️ GPU Device: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU detected, using CPU")
    # Fallback to CPU optimization
    tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())
    tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())

from games.STM.runner import train_and_eval_single_run, convert_numpy_types
from games.STM.config import GAME_CONFIGS, WINDOW_SIZE


def build_experiment_grid(
    grid_type: str = "all",
    games: List[str] = ["Game-1"],
    models: List[str] = None,
    classes: List[int] = [3, 4, 5, 7],
    seeds: List[int] = [1, 2, 3],
    placements: List[str] = ["right-pocket", "right-wrist", "left-wrist", "right-ankle", "left-ankle"],
    fusions: List[str] = ["late", "weighted", "attention"],
    featuresets: List[str] = ["base", "micro", "micro+cross", "micro+cross+semantics", "micro+cross+semantics+film", "semantic-14", "semantic-94", "concatenated"],
    losses: List[str] = ["ce", "focal", "balanced_softmax", "logit_adjust"],
    label_smoothing: List[float] = [0.0, 0.05],
    adjacency_modes: List[str] = ["identity", "fixed", "learned"],
    gate_alpha_init: List[float] = [0.1],
    hierarchical: List[bool] = [False, True],
    hier_lambda: List[float] = [0.3],
    samplers: List[str] = ["random", "class_uniform"],
    warmup_epochs: List[int] = [0, 5],
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Build the experiment grid based on parameters.
    
    Args:
        grid_type: "all", "single", "multi", "mini", or "mini_v2"
        classes: List of class counts to test
        seeds: List of random seeds
        placements: List of placements for single-sensor runs
        fusions: List of fusion methods for multi-sensor runs
        losses: List of loss functions to test
        adjacency_modes: List of adjacency modes for GraphSemantics
        gate_alpha_init: List of initial alpha gate values
        hierarchical: List of hierarchical head settings
        hier_lambda: List of hierarchical loss weights
        samplers: List of sampling strategies
        warmup_epochs: List of warmup epoch counts
        **kwargs: Additional parameters
        
    Returns:
        List of experiment configurations
    """
    experiments = []
    
    # Determine the game to use based on grid type
    if grid_type in ["multi", "mini_v2"]:
        # For multi-sensor experiments, use the first game (Game-1 becomes multi-sensor with --grid multi)
        game_name = games[0] if games else "Game-1"
    else:
        # For single-sensor experiments, use the games directly
        game_name = games[0] if games else "Game-1"
    
    if grid_type == "mini_v2":
        # Mini-grid v2 for imbalance and semantics fixes
        # All runs: 7-class, fusion=attention, seed=1, sampler=class_uniform, warmup_epochs=5, select_by=macro_f1
        
        # 1) Imbalance ablation
        experiments.extend([
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'base',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'identity',
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Imbalance ablation: base featureset with balanced_softmax"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'identity',
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Imbalance ablation: micro+cross featureset with balanced_softmax"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'fixed',
                'gate_alpha_init': 0.1,
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Imbalance ablation: micro+cross+semantics with balanced_softmax and α-gate"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics+film',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'fixed',
                'gate_alpha_init': 0.1,
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Imbalance ablation: micro+cross+semantics+film with balanced_softmax and α-gate"
            }
        ])
        
        # 2) Graph sanity
        experiments.extend([
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'identity',
                'gate_alpha_init': 0.1,
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Graph sanity: identity adjacency"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'fixed',
                'gate_alpha_init': 0.1,
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Graph sanity: fixed adjacency"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'learned',
                'gate_alpha_init': 0.1,
                'hierarchical': False,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Graph sanity: learned adjacency"
            },
            {
                'mode': 'multi',
                'game': game_name,
                'placement': 'NA',
                'fusion': 'attention',
                'num_classes': 7,
                'seed': 1,
                'model_type': 'binary_sensor_token_attention',
                'featureset': 'micro+cross+semantics',
                'loss_type': 'balanced_softmax',
                'adjacency_mode': 'fixed',
                'gate_alpha_init': 0.1,
                'hierarchical': True,
                'hier_lambda': 0.3,
                'sampler': 'class_uniform',
                'warmup_epochs': 5,
                'select_by': 'macro_f1',
                'description': "Graph sanity: hierarchical head"
            }
        ])
        
        return experiments
    
    if grid_type in ["all", "single"]:
        # Single-placement experiments
        for game in games:
            for placement in placements:
                for num_classes in classes:
                    for seed in seeds:
                        # If specific models are provided, use them; otherwise use default
                        model_list = models if models else ['advanced_cnn']
                        for model_type in model_list:
                            experiments.append({
                                'mode': 'single',
                                'game': game,
                                'placement': placement,
                                'fusion': 'NA',
                                'num_classes': num_classes,
                                'seed': seed,
                                'model_type': model_type,
                                'description': f"Single-placement {placement} with {num_classes} classes for {game} using {model_type}"
                            })
    
    if grid_type in ["all", "multi"]:
        # If specific models are provided, use them directly
        if models:
            for model_type in models:
                for featureset in featuresets:
                    # Filter loss functions based on model type
                    traditional_models = ['decision_tree', 'naive_bayes', 'random_forest']
                    if model_type in traditional_models:
                        # Traditional models only support cross-entropy loss
                        model_losses = ['ce']
                    else:
                        # Neural models support all loss functions
                        model_losses = losses
                    
                    for loss in model_losses:
                        for smoothing in label_smoothing:
                            for num_classes in classes:
                                for seed in seeds:
                                    # Skip base featureset with focal loss (not meaningful)
                                    if featureset == "base" and loss == "focal":
                                        continue
                                    
                                    # Skip base featureset with label smoothing (not meaningful)
                                    if featureset == "base" and smoothing > 0:
                                        continue
                                    
                                    # Skip label smoothing for non-CE losses
                                    if loss != "ce" and smoothing > 0:
                                        continue
                                    
                                    # Determine if FiLM should be used
                                    use_film = "film" in featureset
                                    
                                    # Determine loss function
                                    use_focal_loss = loss == "focal"
                                    use_label_smoothing = smoothing > 0
                                    
                                    experiments.append({
                                        'mode': 'multi',
                                        'game': game_name,
                                        'placement': 'all',
                                        'fusion': 'NA',
                                        'num_classes': num_classes,
                                        'seed': seed,
                                        'model_type': model_type,
                                        'featureset': featureset,
                                        'use_film': use_film,
                                        'use_focal_loss': use_focal_loss,
                                        'use_label_smoothing': use_label_smoothing,
                                        'label_smoothing': smoothing,
                                        'loss_type': loss,
                                        'description': f"Multi-sensor {model_type} with {featureset} featureset, {loss} loss, {num_classes} classes"
                                    })
        else:
            # Original logic for automatic model selection
            for fusion in fusions:
                for featureset in featuresets:
                    # Multi-sensor experiments with semantic features
                    # Use simple concatenated model for concatenated featureset
                    if featureset == 'concatenated':
                        model_type = 'simple_concatenated_cnn'
                    # Use binary models for Game-1 (binary-only), continuous models for other games
                    elif games[0] == "Game-1":
                        fusion_to_model = {
                            'late': lambda f: 'binary_late_fusion',
                            'weighted': lambda f: 'binary_weighted_fusion',
                            'attention': lambda f: 'binary_sensor_token_attention'
                        }
                        model_type = fusion_to_model.get(fusion, lambda f: 'unknown')(featureset)
                    else:
                        # For other games with continuous sensor data (accelerometer + gyroscope + binary)
                        fusion_to_model = {
                            'late': lambda f: 'continuous_late_fusion',
                            'weighted': lambda f: 'continuous_weighted_fusion',
                            'attention': lambda f: 'binary_sensor_token_attention'
                        }
                        model_type = fusion_to_model.get(fusion, lambda f: 'unknown')(featureset)
                for loss in losses:
                    for smoothing in label_smoothing:
                        for num_classes in classes:
                            for seed in seeds:
                                # Skip base featureset with focal loss (not meaningful)
                                if featureset == "base" and loss == "focal":
                                    continue
                                
                                # Skip base featureset with label smoothing (not meaningful)
                                if featureset == "base" and smoothing > 0:
                                    continue
                                
                                # Skip label smoothing for non-CE losses
                                if loss != "ce" and smoothing > 0:
                                    continue
                                
                                # Determine if FiLM should be used
                                use_film = "film" in featureset
                                
                                # Determine loss function
                                use_focal_loss = loss == "focal"
                                use_label_smoothing = smoothing > 0
                                
                                # model_type is already set from the loop - don't override it
                                
                                experiments.append({
                                    'mode': 'multi',
                                    'game': game_name,
                                    'placement': 'NA',
                                    'fusion': fusion,
                                    'num_classes': num_classes,
                                    'seed': seed,
                                    'model_type': model_type,
                                    'featureset': featureset,
                                    'use_film': use_film,
                                    'use_focal_loss': use_focal_loss,
                                    'use_label_smoothing': use_label_smoothing,
                                    'label_smoothing': smoothing,
                                    'loss_type': loss,
                                    'description': f"Multi-sensor {fusion} fusion, {featureset}, {loss}, smoothing={smoothing}, {num_classes} classes"
                                })
    
    return experiments


def run_single_experiment(
    exp_config: Dict[str, Any],
    epochs: int = 25,
    batch_size: int = 128,
    use_per_user_norm: bool = True,
    select_by: str = 'accuracy',
    grid_type: str = 'single',
    **kwargs
) -> Dict[str, Any]:
    """
    Run a single experiment.
    
    Args:
        exp_config: Experiment configuration
        epochs: Number of training epochs
        batch_size: Batch size
        use_per_user_norm: Whether to use per-user normalization
        **kwargs: Additional parameters
        
    Returns:
        Results dictionary
    """
    start_time = time.time()
    
    try:
        # Extract semantic features parameters
        featureset = exp_config.get('featureset', 'base')
        use_film = exp_config.get('use_film', False)
        use_focal_loss = exp_config.get('use_focal_loss', False)
        use_label_smoothing = exp_config.get('use_label_smoothing', False)
        label_smoothing = exp_config.get('label_smoothing', 0.0)
        
        # Extract new parameters
        loss_type = exp_config.get('loss_type', 'ce')
        adjacency_mode = exp_config.get('adjacency_mode', 'fixed')
        gate_alpha_init = exp_config.get('gate_alpha_init', 0.1)
        hierarchical = exp_config.get('hierarchical', False)
        hier_lambda = exp_config.get('hier_lambda', 0.3)
        sampler = exp_config.get('sampler', 'random')
        warmup_epochs = exp_config.get('warmup_epochs', 0)
        select_by = exp_config.get('select_by', select_by)
        
        # Map fusion to model_type
        # Use the model_type from exp_config (preserves command-line specified models)
        model_type = exp_config.get('model_type')
        
        # Only override for neural models with concatenated featureset if no model specified
        if not model_type and featureset == 'concatenated':
            model_type = 'advanced_cnn'  # Default for concatenated featureset
        # Use binary models for Game-1 (binary-only), continuous models for other games
        elif exp_config['game'] == "Game-1":
            fusion_to_model = {
                'late': lambda f: 'binary_late_fusion',
                'weighted': lambda f: 'binary_weighted_fusion',
                'attention': lambda f: 'binary_sensor_token_attention'
            }
            model_type = fusion_to_model.get(exp_config['fusion'], lambda f: exp_config.get('model_type', 'unknown'))(featureset)
        else:
            # For other games with continuous sensor data (accelerometer + gyroscope + binary)
            fusion_to_model = {
                'late': lambda f: 'continuous_late_fusion',
                'weighted': lambda f: 'continuous_weighted_fusion',
                'attention': lambda f: 'binary_sensor_token_attention'
            }
            model_type = fusion_to_model.get(exp_config['fusion'], lambda f: exp_config.get('model_type', 'unknown'))(featureset)
        
        # Run the experiment with semantic features support
        results = train_and_eval_single_run(
            game_name=exp_config['game'],
            model_type=model_type,
            split_name="cross-user",
            use_per_user_norm=use_per_user_norm,
            epochs=epochs,
            batch_size=batch_size,
            seed=exp_config['seed'],
            featureset=featureset,
            use_film=use_film,
            use_focal_loss=use_focal_loss,
            use_label_smoothing=use_label_smoothing,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            adjacency_mode=adjacency_mode,
            gate_alpha_init=gate_alpha_init,
            hierarchical=hierarchical,
            hier_lambda=hier_lambda,
            sampler=sampler,
            warmup_epochs=warmup_epochs,
            select_by=select_by,
            use_normalization=kwargs.get('use_normalization', False),
            use_attention=kwargs.get('use_attention', False),
            use_augmentation=kwargs.get('use_augmentation', False),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            focal_alpha=kwargs.get('focal_alpha', 0.25),
            placement=exp_config.get('placement', 'all'),  # Pass placement information
            num_classes=exp_config.get('num_classes', None),  # Pass num_classes for filtering
            grid_type=grid_type  # Pass grid_type for multi-sensor detection
        )
        
        # Add experiment metadata
        results.update({
            'run_id': exp_config.get('run_id', 'unknown'),
            'mode': exp_config['mode'],
            'placement': exp_config['placement'],
            'fusion': exp_config['fusion'],
            'num_classes': exp_config['num_classes'],
            'seed': exp_config['seed'],  # Always include seed
            'featureset': featureset,
            'use_film': use_film,
            'use_focal_loss': use_focal_loss,
            'use_label_smoothing': use_label_smoothing,
            'label_smoothing': label_smoothing,
            'loss_type': loss_type,
            'adjacency_mode': adjacency_mode,
            'gate_alpha_init': gate_alpha_init,
            'hierarchical': hierarchical,
            'hier_lambda': hier_lambda,
            'sampler': sampler,
            'warmup_epochs': warmup_epochs,
            'users_split': 'cross-user',
            'window_size': WINDOW_SIZE,
            'features': results.get('input_shape', [None, None, None])[2] if results.get('input_shape') else None,
            'epochs': epochs,
            'eval_time_s': time.time() - start_time
        })
        
        return results
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return {
            'error': str(e),
            'run_id': exp_config.get('run_id', 'unknown'),
            'mode': exp_config.get('mode', 'unknown'),
            'placement': exp_config.get('placement', 'unknown'),
            'fusion': exp_config.get('fusion', 'unknown'),
            'num_classes': exp_config.get('num_classes', 0),
            'seed': exp_config['seed'], # Ensure seed is always included
            'featureset': exp_config.get('featureset', 'unknown'),
            'use_film': exp_config.get('use_film', False),
            'use_focal_loss': exp_config.get('use_focal_loss', False),
            'use_label_smoothing': exp_config.get('use_label_smoothing', False),
            'label_smoothing': exp_config.get('label_smoothing', 0.0),
            'loss_type': exp_config.get('loss_type', 'unknown'),
            'adjacency_mode': exp_config.get('adjacency_mode', 'unknown'),
            'gate_alpha_init': exp_config.get('gate_alpha_init', 0.0),
            'hierarchical': exp_config.get('hierarchical', False),
            'hier_lambda': exp_config.get('hier_lambda', 0.0),
            'sampler': exp_config.get('sampler', 'unknown'),
            'warmup_epochs': exp_config.get('warmup_epochs', 0),
            'users_split': 'cross-user',
            'window_size': WINDOW_SIZE,
            'features': None,
            'epochs': epochs,
            'eval_time_s': time.time() - start_time
        }


def _save_metrics_and_artifacts(
    true_labels, 
    predictions, 
    save_dir, 
    result_meta, 
    split_type='test'
):
    """Save all metrics and artifacts for a specific split"""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, 
        recall_score, classification_report, confusion_matrix
    )
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_micro = f1_score(true_labels, predictions, average='micro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    
    # 1. Save metrics.json
    metrics = {
        'split_type': split_type,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro), 
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'precision_weighted': float(precision_weighted),
        'recall_macro': float(recall_macro),
        'recall_weighted': float(recall_weighted),
        'model_type': result_meta['model_type'],
        'seed': result_meta['seed'],
        'num_classes': result_meta['num_classes'],
        'num_samples': len(true_labels),
        'featureset': result_meta.get('featureset', 'base'),
        'game_name': result_meta['game_name']
    }
    
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 2. Save predictions.csv
    predictions_df = pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions
    })
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # 3. Save confusion_matrix.csv and .png
    cm = confusion_matrix(true_labels, predictions)
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'), index=False)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {result_meta["model_type"]} ({split_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Save classification_report.txt
    report = classification_report(true_labels, predictions, digits=4, zero_division=0)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Classification Report - {result_meta['model_type']} ({split_type})\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    return metrics


def save_individual_model_results(
    result: Dict[str, Any],
    output_dir: str,
    save_train: bool = True,
    save_test: bool = True
):
    """Save train and/or test results in separate folders for individual models"""
    
    model_type = result['model_type']
    model_dir = os.path.join(output_dir, model_type)
    
    print(f"📁 Saving individual results for {model_type} to {model_dir}")
    
    # Save test results (primary)
    if save_test and 'test_predictions' in result and result['test_predictions'] is not None:
        test_dir = os.path.join(model_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        test_metrics = _save_metrics_and_artifacts(
            result['test_true_labels'],
            result['test_predictions'], 
            test_dir,
            result,
            split_type='test'
        )
        print(f"   ✅ Test results saved: accuracy={test_metrics['accuracy']:.4f}")
    
    # Save training results (optional but recommended)
    if save_train and 'train_predictions' in result and result['train_predictions'] is not None:
        train_dir = os.path.join(model_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        train_metrics = _save_metrics_and_artifacts(
            result['train_true_labels'],
            result['train_predictions'],
            train_dir, 
            result,
            split_type='train'
        )
        print(f"   ✅ Train results saved: accuracy={train_metrics['accuracy']:.4f}")
    
    # Save model metadata
    metadata = {
        'model_type': result['model_type'],
        'game_name': result['game_name'],
        'seed': result['seed'],
        'num_classes': result['num_classes'],
        'training_time_s': result['train_time_s'],
        'featureset': result.get('featureset', 'base'),
        'input_shape': result.get('input_shape'),
        'epochs': result.get('epochs', 0),
        'batch_size': result.get('batch_size', 0)
    }
    
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    run_id: str,
    output_dir: str,
    class_names: Optional[List[str]] = None
):
    """Save confusion matrix as CSV and PNG."""
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save CSV
        cm_df = pd.DataFrame(cm)
        if class_names:
            cm_df.index = class_names
            cm_df.columns = class_names
        
        csv_path = os.path.join(output_dir, 'confusions', f'{run_id}_cm.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        cm_df.to_csv(csv_path)
        
        # Save PNG
        plt.figure(figsize=(10, 8))
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {run_id}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        png_path = os.path.join(output_dir, 'confusions', f'{run_id}_cm.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return csv_path, png_path
        
    except Exception as e:
        print(f"⚠️ Could not save confusion matrix: {e}")
        return None, None


def create_per_class_metrics(
    results: List[Dict[str, Any]],
    output_dir: str
) -> str:
    """Create per-class metrics CSV."""
    per_class_rows = []
    
    for result in results:
        if 'error' in result or 'predictions' not in result:
            continue
            
        try:
            y_true = result['true_labels']
            y_pred = result['predictions']
            
            # Get classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Extract per-class metrics
            for class_id in report:
                if class_id in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                    
                per_class_rows.append({
                    'run_id': result['run_id'],
                    'class_id': int(class_id),
                    'precision': report[class_id]['precision'],
                    'recall': report[class_id]['recall'],
                    'f1': report[class_id]['f1-score'],
                    'support': report[class_id]['support']
                })
                
        except Exception as e:
            print(f"⚠️ Could not process per-class metrics for {result.get('run_id', 'unknown')}: {e}")
    
    # Create DataFrame and save
    if per_class_rows:
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_path = os.path.join(output_dir, 'per_class_metrics.csv')
        per_class_df.to_csv(per_class_path, index=False)
        return per_class_path
    
    return None


def run_experiment_grid(
    grid_type: str = "all",
    games: List[str] = ["Game-1"],
    models: List[str] = None,
    classes: List[int] = [3, 4, 5, 7],
    seeds: List[int] = [1, 2, 3],
    placements: List[str] = ["right-pocket", "right-wrist", "left-wrist", "right-ankle", "left-ankle"],
    fusions: List[str] = ["late", "weighted", "attention"],
    featuresets: List[str] = ["base", "micro", "micro+cross", "micro+cross+semantics", "micro+cross+semantics+film", "semantic-14", "semantic-94", "concatenated"],
    losses: List[str] = ["ce", "focal"],
    label_smoothing: List[float] = [0.0, 0.05],
    epochs: int = 25,
    batch_size: int = 128,
    use_per_user_norm: bool = True,
    select_by: str = 'macro_f1',
    output_dir: str = "results/experiments",
    dry_run: bool = False,
    **kwargs
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Run the complete experiment grid.
    
    Returns:
        Tuple of (results_list, results_csv_path)
    """
    # Create output directory (individual model directories will be created as needed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build experiment grid
    experiments = build_experiment_grid(
        grid_type=grid_type,
        games=games,
        models=models,
        classes=classes,
        seeds=seeds,
        placements=placements,
        fusions=fusions,
        featuresets=featuresets,
        losses=losses,
        label_smoothing=label_smoothing
    )
    
    print(f"🔬 Experiment Grid Configuration:")
    print(f"   Grid type: {grid_type}")
    print(f"   Classes: {classes}")
    print(f"   Seeds: {seeds}")
    print(f"   Placements: {placements}")
    print(f"   Fusions: {fusions}")
    print(f"   Total experiments: {len(experiments)}")
    print(f"   Output directory: {output_dir}")
    
    if dry_run:
        print("\n🔍 DRY RUN - No experiments will be executed")
        for i, exp in enumerate(experiments):
            print(f"   {i+1:3d}. {exp['description']} (seed {exp['seed']})")
        return [], None
    
    # Run experiments
    results = []
    total_experiments = len(experiments)
    
    for i, exp_config in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"🏃 Experiment {i+1}/{total_experiments}: {exp_config['description']}")
        print(f"{'='*80}")
        
        # Generate run ID
        run_id = f"{exp_config['mode']}-{exp_config.get('placement', exp_config['fusion'])}-{exp_config['num_classes']}c-s{exp_config['seed']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run experiment
        result = run_single_experiment(
            exp_config=exp_config,
            epochs=epochs,
            batch_size=batch_size,
            use_per_user_norm=use_per_user_norm,
            grid_type=grid_type,
            **kwargs
        )
        
        # Add run ID
        result['run_id'] = run_id
        
        # Save individual model results instead of aggregated results
        if 'error' not in result:
            save_individual_model_results(
                result, 
                output_dir, 
                save_train=True, 
                save_test=True
            )
        else:
            print(f"❌ Experiment failed: {result['error']}")
        
        results.append(result)
        
        print(f"✅ Completed: {run_id}")
        if 'accuracy' in result:
            accuracy = result['accuracy']
            macro_f1 = result.get('f1_score', 0)  # Use consistent key name
            accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
            macro_f1_str = f"{macro_f1:.4f}" if macro_f1 is not None else "N/A"
            print(f"   Test Accuracy: {accuracy_str}, F1: {macro_f1_str}")
            
            # Show train accuracy if available
            if 'train_accuracy' in result and result['train_accuracy'] is not None:
                train_acc_str = f"{result['train_accuracy']:.4f}"
                train_f1_str = f"{result.get('train_f1_score', 0):.4f}"
                print(f"   Train Accuracy: {train_acc_str}, F1: {train_f1_str}")
    
    # Skip aggregated CSV creation - we're saving individual results only
    print(f"\n💾 Individual model results saved to: {output_dir}")
    print(f"   📁 Each model has separate train/ and test/ folders")
    print(f"   📊 Total experiments completed: {len([r for r in results if 'error' not in r])}")
    print(f"   ❌ Failed experiments: {len([r for r in results if 'error' in r])}")
    
    return results, output_dir


def build_mini_grid_for_fixes():
    """
    Build a mini grid to validate the four fixes:
    1. Macro-F1 pipeline fix
    2. Placement graph semantics shape patch
    3. FiLM with real user metadata
    4. Eval checks for class collapse
    """
    
    experiments = []
    
    # Single seed for mini grid
    seed = 1
    
    # Attention fusion with different featuresets
    attention_configs = [
        {'fusion': 'attention', 'featureset': 'base', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross+semantics', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross+semantics+film', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'base', 'use_focal_loss': True, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro', 'use_focal_loss': True, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross', 'use_focal_loss': True, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross+semantics', 'use_focal_loss': True, 'label_smoothing': 0.05},
        {'fusion': 'attention', 'featureset': 'micro+cross+semantics+film', 'use_focal_loss': True, 'label_smoothing': 0.05},
    ]
    
    # Weighted fusion with micro+cross+semantics
    weighted_configs = [
        {'fusion': 'weighted', 'featureset': 'micro+cross+semantics', 'use_focal_loss': False, 'label_smoothing': 0.05},
        {'fusion': 'weighted', 'featureset': 'micro+cross+semantics', 'use_focal_loss': True, 'label_smoothing': 0.05},
    ]
    
    # Combine all configs
    all_configs = attention_configs + weighted_configs
    
    # Generate experiments
    for config in all_configs:
        exp = {
            'game': 'Game-1',
            'mode': 'multi',
            'placement': 'NA',
            'fusion': config['fusion'],
            'num_classes': 7,
            'seed': seed,
            'featureset': config['featureset'],
            'use_film': 'film' in config['featureset'],
            'use_focal_loss': config['use_focal_loss'],
            'use_label_smoothing': True,  # Always use label smoothing in mini grid
            'label_smoothing': config['label_smoothing'],
            'users_split': 'cross-user',
            'window_size': 100,
            'features': 5 if config['featureset'] == 'base' else 99,  # Approximate
            'epochs': 25,
            'run_id': f"mini-{config['fusion']}-{config['featureset']}-{'focal' if config['use_focal_loss'] else 'ce'}-s{seed}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        experiments.append(exp)
    
    return experiments


def run_mini_grid_for_fixes(output_dir="results/experiments_fixcheck"):
    """
    Run mini grid to validate the four fixes.
    """
    
    print("🔧 Running mini grid to validate fixes...")
    print("=" * 60)
    
    # Build mini grid
    experiments = build_mini_grid_for_fixes()
    
    print(f"📊 Mini grid: {len(experiments)} experiments")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "confusions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failures"), exist_ok=True)
    
    # Run experiments
    results = []
    for i, exp_config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"🏃 Mini experiment {i+1}/{len(experiments)}: {exp_config['run_id']}")
        print(f"{'='*60}")
        
        try:
            # Run single experiment
            result = run_single_experiment(exp_config, output_dir)
            results.append(result)
            
            print(f"✅ Completed: {exp_config['run_id']}")
            if 'accuracy' in result:
                accuracy = result['accuracy']
                macro_f1 = result.get('f1_score', 0)
                accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                macro_f1_str = f"{macro_f1:.4f}" if macro_f1 is not None else "N/A"
                print(f"   Accuracy: {accuracy_str}, F1: {macro_f1_str}")
                if result.get('has_zero_recall', False):
                    print(f"   ⚠️ Zero recall classes: {result.get('zero_recall_classes', [])}")
        
        except Exception as e:
            print(f"❌ Failed: {e}")
            result = {
                'run_id': exp_config['run_id'],
                'error': str(e),
                **exp_config
            }
            results.append(result)
    
    # Create tidy CSV
    tidy_results = []
    for result in results:
        if 'error' in result:
            # Failed experiment
            tidy_results.append({
                'run_id': result['run_id'],
                'seed': result['seed'],
                'game': result.get('game_name', 'unknown'),
                'mode': result['mode'],
                'placement': result['placement'],
                'fusion': result['fusion'],
                'num_classes': result['num_classes'],
                'featureset': result.get('featureset', 'base'),
                'use_film': result.get('use_film', False),
                'use_focal_loss': result.get('use_focal_loss', False),
                'use_label_smoothing': result.get('use_label_smoothing', False),
                'label_smoothing': result.get('label_smoothing', 0.0),
                'users_split': result['users_split'],
                'window_size': result['window_size'],
                'features': result['features'],
                'epochs': result['epochs'],
                'best_epoch': result.get('best_epoch', 0),
                'accuracy': None,
                'macro_f1': None,
                'loss': None,
                'train_time_s': None,
                'eval_time_s': result.get('eval_time_s', 0),
                'artifacts_dir': None,
                'error': result['error'],
                'has_zero_recall': None,
                'zero_recall_classes': None
            })
        else:
            # Successful experiment
            tidy_results.append({
                'run_id': result['run_id'],
                'seed': result['seed'],
                'game': result['game_name'],
                'mode': result['mode'],
                'placement': result['placement'],
                'fusion': result['fusion'],
                'num_classes': result['num_classes'],
                'featureset': result.get('featureset', 'base'),
                'use_film': result.get('use_film', False),
                'use_focal_loss': result.get('use_focal_loss', False),
                'use_label_smoothing': result.get('use_label_smoothing', False),
                'label_smoothing': result.get('label_smoothing', 0.0),
                'users_split': result['users_split'],
                'window_size': result['window_size'],
                'features': result['features'],
                'epochs': result['epochs'],
                'best_epoch': result['best_epoch'],
                'accuracy': result['accuracy'],
                'macro_f1': result.get('f1_score', 0),
                'loss': result.get('loss', 0),
                'train_time_s': result['train_time_s'],
                'eval_time_s': result['eval_time_s'],
                'artifacts_dir': os.path.join(output_dir, 'confusions'),
                'error': None,
                'has_zero_recall': result.get('has_zero_recall', False),
                'zero_recall_classes': result.get('zero_recall_classes', [])
            })
    
    # Save tidy CSV
    results_df = pd.DataFrame(tidy_results)
    results_csv_path = os.path.join(output_dir, 'results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    # Create per-class metrics
    per_class_path = create_per_class_metrics(results, output_dir)
    
    print(f"\n💾 Mini grid results saved:")
    print(f"   📊 Tidy CSV: {results_csv_path}")
    if per_class_path:
        print(f"   📈 Per-class metrics: {per_class_path}")
    print(f"   📁 Confusion matrices: {os.path.join(output_dir, 'confusions')}")
    print(f"   📁 Failures: {os.path.join(output_dir, 'failures')}")
    
    return results, results_csv_path


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="STM Games Comprehensive Experiment Runner")
    parser.add_argument("--games", nargs="+", default=["Game-1"], 
                       help="Games to run (Game-1, Game-2-accel, Game-2-gyro, Game-3)")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to run (advanced_cnn, lstm_attention, multimodal_transformer, hybrid_cnn_transformer, decision_tree, naive_bayes, random_forest, binary_late_fusion, binary_weighted_fusion, binary_sensor_token_attention)")
    parser.add_argument("--grid", choices=["all", "single", "multi", "mini", "mini_v2"], default="all",
                       help="Grid type to run")
    parser.add_argument("--classes", nargs="+", type=int, default=[3, 4, 5, 7],
                       help="Number of classes to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3],
                       help="Random seeds")
    parser.add_argument("--placements", nargs="+", 
                       default=["right-pocket", "right-wrist", "left-wrist", "right-ankle", "left-ankle"],
                       help="Placements for single-sensor experiments")
    parser.add_argument("--fusions", nargs="+", 
                       default=["late", "weighted", "attention"],
                       help="Fusion methods for multi-sensor experiments")
    parser.add_argument("--featuresets", nargs="+",
                       default=["base", "micro", "micro+cross", "micro+cross+semantics", "micro+cross+semantics+film", "semantic-14", "semantic-94", "concatenated"],
                       help="Feature sets to test")
    parser.add_argument("--losses", nargs="+",
                       default=["ce", "focal", "balanced_softmax", "logit_adjust"],
                       help="Loss functions to test")
    parser.add_argument("--label_smoothing", nargs="+", type=float,
                       default=[0.0, 0.05],
                       help="Label smoothing values to test")
    parser.add_argument("--adjacency", nargs="+",
                       default=["identity", "fixed", "learned"],
                       help="Adjacency modes for GraphSemantics")
    parser.add_argument("--gate_alpha_init", nargs="+", type=float,
                       default=[0.1],
                       help="Initial alpha gate values for GraphSemantics")
    parser.add_argument("--hierarchical", nargs="+", type=bool,
                       default=[False, True],
                       help="Whether to use hierarchical heads")
    parser.add_argument("--hier_lambda", nargs="+", type=float,
                       default=[0.3],
                       help="Hierarchical loss weights")
    parser.add_argument("--sampler", nargs="+",
                       default=["random", "class_uniform"],
                       help="Sampling strategies")
    parser.add_argument("--warmup_epochs", nargs="+", type=int,
                       default=[0, 5],
                       help="Warmup epoch counts")
    parser.add_argument("--select_by", choices=["accuracy", "macro_f1"], default="accuracy",
                       help="Metric to use for model selection")
    
    # Optimization parameters
    parser.add_argument("--use_normalization", type=str, default="false",
                       help="Use semantic feature normalization")
    parser.add_argument("--use_attention", type=str, default="false",
                       help="Use attention-based fusion for semantic features")
    parser.add_argument("--use_augmentation", type=str, default="false",
                       help="Use data augmentation")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma parameter")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Focal loss alpha parameter")
    
    parser.add_argument("--output_dir", default="results/experiments",
                       help="Output directory for results")
    parser.add_argument("--dry_run", action="store_true",
                       help="Print experiment grid without running")
    
    args = parser.parse_args()
    
    # Parse optimization parameters
    use_normalization = args.use_normalization.lower() == 'true'
    use_attention = args.use_attention.lower() == 'true'
    use_augmentation = args.use_augmentation.lower() == 'true'
    
    optimization_kwargs = {
        'use_normalization': use_normalization,
        'use_attention': use_attention,
        'use_augmentation': use_augmentation,
        'focal_gamma': args.focal_gamma,
        'focal_alpha': args.focal_alpha
    }
    
    if args.grid == "mini":
        # Run mini grid for validating fixes
        run_mini_grid_for_fixes(args.output_dir)
    elif args.grid == "mini_v2":
        # Run mini grid v2 for imbalance and semantics fixes
        run_experiment_grid(
            grid_type=args.grid,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            **optimization_kwargs
        )
    else:
        # Run regular experiment grid
        run_experiment_grid(
            grid_type=args.grid,
            games=args.games,
            models=args.models,
            classes=args.classes,
            seeds=args.seeds,
            placements=args.placements,
            fusions=args.fusions,
            featuresets=args.featuresets,
            losses=args.losses,
            label_smoothing=args.label_smoothing,
            select_by=args.select_by,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            **optimization_kwargs
        )


if __name__ == "__main__":
    main()
