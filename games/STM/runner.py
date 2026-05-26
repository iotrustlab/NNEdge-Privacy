#!/usr/bin/env python3
"""
Modular runner for STM games training and evaluation.
Can be used both from CLI and programmatically for experiments.
"""

import os
import sys
import time
import json
import argparse
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import tensorflow as tf

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

from games.STM.config import GAME_CONFIGS, DATA_ROOT, RESULTS_ROOT
from games.STM.data_loader import load_data_for_game, prepare_cross_user_data
from games.STM.evaluate import train_all_models_for_game
from games.STM.models.neural_models import train_and_evaluate_advanced_model
from games.STM.models.traditional_models import train_traditional_model
from games.STM.models.ensemble_models import create_ensemble_results


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def train_and_eval_single_run(
    game_name: str,
    model_type: str,
    split_name: str = "cross-user",
    use_per_user_norm: bool = True,
    epochs: int = 25,
    batch_size: int = 128,
    seed: int = 42,
    featureset: str = 'base',
    use_film: bool = False,
    use_focal_loss: bool = False,
    use_label_smoothing: bool = False,
    label_smoothing: float = 0.0,
    loss_type: str = 'ce',
    adjacency_mode: str = 'fixed',
    gate_alpha_init: float = 0.1,
    hierarchical: bool = False,
    hier_lambda: float = 0.3,
    sampler: str = 'random',
    warmup_epochs: int = 0,
    select_by: str = 'accuracy',
    use_normalization: bool = False,
    use_attention: bool = False,
    use_augmentation: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    placement: str = 'all',
    num_classes: int = None,
    grid_type: str = 'single',
    **kwargs
) -> Dict[str, Any]:
    """
    Train and evaluate a single model run.
    
    Args:
        game_name: Name of the game (e.g., "Game-1", "Game-1-multi-binary")
        model_type: Type of model to train
        split_name: Data split to use
        use_per_user_norm: Whether to use per-user normalization
        epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
        featureset: Feature set to use
        use_film: Whether to use FiLM conditioning
        use_focal_loss: Whether to use focal loss
        use_label_smoothing: Whether to use label smoothing
        label_smoothing: Label smoothing factor
        loss_type: Loss function type ('ce', 'focal', 'balanced_softmax', 'logit_adjust')
        adjacency_mode: Adjacency mode for GraphSemantics ('identity', 'fixed', 'learned')
        gate_alpha_init: Initial alpha gate value for GraphSemantics
        hierarchical: Whether to use hierarchical head
        hier_lambda: Hierarchical loss weight
        sampler: Sampling strategy ('random', 'class_uniform')
        warmup_epochs: Number of warmup epochs
        select_by: Metric to use for model selection
        grid_type: Grid type from experiments.py ('single', 'multi', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing results and metadata
    """
    start_time = time.time()
    
    # Set random seed
    np.random.seed(seed)
    
    # Load data
    print(f"📂 Loading data for {game_name}...")
    user_data, all_labels = load_data_for_game(game_name, DATA_ROOT, num_classes, grid_type)
    
    if not user_data:
        raise ValueError(f"No data found for {game_name}")
    
    # Prepare data split
    if split_name == "cross-user":
        # Split users for cross-user experiment (80% train, 20% test)
        user_ids = list(user_data.keys())
        np.random.shuffle(user_ids)
        split_idx = int(0.8 * len(user_ids))
        train_users = user_ids[:split_idx]
        test_users = user_ids[split_idx:]
        
        print(f"📊 Cross-user split: {len(train_users)} train users, {len(test_users)} test users")
        
        X_train, y_train, X_test, y_test, scaler, train_user_metadata, test_user_metadata = prepare_cross_user_data(
            user_data, train_users, test_users, use_per_user_norm=use_per_user_norm, 
            game_name=game_name, featureset=featureset, use_summary_token=True,
            use_normalization=use_normalization, use_attention=use_attention, use_augmentation=use_augmentation
        )
    else:
        raise ValueError(f"Split {split_name} not implemented yet")
    
    # Determine model category
    neural_models = ["advanced_cnn", "lstm_attention", "multimodal_transformer", "hybrid_cnn_transformer"]
    fusion_models = ["binary_late_fusion", "binary_weighted_fusion", "binary_sensor_token_attention", "continuous_late_fusion", "continuous_weighted_fusion"]
    traditional_models = ["decision_tree", "naive_bayes", "random_forest"]
    
    # Train model
    print(f"🧠 Training {model_type} on {game_name}...")
    
    if model_type in neural_models or model_type in fusion_models:
        # Neural network model
        (y_pred, accuracy, y_train_pred, train_accuracy, train_f1, train_precision, train_recall,
         y_train_processed, X_train_processed, history, best_epoch_info, has_zero_recall, zero_recall_classes) = train_and_evaluate_advanced_model(
            X_train, y_train, X_test, y_test, model_type,
            game_name, split_name,
            use_class_weights=False,
            use_per_user_norm=use_per_user_norm,
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
            train_user_metadata=train_user_metadata if use_film else None,
            test_user_metadata=test_user_metadata if use_film else None,
            placement=placement
        )
        
        # Calculate F1 score from predictions
        from sklearn.metrics import f1_score
        f1_score = f1_score(y_test, y_pred, average='macro')
        
        # Extract other metrics
        loss = best_epoch_info.get('loss', 0.0)
        best_epoch = best_epoch_info.get('epoch', 0)
        
    elif model_type in traditional_models:
        # Traditional ML model
        model_results = train_traditional_model(
            model_type, X_train, y_train, X_test, y_test
        )
        
        # Extract results from the returned dictionary
        y_pred = model_results["predictions"]
        accuracy = model_results["accuracy"]
        f1_score = model_results["f1_score"]
        best_epoch_info = {"epoch": 0, "loss": 0.0}  # Traditional models don't have epochs/loss
        
        # Extract train predictions
        y_train_pred = model_results["train_predictions"]
        train_accuracy = model_results["train_accuracy"]
        train_f1_score = model_results["train_f1_score"]
        
        loss = 0.0  # Traditional models don't have loss in the same sense
        best_epoch = 0
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    train_time = time.time() - start_time
    
    # Prepare results
    results = {
        'game_name': game_name,
        'model_type': model_type,
        'split_name': split_name,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'loss': loss,
        'best_epoch': best_epoch,
        'train_time_s': train_time,
        'input_shape': X_train.shape if hasattr(X_train, 'shape') else None,
        'num_classes': len(np.unique(y_train)),
        'num_samples': len(X_train) + len(X_test),
        # Test results
        'test_predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        'test_true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
        # Train results 
        'train_predictions': (y_train_pred.tolist() if hasattr(y_train_pred, 'tolist') else y_train_pred) if (model_type in traditional_models or model_type in neural_models or model_type in fusion_models) else None,
        'train_true_labels': (y_train_processed.tolist() if hasattr(y_train_processed, 'tolist') else y_train_processed) if (model_type in neural_models or model_type in fusion_models) else (y_train.tolist() if hasattr(y_train, 'tolist') else y_train),
        'train_accuracy': train_accuracy,
        'train_f1_score': train_f1_score if model_type in traditional_models else (train_f1 if model_type in neural_models or model_type in fusion_models else None),
        'train_precision': model_results.get("train_precision", None) if model_type in traditional_models else (train_precision if model_type in neural_models or model_type in fusion_models else None),
        'train_recall': model_results.get("train_recall", None) if model_type in traditional_models else (train_recall if model_type in neural_models or model_type in fusion_models else None),
        # Legacy compatibility
        'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
        'true_labels': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
        # Meta
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'use_per_user_norm': use_per_user_norm,
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
        'has_zero_recall': has_zero_recall if model_type in neural_models or model_type in fusion_models else False,
        'zero_recall_classes': zero_recall_classes if model_type in neural_models or model_type in fusion_models else []
    }
    
    return results


def run_experiment_grid(
    games: List[str],
    models: List[str],
    splits: List[str] = ["cross-user"],
    seeds: List[int] = [42],
    epochs: int = 25,
    batch_size: int = 128,
    use_per_user_norm: bool = True,
    output_dir: str = "results/experiments",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Run a grid of experiments.
    
    Args:
        games: List of game names to run
        models: List of model types to run
        splits: List of data splits to use
        seeds: List of random seeds
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_per_user_norm: Whether to use per-user normalization
        output_dir: Directory to save results
        **kwargs: Additional arguments
        
    Returns:
        List of result dictionaries
    """
    all_results = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    total_runs = len(games) * len(models) * len(splits) * len(seeds)
    current_run = 0
    
    print(f"🚀 Starting experiment grid with {total_runs} total runs")
    print(f"📁 Output directory: {output_dir}")
    
    for game in games:
        for model in models:
            for split in splits:
                for seed in seeds:
                    current_run += 1
                    run_id = f"{game}-{model}-{split}-s{seed}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    print(f"\n{'='*60}")
                    print(f"🏃 Run {current_run}/{total_runs}: {run_id}")
                    print(f"{'='*60}")
                    
                    try:
                        results = train_and_eval_single_run(
                            game_name=game,
                            model_type=model,
                            split_name=split,
                            use_per_user_norm=use_per_user_norm,
                            epochs=epochs,
                            batch_size=batch_size,
                            seed=seed,
                            **kwargs
                        )
                        
                        # Add run metadata
                        results['run_id'] = run_id
                        results['timestamp'] = datetime.now().isoformat()
                        
                        all_results.append(results)
                        
                        print(f"✅ Run completed: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
                        
                    except Exception as e:
                        print(f"❌ Run failed: {e}")
                        # Add failed run record
                        failed_result = {
                            'run_id': run_id,
                            'game_name': game,
                            'model_type': model,
                            'split_name': split,
                            'seed': seed,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        all_results.append(failed_result)
    
    # Save results
    results_file = os.path.join(output_dir, f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(all_results), f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print(f"📊 Total runs: {len(all_results)}")
    
    return all_results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="STM Games Experiment Runner")
    parser.add_argument("--games", nargs="+", default=["Game-1"], help="Games to run")
    parser.add_argument("--models", nargs="+", default=["advanced_cnn"], help="Models to run")
    parser.add_argument("--splits", nargs="+", default=["cross-user"], help="Data splits to use")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--per-user-norm", action="store_true", help="Use per-user normalization")
    parser.add_argument("--output-dir", default="results/experiments", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    
    args = parser.parse_args()
    
    # Run the experiment grid
    results = run_experiment_grid(
        games=args.games,
        models=args.models,
        splits=args.splits,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_per_user_norm=args.per_user_norm,
        output_dir=args.output_dir
    )
    
    print(f"\n🎉 Experiment grid completed with {len(results)} runs!")


if __name__ == "__main__":
    main()
