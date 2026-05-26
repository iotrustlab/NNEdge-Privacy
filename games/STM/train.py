#!/usr/bin/env python3
"""
Main training script for STM games.

This script orchestrates training of all models across all games
with proper checkpointing and result management.
"""

import os
import sys
import argparse
import json
import time
import multiprocessing
import numpy as np
import tensorflow as tf
from games.STM.config import get_data_paths, get_game_config, VALID_ACTIVITIES, WINDOW_SIZE, TRAIN_CONFIGS, DATA_ROOT, RESULTS_ROOT
from games.STM.evaluate import train_all_models_for_game, create_comprehensive_summary

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Define all models to train
NEURAL_MODELS = ["advanced_cnn", "lstm_attention", "multimodal_transformer", "hybrid_cnn_transformer"]
FUSION_MODELS = ["binary_late_fusion", "binary_weighted_fusion", "binary_sensor_token_attention"]
TRADITIONAL_MODELS = ["decision_tree", "naive_bayes", "random_forest"]
ALL_MODELS = NEURAL_MODELS + FUSION_MODELS + TRADITIONAL_MODELS

# Define all games
GAMES = ["Game-1", "Game-1-multi-binary", "Game-2-accel", "Game-2-gyro", "Game-3"]

# Define all splits
SPLITS = ["cross-user", "intra-user", "leave_one_user_out"]


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train all models for STM games")
    parser.add_argument("--games", nargs="+", default=GAMES, 
                       help="Games to train (default: all)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                       help="Models to train (default: all)")
    parser.add_argument("--splits", nargs="+", default=["cross-user"],
                       help="Data splits to use (default: cross-user)")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain even if results exist")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (3 classes only)")
    parser.add_argument("--per-user-norm", action="store_true",
                       help="Use per-user normalization for better cross-user generalization")
    parser.add_argument("--ensemble", action="store_true",
                       help="Create ensemble predictions from multiple models")
    parser.add_argument("--fusion", choices=["late", "weighted", "attention"],
                       help="Fusion method for multi-sensor models (Game-1-multi-binary)")
    parser.add_argument("--placement", choices=["right-wrist", "left-wrist", "right-pocket", "right-ankle", "left-ankle"],
                       help="Specific placement for single-sensor training")
    
    args = parser.parse_args()
    
    # Get data paths
    data_root = DATA_ROOT
    results_root = RESULTS_ROOT
    
    print("🎮 STM Games Training")
    print("=" * 60)
    print(f"📁 Data root: {data_root}")
    print(f"📁 Results root: {results_root}")
    print(f"🎯 Games: {args.games}")
    print(f"🧠 Models: {args.models}")
    print(f"📊 Splits: {args.splits}")
    print(f"🔄 Force retrain: {args.force}")
    print(f"🐛 Debug mode: {args.debug}")
    print(f"🔧 Per-user normalization: {args.per_user_norm}")
    print(f"🎯 Ensemble: {args.ensemble}")
    print(f"🔗 Fusion method: {args.fusion}")
    print(f"📍 Placement: {args.placement}")
    
    # Handle fusion model selection
    if args.fusion:
        fusion_map = {
            "late": "binary_late_fusion",
            "weighted": "binary_weighted_fusion", 
            "attention": "binary_sensor_token_attention"
        }
        fusion_model = fusion_map[args.fusion]
        if "Game-1-multi-binary" not in args.games:
            args.games.append("Game-1-multi-binary")
        # Filter to only train the specified fusion model for Game-1-multi-binary
        if "all" in args.models or len(args.models) > 1:
            args.models = [fusion_model]
    
    # Handle placement-specific training
    if args.placement:
        if "Game-1" not in args.games:
            args.games.append("Game-1")
        # For placement-specific training, we'll use the existing Game-1 path
        # but the data loader will need to be updated to select specific placement
    
    # Set debug mode if requested
    if args.debug:
        from .config import DEBUG_MODE
        DEBUG_MODE = True
        print("🐛 Debug mode enabled - using 3 classes only")
    
    # Train all games
    all_results = {}
    
    for game_name in args.games:
        try:
            game_results = train_all_models_for_game(
                game_name, data_root, results_root, 
                force_retrain=args.force,
                use_per_user_norm=args.per_user_norm,
                create_ensemble=args.ensemble,
                models_to_train=args.models  # Pass the specified models
            )
            all_results[game_name] = game_results
            
        except Exception as e:
            print(f"❌ Error training {game_name}: {e}")
            all_results[game_name] = {"error": str(e)}
    
    # Create comprehensive summary
    create_comprehensive_summary(all_results)
    
    # Save complete results
    timestamp = int(time.time())
    results_file = f"comprehensive_training_results_{timestamp}.json"
    
    # Convert NumPy types to native Python types for JSON serialization
    all_results_serializable = convert_numpy_types(all_results)
    
    with open(results_file, "w") as f:
        json.dump(all_results_serializable, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    print("🎉 Training completed!")


if __name__ == "__main__":
    main() 