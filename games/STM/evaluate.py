"""
Evaluation utilities for STM games.

This module provides functions for evaluating model performance
and generating comprehensive results.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from games.STM.models.neural_models import train_and_evaluate_advanced_model, save_advanced_results
from games.STM.models.traditional_models import train_traditional_model, save_traditional_results


def check_existing_results(model_type, game_name, split_name, results_root, featureset='base', placement='all'):
    """Check if results already exist for this experiment"""
    config_suffix = f"{placement}_{featureset}"
    results_dir = os.path.join(results_root, game_name, model_type, split_name, config_suffix)
    
    # Check for metrics.json
    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            print(f"   ✅ Found existing results: {metrics.get('accuracy', 'N/A'):.4f}")
            return True, metrics
        except:
            pass
    
    return False, None


def estimate_time_per_model(data_size, num_classes, model_type):
    """Estimate training time for a model"""
    base_time = 6  # minutes for GPU training
    
    # Model complexity factors
    complexity_factors = {
        "advanced_cnn": 1.0,
        "lstm_attention": 1.2,
        "multimodal_transformer": 1.5,
        "hybrid_cnn_transformer": 1.3,
        "decision_tree": 0.01,
        "naive_bayes": 0.005,
        "random_forest": 0.05
    }
    
    factor = complexity_factors.get(model_type, 1.0)
    estimated_time = base_time * factor * (data_size / 10000) * (num_classes / 7)
    
    return estimated_time


def train_all_models_for_game(game_name, data_root, results_root, force_retrain=False, use_per_user_norm=False, create_ensemble=False, models_to_train=None, featureset='base', use_film=False, use_focal_loss=False, use_label_smoothing=False, label_smoothing=0.0):
    """Train all models for a specific game with optional per-user normalization and ensemble creation"""
    
    print(f"\n🎮 Training all models for {game_name}")
    if use_per_user_norm:
        print("🔧 Using per-user normalization for better cross-user generalization")
    if create_ensemble:
        print("🎯 Will create ensemble predictions from multiple models")
    print("=" * 60)
    
    # Load data
    from .data_loader import load_data_for_game, prepare_cross_user_data
    
    user_data, all_labels = load_data_for_game(game_name, data_root)
    
    if not user_data:
        print(f"❌ No data found for {game_name}")
        return {}
    
    # Get user list
    users = list(user_data.keys())
    print(f"👥 Users: {users}")
    
    # For now, use cross-user split (train on first 7 users, test on last 3)
    train_users = users[:7]
    test_users = users[7:]
    
    if len(test_users) == 0:
        print("⚠️ Not enough users for cross-user split, using last user as test")
        train_users = users[:-1]
        test_users = users[-1:]
    
    print(f"📚 Train users: {train_users}")
    print(f"🧪 Test users: {test_users}")
    
    # Prepare data with optional per-user normalization
    X_train, y_train, X_test, y_test, label_encoder, train_user_metadata, test_user_metadata = prepare_cross_user_data(
        user_data, train_users, test_users, 
        use_per_user_norm=use_per_user_norm, 
        game_name=game_name
    )
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"📊 Classes: {len(np.unique(y_train))}")
    
    # Define models to train - FIXED: Use models_to_train parameter if provided
    neural_models = ["advanced_cnn", "lstm_attention", "multimodal_transformer", "hybrid_cnn_transformer"]
    fusion_models = ["binary_late_fusion", "binary_weighted_fusion", "binary_sensor_token_attention"]
    traditional_models = ["decision_tree", "naive_bayes", "random_forest"]
    all_models = neural_models + fusion_models + traditional_models
    
    # Use specified models if provided, otherwise use all models
    if models_to_train is not None:
        # Filter to only requested models
        models_to_train = [m for m in models_to_train if m in all_models]
        if not models_to_train:
            print(f"❌ No valid models specified for {game_name}")
            return {}
        print(f"🎯 Training only specified models: {models_to_train}")
    else:
        models_to_train = all_models
    
    # Estimate total time
    total_time = sum([estimate_time_per_model(len(X_train), len(np.unique(y_train)), model) 
                     for model in models_to_train])
    print(f"⏱️ Estimated total time: {total_time:.1f} minutes")
    
    # Train each model
    results = {}
    split_name = "cross-user"
    
    for model_type in models_to_train:
        print(f"\n🧠 Training {model_type}...")
        
        # Check if results already exist
        if not force_retrain:
            exists, existing_metrics = check_existing_results(model_type, game_name, split_name, results_root, featureset, 'all')
            if exists:
                results[model_type] = existing_metrics
                continue
        
        try:
            if model_type in neural_models or model_type in fusion_models:
                # Train neural network model with per-user normalization
                y_pred, accuracy, history, best_epoch_info = train_and_evaluate_advanced_model(
                    X_train, y_train, X_test, y_test, model_type, 
                    game_name, split_name, 
                    use_class_weights=False,
                    use_per_user_norm=use_per_user_norm,
                    featureset=featureset,
                    use_film=use_film,
                    use_focal_loss=use_focal_loss,
                    use_label_smoothing=use_label_smoothing,
                    label_smoothing=label_smoothing
                )
                
                # Save results
                from .models.neural_models import save_advanced_results
                results[model_type] = save_advanced_results(
                    y_pred, y_test, accuracy, None, history, model_type, 
                    game_name, split_name, results_root, best_epoch_info,
                    featureset, use_film, use_focal_loss, use_label_smoothing, label_smoothing, 'all'
                )
                
            else:
                # Train traditional ML model
                model_results = train_traditional_model(model_type, X_train, y_train, X_test, y_test)
                save_traditional_results(model_results, model_type, game_name, split_name, results_root, featureset, 'all')
                results[model_type] = {
                    "accuracy": model_results["accuracy"],
                    "f1_score": model_results["f1_score"],
                    "precision": model_results["precision"],
                    "recall": model_results["recall"],
                    "training_time": model_results["training_time"]
                }
                
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    # Create ensemble results if we have at least 2 successful models
    successful_models = {k: v for k, v in results.items() 
                        if isinstance(v, dict) and "accuracy" in v and "error" not in v}
    
    if len(successful_models) >= 2 and create_ensemble:
        print(f"\n🎯 Creating ensemble from {len(successful_models)} models...")
        
        try:
            from .models.ensemble_models import create_ensemble_results, plot_ensemble_comparison, save_ensemble_results
            
            ensemble_results = create_ensemble_results(
                game_name, successful_models, X_test, y_test
            )
            
            if ensemble_results:
                # Add ensemble results to main results
                results.update(ensemble_results)
                
                # Create ensemble comparison plot
                plot_ensemble_comparison(game_name, successful_models, ensemble_results)
                
                # Save ensemble results
                save_ensemble_results(ensemble_results, game_name, results_root)
                
                print(f"✅ Ensemble created with {len(ensemble_results)} methods")
            else:
                print("⚠️ No ensemble results created")
                
        except Exception as e:
            print(f"❌ Error creating ensemble: {e}")
    else:
        print(f"⚠️ Need at least 2 successful models for ensemble (got {len(successful_models)}) or ensemble creation is off.")
    
    print(f"\n✅ {game_name} completed successfully")
    return results


def create_comprehensive_summary(all_results):
    """Create a comprehensive summary of all results"""
    
    print("\n📊 COMPREHENSIVE TRAINING SUMMARY")
    print("=" * 60)
    
    # Create summary DataFrame
    summary_data = []
    
    for game_name, game_results in all_results.items():
        for model_name, model_results in game_results.items():
            if isinstance(model_results, dict) and "accuracy" in model_results:
                summary_data.append({
                    "Game": game_name,
                    "Model": model_name,
                    "Accuracy": model_results.get("accuracy", 0),
                    "F1_Score": model_results.get("f1_score", 0),
                    "Precision": model_results.get("precision", 0),
                    "Recall": model_results.get("recall", 0),
                    "Training_Time": model_results.get("training_time", 0)
                })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Print summary statistics
        print("\n🏆 Best Model per Game:")
        for game in df["Game"].unique():
            game_df = df[df["Game"] == game]
            best_model = game_df.loc[game_df["Accuracy"].idxmax()]
            print(f"  {game}: {best_model['Model']} ({best_model['Accuracy']:.4f})")
        
        print("\n📈 Average Performance by Model:")
        model_avg = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
        for model, acc in model_avg.items():
            print(f"  {model}: {acc:.4f}")
        
        print("\n🎯 Performance Improvement (Game-1 to Game-3):")
        for model in df["Model"].unique():
            model_df = df[df["Model"] == model]
            if len(model_df) >= 2:
                game1_acc = model_df[model_df["Game"] == "Game-1"]["Accuracy"].iloc[0] if "Game-1" in model_df["Game"].values else 0
                game3_acc = model_df[model_df["Game"] == "Game-3"]["Accuracy"].iloc[0] if "Game-3" in model_df["Game"].values else 0
                improvement = game3_acc - game1_acc
                print(f"  {model}: {improvement:+.4f}")
        
        # Save summary
        timestamp = int(time.time())
        summary_file = f"comprehensive_training_summary_{timestamp}.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to: {summary_file}")
        
        return df
    else:
        print("❌ No valid results found")
        return None 