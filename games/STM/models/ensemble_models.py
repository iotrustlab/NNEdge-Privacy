"""
Ensemble models for STM games.

This module provides ensemble methods to combine predictions
from multiple models for more robust activity recognition.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns


class EnsemblePredictor:
    """Ensemble predictor that combines multiple model predictions"""
    
    def __init__(self, model_paths, model_types, weights=None):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: List of paths to saved models
            model_types: List of model types ('neural' or 'traditional')
            weights: Optional weights for each model (default: equal weights)
        """
        self.model_paths = model_paths
        self.model_types = model_types
        self.weights = weights if weights is not None else [1.0] * len(model_paths)
        self.models = []
        self.load_models()
    
    def load_models(self):
        """Load all models"""
        print("🔧 Loading ensemble models...")
        
        for i, (model_path, model_type) in enumerate(zip(self.model_paths, self.model_types)):
            try:
                if model_type == "neural":
                    # Load neural network model
                    model = load_model(model_path, compile=False)
                    self.models.append(("neural", model))
                    print(f"  ✅ Loaded neural model: {os.path.basename(model_path)}")
                elif model_type == "traditional":
                    # Load traditional ML model
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    self.models.append(("traditional", model))
                    print(f"  ✅ Loaded traditional model: {os.path.basename(model_path)}")
                else:
                    print(f"  ⚠️ Unknown model type: {model_type}")
                    
            except Exception as e:
                print(f"  ❌ Error loading model {model_path}: {e}")
    
    def predict(self, X):
        """Get ensemble predictions"""
        predictions = []
        
        for (model_type, model) in self.models:
            try:
                if model_type == "neural":
                    # Neural network prediction
                    pred = model.predict(X, verbose=0)
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        # Multi-class: get class probabilities
                        pred_proba = pred
                        pred_class = np.argmax(pred, axis=1)
                    else:
                        # Binary: get probabilities
                        pred_proba = pred
                        pred_class = (pred > 0.5).astype(int)
                else:
                    # Traditional ML prediction
                    pred_class = model.predict(X)
                    pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions.append({
                    'class': pred_class,
                    'proba': pred_proba,
                    'type': model_type
                })
                
            except Exception as e:
                print(f"❌ Error getting prediction from {model_type} model: {e}")
                continue
        
        return predictions
    
    def ensemble_predict(self, X, method="weighted_voting"):
        """
        Get ensemble prediction using specified method
        
        Args:
            X: Input data
            method: Ensemble method ('weighted_voting', 'soft_voting', 'stacking')
        """
        predictions = self.predict(X)
        
        if not predictions:
            raise ValueError("No valid predictions from any model")
        
        if method == "weighted_voting":
            return self._weighted_voting(predictions)
        elif method == "soft_voting":
            return self._soft_voting(predictions)
        elif method == "stacking":
            return self._stacking(predictions, X)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _weighted_voting(self, predictions):
        """Weighted voting ensemble"""
        # Get class predictions and weights
        class_preds = []
        valid_weights = []
        
        for i, pred in enumerate(predictions):
            if pred['class'] is not None:
                class_preds.append(pred['class'])
                valid_weights.append(self.weights[i])
        
        if not class_preds:
            raise ValueError("No valid class predictions")
        
        # Weighted voting
        num_samples = len(class_preds[0])
        num_classes = len(np.unique(class_preds[0]))
        weighted_votes = np.zeros((num_samples, num_classes))
        
        for pred, weight in zip(class_preds, valid_weights):
            for i, class_idx in enumerate(pred):
                weighted_votes[i, class_idx] += weight
        
        # Get majority vote
        ensemble_pred = np.argmax(weighted_votes, axis=1)
        
        return ensemble_pred
    
    def _soft_voting(self, predictions):
        """Soft voting ensemble using probabilities"""
        # Get probability predictions
        proba_preds = []
        valid_weights = []
        
        for i, pred in enumerate(predictions):
            if pred['proba'] is not None:
                proba_preds.append(pred['proba'])
                valid_weights.append(self.weights[i])
        
        if not proba_preds:
            # Fallback to weighted voting if no probabilities
            return self._weighted_voting(predictions)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(proba_preds[0])
        total_weight = sum(valid_weights)
        
        for proba, weight in zip(proba_preds, valid_weights):
            weighted_proba += (weight / total_weight) * proba
        
        # Get class with highest probability
        ensemble_pred = np.argmax(weighted_proba, axis=1)
        
        return ensemble_pred
    
    def _stacking(self, predictions, X):
        """Stacking ensemble using meta-learner"""
        # Get probability predictions for stacking
        proba_features = []
        
        for pred in predictions:
            if pred['proba'] is not None:
                proba_features.append(pred['proba'])
        
        if not proba_features:
            # Fallback to weighted voting if no probabilities
            return self._weighted_voting(predictions)
        
        # Stack probability features
        stacked_features = np.hstack(proba_features)
        
        # Use simple logistic regression as meta-learner
        # Note: In practice, you'd train this on validation data
        meta_learner = LogisticRegression(random_state=42)
        
        # For now, use simple averaging as fallback
        ensemble_pred = np.argmax(np.mean(proba_features, axis=0), axis=1)
        
        return ensemble_pred


def create_ensemble_results(game_name, model_results, X_test, y_test, ensemble_methods=None):
    """
    Create ensemble results from individual model results
    
    Args:
        game_name: Name of the game
        model_results: Dictionary of model results with predictions
        X_test: Test features
        y_test: Test labels
        ensemble_methods: List of ensemble methods to try
    """
    if ensemble_methods is None:
        ensemble_methods = ["weighted_voting", "soft_voting"]
    
    print(f"\n🎯 Creating ensemble results for {game_name}")
    print("=" * 50)
    
    ensemble_results = {}
    
    # Get model paths and types
    model_paths = []
    model_types = []
    model_names = []
    
    for model_name, results in model_results.items():
        if isinstance(results, dict):
            # Check if this is a neural model with model_path
            if "model_path" in results:
                model_paths.append(results["model_path"])
                # Determine if it's neural or traditional based on model name
                if model_name in ["decision_tree", "naive_bayes", "random_forest"]:
                    model_types.append("traditional")
                else:
                    model_types.append("neural")
                model_names.append(model_name)
            # Skip models without model_path
            elif "accuracy" in results:
                print(f"  ⚠️ Model {model_name} has no saved model path, skipping ensemble")
                continue
    
    if len(model_paths) < 2:
        print("⚠️ Need at least 2 neural models with saved model paths for ensemble")
        return ensemble_results
    
    # Create ensemble predictor
    try:
        ensemble = EnsemblePredictor(model_paths, model_types)
        
        # Test each ensemble method
        for method in ensemble_methods:
            print(f"🔧 Testing {method} ensemble...")
            
            try:
                # Get ensemble predictions
                y_pred_ensemble = ensemble.ensemble_predict(X_test, method=method)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred_ensemble)
                f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
                precision = precision_score(y_test, y_pred_ensemble, average='weighted')
                recall = recall_score(y_test, y_pred_ensemble, average='weighted')
                
                ensemble_results[f"ensemble_{method}"] = {
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "method": method,
                    "models": model_names
                }
                
                print(f"  ✅ {method}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with {method}: {e}")
                ensemble_results[f"ensemble_{method}"] = {"error": str(e)}
    
    except Exception as e:
        print(f"❌ Error creating ensemble: {e}")
    
    return ensemble_results


def plot_ensemble_comparison(game_name, model_results, ensemble_results):
    """Plot comparison of individual models vs ensemble"""
    
    # Prepare data for plotting
    model_accuracies = []
    model_names = []
    
    # Individual model accuracies
    for model_name, results in model_results.items():
        if isinstance(results, dict) and "accuracy" in results:
            model_accuracies.append(results["accuracy"])
            model_names.append(model_name)
    
    # Ensemble accuracies
    for ensemble_name, results in ensemble_results.items():
        if isinstance(results, dict) and "accuracy" in results:
            model_accuracies.append(results["accuracy"])
            model_names.append(ensemble_name)
    
    if not model_accuracies:
        return
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(range(len(model_names)), model_accuracies, 
                   color=['skyblue'] * (len(model_names) - len(ensemble_results)) + 
                         ['orange'] * len(ensemble_results))
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(f'Model vs Ensemble Performance - {game_name}')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, model_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "ensemble_plots"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{game_name}_ensemble_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Ensemble comparison plot saved to: {plot_path}")


def save_ensemble_results(ensemble_results, game_name, results_root):
    """Save ensemble results to file"""
    
    results_dir = os.path.join(results_root, game_name, "ensemble")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, "ensemble_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(ensemble_results, f, indent=2)
    
    print(f"💾 Ensemble results saved to: {metrics_path}")
    
    return metrics_path
