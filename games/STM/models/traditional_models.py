"""
Traditional ML models for STM games.

This module implements Decision Tree, Random Forest, and Naive Bayes models
for privacy leakage evaluation using statistical features extracted from time series data.
"""

import os
import json
import time
import numpy as np
import itertools
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def extract_statistical_features(window):
    """Extract statistical features from time series window for traditional ML"""
    # Flatten the window
    if len(window.shape) == 3:  # (1, 100, features)
        window = window.squeeze(0)
    elif len(window.shape) == 2:  # (100, features)
        pass
    else:
        window = window.flatten()
    
    # For binary data (Game-1), extract binary-specific features
    if window.shape[1] == 1:  # Binary data
        binary_series = window.flatten()
        features = [
            np.mean(binary_series),           # Mean (proportion of 1s)
            np.std(binary_series),            # Standard deviation
            np.min(binary_series),            # Minimum
            np.max(binary_series),            # Maximum
            np.sum(binary_series),            # Total count of 1s
            np.sum(binary_series) / len(binary_series),  # Proportion of 1s
            # Transition features
            np.sum(np.diff(binary_series) != 0),  # Number of transitions
            np.sum(np.diff(binary_series) == 1),  # Number of 0->1 transitions
            np.sum(np.diff(binary_series) == -1), # Number of 1->0 transitions
            # Run length features
            np.mean([len(list(g)) for k, g in itertools.groupby(binary_series)]),  # Average run length
            np.max([len(list(g)) for k, g in itertools.groupby(binary_series)]),   # Max run length
            np.min([len(list(g)) for k, g in itertools.groupby(binary_series)]),   # Min run length
        ]
    else:
        # For multi-feature data (Game-2, Game-3), extract statistical features per feature
        features = []
        for i in range(window.shape[1]):
            feature_series = window[:, i]
            features.extend([
                np.mean(feature_series),
                np.std(feature_series),
                np.min(feature_series),
                np.max(feature_series),
                np.percentile(feature_series, 25),
                np.percentile(feature_series, 75),
            ])
    
    return np.array(features)


def train_traditional_model(model_type, X_train, y_train, X_test, y_test):
    """Train and evaluate a traditional ML model"""
    
    print(f"🧠 Training {model_type}...")
    
    # Convert time series data to statistical features
    X_train_features = np.array([extract_statistical_features(window) for window in X_train])
    X_test_features = np.array([extract_statistical_features(window) for window in X_test])
    
    # Initialize model
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    elif model_type == "naive_bayes":
        model = GaussianNB()
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train model
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   ✅ {model_type}: {accuracy:.4f} (F1: {f1:.4f})")
    print(f"   ⏱️ Training time: {training_time:.2f}s")
    
    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "training_time": training_time,
        "confusion_matrix": cm,
        "predictions": y_pred
    }


def save_traditional_results(model_results, model_type, game_name, split_name, results_root, featureset='base', placement='all'):
    """Save traditional ML model results"""
    
    # Create results directory with configuration details to prevent overwrites
    config_suffix = f"{placement}_{featureset}"
    results_dir = os.path.join(results_root, game_name, model_type, split_name, config_suffix)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(results_dir, f"{model_type}_model.pkl")
    scaler_path = os.path.join(results_dir, f"{model_type}_scaler.pkl")
    
    joblib.dump(model_results["model"], model_path)
    joblib.dump(model_results["scaler"], scaler_path)
    
    # Save metrics
    metrics = {
        "accuracy": float(model_results["accuracy"]),
        "f1_score": float(model_results["f1_score"]),
        "precision": float(model_results["precision"]),
        "recall": float(model_results["recall"]),
        "training_time": float(model_results["training_time"]),
        "model_type": model_type,
        "game_name": game_name,
        "split_name": split_name,
        "model_path": model_path,  # Add model path for ensemble use
        "scaler_path": scaler_path  # Add scaler path for ensemble use
    }
    
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix_traditional(
        model_results["confusion_matrix"], 
        model_type, game_name, split_name
    )
    
    print(f"   💾 Results saved to: {results_dir}")
    
    return metrics


def plot_confusion_matrix_traditional(cm, model_type, game_name, split_name):
    """Plot confusion matrix for traditional models"""
    
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_type} ({game_name}, {split_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        plot_dir = "confusion_matrices_traditional"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{model_type}_{game_name}_{split_name}_cm.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Confusion matrix saved to: {plot_path}")
        
    except Exception as e:
        print(f"   ⚠️ Error plotting confusion matrix: {e}") 