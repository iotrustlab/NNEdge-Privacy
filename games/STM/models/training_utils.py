"""
Training utilities for STM games.

Implements fixes for macro-F1 collapse including focal loss, label smoothing,
class weights, and balanced sampling.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss reduces the relative loss for well-classified examples
    and puts more focus on hard, misclassified examples.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, 
                 from_logits: bool = False, name: str = "focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
        
        Returns:
            Focal loss tensor
        """
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Cross-entropy loss
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Focal loss component
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * tf.reduce_sum(ce_loss, axis=-1)
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config


class LabelSmoothingLoss(keras.losses.Loss):
    """
    Loss function with label smoothing.
    
    Applies label smoothing to prevent overconfidence and improve generalization.
    """
    
    def __init__(self, smoothing: float = 0.05, from_logits: bool = False, 
                 name: str = "label_smoothing_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smoothing = smoothing
        self.from_logits = from_logits
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
        
        Returns:
            Label smoothing loss tensor
        """
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Apply label smoothing
        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        smoothing = tf.cast(self.smoothing, tf.float32)
        smooth_labels = y_true * (1 - smoothing) + smoothing / num_classes
        
        # Cross-entropy loss
        loss = -tf.reduce_sum(smooth_labels * tf.math.log(y_pred), axis=-1)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'smoothing': self.smoothing,
            'from_logits': self.from_logits
        })
        return config


class BalancedSoftmaxLoss(keras.losses.Loss):
    """
    Balanced Softmax Loss for addressing class imbalance.
    
    Replaces logits z by z' = z + log(π) where π are class priors (probabilities).
    Uses standard softmax CE on z'.
    """
    
    def __init__(self, class_priors: np.ndarray, from_logits: bool = True, 
                 name: str = "balanced_softmax_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_priors = tf.constant(class_priors, dtype=tf.float32)
        self.from_logits = from_logits
        
        # Log of class priors
        self.log_priors = tf.math.log(tf.clip_by_value(self.class_priors, 1e-7, 1.0))
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute balanced softmax loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted logits
        
        Returns:
            Balanced softmax loss tensor
        """
        # Adjust logits with class priors: z' = z + log(π)
        adjusted_logits = y_pred + self.log_priors
        
        # Use standard cross-entropy with softmax
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, 
            logits=adjusted_logits
        )
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'class_priors': self.class_priors.numpy(),
            'from_logits': self.from_logits
        })
        return config


class LogitAdjustedLoss(keras.losses.Loss):
    """
    Logit-Adjusted Loss using effective number of samples.
    
    Computes weights α_y = (1 - β) / (1 - β^{n_y}) with β=0.9995 and counts n_y.
    Adds log α to logits of class y.
    """
    
    def __init__(self, class_counts: np.ndarray, beta: float = 0.9995, 
                 from_logits: bool = True, name: str = "logit_adjusted_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta
        self.from_logits = from_logits
        
        # Compute effective number weights
        self.effective_weights = self._compute_effective_weights(class_counts)
        self.log_weights = tf.math.log(tf.clip_by_value(self.effective_weights, 1e-7, 1.0))
    
    def _compute_effective_weights(self, class_counts: np.ndarray) -> tf.Tensor:
        """Compute effective number weights for each class."""
        counts = tf.constant(class_counts, dtype=tf.float32)
        beta = tf.constant(self.beta, dtype=tf.float32)
        
        # α_y = (1 - β) / (1 - β^{n_y})
        effective_weights = (1 - beta) / (1 - tf.pow(beta, counts))
        
        return effective_weights
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute logit-adjusted loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted logits
        
        Returns:
            Logit-adjusted loss tensor
        """
        # Adjust logits with effective weights: z'_k = z_k + log(α_k)
        adjusted_logits = y_pred + self.log_weights
        
        # Use standard cross-entropy with softmax
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, 
            logits=adjusted_logits
        )
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'class_counts': self.effective_weights.numpy(),
            'beta': self.beta,
            'from_logits': self.from_logits
        })
        return config


def compute_class_priors(y_train: np.ndarray) -> np.ndarray:
    """
    Compute class priors (probabilities) from training data.
    
    Args:
        y_train: Training labels (integer or one-hot encoded)
    
    Returns:
        Array of class priors (probabilities)
    """
    # Convert to integer labels if one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Count classes
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    
    # Compute priors
    num_classes = len(unique)
    priors = np.zeros(num_classes)
    
    for cls, count in zip(unique, counts):
        priors[cls] = count / total
    
    return priors


def compute_class_counts(y_train: np.ndarray) -> np.ndarray:
    """
    Compute class counts from training data.
    
    Args:
        y_train: Training labels (integer or one-hot encoded)
    
    Returns:
        Array of class counts
    """
    # Convert to integer labels if one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Count classes
    unique, counts = np.unique(y_train, return_counts=True)
    
    # Create counts array
    num_classes = len(unique)
    class_counts = np.zeros(num_classes)
    
    for cls, count in zip(unique, counts):
        class_counts[cls] = count
    
    return class_counts


def compute_class_weights(y_train: np.ndarray, method: str = 'balanced') -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels (integer encoded)
        method: Weighting method ('balanced', 'balanced_subsample', or None)
    
    Returns:
        Dictionary mapping class indices to weights
    """
    if method is None:
        return {}
    
    # Convert to integer labels if one-hot encoded
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
    
    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight=method,
        classes=classes,
        y=y_train
    )
    
    # Create dictionary
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    
    return class_weights


class BalancedBatchSampler:
    """
    Balanced batch sampler for addressing class imbalance.
    
    Ensures each batch contains samples from all classes.
    """
    
    def __init__(self, y: np.ndarray, batch_size: int, samples_per_class: int = 1):
        """
        Initialize balanced batch sampler.
        
        Args:
            y: Labels (integer encoded)
            batch_size: Batch size
            samples_per_class: Number of samples per class per batch
        """
        self.y = y
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # Convert to integer labels if one-hot encoded
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.y = np.argmax(y, axis=1)
        
        # Group indices by class
        self.class_indices = {}
        for class_idx in np.unique(self.y):
            self.class_indices[class_idx] = np.where(self.y == class_idx)[0]
        
        self.num_classes = len(self.class_indices)
        self.classes = list(self.class_indices.keys())
    
    def __iter__(self):
        """Generate balanced batches."""
        while True:
            batch_indices = []
            
            # Sample from each class
            for _ in range(self.samples_per_class):
                for class_idx in self.classes:
                    if len(self.class_indices[class_idx]) > 0:
                        # Randomly sample from this class
                        idx = np.random.choice(self.class_indices[class_idx])
                        batch_indices.append(idx)
            
            # If we don't have enough samples, pad with random samples
            while len(batch_indices) < self.batch_size:
                class_idx = np.random.choice(self.classes)
                if len(self.class_indices[class_idx]) > 0:
                    idx = np.random.choice(self.class_indices[class_idx])
                    batch_indices.append(idx)
            
            # Trim to batch size
            batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        """Number of batches per epoch."""
        return len(self.y) // self.batch_size


class ClassUniformSampler:
    """
    Class-uniform sampler for windowed sequences.
    
    Approximates equal class counts per batch, falling back to random if not feasible.
    """
    
    def __init__(self, y: np.ndarray, batch_size: int, window_size: int = 100):
        """
        Initialize class-uniform sampler.
        
        Args:
            y: Labels (integer encoded)
            batch_size: Batch size
            window_size: Window size for sequences
        """
        self.y = y
        self.batch_size = batch_size
        self.window_size = window_size
        
        # Convert to integer labels if one-hot encoded
        if len(y.shape) > 1 and y.shape[1] > 1:
            self.y = np.argmax(y, axis=1)
        
        # Group indices by class
        self.class_indices = {}
        for class_idx in np.unique(self.y):
            self.class_indices[class_idx] = np.where(self.y == class_idx)[0]
        
        self.num_classes = len(self.class_indices)
        self.classes = list(self.class_indices.keys())
        
        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
    
    def __iter__(self):
        """Generate class-uniform batches."""
        while True:
            batch_indices = []
            
            # Try to sample equally from each class
            for class_idx in self.classes:
                if len(self.class_indices[class_idx]) > 0:
                    # Sample from this class
                    num_samples = min(self.samples_per_class, len(self.class_indices[class_idx]))
                    indices = np.random.choice(self.class_indices[class_idx], 
                                             size=num_samples, replace=False)
                    batch_indices.extend(indices)
            
            # If we don't have enough samples, pad with random samples
            while len(batch_indices) < self.batch_size:
                class_idx = np.random.choice(self.classes)
                if len(self.class_indices[class_idx]) > 0:
                    idx = np.random.choice(self.class_indices[class_idx])
                    batch_indices.append(idx)
            
            # Trim to batch size
            batch_indices = batch_indices[:self.batch_size]
            
            yield batch_indices
    
    def __len__(self):
        """Number of batches per epoch."""
        return len(self.y) // self.batch_size


class MacroF1Callback(keras.callbacks.Callback):
    """
    Callback for monitoring and early stopping based on macro-F1.
    """
    
    def __init__(self, validation_data: Tuple[np.ndarray, np.ndarray], 
                 patience: int = 10, restore_best_weights: bool = True):
        super().__init__()
        self.validation_data = validation_data
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_macro_f1 = 0.0
        self.wait = 0
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Evaluate macro-F1 at end of epoch."""
        X_val, y_val = self.validation_data
        
        # Get predictions
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Handle hierarchical models (multiple outputs)
        if isinstance(y_pred, list):
            # For hierarchical models, use the fine-grained predictions (first output)
            y_pred_classes = np.argmax(y_pred[0], axis=1)
        else:
            # For single-output models
            y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Convert y_val to classes if one-hot encoded
        if isinstance(y_val, list):
            # For hierarchical models, use the fine-grained targets (first output)
            y_val_fine = y_val[0]
            if len(y_val_fine.shape) > 1 and y_val_fine.shape[1] > 1:
                y_val_classes = np.argmax(y_val_fine, axis=1)
            else:
                y_val_classes = y_val_fine
        else:
            # For single-output models
            if len(y_val.shape) > 1 and y_val.shape[1] > 1:
                y_val_classes = np.argmax(y_val, axis=1)
            else:
                y_val_classes = y_val
        
        # Compute macro-F1
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(y_val_classes, y_pred_classes, average='macro')
        
        # Log the metric
        logs = logs or {}
        logs['val_macro_f1'] = macro_f1
        
        # Early stopping logic
        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(f"\nEarly stopping triggered. Best macro-F1: {self.best_macro_f1:.4f}")


def create_training_callbacks(validation_data: Tuple[np.ndarray, np.ndarray],
                            monitor: str = 'val_macro_f1',
                            patience: int = 10,
                            restore_best_weights: bool = True,
                            model=None) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks with macro-F1 monitoring.
    
    Args:
        validation_data: Validation data tuple (X_val, y_val)
        monitor: Metric to monitor ('val_macro_f1' or 'val_accuracy')
        patience: Early stopping patience
        restore_best_weights: Whether to restore best weights
        model: Model instance for GraphSemantics logging
    
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Early stopping
    if monitor == 'val_macro_f1':
        callbacks.append(MacroF1Callback(validation_data, patience, restore_best_weights))
    else:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=1
        ))
    
    # Reduce learning rate on plateau
    mode = 'max' if monitor == 'val_macro_f1' else 'min'
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-7,
        verbose=1
    ))
    
    # Model checkpoint
    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras.h5',
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        verbose=1
    ))
    
    # Add GraphSemantics logger if model is provided
    if model is not None:
        callbacks.append(GraphSemanticsLogger(model))
    
    return callbacks


class GraphSemanticsLogger(keras.callbacks.Callback):
    """
    Callback to log GraphSemantics alpha gate values during training.
    """
    
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self.alpha_values = []
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
    
    def on_epoch_end(self, epoch, logs=None):
        """Log alpha gate values at the end of each epoch."""
        alpha_values = []
        
        # Find all GraphSemantics layers in the model
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # For nested models
                for sublayer in layer.layers:
                    if hasattr(sublayer, 'alpha') and hasattr(sublayer, 'name') and 'graph_semantics' in sublayer.name.lower():
                        alpha_values.append(sublayer.alpha.numpy())
            elif hasattr(layer, 'alpha') and hasattr(layer, 'name') and 'graph_semantics' in layer.name.lower():
                alpha_values.append(layer.alpha.numpy())
        
        if alpha_values:
            mean_alpha = np.mean(alpha_values)
            self.alpha_values.append(mean_alpha)
            print(f"   📊 GraphSemantics α gate (epoch {epoch+1}): {mean_alpha:.4f}")
            
            # Log to tensorboard if available
            if logs is not None:
                logs['graph_semantics_alpha'] = mean_alpha


def log_class_distribution(y: np.ndarray, split_name: str = "dataset"):
    """
    Log class distribution for debugging.
    
    Args:
        y: Labels (integer or one-hot encoded)
        split_name: Name of the data split
    """
    # Convert to integer labels if one-hot encoded
    if len(y.shape) > 1 and y.shape[1] > 1:
        y_classes = np.argmax(y, axis=1)
    else:
        y_classes = y
    
    # Count classes
    unique, counts = np.unique(y_classes, return_counts=True)
    total = len(y_classes)
    
    print(f"\nClass distribution for {split_name}:")
    print(f"Total samples: {total}")
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
    
    # Check for imbalance
    min_count = np.min(counts)
    max_count = np.max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("⚠️  Significant class imbalance detected!")
    
    return dict(zip(unique, counts))
