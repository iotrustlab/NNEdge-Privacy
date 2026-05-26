#!/usr/bin/env python3
"""Shared helpers for the Identity + Placement branch experiments."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

MODEL_LABELS = ("RF", "DT", "NB", "CNN", "RNN", "Transformer")
DL_MODEL_LABELS = ("CNN", "RNN", "Transformer")
EPSILON = 1e-10

_TF_MODULE = None
_TF_INITIALIZED = False
_TF_GPU_NAMES: list[str] = []


def set_global_seeds(seed: int) -> None:
    """Keep sklearn and NumPy sampling reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)


def calculate_entropy(labels: np.ndarray) -> float:
    """Calculate entropy of a label distribution."""
    if len(labels) == 0:
        return 0.0

    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return float(-np.sum(probabilities * np.log(probabilities + EPSILON)))


def calculate_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mutual information between true and predicted labels."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    contingency = np.zeros((len(unique_true), len(unique_pred)))

    for i, true_val in enumerate(unique_true):
        for j, pred_val in enumerate(unique_pred):
            contingency[i, j] = np.sum((y_true == true_val) & (y_pred == pred_val))

    joint_prob = contingency / len(y_true)
    prob_true = np.sum(joint_prob, axis=1)
    prob_pred = np.sum(joint_prob, axis=0)

    mi = 0.0
    for i in range(len(unique_true)):
        for j in range(len(unique_pred)):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(
                    joint_prob[i, j] / (prob_true[i] * prob_pred[j] + EPSILON) + EPSILON
                )

    return float(max(0.0, mi))


def calculate_normalized_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate normalized mutual information as a percentage."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    mutual_information = calculate_mutual_information(y_true, y_pred)
    entropy_true = calculate_entropy(y_true)
    entropy_pred = calculate_entropy(y_pred)

    if entropy_true > 0 and entropy_pred > 0:
        nmi_value = mutual_information / np.sqrt(entropy_true * entropy_pred)
    else:
        nmi_value = 0.0

    return float(min(100.0, max(0.0, nmi_value * 100.0)))


def calculate_kl_divergence_from_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate the average KL divergence from prediction probabilities."""
    if len(y_true) == 0 or y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0

    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else len(np.unique(y_true))
    total_kl = 0.0

    for index in range(n_samples):
        true_dist = np.zeros(n_classes)
        if y_true[index] < n_classes:
            true_dist[y_true[index]] = 1.0

        if len(y_pred_proba.shape) > 1:
            pred_dist = y_pred_proba[index]
        else:
            pred_dist = np.ones(n_classes) / n_classes

        pred_dist = pred_dist + EPSILON
        pred_dist = pred_dist / np.sum(pred_dist)

        kl_value = 0.0
        for class_index in range(n_classes):
            if true_dist[class_index] > 0:
                kl_value += true_dist[class_index] * np.log(true_dist[class_index] / pred_dist[class_index])

        total_kl += kl_value

    return float(total_kl / n_samples)


def calculate_normalized_kl_divergence(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate reverse-KL percentage, retained for backward compatibility."""
    if len(y_true) == 0 or y_pred_proba is None:
        return 0.0

    average_kl = calculate_kl_divergence_from_predictions(y_true, y_pred_proba)
    n_classes = len(np.unique(y_true))
    kl_max = np.log(n_classes) if n_classes > 0 else 0.0

    if kl_max > 0:
        kl_normalized = min(1.0, average_kl / kl_max)
    else:
        kl_normalized = 0.0

    reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
    return float(max(0.0, min(100.0, reverse_kl_percentage)))


def calculate_vulnerability(y_pred_proba: np.ndarray) -> float:
    """Average the maximum predicted class probability across the test set."""
    if y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0

    normalized_proba = normalize_probabilities(y_pred_proba)
    return float(np.mean(np.max(normalized_proba, axis=1)))


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip and normalize class probabilities row-wise."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, EPSILON, None)
    row_sums = np.sum(probabilities, axis=1, keepdims=True)
    return probabilities / row_sums


def build_metric_summary(values: list[float]) -> dict[str, float]:
    """Build a consistent metric summary for aggregate outputs."""
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
    }


def build_aggregate_summary(
    *,
    combination_results: list[dict],
    model_label: str,
    split_name: str,
    frequency_hz: int,
    source_frequency_hz: int,
    max_combinations_requested: int,
    data_root: str,
    results_dir: str,
    extra_fields: dict | None = None,
) -> dict:
    """Build a split-level summary with required max/std fields and aliases."""
    accuracies = [result["accuracy"] for result in combination_results]
    f1_scores = [result["f1_score"] for result in combination_results]
    precisions = [result["precision"] for result in combination_results]
    recalls = [result["recall"] for result in combination_results]
    nmi_percentages = [result["nmi_percentage"] for result in combination_results]
    nrkl_percentages = [result["nrkl_percentage"] for result in combination_results]
    reverse_kl_percentages = [result["reverse_kl_percentage"] for result in combination_results]
    vulnerabilities = [result["vulnerability"] for result in combination_results]

    accuracy_summary = build_metric_summary(accuracies)
    f1_summary = build_metric_summary(f1_scores)
    precision_summary = build_metric_summary(precisions)
    recall_summary = build_metric_summary(recalls)
    nmi_summary = build_metric_summary(nmi_percentages)
    nrkl_summary = build_metric_summary(nrkl_percentages)
    reverse_kl_summary = build_metric_summary(reverse_kl_percentages)
    vulnerability_summary = build_metric_summary(vulnerabilities)

    summary = {
        "model_label": model_label,
        "split_name": split_name,
        "frequency_hz": frequency_hz,
        "source_frequency_hz": source_frequency_hz,
        "max_combinations_requested": max_combinations_requested,
        "tested_combinations": len(combination_results),
        "valid_combinations": len(combination_results),
        "data_root": data_root,
        "results_dir": results_dir,
        "accuracy": accuracy_summary,
        "f1_score": f1_summary,
        "precision": precision_summary,
        "recall": recall_summary,
        "nmi_percentage": nmi_summary,
        "nrkl_percentage": nrkl_summary,
        "reverse_kl_percentage": reverse_kl_summary,
        "vulnerability": vulnerability_summary,
        "accuracy_max": accuracy_summary["max"],
        "accuracy_std": accuracy_summary["std"],
        "f1_max": f1_summary["max"],
        "f1_std": f1_summary["std"],
        "nmi_max": nmi_summary["max"],
        "nmi_std": nmi_summary["std"],
        "nrkl_max": nrkl_summary["max"],
        "nrkl_std": nrkl_summary["std"],
        "reverse_kl_max": reverse_kl_summary["max"],
        "reverse_kl_std": reverse_kl_summary["std"],
        "vulnerability_max": vulnerability_summary["max"],
        "vulnerability_std": vulnerability_summary["std"],
        "detailed_results": combination_results,
    }

    if extra_fields:
        summary.update(extra_fields)

    return summary


def ensure_tensorflow_gpu(random_seed: int, require_gpu: bool = True):
    """Import TensorFlow, configure memory growth, and fail fast if GPU is missing."""
    global _TF_MODULE, _TF_INITIALIZED, _TF_GPU_NAMES

    if _TF_MODULE is None:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel

        _TF_MODULE = tf

    tf = _TF_MODULE
    tf.keras.utils.set_random_seed(random_seed)

    if not _TF_INITIALIZED:
        physical_gpus = tf.config.list_physical_devices("GPU")
        _TF_GPU_NAMES = [gpu.name for gpu in physical_gpus]

        print(f"[DL] TensorFlow version: {tf.__version__}")
        print(f"[DL] Physical GPUs: {_TF_GPU_NAMES}")

        if not physical_gpus:
            if require_gpu:
                raise RuntimeError(
                    "GPU is required for CNN/RNN/Transformer runs on this branch, "
                    "but TensorFlow did not detect any physical GPU devices."
                )
            print("[DL] GPU-backed execution path: DISABLED")
        else:
            for gpu in physical_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as exc:
                    print(f"[DL] Memory-growth setup skipped for {gpu.name}: {exc}")

            logical_gpu_names = [gpu.name for gpu in tf.config.list_logical_devices("GPU")]
            print(f"[DL] Logical GPUs: {logical_gpu_names}")
            print("[DL] GPU-backed execution path: ENABLED")

        _TF_INITIALIZED = True
    elif require_gpu and not _TF_GPU_NAMES:
        raise RuntimeError(
            "GPU is required for CNN/RNN/Transformer runs on this branch, "
            "but no GPU devices were detected during TensorFlow initialization."
        )

    return tf


def train_tabular_model(
    *,
    model_label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_seed: int,
    rf_builder: Callable[[], object],
) -> tuple[np.ndarray, np.ndarray]:
    """Train one configured model family and return encoded predictions and probabilities."""
    if model_label == "RF":
        model = rf_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        return y_pred, normalize_probabilities(y_pred_proba)

    if model_label == "DT":
        model = DecisionTreeClassifier(
            random_state=random_seed,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        return y_pred, normalize_probabilities(y_pred_proba)

    if model_label == "NB":
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        return y_pred, normalize_probabilities(y_pred_proba)

    if model_label in DL_MODEL_LABELS:
        return train_dl_tabular_model(
            model_label=model_label,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            random_seed=random_seed,
        )

    raise ValueError(f"Unsupported model label: {model_label}")


def train_dl_tabular_model(
    *,
    model_label: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a GPU-backed Keras model over the branch's tabular feature vectors."""
    tf = ensure_tensorflow_gpu(random_seed=random_seed, require_gpu=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_seed)

    X_train_dl = np.asarray(X_train, dtype=np.float32).reshape((len(X_train), X_train.shape[1], 1))
    X_test_dl = np.asarray(X_test, dtype=np.float32).reshape((len(X_test), X_test.shape[1], 1))
    y_train = np.asarray(y_train, dtype=np.int32)

    num_classes = int(np.max(y_train) + 1)
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    input_shape = (X_train_dl.shape[1], X_train_dl.shape[2])

    if model_label == "CNN":
        model = _build_cnn_model(tf, input_shape, num_classes)
    elif model_label == "RNN":
        model = _build_rnn_model(tf, input_shape, num_classes)
    elif model_label == "Transformer":
        model = _build_transformer_model(tf, input_shape, num_classes)
    else:
        raise ValueError(f"Unsupported DL model label: {model_label}")

    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    class_weight_map = {int(class_id): float(weight) for class_id, weight in zip(unique_classes, class_weights)}

    validation_split = 0.2 if len(X_train_dl) >= 20 else 0.0
    monitor = "val_loss" if validation_split > 0 else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
        )
    ]

    batch_size = min(32, max(1, len(X_train_dl)))

    with tf.device("/GPU:0"):
        model.fit(
            X_train_dl,
            y_train_one_hot,
            epochs=25,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight_map,
            verbose=0,
            shuffle=True,
        )
        y_pred_proba = model.predict(X_test_dl, verbose=0)

    y_pred_proba = normalize_probabilities(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    tf.keras.backend.clear_session()
    return y_pred, y_pred_proba


def _build_cnn_model(tf, input_shape: tuple[int, int], num_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placement_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_rnn_model(tf, input_shape: tuple[int, int], num_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placement_rnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_transformer_model(tf, input_shape: tuple[int, int], num_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(32)(inputs)

    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    feed_forward = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32),
        ]
    )
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + feed_forward(x))
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placement_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_combination_result(
    *,
    model_label: str,
    train_users: list[int],
    test_users: list[int],
    train_samples: int,
    test_samples: int,
    n_features: int,
    y_true_encoded: np.ndarray,
    y_pred_encoded: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """Build a consistent per-combination result payload."""
    accuracy = float(accuracy_score(y_true_encoded, y_pred_encoded))
    f1_value = float(f1_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0))
    precision_value = float(precision_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0))
    recall_value = float(recall_score(y_true_encoded, y_pred_encoded, average="weighted", zero_division=0))
    nmi_percentage = calculate_normalized_mutual_information(y_true_encoded, y_pred_encoded)
    nrkl_percentage = calculate_normalized_kl_divergence(y_true_encoded, y_pred_proba)
    vulnerability = calculate_vulnerability(y_pred_proba)

    return {
        "model_label": model_label,
        "train_users": train_users,
        "test_users": test_users,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "n_features": n_features,
        "accuracy": accuracy,
        "f1_score": f1_value,
        "precision": precision_value,
        "recall": recall_value,
        "nmi_percentage": nmi_percentage,
        "nrkl_percentage": nrkl_percentage,
        "reverse_kl_percentage": nrkl_percentage,
        "vulnerability": vulnerability,
    }


def make_rf_voting_baseline(random_seed: int) -> VotingClassifier:
    """Build the branch-specific Game-2 baseline to be stored as RF."""
    from sklearn.ensemble import RandomForestClassifier  # pylint: disable=import-outside-toplevel
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel
    from sklearn.svm import SVC  # pylint: disable=import-outside-toplevel

    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=random_seed,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
    )
    logistic_model = LogisticRegression(random_state=random_seed, max_iter=1000)
    svc_model = SVC(random_state=random_seed, probability=True, kernel="rbf")
    return VotingClassifier(
        estimators=[("rf", rf_model), ("lr", logistic_model), ("svc", svc_model)],
        voting="soft",
    )
