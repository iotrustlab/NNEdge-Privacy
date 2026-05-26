#!/usr/bin/env python3
"""Shared helpers for the identity + anthropometrics + placements branch."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from project_paths import DEFAULT_RANDOM_SEED, SUPPORTED_MODEL_LABELS, SUPPORTED_SPLIT_NAMES

MODEL_LABELS = SUPPORTED_MODEL_LABELS
DL_MODEL_LABELS = frozenset({"CNN", "RNN", "Transformer"})
TARGET_SPLIT_SPECS = {"8_3": {"train_size": 8, "test_size": 3}}
DEFAULT_SPLIT_NAME = SUPPORTED_SPLIT_NAMES[0]
_ALLOW_CPU_OVERRIDE_ENV = "NNEDGE_PRIVACY_ALLOW_CPU_DL"
_TF_RUNTIME_STATE: dict[str, Any] | None = None


def calculate_entropy(labels: Sequence[int] | np.ndarray) -> float:
    """Calculate entropy of a label distribution."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0

    unique_labels, counts = np.unique(labels, return_counts=True)
    del unique_labels
    probabilities = counts / labels.size
    entropy_val = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return float(entropy_val)


def calculate_mutual_information(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float:
    """Calculate mutual information between true and predicted labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    contingency = np.zeros((len(unique_true), len(unique_pred)), dtype=np.float64)

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
                    joint_prob[i, j] / (prob_true[i] * prob_pred[j] + 1e-10) + 1e-10
                )

    return float(max(0.0, mi))


def calculate_normalized_mutual_information(
    y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray
) -> float:
    """Calculate NMI as a percentage bounded to [0, 100]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0

    mi = calculate_mutual_information(y_true, y_pred)
    h_true = calculate_entropy(y_true)
    h_pred = calculate_entropy(y_pred)

    if h_true > 0 and h_pred > 0:
        nmi = mi / np.sqrt(h_true * h_pred)
    else:
        nmi = 0.0

    return float(min(100.0, max(0.0, nmi * 100.0)))


def calculate_kl_divergence_from_predictions(
    y_true: Sequence[int] | np.ndarray, y_pred_proba: np.ndarray | Sequence[Sequence[float]] | None
) -> float:
    """Calculate average KL divergence from prediction probabilities."""
    y_true = np.asarray(y_true)
    if y_true.size == 0 or y_pred_proba is None:
        return 0.0

    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    if y_pred_proba.size == 0:
        return 0.0

    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else len(np.unique(y_true))
    total_kl = 0.0

    for i in range(n_samples):
        true_dist = np.zeros(n_classes, dtype=np.float64)
        if y_true[i] < n_classes:
            true_dist[y_true[i]] = 1.0

        if y_pred_proba.ndim > 1:
            pred_dist = np.asarray(y_pred_proba[i], dtype=np.float64)
        else:
            pred_dist = np.ones(n_classes, dtype=np.float64) / n_classes

        pred_dist = pred_dist + 1e-10
        pred_dist = pred_dist / np.sum(pred_dist)

        kl = 0.0
        for j in range(n_classes):
            if true_dist[j] > 0:
                kl += true_dist[j] * np.log(true_dist[j] / pred_dist[j])

        total_kl += kl

    return float(total_kl / n_samples)


def calculate_normalized_kl_divergence(
    y_true: Sequence[int] | np.ndarray, y_pred_proba: np.ndarray | Sequence[Sequence[float]] | None
) -> float:
    """Calculate reverse KL as a percentage bounded to [0, 100]."""
    y_true = np.asarray(y_true)
    if y_true.size == 0 or y_pred_proba is None:
        return 0.0

    avg_kl = calculate_kl_divergence_from_predictions(y_true, y_pred_proba)
    n_classes = len(np.unique(y_true))
    kl_max = np.log(n_classes) if n_classes > 0 else 0.0

    if kl_max > 0:
        kl_normalized = min(1.0, avg_kl / kl_max)
    else:
        kl_normalized = 0.0

    reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
    return float(max(0.0, min(100.0, reverse_kl_percentage)))


def calculate_vulnerability(y_pred_proba: np.ndarray | Sequence[Sequence[float]] | None) -> float:
    """Average maximum predicted class probability across the test set."""
    if y_pred_proba is None:
        return 0.0

    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    if y_pred_proba.size == 0:
        return 0.0

    if y_pred_proba.ndim == 1:
        return float(np.mean(y_pred_proba))

    return float(np.mean(np.max(y_pred_proba, axis=1)))


def ensure_feature_frame(features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> pd.DataFrame:
    """Return a dense, NaN-free feature frame."""
    if isinstance(features, pd.DataFrame):
        return features.copy().fillna(0)

    return pd.DataFrame(features).fillna(0)


def align_feature_frames(
    train_features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    test_features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align train/test feature frames using the union of columns."""
    train_df = ensure_feature_frame(train_features)
    test_df = ensure_feature_frame(test_features)

    all_columns = sorted({str(column) for column in train_df.columns} | {str(column) for column in test_df.columns})
    train_df.columns = [str(column) for column in train_df.columns]
    test_df.columns = [str(column) for column in test_df.columns]

    for column in all_columns:
        if column not in train_df.columns:
            train_df[column] = 0.0
        if column not in test_df.columns:
            test_df[column] = 0.0

    aligned_train = train_df[all_columns].to_numpy(dtype=np.float32)
    aligned_test = test_df[all_columns].to_numpy(dtype=np.float32)
    return aligned_train, aligned_test, all_columns


def requires_dl_runtime(model_labels: Sequence[str]) -> bool:
    """Return True when any requested model uses TensorFlow."""
    return any(model_label in DL_MODEL_LABELS for model_label in model_labels)


def validate_dl_runtime_for_models(model_labels: Sequence[str]) -> None:
    """Validate the TensorFlow runtime when DL models were requested."""
    if requires_dl_runtime(model_labels):
        get_tensorflow_runtime(require_gpu=True)


def get_tensorflow_runtime(require_gpu: bool = True):
    """Return TensorFlow after validating a GPU-backed runtime."""
    global _TF_RUNTIME_STATE

    if _TF_RUNTIME_STATE is None:
        import tensorflow as tf

        gpu_devices = tf.config.list_physical_devices("GPU")
        gpu_names = [device.name for device in gpu_devices]

        for device in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                # Memory growth can only be set before runtime initialization.
                pass

        has_gpu = bool(gpu_devices)
        allow_cpu_override = os.environ.get(_ALLOW_CPU_OVERRIDE_ENV, "").strip().lower() in {"1", "true", "yes"}
        backend = "GPU" if has_gpu else "CPU override"

        print(f"🧠 TensorFlow version: {tf.__version__}")
        print(f"🧠 Detected GPU devices: {gpu_names}")
        print(f"🧠 DL execution backend: {backend}")

        if require_gpu and not has_gpu and not allow_cpu_override:
            raise RuntimeError(
                "No TensorFlow GPU detected for DL model execution. "
                "Run `./.venv-identity-anthro-placements/bin/python -c "
                "\"import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))\"` "
                "and only set NNEDGE_PRIVACY_ALLOW_CPU_DL=1 if you explicitly approve CPU fallback."
            )

        _TF_RUNTIME_STATE = {
            "tf": tf,
            "has_gpu": has_gpu,
            "gpu_names": gpu_names,
            "backend": "GPU" if has_gpu else "CPU",
        }

    if require_gpu and not _TF_RUNTIME_STATE["has_gpu"]:
        allow_cpu_override = os.environ.get(_ALLOW_CPU_OVERRIDE_ENV, "").strip().lower() in {"1", "true", "yes"}
        if not allow_cpu_override:
            raise RuntimeError(
                "DL models require a GPU on this branch. "
                "Set NNEDGE_PRIVACY_ALLOW_CPU_DL=1 only if you explicitly approve CPU fallback."
            )

    return _TF_RUNTIME_STATE["tf"], dict(_TF_RUNTIME_STATE)


def build_model_backend(model_label: str) -> str:
    """Return the serving backend for a model label."""
    return "tensorflow" if model_label in DL_MODEL_LABELS else "sklearn"


def fit_predict_model(
    model_label: str,
    baseline_factory: Callable[[], Any],
    X_train_scaled: np.ndarray,
    y_train_encoded: np.ndarray,
    X_test_scaled: np.ndarray,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train the requested model and return predictions plus execution metadata."""
    if model_label == "RF":
        model = baseline_factory()
        model.fit(X_train_scaled, y_train_encoded)
        return (
            np.asarray(model.predict(X_test_scaled)),
            np.asarray(model.predict_proba(X_test_scaled), dtype=np.float64),
            {
                "model_backend": "sklearn",
                "execution_backend": "CPU",
            },
        )

    if model_label == "DT":
        model = DecisionTreeClassifier(random_state=random_seed)
        model.fit(X_train_scaled, y_train_encoded)
        return (
            np.asarray(model.predict(X_test_scaled)),
            np.asarray(model.predict_proba(X_test_scaled), dtype=np.float64),
            {
                "model_backend": "sklearn",
                "execution_backend": "CPU",
            },
        )

    if model_label == "NB":
        model = GaussianNB()
        model.fit(X_train_scaled, y_train_encoded)
        return (
            np.asarray(model.predict(X_test_scaled)),
            np.asarray(model.predict_proba(X_test_scaled), dtype=np.float64),
            {
                "model_backend": "sklearn",
                "execution_backend": "CPU",
            },
        )

    if model_label not in DL_MODEL_LABELS:
        raise ValueError(f"Unsupported model label: {model_label}")

    tf, runtime_state = get_tensorflow_runtime(require_gpu=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_seed)

    n_classes = len(np.unique(y_train_encoded))
    X_train_dl = np.asarray(X_train_scaled, dtype=np.float32)[..., np.newaxis]
    X_test_dl = np.asarray(X_test_scaled, dtype=np.float32)[..., np.newaxis]
    input_shape = X_train_dl.shape[1:]

    model = _build_dl_classifier(tf, model_label, input_shape, n_classes)
    batch_size = _choose_dl_batch_size(len(X_train_dl))
    validation_split = 0.2 if len(X_train_dl) >= 20 else 0.0

    print(
        f"🧠 Training {model_label} with TensorFlow on {runtime_state['backend']} "
        f"(input_shape={input_shape}, batch_size={batch_size})"
    )

    callbacks = []
    if validation_split > 0:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
            ),
        ]

    fit_kwargs: dict[str, Any] = {
        "epochs": 15,
        "batch_size": batch_size,
        "verbose": 0,
        "shuffle": True,
    }
    if validation_split > 0:
        fit_kwargs["validation_split"] = validation_split
        fit_kwargs["callbacks"] = callbacks

    try:
        model.fit(X_train_dl, y_train_encoded, **fit_kwargs)
        y_pred_proba = np.asarray(model.predict(X_test_dl, batch_size=batch_size, verbose=0), dtype=np.float64)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return (
            y_pred,
            y_pred_proba,
            {
                "model_backend": "tensorflow",
                "execution_backend": runtime_state["backend"],
                "gpu_devices": runtime_state["gpu_names"],
                "batch_size": batch_size,
                "epochs": fit_kwargs["epochs"],
                "validation_split": validation_split,
            },
        )
    finally:
        tf.keras.backend.clear_session()


def build_summary_statistics(
    results: Sequence[dict[str, Any]],
    *,
    split_name: str,
    frequency_hz: int,
    source_frequency_hz: int,
    max_combinations_requested: int,
    total_combinations: int,
    successful_combinations: int,
    data_root: str,
    results_dir: str,
    model_label: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-combination results into the branch summary schema."""
    if not results:
        raise ValueError("Cannot summarize an empty result set.")

    accuracies = [float(result["accuracy"]) for result in results]
    f1_scores = [float(result["f1_score"]) for result in results]
    precisions = [float(result.get("precision", 0.0)) for result in results]
    recalls = [float(result.get("recall", 0.0)) for result in results]
    nmi_percentages = [float(result["nmi_percentage"]) for result in results]
    nrkl_percentages = [float(result["nrkl_percentage"]) for result in results]
    vulnerability_scores = [float(result["vulnerability"]) for result in results]

    summary = {
        "split_name": split_name,
        "split_ratio": split_name,
        "model_label": model_label,
        "model_backend": build_model_backend(model_label),
        "frequency_hz": frequency_hz,
        "source_frequency_hz": source_frequency_hz,
        "max_combinations_requested": max_combinations_requested,
        "total_combinations": total_combinations,
        "successful_combinations": successful_combinations,
        "n_combinations": len(results),
        "tested_combinations": len(results),
        "accuracy_mean": _mean(accuracies),
        "accuracy_std": _std(accuracies),
        "accuracy_min": _min(accuracies),
        "accuracy_max": _max(accuracies),
        "f1_mean": _mean(f1_scores),
        "f1_std": _std(f1_scores),
        "f1_min": _min(f1_scores),
        "f1_max": _max(f1_scores),
        "precision_mean": _mean(precisions),
        "recall_mean": _mean(recalls),
        "nmi_mean": _mean(nmi_percentages),
        "nmi_std": _std(nmi_percentages),
        "nmi_min": _min(nmi_percentages),
        "nmi_max": _max(nmi_percentages),
        "nrkl_mean": _mean(nrkl_percentages),
        "nrkl_std": _std(nrkl_percentages),
        "nrkl_min": _min(nrkl_percentages),
        "nrkl_max": _max(nrkl_percentages),
        "reverse_kl_mean": _mean(nrkl_percentages),
        "reverse_kl_std": _std(nrkl_percentages),
        "reverse_kl_min": _min(nrkl_percentages),
        "reverse_kl_max": _max(nrkl_percentages),
        "vulnerability_mean": _mean(vulnerability_scores),
        "vulnerability_std": _std(vulnerability_scores),
        "vulnerability_min": _min(vulnerability_scores),
        "vulnerability_max": _max(vulnerability_scores),
        "avg_train_samples": _mean([result["n_train_samples"] for result in results]),
        "avg_test_samples": _mean([result["n_test_samples"] for result in results]),
        "avg_features": _mean([result["n_features"] for result in results]),
        "data_root": data_root,
        "results_dir": results_dir,
    }

    if extra_fields:
        summary.update(extra_fields)

    return summary


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with numpy-safe serialization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64)))


def _min(values: Sequence[float]) -> float:
    return float(np.min(np.asarray(values, dtype=np.float64)))


def _max(values: Sequence[float]) -> float:
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _choose_dl_batch_size(n_samples: int) -> int:
    if n_samples <= 0:
        return 32
    return max(1, min(256, n_samples, max(16, n_samples // 4)))


def _build_dl_classifier(tf, model_label: str, input_shape: tuple[int, int], n_classes: int):
    if model_label == "CNN":
        return _build_cnn_classifier(tf, input_shape, n_classes)
    if model_label == "RNN":
        return _build_rnn_classifier(tf, input_shape, n_classes)
    if model_label == "Transformer":
        return _build_transformer_classifier(tf, input_shape, n_classes)
    raise ValueError(f"Unsupported DL model label: {model_label}")


def _build_cnn_classifier(tf, input_shape: tuple[int, int], n_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placements_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_rnn_classifier(tf, input_shape: tuple[int, int], n_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placements_rnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_transformer_classifier(tf, input_shape: tuple[int, int], n_classes: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64)(inputs)
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
    feed_forward = tf.keras.layers.Dense(128, activation="relu")(x)
    feed_forward = tf.keras.layers.Dense(64)(feed_forward)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + feed_forward)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_placements_transformer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
