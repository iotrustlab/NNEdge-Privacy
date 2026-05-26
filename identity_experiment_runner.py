#!/usr/bin/env python3
"""Shared runner for the Identity branch experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from project_paths import SOURCE_FREQUENCY_HZ, sample_experiment_combinations

DL_MODEL_LABELS = ("CNN", "RNN", "Transformer")
ALL_MODEL_LABELS = ("RF", "DT", "NB", "CNN", "RNN", "Transformer")
BRANCH_MAX_COMBINATIONS = 5

_TF_RUNTIME: dict[str, Any] | None = None


@dataclass(frozen=True)
class PreparedCombinationData:
    train_users: list[int]
    test_users: list[int]
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_train_samples: int
    n_test_samples: int
    n_features: int
    feature_names: tuple[str, ...]
    user_feature_mask: np.ndarray


def calculate_entropy(labels: np.ndarray) -> float:
    """Calculate entropy of a label distribution."""
    if len(labels) == 0:
        return 0.0

    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return float(entropy_value)


def calculate_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mutual information between true and predicted labels."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    contingency = np.zeros((len(unique_true), len(unique_pred)))

    for true_index, true_value in enumerate(unique_true):
        for pred_index, pred_value in enumerate(unique_pred):
            contingency[true_index, pred_index] = np.sum(
                (y_true == true_value) & (y_pred == pred_value)
            )

    joint_prob = contingency / len(y_true)
    prob_true = np.sum(joint_prob, axis=1)
    prob_pred = np.sum(joint_prob, axis=0)

    mutual_information = 0.0
    for true_index in range(len(unique_true)):
        for pred_index in range(len(unique_pred)):
            if joint_prob[true_index, pred_index] > 0:
                mutual_information += joint_prob[true_index, pred_index] * np.log(
                    joint_prob[true_index, pred_index]
                    / (prob_true[true_index] * prob_pred[pred_index] + 1e-10)
                    + 1e-10
                )

    return float(max(0.0, mutual_information))


def calculate_normalized_mutual_information(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate normalized mutual information as a percentage."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0

    mutual_information = calculate_mutual_information(y_true, y_pred)
    entropy_true = calculate_entropy(y_true)
    entropy_pred = calculate_entropy(y_pred)

    if entropy_true > 0 and entropy_pred > 0:
        nmi = mutual_information / np.sqrt(entropy_true * entropy_pred)
    else:
        nmi = 0.0

    return float(min(100.0, max(0.0, nmi * 100.0)))


def calculate_kl_divergence_from_predictions(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """Calculate average KL divergence from prediction probabilities."""
    if len(y_true) == 0 or y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0

    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    total_kl = 0.0

    for sample_index in range(n_samples):
        true_dist = np.zeros(n_classes)
        if y_true[sample_index] < n_classes:
            true_dist[y_true[sample_index]] = 1.0

        pred_dist = np.asarray(y_pred_proba[sample_index], dtype=np.float64) + 1e-10
        pred_dist = pred_dist / np.sum(pred_dist)

        kl_value = 0.0
        for class_index in range(n_classes):
            if true_dist[class_index] > 0:
                kl_value += true_dist[class_index] * np.log(
                    true_dist[class_index] / pred_dist[class_index]
                )

        total_kl += kl_value

    return float(total_kl / n_samples)


def calculate_normalized_kl_divergence(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> float:
    """Calculate reverse-KL/NRKL as a percentage."""
    if len(y_true) == 0 or y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0

    average_kl = calculate_kl_divergence_from_predictions(y_true, y_pred_proba)
    n_classes = max(1, len(np.unique(y_true)))
    kl_max = np.log(n_classes)

    if kl_max > 0:
        kl_normalized = min(1.0, average_kl / kl_max)
    else:
        kl_normalized = 0.0

    reverse_kl_percentage = (1.0 - kl_normalized) * 100.0
    return float(max(0.0, min(100.0, reverse_kl_percentage)))


def calculate_vulnerability(y_pred_proba: np.ndarray) -> float:
    """Average maximum predicted class probability across the test set."""
    if y_pred_proba is None or len(y_pred_proba) == 0:
        return 0.0

    return float(np.mean(np.max(y_pred_proba, axis=1)))


def normalize_combination(combination: Any) -> tuple[list[int], list[int]]:
    """Normalize tuple/dict combinations to train/test user lists."""
    if isinstance(combination, dict):
        return list(combination["train_users"]), list(combination["test_users"])

    if isinstance(combination, tuple) and len(combination) == 2:
        return list(combination[0]), list(combination[1])

    raise TypeError(f"Unsupported combination format: {combination!r}")


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    return value


def _get_model_results_dir(base_results_dir: str, model_label: str) -> Path:
    model_results_dir = Path(base_results_dir) / model_label
    model_results_dir.mkdir(parents=True, exist_ok=True)
    return model_results_dir


def _prepare_user_identity_columns(
    features_df: pd.DataFrame, all_users: list[int]
) -> pd.DataFrame:
    if "user_id" not in features_df.columns:
        return features_df

    categorical_users = pd.Categorical(features_df["user_id"], categories=all_users)
    user_dummies = pd.get_dummies(categorical_users, prefix="user")
    expected_columns = [f"user_{user_id}" for user_id in all_users]
    user_dummies = user_dummies.reindex(columns=expected_columns, fill_value=0)
    features_df = pd.concat([features_df.drop(columns=["user_id"]), user_dummies], axis=1)
    return features_df


def build_dataset_frame_from_rows(
    feature_rows: list[dict[str, Any]],
    labels: list[str],
    metadata: list[dict[str, Any]],
    all_users: list[int],
) -> tuple[pd.DataFrame | None, np.ndarray | None, list[dict[str, Any]] | None]:
    """Convert cached feature rows into the aligned feature DataFrame used by the sweep."""
    if not feature_rows:
        return None, None, None

    feature_frame = pd.DataFrame(feature_rows).fillna(0.0)
    feature_frame = _prepare_user_identity_columns(feature_frame, all_users)
    feature_frame = feature_frame.fillna(0.0).astype(np.float32)

    return feature_frame, np.asarray(labels), metadata


def _get_tensorflow_runtime(require_gpu: bool) -> dict[str, Any]:
    global _TF_RUNTIME

    if _TF_RUNTIME is not None:
        if require_gpu and not _TF_RUNTIME["gpu_names"]:
            raise RuntimeError(
                "No TensorFlow GPU devices detected. DL models on this branch require GPU execution."
            )
        return _TF_RUNTIME

    import tensorflow as tf

    print(f"[DL] TensorFlow version: {tf.__version__}")
    print(f"[DL] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [device.name for device in physical_gpus]

    if not physical_gpus:
        print("[DL] Physical GPU devices: []")
        print("[DL] Selected execution path: CPU")
        if require_gpu:
            raise RuntimeError(
                "No TensorFlow GPU devices detected. DL models on this branch require GPU execution."
            )
    else:
        for device in physical_gpus:
            tf.config.experimental.set_memory_growth(device, True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        logical_gpu_names = [device.name for device in logical_gpus]
        print(f"[DL] Physical GPU devices: {gpu_names}")
        print(f"[DL] Logical GPU devices: {logical_gpu_names}")
        print("[DL] Selected execution path: GPU-backed TensorFlow/Keras")

    _TF_RUNTIME = {
        "tf": tf,
        "gpu_names": gpu_names,
        "device_name": "/GPU:0" if gpu_names else "/CPU:0",
    }
    return _TF_RUNTIME


def _build_cnn_model(
    tf: Any,
    input_dim: int,
    n_classes: int,
    profile: dict[str, Any],
) -> Any:
    regularizer = tf.keras.regularizers.l2(profile["l2"])

    inputs = tf.keras.Input(shape=(input_dim,), name="tabular_features")
    normalized = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    shortcut = tf.keras.layers.Dense(
        profile["dense_units"] // 2,
        activation="gelu",
        kernel_regularizer=regularizer,
    )(normalized)

    token_projection = tf.keras.layers.Dense(
        profile["token_count"] * profile["projection_dim"],
        activation="gelu",
        kernel_regularizer=regularizer,
    )(normalized)
    token_sequence = tf.keras.layers.Reshape(
        (profile["token_count"], profile["projection_dim"])
    )(token_projection)

    x = tf.keras.layers.Conv1D(
        profile["cnn_filters"],
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(token_sequence)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SeparableConv1D(
        profile["cnn_filters"],
        kernel_size=3,
        padding="same",
        activation="relu",
        depthwise_regularizer=regularizer,
        pointwise_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([x, shortcut])
    x = tf.keras.layers.Dense(
        profile["dense_units"],
        activation="gelu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.Dropout(profile["dropout"])(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_cnn")


def _build_rnn_model(
    tf: Any,
    input_dim: int,
    n_classes: int,
    profile: dict[str, Any],
) -> Any:
    regularizer = tf.keras.regularizers.l2(profile["l2"])

    inputs = tf.keras.Input(shape=(input_dim,), name="tabular_features")
    normalized = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    shortcut = tf.keras.layers.Dense(
        profile["dense_units"] // 2,
        activation="gelu",
        kernel_regularizer=regularizer,
    )(normalized)

    token_projection = tf.keras.layers.Dense(
        profile["token_count"] * profile["projection_dim"],
        activation="gelu",
        kernel_regularizer=regularizer,
    )(normalized)
    token_sequence = tf.keras.layers.Reshape(
        (profile["token_count"], profile["projection_dim"])
    )(token_projection)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            profile["rnn_units"],
            return_sequences=True,
            dropout=profile["recurrent_dropout"],
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
        )
    )(token_sequence)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            max(16, profile["rnn_units"] // 2),
            dropout=profile["recurrent_dropout"],
            recurrent_dropout=0.0,
            kernel_regularizer=regularizer,
        )
    )(x)
    x = tf.keras.layers.Concatenate()([x, shortcut])
    x = tf.keras.layers.Dense(
        profile["dense_units"],
        activation="gelu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.Dropout(profile["dropout"])(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_rnn")


def _build_transformer_model(tf: Any, input_shape: tuple[int, int], n_classes: int) -> Any:
    inputs = tf.keras.Input(shape=input_shape)
    projected = tf.keras.layers.Dense(32)(inputs)
    normalized = tf.keras.layers.LayerNormalization(epsilon=1e-6)(projected)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(
        normalized, normalized
    )
    attended = tf.keras.layers.Add()([projected, attention])
    normalized_attended = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attended)
    feed_forward = tf.keras.layers.Dense(64, activation="relu")(normalized_attended)
    feed_forward = tf.keras.layers.Dense(32)(feed_forward)
    transformer_block = tf.keras.layers.Add()([normalized_attended, feed_forward])
    pooled = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
    dense = tf.keras.layers.Dense(64, activation="relu")(pooled)
    dense = tf.keras.layers.Dropout(0.2)(dense)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(dense)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="identity_transformer")


def _build_dl_model(
    model_label: str,
    tf: Any,
    input_dim: int,
    n_classes: int,
    profile: dict[str, Any],
) -> Any:
    if model_label == "CNN":
        return _build_cnn_model(tf, input_dim, n_classes, profile)
    if model_label == "RNN":
        return _build_rnn_model(tf, input_dim, n_classes, profile)
    if model_label == "Transformer":
        return _build_transformer_model(tf, (input_dim, 1), n_classes)
    raise ValueError(f"Unsupported DL model label: {model_label}")


def _get_dl_profile(analyzer: Any, model_label: str) -> dict[str, Any]:
    experiment_name = analyzer.experiment_name
    base_profile: dict[str, Any] = {
        "epochs": 60,
        "batch_size": 8,
        "learning_rate": 5e-4,
        "dropout": 0.30,
        "recurrent_dropout": 0.15,
        "l2": 1e-4,
        "augment_rounds": 3,
        "noise_std": 0.03,
        "token_count": 8,
        "projection_dim": 8,
        "cnn_filters": 48,
        "rnn_units": 32,
        "dense_units": 64,
        "patience": 8,
    }

    if experiment_name == "game-1-user_identity_only":
        base_profile.update(
            epochs=90,
            augment_rounds=4,
            noise_std=0.02,
            token_count=4,
            projection_dim=8,
            cnn_filters=24,
            rnn_units=20,
            dense_units=32,
            dropout=0.35,
            patience=10,
        )
    elif experiment_name == "game-3-user_identity_only":
        base_profile.update(
            epochs=80,
            augment_rounds=3,
            noise_std=0.035,
            token_count=10,
            projection_dim=10,
            cnn_filters=64,
            rnn_units=40,
            dense_units=96,
            dropout=0.25,
        )
    else:
        base_profile.update(
            epochs=75,
            augment_rounds=3,
            noise_std=0.03,
            token_count=8,
            projection_dim=10,
            cnn_filters=48,
            rnn_units=32,
            dense_units=64,
        )

    if model_label == "RNN":
        base_profile["learning_rate"] = 3e-4
        base_profile["dropout"] = max(base_profile["dropout"], 0.30)

    return base_profile


def _prepare_dl_inputs(prepared_data: PreparedCombinationData) -> tuple[np.ndarray, np.ndarray, int]:
    user_mask = prepared_data.user_feature_mask.astype(bool)
    continuous_mask = ~user_mask

    X_train_raw = prepared_data.X_train_raw.astype(np.float32)
    X_test_raw = prepared_data.X_test_raw.astype(np.float32)

    continuous_count = int(np.sum(continuous_mask))
    if continuous_count > 0:
        scaler = StandardScaler()
        X_train_cont = scaler.fit_transform(X_train_raw[:, continuous_mask]).astype(np.float32)
        X_test_cont = scaler.transform(X_test_raw[:, continuous_mask]).astype(np.float32)
    else:
        X_train_cont = np.empty((len(X_train_raw), 0), dtype=np.float32)
        X_test_cont = np.empty((len(X_test_raw), 0), dtype=np.float32)

    if np.any(user_mask):
        X_train_user = X_train_raw[:, user_mask].astype(np.float32)
        X_test_user = X_test_raw[:, user_mask].astype(np.float32)
        X_train_dl = np.concatenate([X_train_cont, X_train_user], axis=1)
        X_test_dl = np.concatenate([X_test_cont, X_test_user], axis=1)
    else:
        X_train_dl = X_train_cont
        X_test_dl = X_test_cont

    return X_train_dl.astype(np.float32), X_test_dl.astype(np.float32), continuous_count


def _augment_dl_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    continuous_count: int,
    augment_rounds: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if augment_rounds <= 0 or len(X_train) == 0:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    feature_batches = [X_train]
    label_batches = [y_train]

    for _ in range(augment_rounds):
        augmented = np.array(X_train, copy=True)
        if continuous_count > 0:
            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=(len(X_train), continuous_count),
            ).astype(np.float32)
            scales = rng.normal(
                loc=1.0,
                scale=noise_std / 2.0,
                size=(len(X_train), continuous_count),
            ).astype(np.float32)
            augmented[:, :continuous_count] = (augmented[:, :continuous_count] * scales) + noise

        feature_batches.append(augmented)
        label_batches.append(y_train.copy())

    X_aug = np.concatenate(feature_batches, axis=0)
    y_aug = np.concatenate(label_batches, axis=0)
    shuffle_indices = rng.permutation(len(X_aug))
    return X_aug[shuffle_indices], y_aug[shuffle_indices]


def _train_dl_model(
    analyzer: Any,
    model_label: str,
    prepared_data: PreparedCombinationData,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    runtime = _get_tensorflow_runtime(require_gpu=True)
    tf = runtime["tf"]
    profile = _get_dl_profile(analyzer, model_label)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    X_train_base, X_test, continuous_count = _prepare_dl_inputs(prepared_data)
    y_train = prepared_data.y_train.astype(np.int32)

    fit_kwargs: dict[str, Any] = {
        "epochs": profile["epochs"],
        "batch_size": max(4, min(profile["batch_size"], len(X_train_base))),
        "verbose": 0,
        "shuffle": True,
    }

    callbacks = []
    if len(X_train_base) >= 21 and np.min(np.bincount(y_train)) >= 3:
        X_fit_base, X_val, y_fit, y_val = train_test_split(
            X_train_base,
            y_train,
            test_size=0.25,
            random_state=seed,
            stratify=y_train,
        )
        X_fit, y_fit = _augment_dl_training_data(
            X_fit_base,
            y_fit,
            continuous_count=continuous_count,
            augment_rounds=profile["augment_rounds"],
            noise_std=profile["noise_std"],
            seed=seed,
        )
        fit_kwargs["x"] = X_fit
        fit_kwargs["y"] = y_fit
        fit_kwargs["validation_data"] = (X_val, y_val)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=profile["patience"],
                restore_best_weights=True,
            )
        )
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, profile["patience"] // 3),
                min_lr=1e-5,
            )
        )
    else:
        X_fit, y_fit = _augment_dl_training_data(
            X_train_base,
            y_train,
            continuous_count=continuous_count,
            augment_rounds=profile["augment_rounds"],
            noise_std=profile["noise_std"],
            seed=seed,
        )
        fit_kwargs["x"] = X_fit
        fit_kwargs["y"] = y_fit

    with tf.device(runtime["device_name"]):
        model = _build_dl_model(
            model_label=model_label,
            tf=tf,
            input_dim=X_train_base.shape[1],
            n_classes=len(np.unique(y_train)),
            profile=profile,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=profile["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(callbacks=callbacks, **fit_kwargs)
        y_pred_proba = model.predict(X_test, verbose=0)

    y_pred = np.argmax(y_pred_proba, axis=1)
    details = {
        "execution_backend": "tensorflow_gpu",
        "gpu_devices": runtime["gpu_names"],
        "device_name": runtime["device_name"],
        "dl_profile": profile,
    }
    tf.keras.backend.clear_session()
    return y_pred, y_pred_proba, details


def _train_sklearn_model(
    analyzer: Any,
    model_label: str,
    prepared_data: PreparedCombinationData,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model_label == "RF":
        model = analyzer.build_rf_model(seed)
    elif model_label == "DT":
        model = DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    elif model_label == "NB":
        model = GaussianNB()
    else:
        raise ValueError(f"Unsupported sklearn model label: {model_label}")

    model.fit(prepared_data.X_train, prepared_data.y_train)
    y_pred = model.predict(prepared_data.X_test)
    y_pred_proba = model.predict_proba(prepared_data.X_test)

    details = {
        "execution_backend": "sklearn",
        "underlying_model": (
            "VotingClassifier" if isinstance(model, VotingClassifier) else type(model).__name__
        ),
    }
    return y_pred, y_pred_proba, details


def _prepare_combination_data(
    analyzer: Any,
    train_users: list[int],
    test_users: list[int],
) -> PreparedCombinationData | None:
    X_train_df, y_train, _train_metadata = analyzer.create_dataset(train_users)
    X_test_df, y_test, _test_metadata = analyzer.create_dataset(test_users)

    if X_train_df is None or X_test_df is None or y_train is None or y_test is None:
        return None

    if len(X_train_df) == 0 or len(X_test_df) == 0:
        return None

    all_columns = sorted(set(X_train_df.columns) | set(X_test_df.columns))
    X_train_aligned = X_train_df.reindex(columns=all_columns, fill_value=0.0)
    X_test_aligned = X_test_df.reindex(columns=all_columns, fill_value=0.0)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    unseen_classes = sorted(set(y_test) - set(label_encoder.classes_))
    if unseen_classes:
        print(
            f"[WARN] Skipping combination train={train_users} test={test_users} because "
            f"the test set contains unseen classes: {unseen_classes}"
        )
        return None

    y_test_encoded = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aligned).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_aligned).astype(np.float32)

    return PreparedCombinationData(
        train_users=train_users,
        test_users=test_users,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        X_train_raw=X_train_aligned.to_numpy(dtype=np.float32, copy=True),
        X_test_raw=X_test_aligned.to_numpy(dtype=np.float32, copy=True),
        y_train=y_train_encoded,
        y_test=y_test_encoded,
        n_train_samples=len(X_train_aligned),
        n_test_samples=len(X_test_aligned),
        n_features=len(all_columns),
        feature_names=tuple(all_columns),
        user_feature_mask=np.asarray(
            [column_name.startswith("user_") for column_name in all_columns],
            dtype=bool,
        ),
    )


def _evaluate_model_on_prepared_data(
    analyzer: Any,
    model_label: str,
    prepared_data: PreparedCombinationData,
    seed: int,
) -> dict[str, Any]:
    if model_label in DL_MODEL_LABELS:
        y_pred, y_pred_proba, backend_details = _train_dl_model(
            analyzer=analyzer,
            model_label=model_label,
            prepared_data=prepared_data,
            seed=seed,
        )
    else:
        y_pred, y_pred_proba, backend_details = _train_sklearn_model(
            analyzer=analyzer,
            model_label=model_label,
            prepared_data=prepared_data,
            seed=seed,
        )

    accuracy = accuracy_score(prepared_data.y_test, y_pred)
    precision = precision_score(
        prepared_data.y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        prepared_data.y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    f1_value = f1_score(
        prepared_data.y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    nmi_percentage = calculate_normalized_mutual_information(prepared_data.y_test, y_pred)
    nrkl_percentage = calculate_normalized_kl_divergence(prepared_data.y_test, y_pred_proba)
    vulnerability = calculate_vulnerability(y_pred_proba)

    result = {
        "model_label": model_label,
        "train_users": prepared_data.train_users,
        "test_users": prepared_data.test_users,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1_value),
        "f1_score": float(f1_value),
        "nmi_percentage": float(nmi_percentage),
        "nrkl_percentage": float(nrkl_percentage),
        "reverse_kl_percentage": float(nrkl_percentage),
        "vulnerability": float(vulnerability),
        "n_train_samples": prepared_data.n_train_samples,
        "n_test_samples": prepared_data.n_test_samples,
        "n_features": prepared_data.n_features,
    }
    result.update(backend_details)
    return result


def _build_model_summary(
    analyzer: Any,
    model_label: str,
    split_name: str,
    total_combinations_available: int,
    sampled_combinations: list[Any],
    results: list[dict[str, Any]],
    model_results_dir: Path,
) -> dict[str, Any]:
    accuracy_values = np.asarray([result["accuracy"] for result in results], dtype=np.float64)
    f1_values = np.asarray([result["f1"] for result in results], dtype=np.float64)
    nmi_values = np.asarray([result["nmi_percentage"] for result in results], dtype=np.float64)
    nrkl_values = np.asarray([result["nrkl_percentage"] for result in results], dtype=np.float64)
    vulnerability_values = np.asarray([result["vulnerability"] for result in results], dtype=np.float64)
    precision_values = np.asarray([result["precision"] for result in results], dtype=np.float64)
    recall_values = np.asarray([result["recall"] for result in results], dtype=np.float64)

    summary = {
        "model_label": model_label,
        "split_ratio": split_name,
        "frequency_hz": analyzer.config.frequency_hz,
        "source_frequency_hz": SOURCE_FREQUENCY_HZ,
        "total_combinations_available": total_combinations_available,
        "max_combinations_requested": min(analyzer.config.max_combinations, BRANCH_MAX_COMBINATIONS),
        "n_combinations_tested": len(results),
        "successful_combinations": len(results),
        "sampled_combinations": [
            {
                "train_users": train_users,
                "test_users": test_users,
            }
            for train_users, test_users in (normalize_combination(combo) for combo in sampled_combinations)
        ],
        "accuracy_mean": float(np.mean(accuracy_values)),
        "accuracy_std": float(np.std(accuracy_values)),
        "accuracy_min": float(np.min(accuracy_values)),
        "accuracy_max": float(np.max(accuracy_values)),
        "f1_mean": float(np.mean(f1_values)),
        "f1_std": float(np.std(f1_values)),
        "f1_min": float(np.min(f1_values)),
        "f1_max": float(np.max(f1_values)),
        "precision_mean": float(np.mean(precision_values)),
        "recall_mean": float(np.mean(recall_values)),
        "nmi_mean": float(np.mean(nmi_values)),
        "nmi_std": float(np.std(nmi_values)),
        "nmi_min": float(np.min(nmi_values)),
        "nmi_max": float(np.max(nmi_values)),
        "nrkl_mean": float(np.mean(nrkl_values)),
        "nrkl_std": float(np.std(nrkl_values)),
        "nrkl_min": float(np.min(nrkl_values)),
        "nrkl_max": float(np.max(nrkl_values)),
        "reverse_kl_mean": float(np.mean(nrkl_values)),
        "reverse_kl_std": float(np.std(nrkl_values)),
        "reverse_kl_min": float(np.min(nrkl_values)),
        "reverse_kl_max": float(np.max(nrkl_values)),
        "vulnerability_mean": float(np.mean(vulnerability_values)),
        "vulnerability_std": float(np.std(vulnerability_values)),
        "vulnerability_min": float(np.min(vulnerability_values)),
        "vulnerability_max": float(np.max(vulnerability_values)),
        "avg_train_samples": float(np.mean([result["n_train_samples"] for result in results])),
        "avg_test_samples": float(np.mean([result["n_test_samples"] for result in results])),
        "avg_features": float(np.mean([result["n_features"] for result in results])),
        "data_root": str(analyzer.config.data_root),
        "results_dir": str(model_results_dir),
        "execution_backend": results[0]["execution_backend"],
        "gpu_devices": results[0].get("gpu_devices", []),
    }
    return summary


def _save_model_outputs(
    analyzer: Any,
    model_label: str,
    split_name: str,
    total_combinations_available: int,
    sampled_combinations: list[Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    model_results_dir = _get_model_results_dir(analyzer.base_results_dir, model_label)
    summary = _build_model_summary(
        analyzer=analyzer,
        model_label=model_label,
        split_name=split_name,
        total_combinations_available=total_combinations_available,
        sampled_combinations=sampled_combinations,
        results=results,
        model_results_dir=model_results_dir,
    )

    split_results_file = model_results_dir / analyzer.split_results_filename
    report_file = model_results_dir / analyzer.report_filename

    split_payload = {
        "model_label": model_label,
        "summary_stats": summary,
        "individual_results": results,
        "timestamp": datetime.now().isoformat(),
    }

    report_payload = {
        "study_type": analyzer.study_type,
        "experiment_name": analyzer.experiment_name,
        "game_type": analyzer.game_type,
        "model_label": model_label,
        "data_available": analyzer.data_available,
        "semantic_info_removed": analyzer.semantic_info_removed,
        "semantic_info_available": ["user_identity"],
        "placement_anonymous": True,
        "activity_labels_in_features": False,
        "frequency_hz": analyzer.config.frequency_hz,
        "source_frequency_hz": SOURCE_FREQUENCY_HZ,
        "split_name": split_name,
        "max_combinations_requested": min(analyzer.config.max_combinations, BRANCH_MAX_COMBINATIONS),
        "combinations_tested": len(results),
        "data_root": str(analyzer.config.data_root),
        "results_dir": str(model_results_dir),
        "model_backend": summary["execution_backend"],
        "results": {split_name: summary},
        "timestamp": datetime.now().isoformat(),
    }

    if model_label == "RF":
        report_payload["baseline_slot"] = "RF"
        report_payload["backing_model"] = analyzer.rf_description
    else:
        report_payload["backing_model"] = model_label

    with split_results_file.open("w", encoding="utf-8") as file_handle:
        json.dump(_sanitize_for_json(split_payload), file_handle, indent=2)

    with report_file.open("w", encoding="utf-8") as file_handle:
        json.dump(_sanitize_for_json(report_payload), file_handle, indent=2)

    print(
        f"[{model_label}] saved {split_results_file.name} and {report_file.name} under {model_results_dir}"
    )
    print(
        f"[{model_label}] accuracy_max={summary['accuracy_max']:.4f} "
        f"accuracy_std={summary['accuracy_std']:.4f} "
        f"f1_max={summary['f1_max']:.4f} f1_std={summary['f1_std']:.4f} "
        f"nmi_max={summary['nmi_max']:.2f} nmi_std={summary['nmi_std']:.2f} "
        f"nrkl_max={summary['nrkl_max']:.2f} nrkl_std={summary['nrkl_std']:.2f} "
        f"vulnerability_max={summary['vulnerability_max']:.4f} "
        f"vulnerability_std={summary['vulnerability_std']:.4f}"
    )
    return summary


def run_identity_experiment_suite(analyzer: Any) -> dict[str, Any]:
    """Run the branch-required Identity experiment sweep for a single script/frequency."""
    split_name = analyzer.config.split_name
    requested_models = [label for label in analyzer.config.model_labels if label in ALL_MODEL_LABELS]
    max_combinations = min(analyzer.config.max_combinations, BRANCH_MAX_COMBINATIONS)

    print(
        f"[RUN] {analyzer.experiment_name}: split={split_name} "
        f"frequency={analyzer.config.frequency_hz}Hz models={requested_models} "
        f"max_combinations={max_combinations}"
    )

    if any(model_label in DL_MODEL_LABELS for model_label in requested_models):
        _get_tensorflow_runtime(require_gpu=True)

    all_combinations = analyzer.generate_all_combinations(split_name)
    sampled_combinations = sample_experiment_combinations(
        all_combinations,
        max_combinations=max_combinations,
        seed=analyzer.config.random_seed,
    )
    print(
        f"[RUN] sampled {len(sampled_combinations)} of {len(all_combinations)} combinations for {split_name}"
    )

    results_by_model: dict[str, list[dict[str, Any]]] = {
        model_label: [] for model_label in requested_models
    }

    for combination_index, combination in enumerate(sampled_combinations, start=1):
        train_users, test_users = normalize_combination(combination)
        print(
            f"[COMBO {combination_index}/{len(sampled_combinations)}] "
            f"train={train_users} test={test_users}"
        )

        prepared_data = _prepare_combination_data(
            analyzer=analyzer,
            train_users=train_users,
            test_users=test_users,
        )
        if prepared_data is None:
            print("[WARN] Skipping combination because the dataset could not be prepared.")
            continue

        for model_offset, model_label in enumerate(requested_models):
            model_seed = analyzer.config.random_seed + (combination_index * 100) + model_offset
            result = _evaluate_model_on_prepared_data(
                analyzer=analyzer,
                model_label=model_label,
                prepared_data=prepared_data,
                seed=model_seed,
            )
            result["combination_index"] = combination_index
            results_by_model[model_label].append(result)
            print(
                f"  [{model_label}] accuracy={result['accuracy']:.4f} "
                f"f1={result['f1']:.4f} nmi={result['nmi_percentage']:.2f} "
                f"nrkl={result['nrkl_percentage']:.2f} "
                f"vulnerability={result['vulnerability']:.4f}"
            )

    summaries: dict[str, Any] = {}
    for model_label in requested_models:
        model_results = results_by_model[model_label]
        if not model_results:
            raise RuntimeError(
                f"No successful results were produced for {analyzer.experiment_name} model {model_label}."
            )

        summaries[model_label] = _save_model_outputs(
            analyzer=analyzer,
            model_label=model_label,
            split_name=split_name,
            total_combinations_available=len(all_combinations),
            sampled_combinations=sampled_combinations,
            results=model_results,
        )

    return summaries
