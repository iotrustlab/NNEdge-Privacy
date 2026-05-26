import os

# Data paths - centralized configuration
# Update these paths to match your local setup
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Data")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "STM")

# Placement file mapping for multi-sensor support
# Based on actual file names found in the dataset
PLACEMENT_FILES = {
    "right-wrist": ["right-wrist.csv"],
    "left-wrist": ["left-wrist.csv"],
    "right-pocket": ["right-pocket.csv"],
    "right-ankle": ["right-ankle.csv"],
    "left-ankle": ["left-ankle.csv"],
}

# Expected order of placements in the stacked data
PLACEMENT_ORDER = ["right-wrist", "left-wrist", "right-pocket", "right-ankle", "left-ankle"]

# Binary feature name within each placement CSV
BINARY_FEATURE_NAME = "dec_tree_out_1"

# Toggle for timestamp alignment in multi-sensor data
USE_TIMESTAMP_ALIGNMENT = True

# Game-specific configurations
GAME_CONFIGS = {
    "Game-1": {
        "input_features": ["dec_tree_out_1"],  # Binary time series
        "input_shape": (100, 1),
        "description": "Binary shake/no-shake classification"
    },
    "Game-2-accel": {
        "input_features": ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]", "dec_tree_out_1"],  # Accelerometer + Binary
        "input_shape": (100, 4),
        "description": "Accelerometer + Binary classification"
    },
    "Game-2-gyro": {
        "input_features": ["gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]", "dec_tree_out_1"],  # Gyroscope + Binary
        "input_shape": (100, 4),
        "description": "Gyroscope + Binary classification"
    },
    "Game-3": {
        "input_features": ["acc_x[mg]", "acc_y[mg]", "acc_z[mg]", "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]", "dec_tree_out_1"],  # Full IMU + Binary
        "input_shape": (100, 7),
        "description": "Full IMU + Binary classification"
    }
}

# Model configurations
MODEL_CONFIGS = {
    "Transformer": {
        "num_layers": 2,  # Reduced from 3
        "head_size": 32,   # Reduced from 64
        "num_heads": 2,    # Reduced from 4
        "ff_dim": 64,      # Reduced from 128
        "dropout": 0.05,   # Reduced from 0.1
        "embedding_dim": 16  # New: embedding for binary features
    },
    "CNN": {
        "filters": [32, 64, 128],
        "kernel_sizes": [3, 3, 3],
        "dropout": 0.3
    },
    "RNN": {
        "units": [64, 32],
        "dropout": 0.2
    }
}

# Training configurations optimized for Metal GPU
TRAIN_CONFIGS = {
    "epochs": 25,  # Reduced for faster training
    "batch_size": 128,  # Increased for better GPU utilization (Metal can handle larger batches)
    "validation_split": 0.2,
    "early_stopping_patience": 8  # Reduced for faster convergence
}

# Valid activities (matching actual data directory names)
VALID_ACTIVITIES = {"Jogging", "Laying", "Sitting", "Standing", "Upstairs", "Downstairs", "Walking"}

# Window size for segmentation
WINDOW_SIZE = 100

# Train/test ratios
TRAIN_TEST_RATIOS = [(0.8, 0.2), (0.7, 0.3), (0.5, 0.5)]

# Debug mode for 3-class training (jogging, laying, walking)
DEBUG_MODE = False
DEBUG_ACTIVITIES = ["Jogging", "Laying", "Walking"]

def get_data_paths(game_name):
    """Get data and results paths for a specific game"""
    return {
        "data_root": DATA_ROOT,
        "results_root": os.path.join(RESULTS_ROOT, game_name)
    }

def get_model_config(model_name):
    """Get model configuration for a specific model"""
    return MODEL_CONFIGS.get(model_name, {})

def get_game_config(game_name):
    """Get game configuration for a specific game"""
    config = GAME_CONFIGS.get(game_name, {})
    
    # Override activities for debug mode
    if DEBUG_MODE:
        config["valid_activities"] = DEBUG_ACTIVITIES
    
    return config 