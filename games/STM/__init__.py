"""
Sensor Time-series Model (STM) games for privacy leakage evaluation.

This module implements privacy leakage experiments using IMU-derived activity data
across different adversarial scenarios ("Games").
"""

from .config import *
from .data_loader import *
from .models import *
from .train import *
from .evaluate import *

__all__ = [
    'GAME_CONFIGS',
    'MODEL_CONFIGS', 
    'TRAIN_CONFIGS',
    'VALID_ACTIVITIES',
    'load_data_for_game',
    'build_improved_transformer',
    'build_advanced_cnn',
    'build_lstm_attention',
    'build_multimodal_transformer',
    'build_hybrid_cnn_transformer',
    'train_and_evaluate_advanced_model',
    'save_advanced_results',
    'plot_confusion_matrix'
] 