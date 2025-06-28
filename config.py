"""
Configuration file for Speed Sensor Anomaly Detection
"""

import os

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data paths
LABELED_DATA_PATH = 'Data\labelled_1_23.csv'
UNLABELED_DATA_PATH = 'Data\input_data.csv'
MODEL_SAVE_PATH = 'models/'
RESULTS_PATH = 'results/'

# Model parameters
WINDOW_SIZE = 60  # 60 seconds of data
STEP_SIZE = 1     # 1 second step for sliding window
THRESHOLD_PERCENTILE = 95  # More reasonable than 99

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32  # Optimized for CPU
VALIDATION_SPLIT = 0.15

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature engineering parameters
ROLLING_WINDOW_SIZES = [30, 60]  # seconds
LAG_FEATURES = [1, 5, 10]
FFT_FREQ_BANDS = [(0.1, 1.0)]  # Hz

# Threshold adjustment parameters
MIN_THRESHOLD = 0.01
MAX_THRESHOLD = 1.0
THRESHOLD_ADJUSTMENT_FACTOR = 0.1

# Visualization parameters
MAX_PLOT_POINTS = 10000
FIGURE_DPI = 120

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)