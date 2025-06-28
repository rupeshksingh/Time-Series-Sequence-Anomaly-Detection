"""
Main Anomaly Detector Class - CPU Optimized
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import json
from datetime import datetime
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

import config

# Configure TensorFlow for CPU
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)


class SpeedAnomalyDetector:
    """
    CPU-optimized anomaly detector for speed sensor data.
    Incorporates all findings from the analysis.
    """
    
    def __init__(self, window_size=None, step_size=None, threshold_percentile=None):
        self.window_size = window_size or config.WINDOW_SIZE
        self.step_size = step_size or config.STEP_SIZE
        self.threshold_percentile = threshold_percentile or config.THRESHOLD_PERCENTILE
        
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.training_stats = {}
        
    def load_and_split_data(self, filepath):
        """Load data with proper chronological splitting."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Convert timestamp
        df['indo_time'] = pd.to_datetime(df['indo_time'])
        df = df.sort_values('indo_time').reset_index(drop=True)
        
        # Chronological split
        n_total = len(df)
        train_idx = int(config.TRAIN_RATIO * n_total)
        val_idx = int((config.TRAIN_RATIO + config.VAL_RATIO) * n_total)
        
        train_data = df[:train_idx].copy()
        val_data = df[train_idx:val_idx].copy()
        test_data = df[val_idx:].copy()
        
        # Store split information
        self.training_stats['data_splits'] = {
            'total': n_total,
            'train': len(train_data),
            'val': len(val_data),
            'test': len(test_data),
            'train_anomalies': int(train_data['pred_label'].sum()),
            'val_anomalies': int(val_data['pred_label'].sum()),
            'test_anomalies': int(test_data['pred_label'].sum())
        }
        
        print(f"Data split - Train: {len(train_data)} ({self.training_stats['data_splits']['train_anomalies']} anomalies)")
        print(f"           Val: {len(val_data)} ({self.training_stats['data_splits']['val_anomalies']} anomalies)")
        print(f"           Test: {len(test_data)} ({self.training_stats['data_splits']['test_anomalies']} anomalies)")
        
        # Warning if no anomalies in validation/test
        if self.training_stats['data_splits']['val_anomalies'] == 0:
            print("⚠️  WARNING: No anomalies in validation set!")
        if self.training_stats['data_splits']['test_anomalies'] == 0:
            print("⚠️  WARNING: No anomalies in test set! This explains perfect scores.")
            
        return train_data, val_data, test_data
    
    def normalize_data(self, train_data, val_data=None, test_data=None):
        """Normalize data using training statistics only."""
        # Fit scaler on training data only
        self.scaler.fit(train_data[['speed']].values)
        
        # Store normalization parameters
        self.training_stats['normalization'] = {
            'mean': float(self.scaler.mean_[0]),
            'std': float(self.scaler.scale_[0])
        }
        
        # Transform all sets
        train_data['speed_norm'] = self.scaler.transform(train_data[['speed']].values)
        
        if val_data is not None:
            val_data['speed_norm'] = self.scaler.transform(val_data[['speed']].values)
        if test_data is not None:
            test_data['speed_norm'] = self.scaler.transform(test_data[['speed']].values)
        return train_data, val_data, test_data
    
    def create_sequences(self, data, target_col='speed_norm', label_col='pred_label'):
        """Create sequences with proper windowing."""
        sequences = []
        labels = []
        
        values = data[target_col].values
        label_values = data[label_col].values if label_col in data.columns else None
        
        # Create sequences
        for i in range(0, len(values) - self.window_size + 1, self.step_size):
            seq = values[i:i + self.window_size]
            sequences.append(seq)
            
            if label_values is not None:
                # Label is 1 if any point in window is anomalous
                label = 1 if np.any(label_values[i:i + self.window_size] == 1) else 0
                labels.append(label)
        
        return np.array(sequences), np.array(labels) if labels else None
    
    def extract_features(self, sequence):
        """Extract comprehensive features from sequence."""
        features = {}
        
        # Basic features
        features['current'] = sequence[-1]
        features['mean'] = np.mean(sequence)
        features['std'] = np.std(sequence)
        features['min'] = np.min(sequence)
        features['max'] = np.max(sequence)
        
        # Lag features
        for lag in config.LAG_FEATURES:
            if len(sequence) > lag:
                features[f'lag_{lag}'] = sequence[-lag-1]
        
        # Rolling statistics
        for window in config.ROLLING_WINDOW_SIZES:
            if len(sequence) >= window:
                window_data = sequence[-window:]
                features[f'mean_{window}s'] = np.mean(window_data)
                features[f'std_{window}s'] = np.std(window_data)
        
        # Difference features
        if len(sequence) >= 2:
            features['diff_1'] = sequence[-1] - sequence[-2]
        if len(sequence) >= 5:
            features['diff_5'] = sequence[-1] - sequence[-6]
        
        # Spectral features
        if len(sequence) >= self.window_size:
            fft_vals = np.abs(fft(sequence))
            freqs = fftfreq(len(sequence), d=1.0)
            
            for low_freq, high_freq in config.FFT_FREQ_BANDS:
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                features[f'fft_energy_{low_freq}_{high_freq}Hz'] = np.sum(fft_vals[band_mask])
        
        return features
    
    def build_model(self):
        """Build CPU-optimized LSTM autoencoder."""
        model = models.Sequential([
            # Encoder
            layers.LSTM(64, activation='relu', return_sequences=True,
                       input_shape=(self.window_size, 1)),
            layers.LSTM(32, activation='relu', return_sequences=False),
            
            # Decoder
            layers.RepeatVector(self.window_size),
            layers.LSTM(32, activation='relu', return_sequences=True),
            layers.LSTM(64, activation='relu', return_sequences=True),
            layers.TimeDistributed(layers.Dense(1))
        ])
        
        # Compile with CPU-friendly optimizer
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train(self, train_sequences, val_sequences, train_labels=None):
        """Train model with proper threshold calculation."""
        print("Training model...")
        
        # Use only normal sequences for training
        if train_labels is not None:
            normal_mask = train_labels == 0
            train_sequences = train_sequences[normal_mask]
            print(f"Training on {len(train_sequences)} normal sequences")
        
        # Reshape for LSTM
        train_sequences = train_sequences.reshape((-1, self.window_size, 1))
        val_sequences = val_sequences.reshape((-1, self.window_size, 1))
        
        # Build model
        self.model = self.build_model()
        
        # Train
        history = self.model.fit(
            train_sequences, train_sequences,
            validation_data=(val_sequences, val_sequences),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=1
        )
        
        # Calculate threshold
        self._calculate_threshold(train_sequences)
        
        # Store training history
        self.training_stats['history'] = history.history
        
        return history
    
    def _calculate_threshold(self, train_sequences):
        """Calculate and validate threshold."""
        print("Calculating threshold...")
        
        # Get reconstruction errors
        predictions = self.model.predict(train_sequences, batch_size=config.BATCH_SIZE)
        mse_scores = np.mean((train_sequences - predictions) ** 2, axis=(1, 2))
        
        # Calculate threshold
        self.threshold = np.percentile(mse_scores, self.threshold_percentile)
        
        # Validate threshold
        if self.threshold > config.MAX_THRESHOLD:
            print(f"⚠️  Threshold too high ({self.threshold:.4f}), adjusting to {config.THRESHOLD_ADJUSTMENT_FACTOR}")
            self.threshold = config.THRESHOLD_ADJUSTMENT_FACTOR
        elif self.threshold < config.MIN_THRESHOLD:
            print(f"⚠️  Threshold too low ({self.threshold:.4f}), adjusting to {config.MIN_THRESHOLD}")
            self.threshold = config.MIN_THRESHOLD
        
        print(f"Anomaly threshold set at: {self.threshold:.6f}")
        
        # Store statistics
        self.training_stats['threshold_stats'] = {
            'threshold': float(self.threshold),
            'percentile': float(self.threshold_percentile),
            'train_errors_mean': float(np.mean(mse_scores)),
            'train_errors_std': float(np.std(mse_scores)),
            'train_errors_min': float(np.min(mse_scores)),
            'train_errors_max': float(np.max(mse_scores))
        }
    
    def predict_anomaly(self, sequence):
        """Predict if sequence is anomalous."""
        sequence = sequence.reshape((1, self.window_size, 1))
        prediction = self.model.predict(sequence, verbose=0)
        mse = np.mean((sequence - prediction) ** 2)
        
        return mse > self.threshold, float(mse)
    
    def evaluate(self, test_sequences, test_labels):
        """Evaluate model performance."""
        test_sequences = test_sequences.reshape((-1, self.window_size, 1))
        
        # Get predictions
        predictions = self.model.predict(test_sequences, batch_size=config.BATCH_SIZE)
        mse_scores = np.mean((test_sequences - predictions) ** 2, axis=(1, 2))
        
        # Binary predictions
        pred_labels = (mse_scores > self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(test_labels, pred_labels, zero_division=0),
            'recall': recall_score(test_labels, pred_labels, zero_division=0),
            'f1_score': f1_score(test_labels, pred_labels, zero_division=0),
            'support': int(np.sum(test_labels)),
            'anomaly_rate': float(np.mean(pred_labels))
        }
        
        # False alarm rate
        if np.sum(test_labels == 0) > 0:
            metrics['false_alarm_rate'] = float(
                np.sum((pred_labels == 1) & (test_labels == 0)) / np.sum(test_labels == 0)
            )
        else:
            metrics['false_alarm_rate'] = 0.0
        
        return metrics, mse_scores, pred_labels
    
    def save_model(self, base_path=None):
        """Save model and parameters."""
        base_path = base_path or config.MODEL_SAVE_PATH
        
        # Save model
        self.model.save(f'{base_path}/model.keras')
        
        # Save parameters
        params = {
            'window_size': self.window_size,
            'step_size': self.step_size,
            'threshold': float(self.threshold),
            'threshold_percentile': self.threshold_percentile,
            'training_stats': self.training_stats,
            'config': {
                'version': '1.0',
                'created': datetime.now().isoformat()
            }
        }
        
        with open(f'{base_path}/params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        # Save scaler
        joblib.dump(self.scaler, f'{base_path}/scaler.pkl')
        
        print(f"Model saved to {base_path}")
    
    def load_model(self, base_path=None):
        """Load saved model."""
        base_path = base_path or config.MODEL_SAVE_PATH
        
        # Load model
        self.model = tf.keras.models.load_model(f'{base_path}model.keras')
        
        # Load parameters
        with open(f'{base_path}params.json', 'r') as f:
            params = json.load(f)
        
        self.window_size = params['window_size']
        self.step_size = params['step_size']
        self.threshold = params['threshold']
        self.threshold_percentile = params['threshold_percentile']
        self.training_stats = params.get('training_stats', {})
        
        # Load scaler
        self.scaler = joblib.load(f'{base_path}scaler.pkl')
        
        print(f"Model loaded from {base_path}")
        print(f"Threshold: {self.threshold:.6f}")