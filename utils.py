"""
Utility functions for anomaly detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

import config


def analyze_data_distribution(train_data, val_data, test_data):
    """Analyze and visualize data distribution across splits."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Speed distribution
    ax1 = axes[0, 0]
    ax1.hist(train_data['speed'], bins=50, alpha=0.5, label='Train', density=True)
    ax1.hist(val_data['speed'], bins=50, alpha=0.5, label='Val', density=True)
    ax1.hist(test_data['speed'], bins=50, alpha=0.5, label='Test', density=True)
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Density')
    ax1.set_title('Speed Distribution by Split')
    ax1.legend()
    
    # Label distribution
    ax2 = axes[0, 1]
    splits = ['Train', 'Val', 'Test']
    anomaly_counts = [
        train_data['pred_label'].sum(),
        val_data['pred_label'].sum(),
        test_data['pred_label'].sum()
    ]
    normal_counts = [
        len(train_data) - anomaly_counts[0],
        len(val_data) - anomaly_counts[1],
        len(test_data) - anomaly_counts[2]
    ]
    
    x = np.arange(len(splits))
    width = 0.35
    
    ax2.bar(x - width/2, normal_counts, width, label='Normal')
    ax2.bar(x + width/2, anomaly_counts, width, label='Anomaly')
    ax2.set_xlabel('Split')
    ax2.set_ylabel('Count')
    ax2.set_title('Label Distribution by Split')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.legend()
    
    # Time series view
    ax3 = axes[1, 0]
    all_data = pd.concat([train_data, val_data, test_data])
    ax3.plot(all_data.index, all_data['speed'], 'b-', alpha=0.5, linewidth=0.5)
    ax3.axvline(len(train_data), color='green', linestyle='--', alpha=0.7, label='Train/Val')
    ax3.axvline(len(train_data) + len(val_data), color='orange', linestyle='--', alpha=0.7, label='Val/Test')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Speed')
    ax3.set_title('Speed Time Series with Splits')
    ax3.legend()
    
    # Anomaly timeline
    ax4 = axes[1, 1]
    ax4.plot(all_data.index, all_data['pred_label'], 'r-', alpha=0.7)
    ax4.axvline(len(train_data), color='green', linestyle='--', alpha=0.7)
    ax4.axvline(len(train_data) + len(val_data), color='orange', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Anomaly Label')
    ax4.set_title('Anomaly Labels Timeline')
    ax4.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/data_distribution.png', dpi=config.FIGURE_DPI)
    plt.close()


def plot_training_results(history, train_seq, train_labels, val_seq, val_labels, 
                         test_seq, test_labels, threshold, test_scores=None, test_predictions=None):
    """Plot comprehensive training results."""
    fig = plt.figure(figsize=(15, 10))
    
    # Training history
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Score distributions
    ax2 = plt.subplot(2, 3, 2)
    
    # Calculate scores for each set
    from detector import SpeedAnomalyDetector
    detector = SpeedAnomalyDetector()
    detector.threshold = threshold
    
    if test_scores is not None and len(test_scores) > 0:
        # Separate by label
        normal_scores = test_scores[test_labels == 0] if np.any(test_labels == 0) else []
        anomaly_scores = test_scores[test_labels == 1] if np.any(test_labels == 1) else []
        
        if len(normal_scores) > 0:
            ax2.hist(normal_scores, bins=50, alpha=0.5, label='Normal', density=True)
        if len(anomaly_scores) > 0:
            ax2.hist(anomaly_scores, bins=50, alpha=0.5, label='Anomaly', density=True)
        
        ax2.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Test Score Distribution')
        ax2.legend()
        ax2.set_xlim(0, np.percentile(test_scores, 99))
    
    # Confusion matrix
    ax3 = plt.subplot(2, 3, 3)
    if test_predictions is not None and test_labels is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, test_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')
    
    # Sequence length distribution
    ax4 = plt.subplot(2, 3, 4)
    seq_lengths = [
        len(train_seq),
        len(val_seq),
        len(test_seq) if test_seq is not None else 0
    ]
    ax4.bar(['Train', 'Val', 'Test'], seq_lengths)
    ax4.set_ylabel('Number of Sequences')
    ax4.set_title('Sequence Counts by Split')
    
    # Label balance
    ax5 = plt.subplot(2, 3, 5)
    if train_labels is not None:
        train_anomaly_rate = np.mean(train_labels) * 100
        val_anomaly_rate = np.mean(val_labels) * 100 if val_labels is not None else 0
        test_anomaly_rate = np.mean(test_labels) * 100 if test_labels is not None else 0
        
        ax5.bar(['Train', 'Val', 'Test'], 
                [train_anomaly_rate, val_anomaly_rate, test_anomaly_rate])
        ax5.set_ylabel('Anomaly Rate (%)')
        ax5.set_title('Anomaly Rate by Split')
        ax5.set_ylim(0, max(train_anomaly_rate, val_anomaly_rate, test_anomaly_rate) * 1.2)
    
    # Performance metrics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    if test_predictions is not None and test_labels is not None and np.any(test_labels == 1):
        from sklearn.metrics import classification_report
        report = classification_report(test_labels, test_predictions, output_dict=True)
        
        metrics_text = f"Test Set Performance:\n\n"
        metrics_text += f"Precision: {report['1']['precision']:.3f}\n"
        metrics_text += f"Recall: {report['1']['recall']:.3f}\n"
        metrics_text += f"F1-Score: {report['1']['f1-score']:.3f}\n"
        metrics_text += f"Support: {report['1']['support']}\n\n"
        metrics_text += f"Accuracy: {report['accuracy']:.3f}\n"
        metrics_text += f"Threshold: {threshold:.6f}"
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center')
    else:
        ax6.text(0.1, 0.5, "No anomalies in test set\nfor evaluation", 
                fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/training_results.png', dpi=config.FIGURE_DPI)
    plt.close()


def visualize_anomalies(df, threshold, max_points=None):
    """Visualize anomaly detection results."""
    max_points = max_points or config.MAX_PLOT_POINTS
    
    # Subsample if needed
    if len(df) > max_points:
        sample_rate = len(df) // max_points
        plot_df = df.iloc[::sample_rate].copy()
    else:
        plot_df = df.copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Speed with anomalies
    ax1 = axes[0]
    ax1.plot(plot_df.index, plot_df['speed'], 'b-', alpha=0.7, linewidth=0.5)
    
    anomalies = plot_df[plot_df['is_anomaly'] == 1]
    if len(anomalies) > 0:
        ax1.scatter(anomalies.index, anomalies['speed'], 
                   c='red', s=20, alpha=0.8, label=f'Anomalies ({len(anomalies)})')
    
    ax1.set_ylabel('Speed')
    ax1.set_title('Speed Sensor Data with Detected Anomalies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Anomaly scores
    ax2 = axes[1]
    valid_scores = plot_df.dropna(subset=['anomaly_score'])
    if len(valid_scores) > 0:
        ax2.plot(valid_scores.index, valid_scores['anomaly_score'], 
                'g-', alpha=0.7, linewidth=0.5)
        ax2.axhline(y=threshold, color='r', linestyle='--', 
                   label=f'Threshold ({threshold:.4f})')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Reconstruction Error')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Normalized speed
    ax3 = axes[2]
    if 'speed_norm' in plot_df.columns:
        ax3.plot(plot_df.index, plot_df['speed_norm'], 
                'purple', alpha=0.7, linewidth=0.5)
        ax3.set_ylabel('Normalized Speed')
        ax3.set_title('Normalized Speed Values')
        ax3.grid(True, alpha=0.3)
    
    # Anomaly density
    ax4 = axes[3]
    if 'is_anomaly' in df.columns:
        window = min(1000, len(df) // 10)
        anomaly_density = df['is_anomaly'].rolling(window=window, center=True).mean() * 100
        
        # Subsample density for plotting
        if len(df) > max_points:
            density_plot = anomaly_density.iloc[::sample_rate]
        else:
            density_plot = anomaly_density
        
        ax4.plot(density_plot.index, density_plot, 'orange', alpha=0.7)
        ax4.set_ylabel('Anomaly Rate (%)')
        ax4.set_xlabel('Sample Index')
        ax4.set_title(f'Local Anomaly Rate ({window}-point window)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/anomaly_visualization.png', dpi=config.FIGURE_DPI)
    plt.close()
    
    # Additional distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Score distribution
    ax1 = axes[0]
    if 'anomaly_score' in df.columns:
        valid_scores = df.dropna(subset=['anomaly_score'])['anomaly_score']
        if len(valid_scores) > 0:
            ax1.hist(valid_scores, bins=100, alpha=0.7, density=True, color='green')
            ax1.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
            ax1.set_xlabel('Anomaly Score')
            ax1.set_ylabel('Density')
            ax1.set_title('Anomaly Score Distribution')
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    # Speed distribution by class
    ax2 = axes[1]
    normal_speeds = df[df['is_anomaly'] == 0]['speed']
    anomaly_speeds = df[df['is_anomaly'] == 1]['speed']
    
    if len(normal_speeds) > 0:
        ax2.hist(normal_speeds, bins=50, alpha=0.5, label=f'Normal (n={len(normal_speeds)})', 
                density=True, color='blue')
    if len(anomaly_speeds) > 0:
        ax2.hist(anomaly_speeds, bins=50, alpha=0.5, label=f'Anomaly (n={len(anomaly_speeds)})', 
                density=True, color='red')
    
    ax2.set_xlabel('Speed')
    ax2.set_ylabel('Density')
    ax2.set_title('Speed Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/distributions.png', dpi=config.FIGURE_DPI)
    plt.close()


def create_anomaly_report(df, segments, detector):
    """Create comprehensive anomaly detection report."""
    report_path = f"{config.RESULTS_PATH}/anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    n_anomalies = df['is_anomaly'].sum()
    valid_scores = df.dropna(subset=['anomaly_score'])
    anomaly_rate = (n_anomalies / len(valid_scores)) * 100 if len(valid_scores) > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SPEED SENSOR ANOMALY DETECTION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {config.MODEL_SAVE_PATH}\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total data points: {len(df):,}\n")
        f.write(f"Valid predictions: {len(valid_scores):,}\n")
        f.write(f"Time range: {df['indo_time'].min()} to {df['indo_time'].max()}\n")
        f.write(f"Speed range: {df['speed'].min():.2f} to {df['speed'].max():.2f}\n\n")
        
        f.write("DETECTION RESULTS\n")
        f.write("-"*40 + "\n")
        f.write(f"Anomalies detected: {n_anomalies:,}\n")
        f.write(f"Anomaly rate: {anomaly_rate:.2f}%\n")
        f.write(f"Detection threshold: {detector.threshold:.6f}\n")
        f.write(f"Window size: {detector.window_size}\n\n")
        
        f.write("SPEED STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write("Overall:\n")
        f.write(f"  Mean: {df['speed'].mean():.2f}\n")
        f.write(f"  Std:  {df['speed'].std():.2f}\n")
        f.write(f"  Min:  {df['speed'].min():.2f}\n")
        f.write(f"  Max:  {df['speed'].max():.2f}\n\n")
        
        if n_anomalies > 0:
            f.write("Normal points:\n")
            normal_speeds = df[df['is_anomaly'] == 0]['speed']
            f.write(f"  Mean: {normal_speeds.mean():.2f}\n")
            f.write(f"  Std:  {normal_speeds.std():.2f}\n\n")
            
            f.write("Anomalous points:\n")
            anomaly_speeds = df[df['is_anomaly'] == 1]['speed']
            f.write(f"  Mean: {anomaly_speeds.mean():.2f}\n")
            f.write(f"  Std:  {anomaly_speeds.std():.2f}\n\n")
        
        if segments:
            f.write("ANOMALY SEGMENTS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total segments: {len(segments)}\n")
            
            durations = [s['duration'] for s in segments]
            f.write(f"Duration statistics:\n")
            f.write(f"  Mean: {np.mean(durations):.1f} points\n")
            f.write(f"  Max:  {np.max(durations)} points\n")
            f.write(f"  Min:  {np.min(durations)} points\n\n")
            
            f.write("Top 10 segments by anomaly score:\n")
            sorted_segments = sorted(segments, key=lambda x: x['max_score'], reverse=True)[:10]
            for i, seg in enumerate(sorted_segments):
                f.write(f"{i+1:2d}. Score: {seg['max_score']:8.4f}, "
                       f"Duration: {seg['duration']:4d}, "
                       f"Time: {seg['start_time']}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*60 + "\n")
    
    return report_path