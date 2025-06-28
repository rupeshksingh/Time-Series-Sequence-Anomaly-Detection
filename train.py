"""
Training script for Speed Sensor Anomaly Detection
"""

import numpy as np
from datetime import datetime

import config
from detector import SpeedAnomalyDetector
from utils import plot_training_results, analyze_data_distribution


def train_anomaly_detector():
    """Main training function."""
    print("="*60)
    print("Speed Sensor Anomaly Detection - Training")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize detector
    detector = SpeedAnomalyDetector()
    
    # Load and analyze data
    print("\n1. Loading and analyzing data...")
    train_data, val_data, test_data = detector.load_and_split_data(config.LABELED_DATA_PATH)
    
    # Analyze data distribution
    analyze_data_distribution(train_data, val_data, test_data)
    
    # Normalize data
    print("\n2. Normalizing data...")
    train_data, val_data, test_data = detector.normalize_data(train_data, val_data, test_data)
    
    # Create sequences
    print("\n3. Creating sequences...")
    train_seq, train_labels = detector.create_sequences(train_data)
    val_seq, val_labels = detector.create_sequences(val_data)
    test_seq, test_labels = detector.create_sequences(test_data)
    
    print(f"Sequences created:")
    print(f"  Train: {len(train_seq)} (Normal: {np.sum(train_labels==0)}, Anomaly: {np.sum(train_labels==1)})")
    print(f"  Val: {len(val_seq)} (Normal: {np.sum(val_labels==0)}, Anomaly: {np.sum(val_labels==1)})")
    print(f"  Test: {len(test_seq)} (Normal: {np.sum(test_labels==0)}, Anomaly: {np.sum(test_labels==1)})")
    
    # Handle case where validation has no anomalies
    if np.sum(val_labels == 1) == 0:
        print("\n⚠️  No anomalies in validation set. Adding some from training...")
        # Take some anomalies from training for validation
        train_anomaly_idx = np.where(train_labels == 1)[0]
        if len(train_anomaly_idx) > 20:
            # Move 20% of anomalies to validation
            n_move = max(10, len(train_anomaly_idx) // 5)
            move_idx = np.random.choice(train_anomaly_idx, size=n_move, replace=False)
            
            # Add to validation
            val_seq = np.vstack([val_seq, train_seq[move_idx]])
            val_labels = np.hstack([val_labels, train_labels[move_idx]])
            
            # Remove from training
            keep_idx = np.setdiff1d(np.arange(len(train_seq)), move_idx)
            train_seq = train_seq[keep_idx]
            train_labels = train_labels[keep_idx]
            
            print(f"  Moved {n_move} anomaly sequences to validation")
    
    # Train model
    print("\n4. Training model...")
    history = detector.train(train_seq, val_seq, train_labels)
    
    # Evaluate on test set
    print("\n5. Evaluating model...")
    if np.sum(test_labels == 1) > 0:
        metrics, scores, predictions = detector.evaluate(test_seq, test_labels)
        
        print("\nTest Set Performance:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  False Alarm Rate: {metrics['false_alarm_rate']:.3f}")
        print(f"  Anomalies detected: {np.sum(predictions)} / {len(predictions)}")
    else:
        print("⚠️  Cannot evaluate - no anomalies in test set!")
        print("  This explains perfect scores in your output.")
        metrics = None
        scores = None
        predictions = None
    
    # Plot results
    print("\n6. Generating visualizations...")
    plot_training_results(
        history,
        train_seq, train_labels,
        val_seq, val_labels,
        test_seq, test_labels,
        detector.threshold,
        scores, predictions
    )
    
    # Save model
    print("\n7. Saving model...")
    detector.save_model()
    
    # Save training report
    report = {
        'training_date': datetime.now().isoformat(),
        'data_stats': detector.training_stats,
        'evaluation_metrics': metrics if metrics else 'No anomalies in test set',
        'model_params': {
            'window_size': detector.window_size,
            'threshold': float(detector.threshold),
            'threshold_percentile': detector.threshold_percentile
        }
    }
    
    import json
    with open(f'{config.RESULTS_PATH}/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Report saved to: {config.RESULTS_PATH}/training_report.json")
    print("="*60)
    
    return detector


if __name__ == "__main__":
    detector = train_anomaly_detector()