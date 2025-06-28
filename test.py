"""
Test trained model on unlabeled data
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

import config
from detector import SpeedAnomalyDetector
from utils import create_anomaly_report, visualize_anomalies


class UnlabeledDataProcessor:
    """Process and analyze unlabeled data."""
    
    def __init__(self):
        self.detector = SpeedAnomalyDetector()
        self.results = None
        
    def load_model(self):
        """Load trained model."""
        if not os.path.exists(f'{config.MODEL_SAVE_PATH}model.keras'):
            raise FileNotFoundError(
                f"No trained model found in {config.MODEL_SAVE_PATH}. "
                "Please run train_model.py first."
            )
        
        self.detector.load_model()
        return True
    
    def load_unlabeled_data(self, filepath):
        """Load and preprocess unlabeled data."""
        print(f"Loading unlabeled data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Handle different column names
        column_mapping = {
            'Time_stamp': 'indo_time',
            'A2:MCPGSpeed': 'speed'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert timestamp
        df['indo_time'] = pd.to_datetime(df['indo_time'])
        df = df.sort_values('indo_time').reset_index(drop=True)
        
        # Display info
        print(f"Data shape: {df.shape}")
        print(f"Time range: {df['indo_time'].min()} to {df['indo_time'].max()}")
        print(f"Speed range: {df['speed'].min():.2f} to {df['speed'].max():.2f}")
        
        return df
    
    def process_data(self, df):
        """Process data and detect anomalies."""
        print("\nProcessing data...")
        
        # Normalize using trained scaler
        df['speed_norm'] = self.detector.scaler.transform(df[['speed']].values)
        
        # Initialize results columns
        df['anomaly_score'] = np.nan
        df['is_anomaly'] = 0
        
        # Process in batches for efficiency
        batch_size = 1000
        n_batches = (len(df) - self.detector.window_size) // batch_size + 1
        
        print(f"Processing {n_batches} batches...")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size + self.detector.window_size, len(df))
            
            if end_idx - start_idx < self.detector.window_size:
                continue
            
            # Create sequences for this batch
            sequences = []
            indices = []
            
            for i in range(start_idx, end_idx - self.detector.window_size + 1):
                seq = df['speed_norm'].values[i:i + self.detector.window_size]
                sequences.append(seq)
                indices.append(i + self.detector.window_size - 1)  # Index of last point
            
            if not sequences:
                continue
            
            # Predict anomalies
            sequences = np.array(sequences).reshape(-1, self.detector.window_size, 1)
            predictions = self.detector.model.predict(sequences, batch_size=64, verbose=0)
            scores = np.mean((sequences - predictions) ** 2, axis=(1, 2))
            
            # Update results
            for idx, score in zip(indices, scores):
                df.loc[idx, 'anomaly_score'] = score
                df.loc[idx, 'is_anomaly'] = int(score > self.detector.threshold)
            
            # Progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{n_batches} batches")
        
        # Store results
        self.results = df
        
        # Calculate summary statistics
        valid_scores = df.dropna(subset=['anomaly_score'])
        n_anomalies = valid_scores['is_anomaly'].sum()
        anomaly_rate = (n_anomalies / len(valid_scores)) * 100 if len(valid_scores) > 0 else 0
        
        print(f"\nProcessing complete!")
        print(f"Valid predictions: {len(valid_scores)}/{len(df)}")
        print(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")
        
        return df
    
    def analyze_anomalies(self):
        """Analyze detected anomalies."""
        if self.results is None:
            return
        
        df = self.results
        anomalies = df[df['is_anomaly'] == 1]
        
        if len(anomalies) == 0:
            print("\nNo anomalies detected!")
            return
        
        print(f"\nAnomaly Analysis:")
        print(f"Total anomalies: {len(anomalies)}")
        
        # Find anomaly segments
        segments = []
        in_segment = False
        start_idx = None
        
        for idx in df.index:
            if df.loc[idx, 'is_anomaly'] == 1 and not in_segment:
                in_segment = True
                start_idx = idx
            elif df.loc[idx, 'is_anomaly'] == 0 and in_segment:
                in_segment = False
                segments.append({
                    'start': start_idx,
                    'end': idx - 1,
                    'duration': idx - start_idx,
                    'start_time': df.loc[start_idx, 'indo_time'],
                    'end_time': df.loc[idx - 1, 'indo_time'],
                    'max_score': df.loc[start_idx:idx-1, 'anomaly_score'].max(),
                    'mean_speed': df.loc[start_idx:idx-1, 'speed'].mean()
                })
        
        # Handle last segment
        if in_segment:
            segments.append({
                'start': start_idx,
                'end': len(df) - 1,
                'duration': len(df) - start_idx,
                'start_time': df.loc[start_idx, 'indo_time'],
                'end_time': df.iloc[-1]['indo_time'],
                'max_score': df.loc[start_idx:, 'anomaly_score'].max(),
                'mean_speed': df.loc[start_idx:, 'speed'].mean()
            })
        
        if segments:
            print(f"\nFound {len(segments)} anomaly segments")
            durations = [s['duration'] for s in segments]
            print(f"Duration stats: mean={np.mean(durations):.1f}, max={np.max(durations)}, min={np.min(durations)}")
            
            # Show top anomalies
            print("\nTop 5 anomaly segments by score:")
            sorted_segments = sorted(segments, key=lambda x: x['max_score'], reverse=True)[:5]
            for i, seg in enumerate(sorted_segments):
                print(f"{i+1}. Score: {seg['max_score']:.4f}, Duration: {seg['duration']}, Time: {seg['start_time']}")
        
        return segments


def main():
    """Main execution function."""
    print("="*60)
    print("Speed Sensor Anomaly Detection - Testing on Unlabeled Data")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize processor
    processor = UnlabeledDataProcessor()
    
    # Load model
    try:
        processor.load_model()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Load unlabeled data
    df = processor.load_unlabeled_data(config.UNLABELED_DATA_PATH)
    
    # Process data
    results = processor.process_data(df)
    
    # Analyze anomalies
    segments = processor.analyze_anomalies()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_anomalies(results, processor.detector.threshold)
    
    # Save results
    print("\nSaving results...")
    
    # Save full results
    output_path = f"{config.RESULTS_PATH}/unlabeled_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # Save anomalies only
    anomalies = results[results['is_anomaly'] == 1]
    if len(anomalies) > 0:
        anomaly_path = output_path.replace('.csv', '_anomalies.csv')
        anomalies.to_csv(anomaly_path, index=False)
        print(f"Anomalies saved to: {anomaly_path}")
    
    # Create report
    report_path = create_anomaly_report(results, segments, processor.detector)
    print(f"Report saved to: {report_path}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()