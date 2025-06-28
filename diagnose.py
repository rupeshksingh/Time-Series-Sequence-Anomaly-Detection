#!/usr/bin/env python3
"""
Diagnostic script to analyze data and model issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

import config


def diagnose_data_splits():
    """Analyze data distribution across splits."""
    print("="*60)
    print("DATA SPLIT DIAGNOSIS")
    print("="*60)
    
    # Load data
    df = pd.read_csv(config.LABELED_DATA_PATH)
    df['indo_time'] = pd.to_datetime(df['indo_time'])
    df = df.sort_values('indo_time').reset_index(drop=True)
    
    # Calculate split indices
    n_total = len(df)
    train_idx = int(config.TRAIN_RATIO * n_total)
    val_idx = int((config.TRAIN_RATIO + config.VAL_RATIO) * n_total)
    
    # Analyze each split
    splits = {
        'Train': df[:train_idx],
        'Validation': df[train_idx:val_idx],
        'Test': df[val_idx:]
    }
    
    print(f"\nTotal samples: {n_total:,}")
    print(f"Total anomalies: {df['pred_label'].sum():,} ({df['pred_label'].mean()*100:.2f}%)\n")
    
    print("Split Analysis:")
    print("-" * 50)
    print(f"{'Split':<12} {'Samples':<10} {'Anomalies':<10} {'Rate':<8}")
    print("-" * 50)
    
    for name, split_df in splits.items():
        n_samples = len(split_df)
        n_anomalies = split_df['pred_label'].sum()
        rate = (n_anomalies / n_samples * 100) if n_samples > 0 else 0
        print(f"{name:<12} {n_samples:<10,} {n_anomalies:<10,} {rate:<8.2f}%")
    
    # Check for issues
    print("\n" + "="*60)
    print("ISSUES DETECTED:")
    print("="*60)
    
    issues_found = False
    
    if splits['Validation']['pred_label'].sum() == 0:
        print("⚠️  No anomalies in validation set!")
        print("   This will make threshold optimization impossible.")
        issues_found = True
    
    if splits['Test']['pred_label'].sum() == 0:
        print("⚠️  No anomalies in test set!")
        print("   This explains the perfect scores (1.0 precision/recall).")
        print("   The model is not being tested on any anomalous sequences.")
        issues_found = True
    
    if not issues_found:
        print("✅ No major issues found in data splits.")
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Timeline view
    ax1 = axes[0]
    ax1.scatter(df.index, df['speed'], c=df['pred_label'], 
                cmap='coolwarm', alpha=0.5, s=1)
    ax1.axvline(train_idx, color='green', linestyle='--', label='Train/Val')
    ax1.axvline(val_idx, color='orange', linestyle='--', label='Val/Test')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Speed')
    ax1.set_title('Speed Data with Anomaly Labels (Red = Anomaly)')
    ax1.legend()
    
    # Anomaly distribution
    ax2 = axes[1]
    split_names = list(splits.keys())
    anomaly_counts = [splits[name]['pred_label'].sum() for name in split_names]
    normal_counts = [len(splits[name]) - splits[name]['pred_label'].sum() 
                    for name in split_names]
    
    x = np.arange(len(split_names))
    width = 0.35
    
    ax2.bar(x - width/2, normal_counts, width, label='Normal', color='blue', alpha=0.7)
    ax2.bar(x + width/2, anomaly_counts, width, label='Anomaly', color='red', alpha=0.7)
    
    # Add value labels
    for i, (normal, anomaly) in enumerate(zip(normal_counts, anomaly_counts)):
        ax2.text(i - width/2, normal + 50, str(normal), ha='center')
        ax2.text(i + width/2, anomaly + 50, str(anomaly), ha='center')
    
    ax2.set_xlabel('Data Split')
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Distribution by Split')
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_names)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_PATH}/data_diagnosis.png', dpi=config.FIGURE_DPI)
    plt.show()
    
    return splits


def diagnose_model_threshold():
    """Analyze model threshold issues."""
    print("\n" + "="*60)
    print("MODEL THRESHOLD DIAGNOSIS")
    print("="*60)
    
    params_path = f'{config.MODEL_SAVE_PATH}/params.json'
    
    if not os.path.exists(params_path):
        print("❌ No trained model found. Train a model first.")
        return
    
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    threshold = params['threshold']
    stats = params.get('training_stats', {}).get('threshold_stats', {})
    
    print(f"\nCurrent threshold: {threshold:.6f}")
    print(f"Threshold percentile: {params['threshold_percentile']}%")
    
    if stats:
        print(f"\nTraining error statistics:")
        print(f"  Mean: {stats.get('train_errors_mean', 'N/A'):.6f}")
        print(f"  Std:  {stats.get('train_errors_std', 'N/A'):.6f}")
        print(f"  Min:  {stats.get('train_errors_min', 'N/A'):.6f}")
        print(f"  Max:  {stats.get('train_errors_max', 'N/A'):.6f}")
    
    # Check for issues
    print("\n" + "="*60)
    print("THRESHOLD ISSUES:")
    print("="*60)
    
    if threshold > 1.0:
        print("⚠️  Threshold is too high (>1.0)!")
        print("   This might result in no anomalies being detected.")
        print("   Recommendation: Use --threshold 0.1 when testing")
    elif threshold < 0.001:
        print("⚠️  Threshold is very low (<0.001)!")
        print("   This might result in too many false positives.")
        print("   Recommendation: Use --threshold 0.05 when testing")
    else:
        print("✅ Threshold appears reasonable.")
    
    # Suggest values
    print("\nSuggested threshold values to try:")
    print("  Conservative (fewer detections): 0.2")
    print("  Balanced: 0.1")
    print("  Sensitive (more detections): 0.05")
    print("  Very sensitive: 0.01")


def quick_recommendations():
    """Provide quick recommendations."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. If you got perfect scores (1.0 precision/recall):")
    print("   - This is due to no anomalies in test set")
    print("   - The model is working, but evaluation is misleading")
    print("   - Focus on testing with unlabeled data instead")
    
    print("\n2. To test on unlabeled data with adjusted threshold:")
    print("   python main.py test --threshold 0.1")
    
    print("\n3. To see anomaly detection working:")
    print("   - Start with threshold 0.1")
    print("   - If no detections, try 0.05")
    print("   - If too many detections, try 0.2")
    
    print("\n4. For production use:")
    print("   - Monitor anomaly rate over time")
    print("   - Adjust threshold based on false positive tolerance")
    print("   - Consider retraining periodically")


def main():
    """Run all diagnostics."""
    print("Speed Sensor Anomaly Detection - Diagnostics")
    print("="*60)
    
    # Diagnose data splits
    splits = diagnose_data_splits()
    
    # Diagnose model threshold
    diagnose_model_threshold()
    
    # Provide recommendations
    quick_recommendations()
    
    print("\n" + "="*60)
    print("Diagnostics complete!")
    print("Check results/data_diagnosis.png for visualizations")
    print("="*60)


if __name__ == "__main__":
    main()