#!/usr/bin/env python3
"""
Main script for Speed Sensor Anomaly Detection
Usage: python main.py [train|test|both]
"""

import sys
import os
import argparse
from datetime import datetime

import config
from train import train_anomaly_detector
from test import main as test_unlabeled_data


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_data_files():
    """Check if required data files exist."""
    issues = []
    
    if not os.path.exists(config.LABELED_DATA_PATH):
        issues.append(f"Labeled data not found: {config.LABELED_DATA_PATH}")
    
    if not os.path.exists(config.UNLABELED_DATA_PATH):
        issues.append(f"Unlabeled data not found: {config.UNLABELED_DATA_PATH}")
    
    if issues:
        print("\n❌ Data file issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Speed Sensor Anomaly Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train              # Train new model
  python main.py test               # Test on unlabeled data
  python main.py both               # Train and test
  python main.py test --threshold 0.05  # Test with custom threshold
        """
    )
    
    parser.add_argument(
        'action',
        choices=['train', 'test', 'both'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Override detection threshold for testing'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if model exists'
    )
    
    args = parser.parse_args()
    
    # Header
    print_header("SPEED SENSOR ANOMALY DETECTION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Please ensure data files are in the correct location.")
        return 1
    
    # Execute requested action
    try:
        if args.action in ['train', 'both']:
            # Check if model exists
            model_exists = os.path.exists(f'{config.MODEL_SAVE_PATH}/model.keras')
            
            if model_exists and not args.force_retrain:
                print("\n⚠️  Model already exists. Use --force-retrain to retrain.")
                if args.action == 'train':
                    return 0
            else:
                print_header("TRAINING MODEL")
                detector = train_anomaly_detector()
        
        if args.action in ['test', 'both']:
            print_header("TESTING ON UNLABELED DATA")
            
            # Check if model exists
            if not os.path.exists(f'{config.MODEL_SAVE_PATH}/model.keras'):
                print("\n❌ No trained model found. Please train first:")
                print("   python main.py train")
                return 1
            
            # Override threshold if specified
            if args.threshold:
                print(f"\n📌 Using custom threshold: {args.threshold}")
                # Temporarily modify the loaded model's threshold
                import json
                params_path = f'{config.MODEL_SAVE_PATH}/params.json'
                with open(params_path, 'r') as f:
                    params = json.load(f)
                original_threshold = params['threshold']
                params['threshold'] = args.threshold
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=2)
                
                # Run test
                results = test_unlabeled_data()
                
                # Restore original threshold
                params['threshold'] = original_threshold
                with open(params_path, 'w') as f:
                    json.dump(params, f, indent=2)
            else:
                results = test_unlabeled_data()
        
        print_header("COMPLETE")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ All operations completed successfully!")
        
        # Summary
        print("\n📁 Output locations:")
        print(f"   - Model: {config.MODEL_SAVE_PATH}")
        print(f"   - Results: {config.RESULTS_PATH}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # If no arguments, show help
    if len(sys.argv) == 1:
        sys.argv.append('--help')
    
    sys.exit(main())