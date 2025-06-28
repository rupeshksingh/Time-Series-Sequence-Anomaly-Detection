#!/usr/bin/env python3
"""
Quick setup and test script for Speed Sensor Anomaly Detection
"""

import os
import sys
import subprocess


def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False


def check_data_files():
    """Check if data files exist."""
    files_ok = True
    
    print("\n📁 Checking data files...")
    
    if os.path.exists("Data\labelled_1_23.csv"):
        print("✅ Training Data Found")
    else:
        print("❌ Training Data Not Found")
        files_ok = False
    
    if os.path.exists("Data\input_data.csv"):
        print("✅ Unlabeled Test Data Found")
    else:
        print("❌ Unlabeled Test Data Not Found")
        files_ok = False
    
    return files_ok


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("✅ Directories created")


def quick_test():
    """Run a quick test."""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        print("✅ All imports successful")
        
        # Test TensorFlow CPU
        print(f"✅ TensorFlow {tf.__version__} (CPU mode)")
        
        # Test data loading
        import config
        from detector import SpeedAnomalyDetector
        print("✅ Modules loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("Speed Sensor Anomaly Detection - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Check data files
    if not check_data_files():
        print("\n⚠️  Please add your data files:")
        print("   - (training data)")
        print("   - (unlabeled data)")
    
    # Create directories
    create_directories()
    
    # Quick test
    if quick_test():
        print("\n" + "="*60)
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Train model:  python main.py train")
        print("2. Test data:    python main.py test")
        print("3. Both:         python main.py both")
        print("="*60)
        return 0
    else:
        print("\n❌ Setup failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())