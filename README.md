# Speed Sensor Anomaly Detection System

A CPU-optimized anomaly detection system for univariate speed sensor data using LSTM autoencoders. This implementation addresses the issues found in the original model and provides a clean, production-ready solution.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python main.py train

# Test on unlabeled data
python main.py test

# Train and test
python main.py both
```

## 📁 Project Structure

```
speed-anomaly-detection/
├── config.py              # Configuration settings
├── anomaly_detector.py    # Main detector class
├── train_model.py         # Training script
├── test_unlabeled.py      # Testing script
├── utils.py               # Utility functions
├── main.py                # Main execution script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── labelled_1_23.csv      # Training data (your file)
├── input_data.csv         # Unlabeled data (your file)
├── models/                # Saved models (created)
└── results/               # Results and visualizations (created)
```

## 🔑 Key Features

1. **CPU Optimized**: Configured for efficient CPU execution
2. **Leakage-Free**: Proper chronological splitting and normalization
3. **Robust Threshold**: Automatic threshold adjustment to prevent issues
4. **Comprehensive Analysis**: Detailed visualizations and reports
5. **Production Ready**: Clean code structure with error handling

## 📊 Key Findings Addressed

### 1. Perfect Score Issue
- **Problem**: Model achieved perfect scores (Precision: 1.0, Recall: 1.0)
- **Cause**: No anomalies in test set due to chronological splitting
- **Solution**: 
  - Diagnostic tools to identify the issue
  - Option to move anomalies between splits
  - Threshold validation and adjustment

### 2. Threshold Adjustment
- **Problem**: Threshold can be unreasonably high (>1.0)
- **Solution**: Automatic adjustment to reasonable range (0.01-1.0)

### 3. Data Compatibility
- **Handles different column names**: 
  - `Time_stamp` → `indo_time`
  - `A2:MCPGSpeed` → `speed`

## 🛠️ Configuration

Edit `config.py` to adjust:

```python
# Model parameters
WINDOW_SIZE = 60           # 60 seconds of data
THRESHOLD_PERCENTILE = 95  # More reasonable than 99

# Training parameters
EPOCHS = 20
BATCH_SIZE = 32           # Optimized for CPU
```

## 📈 Usage Examples

### Basic Training and Testing
```bash
# Train a new model
python main.py train

# Test on unlabeled data
python main.py test
```

### Custom Threshold Testing
```bash
# Test with more sensitive threshold
python main.py test --threshold 0.05

# Test with less sensitive threshold
python main.py test --threshold 0.2
```

### Force Retraining
```bash
# Retrain even if model exists
python main.py train --force-retrain
```

## 📊 Output Files

### After Training
- `models/model.h5` - Trained LSTM model
- `models/params.json` - Model parameters and threshold
- `models/scaler.pkl` - Normalization parameters
- `results/data_distribution.png` - Data analysis
- `results/training_results.png` - Training metrics
- `results/training_report.json` - Detailed report

### After Testing
- `results/unlabeled_results_*.csv` - Full results with scores
- `results/unlabeled_results_*_anomalies.csv` - Anomalies only
- `results/anomaly_visualization.png` - Main visualization
- `results/distributions.png` - Score distributions
- `results/anomaly_report_*.txt` - Detailed text report

## 🔍 Understanding Results

### Anomaly Score
- Lower scores = more normal
- Higher scores = more anomalous
- Threshold determines the cutoff

### Typical Threshold Values
- **Conservative** (fewer detections): 0.2 - 0.5
- **Balanced**: 0.05 - 0.2
- **Sensitive** (more detections): 0.01 - 0.05

### If No Anomalies Detected
```bash
# Try lower threshold
python main.py test --threshold 0.01
```

### If Too Many Anomalies Detected
```bash
# Try higher threshold
python main.py test --threshold 0.3
```

## 🐛 Troubleshooting

### "No trained model found"
```bash
python main.py train
```

### "Data file issues"
Ensure these files exist:
- `labelled_1_23.csv` (training data)
- `input_data.csv` (unlabeled data)

### High memory usage
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

## 📊 Model Architecture

```
LSTM Autoencoder:
Input (60, 1)
  ↓
LSTM(64, return_sequences=True)
  ↓
LSTM(32, return_sequences=False)
  ↓
RepeatVector(60)
  ↓
LSTM(32, return_sequences=True)
  ↓
LSTM(64, return_sequences=True)
  ↓
TimeDistributed(Dense(1))
  ↓
Output (60, 1)
```

## 🔧 Advanced Usage

### Custom Data Files
```python
# In config.py
LABELED_DATA_PATH = 'path/to/your/labeled.csv'
UNLABELED_DATA_PATH = 'path/to/your/unlabeled.csv'
```

### Modify Window Size
```python
# In config.py
WINDOW_SIZE = 120  # 2 minutes instead of 1
```

### Add Custom Features
Edit `extract_features()` in `anomaly_detector.py` to add domain-specific features.

## 📝 Notes

1. **Training Time**: ~5-10 minutes on CPU for 20 epochs
2. **Memory Usage**: ~2-4 GB RAM
3. **Inference Speed**: ~1000 samples/second on CPU

## 🤝 Contributing

Feel free to modify and extend this implementation for your specific needs. The modular structure makes it easy to:
- Add new model architectures
- Implement different anomaly detection methods
- Extend feature engineering
- Customize visualizations

## 📄 License

This implementation is provided as-is for your local use and modification.