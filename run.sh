#!/bin/bash

# Speed Sensor Anomaly Detection - Run Script

echo "==================================="
echo "Speed Sensor Anomaly Detection"
echo "==================================="

# Check if model exists
if [ ! -f "models/model.keras" ]; then
    echo "⚠️  No trained model found!"
    echo "Please train a model first:"
    echo "  python main.py train"
    echo ""
fi

# Create necessary directories
mkdir -p templates static uploads api_results

# Check for data files
if [ ! -f "labelled_1_23.csv" ] || [ ! -f "input_data.csv" ]; then
    echo "⚠️  Data files missing!"
    echo "Please ensure these files exist:"
    echo "  - labelled_1_23.csv"
    echo "  - input_data.csv"
    echo ""
fi

# Run mode selection
if [ "$1" == "docker" ]; then
    echo "Starting with Docker..."
    docker-compose up --build
elif [ "$1" == "docker-prod" ]; then
    echo "Starting in production mode with Docker..."
    docker-compose --profile production up --build
elif [ "$1" == "dev" ]; then
    echo "Starting in development mode..."
    python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
else
    echo "Starting FastAPI application..."
    python app.py
fi