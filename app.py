"""
FastAPI application for Speed Sensor Anomaly Detection
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

from pydantic import BaseModel
from typing import Dict, Any

# Import our modules
import config
from test import UnlabeledDataProcessor

# Create FastAPI app
app = FastAPI(
    title="Speed Sensor Anomaly Detection",
    description="Real-time anomaly detection for speed sensor data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
detector = None
processor = None
active_connections: List[WebSocket] = []
processing_status = {}

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("api_results", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Pydantic models
class DetectionRequest(BaseModel):
    threshold: Optional[float] = None
    window_size: Optional[int] = 60


class DetectionResult(BaseModel):
    job_id: str
    status: str
    total_points: int
    anomalies_detected: int
    anomaly_rate: float
    threshold: float
    processing_time: float


class ModelInfo(BaseModel):
    model_loaded: bool
    threshold: float
    window_size: int
    training_date: Optional[str]
    training_stats: Optional[Dict[str, Any]]


class RealtimeData(BaseModel):
    timestamp: str
    speed: float


# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global detector, processor
    
    try:
        processor = UnlabeledDataProcessor()
        if os.path.exists(f'{config.MODEL_SAVE_PATH}model.keras'):
            processor.load_model()
            detector = processor.detector
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No trained model found")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    with open("templates/index.html", "r") as f:
        return f.read()


@app.get("/api/model/info")
async def get_model_info() -> ModelInfo:
    """Get current model information."""
    if detector is None:
        return ModelInfo(
            model_loaded=False,
            threshold=0.0,
            window_size=60,
            training_date=None,
            training_stats=None
        )
    
    # Load params if available
    params_path = f'{config.MODEL_SAVE_PATH}params.json'
    training_stats = {}
    training_date = None
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            training_stats = params.get('training_stats', {})
            training_date = params.get('config', {}).get('created')
    
    return ModelInfo(
        model_loaded=True,
        threshold=detector.threshold,
        window_size=detector.window_size,
        training_date=training_date,
        training_stats=training_stats
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file for analysis."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = f"uploads/{file_id}_{file.filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Quick validation
    try:
        df = pd.read_csv(file_path)
        
        # Check for required columns
        if 'Time_stamp' in df.columns and 'A2:MCPGSpeed' in df.columns:
            columns = ['Time_stamp', 'A2:MCPGSpeed']
        elif 'indo_time' in df.columns and 'speed' in df.columns:
            columns = ['indo_time', 'speed']
        else:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail="CSV must contain either (Time_stamp, A2:MCPGSpeed) or (indo_time, speed) columns"
            )
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "file_path": file_path
        }
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")


@app.post("/api/detect/{file_id}")
async def detect_anomalies(file_id: str, request: DetectionRequest):
    """Run anomaly detection on uploaded file."""
    global processing_status
    
    if detector is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please train a model first.")
    
    # Find file
    file_path = None
    for f in os.listdir("uploads"):
        if f.startswith(file_id):
            file_path = f"uploads/{f}"
            break
    
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Create job
    job_id = str(uuid.uuid4())
    processing_status[job_id] = {
        "status": "processing",
        "progress": 0,
        "start_time": datetime.now()
    }
    
    # Process in background
    asyncio.create_task(process_file_async(job_id, file_path, request.threshold))
    
    return {"job_id": job_id, "status": "processing"}


async def process_file_async(job_id: str, file_path: str, threshold: Optional[float]):
    """Process file asynchronously."""
    global processing_status
    
    try:
        # Update status
        processing_status[job_id]["progress"] = 10
        
        # Load data
        df = processor.load_unlabeled_data(file_path)
        processing_status[job_id]["progress"] = 30
        
        # Override threshold if specified
        if threshold:
            original_threshold = processor.detector.threshold
            processor.detector.threshold = threshold
        
        # Process data
        results = processor.process_data(df)
        processing_status[job_id]["progress"] = 80
        
        # Analyze anomalies
        segments = processor.analyze_anomalies()
        
        # Save results
        result_path = f"api_results/{job_id}_results.csv"
        results.to_csv(result_path, index=False)
        
        # Create summary
        n_anomalies = results['is_anomaly'].sum()
        anomaly_rate = (n_anomalies / len(results)) * 100
        
        processing_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "results": {
                "total_points": len(results),
                "anomalies_detected": int(n_anomalies),
                "anomaly_rate": float(anomaly_rate),
                "threshold": float(processor.detector.threshold),
                "result_file": result_path,
                "segments": segments if segments else []
            },
            "processing_time": (datetime.now() - processing_status[job_id]["start_time"]).total_seconds()
        }
        
        # Restore threshold
        if threshold:
            processor.detector.threshold = original_threshold
            
        # Notify connected clients
        await notify_clients({
            "type": "processing_complete",
            "job_id": job_id,
            "results": processing_status[job_id]["results"]
        })
        
    except Exception as e:
        processing_status[job_id] = {
            "status": "error",
            "error": str(e)
        }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Get job processing status."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get detection results."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if processing_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    return processing_status[job_id]["results"]


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """Download results CSV."""
    result_path = f"api_results/{job_id}_results.csv"
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Results not found")
    
    return FileResponse(
        result_path,
        media_type="text/csv",
        filename=f"anomaly_results_{job_id}.csv"
    )


@app.get("/api/visualize/{job_id}")
async def get_visualization_data(job_id: str, sample_size: int = 1000):
    """Get data for visualization."""
    result_path = f"api_results/{job_id}_results.csv"
    
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Results not found")
    
    # Load results
    df = pd.read_csv(result_path)
    
    # Sample if needed
    if len(df) > sample_size:
        sample_rate = len(df) // sample_size
        df_sample = df.iloc[::sample_rate].copy()
    else:
        df_sample = df
    
    # Prepare data for visualization
    viz_data = {
        "timestamps": df_sample['indo_time'].tolist(),
        "speeds": df_sample['speed'].tolist(),
        "anomaly_scores": df_sample['anomaly_score'].fillna(0).tolist(),
        "is_anomaly": df_sample['is_anomaly'].tolist(),
        "threshold": float(processor.detector.threshold)
    }
    
    return viz_data


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time monitoring."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            if data["type"] == "realtime_data":
                # Process single data point
                speed = data["speed"]
                is_anomaly, score, features = detector.online_detection(speed)
                
                response = {
                    "type": "detection_result",
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "speed": speed,
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_score": float(score),
                    "threshold": float(detector.threshold)
                }
                
                await websocket.send_json(response)
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def notify_clients(message: dict):
    """Notify all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            pass


@app.post("/api/train")
async def train_model():
    """Trigger model training."""
    # This would typically be a background task
    return {
        "status": "Training endpoint not implemented in demo",
        "message": "Please use command line: python main.py train"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        port=8000,
        reload=True
    )