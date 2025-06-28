#!/usr/bin/env python3
"""
Example API client for Speed Sensor Anomaly Detection
"""

import requests
import time
import json
import pandas as pd
from typing import Optional, Dict, Any
import websocket
import threading


class AnomalyDetectionClient:
    """Client for interacting with the Anomaly Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_health(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information."""
        response = self.session.get(f"{self.base_url}/api/model/info")
        response.raise_for_status()
        return response.json()
    
    def upload_file(self, file_path: str) -> str:
        """Upload a CSV file for analysis."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(f"{self.base_url}/api/upload", files=files)
            response.raise_for_status()
            return response.json()['file_id']
    
    def detect_anomalies(self, file_id: str, threshold: Optional[float] = None) -> str:
        """Start anomaly detection on uploaded file."""
        data = {}
        if threshold is not None:
            data['threshold'] = threshold
            
        response = self.session.post(
            f"{self.base_url}/api/detect/{file_id}",
            json=data
        )
        response.raise_for_status()
        return response.json()['job_id']
    
    def wait_for_results(self, job_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for job to complete and return results."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/api/job/{job_id}")
            response.raise_for_status()
            status = response.json()
            
            if status['status'] == 'completed':
                # Get full results
                results_response = self.session.get(f"{self.base_url}/api/results/{job_id}")
                results_response.raise_for_status()
                return results_response.json()
            
            elif status['status'] == 'error':
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            
            # Wait before polling again
            time.sleep(1)
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")
    
    def download_results(self, job_id: str, output_path: str):
        """Download results as CSV file."""
        response = self.session.get(f"{self.base_url}/api/download/{job_id}")
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    def get_visualization_data(self, job_id: str, sample_size: int = 1000) -> Dict[str, Any]:
        """Get visualization data for a completed job."""
        response = self.session.get(
            f"{self.base_url}/api/visualize/{job_id}",
            params={'sample_size': sample_size}
        )
        response.raise_for_status()
        return response.json()
    
    def analyze_file(self, file_path: str, threshold: Optional[float] = None,
                    download_path: Optional[str] = None) -> Dict[str, Any]:
        """Complete workflow: upload, analyze, and get results."""
        print(f"Uploading {file_path}...")
        file_id = self.upload_file(file_path)
        print(f"File uploaded with ID: {file_id}")
        
        print("Starting anomaly detection...")
        job_id = self.detect_anomalies(file_id, threshold)
        print(f"Job started with ID: {job_id}")
        
        print("Waiting for results...")
        results = self.wait_for_results(job_id)
        
        print(f"\nResults:")
        print(f"  Total points: {results['total_points']:,}")
        print(f"  Anomalies detected: {results['anomalies_detected']:,}")
        print(f"  Anomaly rate: {results['anomaly_rate']:.2f}%")
        print(f"  Threshold used: {results['threshold']:.6f}")
        
        if download_path:
            print(f"\nDownloading results to {download_path}...")
            self.download_results(job_id, download_path)
            print("Download complete!")
        
        return results


class RealtimeMonitor:
    """Real-time monitoring via WebSocket."""
    
    def __init__(self, ws_url: str = "ws://localhost:8000/ws"):
        self.ws_url = ws_url
        self.ws = None
        self.running = False
        
    def on_message(self, ws, message):
        """Handle incoming messages."""
        data = json.loads(message)
        if data['type'] == 'detection_result':
            if data['is_anomaly']:
                print(f"🚨 ANOMALY DETECTED!")
                print(f"   Time: {data['timestamp']}")
                print(f"   Speed: {data['speed']:.2f}")
                print(f"   Score: {data['anomaly_score']:.6f}")
                print(f"   Threshold: {data['threshold']:.6f}")
            else:
                print(f"✓ Normal - Speed: {data['speed']:.2f}, Score: {data['anomaly_score']:.6f}")
    
    def on_error(self, ws, error):
        """Handle errors."""
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle connection close."""
        print("WebSocket connection closed")
        self.running = False
    
    def on_open(self, ws):
        """Handle connection open."""
        print("WebSocket connection established")
        self.running = True
        
        # Start sending data
        def send_data():
            import random
            while self.running:
                # Generate random speed data
                speed = 23000 + random.gauss(0, 500)
                data = {
                    'type': 'realtime_data',
                    'speed': speed,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                ws.send(json.dumps(data))
                time.sleep(1)
        
        thread = threading.Thread(target=send_data)
        thread.daemon = True
        thread.start()
    
    def start(self):
        """Start real-time monitoring."""
        print("Starting real-time monitoring...")
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        try:
            self.ws.run_forever()
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            self.stop()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.ws:
            self.ws.close()


def main():
    """Example usage of the API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Anomaly Detection API Client')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--action', choices=['info', 'analyze', 'monitor'], required=True,
                      help='Action to perform')
    parser.add_argument('--file', help='CSV file to analyze')
    parser.add_argument('--threshold', type=float, help='Detection threshold')
    parser.add_argument('--output', help='Output file path for results')
    
    args = parser.parse_args()
    
    # Create client
    client = AnomalyDetectionClient(args.url)
    
    try:
        # Check health
        health = client.check_health()
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        
        if args.action == 'info':
            # Get model info
            info = client.get_model_info()
            print(f"\nModel Information:")
            print(f"  Loaded: {info['model_loaded']}")
            print(f"  Threshold: {info['threshold']:.6f}")
            print(f"  Window Size: {info['window_size']}")
            print(f"  Training Date: {info['training_date']}")
            
        elif args.action == 'analyze':
            if not args.file:
                print("Error: --file required for analyze action")
                return
            
            # Analyze file
            results = client.analyze_file(
                args.file,
                threshold=args.threshold,
                download_path=args.output
            )
            
        elif args.action == 'monitor':
            # Start real-time monitoring
            monitor = RealtimeMonitor(args.url.replace('http', 'ws') + '/ws')
            monitor.start()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())