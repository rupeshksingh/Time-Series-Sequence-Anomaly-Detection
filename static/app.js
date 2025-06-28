// Global variables
let speedChart, scoreChart;
let ws = null;
let currentJobId = null;
let processingInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadModelInfo();
    setupEventListeners();
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Initialize Charts
function initializeCharts() {
    // Speed Chart
    const speedCtx = document.getElementById('speedChart').getContext('2d');
    speedChart = new Chart(speedCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Speed',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                tension: 0.1,
                pointRadius: 0
            }, {
                label: 'Anomalies',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.8)',
                showLine: false,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                title: {
                    display: false
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Speed'
                    }
                }
            }
        }
    });

    // Score Chart
    const scoreCtx = document.getElementById('scoreChart').getContext('2d');
    scoreChart = new Chart(scoreCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Anomaly Score',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                tension: 0.1,
                pointRadius: 0
            }, {
                label: 'Threshold',
                data: [],
                borderColor: 'rgb(255, 159, 64)',
                borderDash: [5, 5],
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    display: true,
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Anomaly Score (log scale)'
                    }
                }
            }
        }
    });
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await axios.get('/api/model/info');
        const data = response.data;
        
        const modelInfoDiv = document.getElementById('modelInfo');
        const modelStatus = document.getElementById('modelStatus');
        
        if (data.model_loaded) {
            modelStatus.innerHTML = '<i class="fas fa-circle text-success"></i> Model Ready';
            modelInfoDiv.innerHTML = `
                <p><strong>Threshold:</strong> ${data.threshold.toFixed(6)}</p>
                <p><strong>Window Size:</strong> ${data.window_size}s</p>
                <p><strong>Training Date:</strong> ${data.training_date ? new Date(data.training_date).toLocaleDateString() : 'Unknown'}</p>
                ${data.training_stats.data_splits ? `
                    <p><strong>Training Samples:</strong> ${data.training_stats.data_splits.train.toLocaleString()}</p>
                    <p><strong>Training Anomalies:</strong> ${data.training_stats.data_splits.train_anomalies.toLocaleString()}</p>
                ` : ''}
            `;
        } else {
            modelStatus.innerHTML = '<i class="fas fa-circle text-danger"></i> No Model';
            modelInfoDiv.innerHTML = '<p class="text-danger">No model loaded. Please train a model first.</p>';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelStatus').innerHTML = '<i class="fas fa-circle text-danger"></i> Error';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Upload form
    document.getElementById('uploadForm').addEventListener('submit', handleUpload);
    
    // Real-time monitoring buttons
    document.getElementById('startRealtimeBtn').addEventListener('click', startRealtime);
    document.getElementById('stopRealtimeBtn').addEventListener('click', stopRealtime);
}

// Handle file upload
async function handleUpload(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a file', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');
    
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Uploading...';
    
    try {
        // Upload file
        const uploadResponse = await axios.post('/api/upload', formData);
        const fileId = uploadResponse.data.file_id;
        
        uploadStatus.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check"></i> File uploaded successfully
                <br><small>${uploadResponse.data.rows.toLocaleString()} rows</small>
            </div>
        `;
        
        // Prepare detection request
        const threshold = document.getElementById('thresholdInput').value;
        const detectionRequest = {};
        if (threshold) {
            detectionRequest.threshold = parseFloat(threshold);
        }
        
        // Start detection
        const detectResponse = await axios.post(`/api/detect/${fileId}`, detectionRequest);
        currentJobId = detectResponse.data.job_id;
        
        // Show processing modal
        const processingModal = new bootstrap.Modal(document.getElementById('processingModal'));
        processingModal.show();
        
        // Start polling for results
        pollJobStatus(processingModal);
        
    } catch (error) {
        console.error('Error:', error);
        uploadStatus.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> ${error.response?.data?.detail || 'Upload failed'}
            </div>
        `;
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-play"></i> Analyze';
    }
}

// Poll job status
async function pollJobStatus(modal) {
    processingInterval = setInterval(async () => {
        try {
            const response = await axios.get(`/api/job/${currentJobId}`);
            const status = response.data;
            
            // Update progress bar
            if (status.progress) {
                document.getElementById('progressBar').style.width = `${status.progress}%`;
            }
            
            if (status.status === 'completed') {
                clearInterval(processingInterval);
                modal.hide();
                
                // Load and display results
                await loadResults(currentJobId);
                
                // Show download button
                const downloadBtn = document.getElementById('downloadBtn');
                downloadBtn.style.display = 'block';
                downloadBtn.onclick = () => downloadResults(currentJobId);
                
            } else if (status.status === 'error') {
                clearInterval(processingInterval);
                modal.hide();
                showAlert(`Processing error: ${status.error}`, 'danger');
            }
        } catch (error) {
            clearInterval(processingInterval);
            modal.hide();
            showAlert('Error checking job status', 'danger');
        }
    }, 1000);
}

// Load and display results
async function loadResults(jobId) {
    try {
        // Get results
        const resultsResponse = await axios.get(`/api/results/${jobId}`);
        const results = resultsResponse.data;
        
        // Update summary
        document.getElementById('resultsSummary').style.display = 'block';
        document.getElementById('totalPoints').textContent = results.total_points.toLocaleString();
        document.getElementById('anomaliesDetected').textContent = results.anomalies_detected.toLocaleString();
        document.getElementById('anomalyRate').textContent = `${results.anomaly_rate.toFixed(2)}%`;
        document.getElementById('thresholdUsed').textContent = results.threshold.toFixed(6);
        
        // Get visualization data
        const vizResponse = await axios.get(`/api/visualize/${jobId}?sample_size=2000`);
        const vizData = vizResponse.data;
        
        // Update charts
        updateCharts(vizData);
        
        // Update segments table if available
        if (results.segments && results.segments.length > 0) {
            updateSegmentsTable(results.segments);
        }
        
    } catch (error) {
        showAlert('Error loading results', 'danger');
        console.error(error);
    }
}

// Update charts with data
function updateCharts(data) {
    // Prepare speed chart data
    const speedData = [];
    const anomalyData = [];
    
    data.speeds.forEach((speed, i) => {
        speedData.push({x: i, y: speed});
        if (data.is_anomaly[i] === 1) {
            anomalyData.push({x: i, y: speed});
        }
    });
    
    speedChart.data.labels = data.timestamps.map((t, i) => i);
    speedChart.data.datasets[0].data = speedData;
    speedChart.data.datasets[1].data = anomalyData;
    speedChart.update();
    
    // Prepare score chart data
    const scoreData = data.anomaly_scores.map((score, i) => ({x: i, y: score || 0.0001}));
    const thresholdData = data.anomaly_scores.map((_, i) => ({x: i, y: data.threshold}));
    
    scoreChart.data.labels = data.timestamps.map((t, i) => i);
    scoreChart.data.datasets[0].data = scoreData;
    scoreChart.data.datasets[1].data = thresholdData;
    scoreChart.update();
}

// Update segments table
function updateSegmentsTable(segments) {
    const segmentsCard = document.getElementById('segmentsCard');
    const tbody = document.querySelector('#segmentsTable tbody');
    
    segmentsCard.style.display = 'block';
    tbody.innerHTML = '';
    
    segments.sort((a, b) => b.max_score - a.max_score);
    
    segments.slice(0, 20).forEach((segment, i) => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${i + 1}</td>
            <td>${new Date(segment.start_time).toLocaleString()}</td>
            <td>${segment.duration} points</td>
            <td>${segment.max_score.toFixed(4)}</td>
            <td>${segment.mean_speed.toFixed(2)}</td>
        `;
    });
}

// Download results
function downloadResults(jobId) {
    window.location.href = `/api/download/${jobId}`;
}

// Start real-time monitoring
function startRealtime() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = function() {
        document.getElementById('startRealtimeBtn').style.display = 'none';
        document.getElementById('stopRealtimeBtn').style.display = 'block';
        document.getElementById('realtimeStatus').innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-circle text-success"></i> Connected
            </div>
        `;
        
        // Start sending simulated data
        simulateRealtimeData();
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.type === 'detection_result') {
            updateRealtimeChart(data);
        }
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
        stopRealtime();
    };
    
    ws.onclose = function() {
        stopRealtime();
    };
}

// Stop real-time monitoring
function stopRealtime() {
    if (ws) {
        ws.close();
        ws = null;
    }
    
    document.getElementById('startRealtimeBtn').style.display = 'block';
    document.getElementById('stopRealtimeBtn').style.display = 'none';
    document.getElementById('realtimeStatus').innerHTML = '';
}

// Simulate real-time data
function simulateRealtimeData() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        return;
    }
    
    // Generate random speed data
    const baseSpeed = 23000 + Math.random() * 500;
    const speed = baseSpeed + (Math.random() - 0.5) * 1000;
    
    ws.send(JSON.stringify({
        type: 'realtime_data',
        speed: speed,
        timestamp: new Date().toISOString()
    }));
    
    // Continue sending data every second
    setTimeout(simulateRealtimeData, 1000);
}

// Update real-time chart
function updateRealtimeChart(data) {
    // Keep only last 100 points
    if (speedChart.data.labels.length > 100) {
        speedChart.data.labels.shift();
        speedChart.data.datasets[0].data.shift();
        speedChart.data.datasets[1].data.shift();
        scoreChart.data.labels.shift();
        scoreChart.data.datasets[0].data.shift();
        scoreChart.data.datasets[1].data.shift();
    }
    
    // Add new data
    const label = new Date(data.timestamp).toLocaleTimeString();
    speedChart.data.labels.push(label);
    speedChart.data.datasets[0].data.push(data.speed);
    speedChart.data.datasets[1].data.push(data.is_anomaly ? data.speed : null);
    
    scoreChart.data.labels.push(label);
    scoreChart.data.datasets[0].data.push(data.anomaly_score);
    scoreChart.data.datasets[1].data.push(data.threshold);
    
    // Update charts
    speedChart.update('none');
    scoreChart.update('none');
    
    // Update status if anomaly
    if (data.is_anomaly) {
        const status = document.getElementById('realtimeStatus');
        status.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> Anomaly detected!
                <br><small>Score: ${data.anomaly_score.toFixed(4)}</small>
            </div>
        `;
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                status.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-circle text-success"></i> Connected
                    </div>
                `;
            }
        }, 3000);
    }
}

// Show alert
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3`;
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}