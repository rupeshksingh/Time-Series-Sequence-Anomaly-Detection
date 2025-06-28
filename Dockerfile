# Use Python 3.9 slim image
FROM python:3.11.13-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application files
COPY config.py .
COPY detector.py .
COPY train.py .
COPY test.py .
COPY utils.py .
COPY main.py .
COPY app.py .

# Copy directories
COPY templates templates/
COPY static static/
COPY models models/

# Create necessary directories
RUN mkdir -p uploads api_results results

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]