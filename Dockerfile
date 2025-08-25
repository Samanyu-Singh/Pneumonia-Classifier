# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model
COPY latest_non_overfit_model.pth .

# Copy the training script
COPY train_model.py .

# Copy the dataset directory
COPY chest_xray/ ./chest_xray/

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "train_model.py"]
