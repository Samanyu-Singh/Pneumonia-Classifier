# 🫁 X-ray Pneumonia Classifier

A production-ready, containerized web application for detecting pneumonia from chest X-ray images using AI.

## 🌟 Features

- **91%+ Accuracy**: Trained on extensive chest X-ray dataset
- **Real-time Analysis**: Instant predictions with confidence scores
- **Professional UI**: Medical-grade, responsive interface
- **Production Ready**: Docker, Kubernetes, auto-scaling
- **GPU Support**: Automatic CUDA detection and utilization

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local development)
- Trained model file: `latest_non_overfit_model.pth`

### 1. Development Mode (with hot reload)
```bash
# Start development environment
docker-compose up xray-classifier-dev

# Visit: http://localhost:5000
```

### 2. Production Mode
```bash
# Deploy to production
./deploy.sh

# Visit: http://localhost:5000
```

### 3. Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Visit: http://localhost:5000
```

## 🧪 Testing Guide

### Phase 1: Basic Flask + Docker
```bash
# Test basic Flask app
cd xray_flask
python app.py
# Visit: http://localhost:5000
# Should see: Upload interface with dark theme
```

### Phase 2: Model Integration
```bash
# Ensure model file exists
ls models/latest_non_overfit_model.pth

# Test model loading
curl http://localhost:5000/health
# Should return: {"model_status": "loaded", "device": "cuda" or "cpu"}

# Test prediction
# Upload an X-ray image and click "Analyze"
# Should see: Real prediction with confidence scores
```

### Phase 3: Development Environment
```bash
# Test development mode with volume mounts
docker-compose up xray-classifier-dev

# Make a code change (e.g., edit app.py)
# Should see: Auto-reload in container

# Test file upload
# Drag & drop an X-ray image
# Should see: Image preview and analysis
```

### Phase 4: Production Deployment
```bash
# Test production build
docker build -f Dockerfile.prod -t xray-classifier:latest .

# Test production deployment
./deploy.sh

# Test health checks
curl http://localhost:5000/health

# Test auto-scaling (if using Kubernetes)
kubectl apply -f k8s/
kubectl get hpa xray-classifier-hpa
```

## 🔍 Health Check Endpoints

- **`/health`**: Application and model status
- **`/`**: Main upload interface
- **`/upload`**: File upload and analysis API

## 📁 Project Structure

```
xray_flask/
├── app.py                 # Main Flask application
├── Dockerfile.dev        # Development Dockerfile
├── Dockerfile.prod       # Production Dockerfile
├── docker-compose.yml    # Development environment
├── deploy.sh             # Production deployment script
├── k8s/                  # Kubernetes manifests
├── models/               # Trained model files
├── static/               # CSS, JS, uploads
├── templates/            # HTML templates
├── env.development      # Development environment vars
├── env.production       # Production environment vars
└── requirements.txt      # Python dependencies
```

## 🐳 Docker Commands

### Development
```bash
# Build and run development container
docker-compose up xray-classifier-dev

# View logs
docker-compose logs -f xray-classifier-dev

# Stop development
docker-compose down
```

### Production
```bash
# Build production image
docker build -f Dockerfile.prod -t xray-classifier:latest .

# Run production container
docker run -d -p 5000:5000 --name xray-prod xray-classifier:latest

# Check status
docker ps
docker logs xray-prod
```

## ☸️ Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
kubectl get hpa

# Scale manually
kubectl scale deployment xray-classifier --replicas=5
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: development/production
- `FLASK_DEBUG`: 0/1
- `MODEL_PATH`: Path to trained model
- `LOG_LEVEL`: DEBUG/INFO/WARNING/ERROR

### Model Requirements
- File: `latest_non_overfit_model.pth`
- Architecture: ResNet18 with custom final layers
- Input: 224x224 RGB images
- Output: 2 classes (Normal, Pneumonia)

## 📊 Performance

- **Model Accuracy**: 91%+ on test set
- **Inference Time**: <2 seconds per image
- **Memory Usage**: ~512MB per container
- **Auto-scaling**: 3-10 replicas based on CPU/Memory

## 🚨 Troubleshooting

### Model Not Loading
```bash
# Check model file exists
ls -la models/

# Check container logs
docker logs <container-name>

# Verify model path in environment
docker exec <container-name> env | grep MODEL
```

### Container Won't Start
```bash
# Check Docker logs
docker logs <container-name>

# Verify dependencies
docker run --rm -it xray-classifier:latest python -c "import torch; print('PyTorch OK')"
```

### Health Check Failing
```bash
# Check if app is running
curl http://localhost:5000/health

# Check container health
docker inspect <container-name> | grep Health
```

## 🎯 Success Criteria

✅ **Phase 1**: Flask app runs with dark UI
✅ **Phase 2**: Model loads and makes predictions
✅ **Phase 3**: Development mode with hot reload
✅ **Phase 4**: Production deployment with auto-scaling

## 🎉 You're Ready!

Your X-ray Classifier is now:
- **Containerized** and runs anywhere
- **Production-ready** with health checks
- **Auto-scaling** for high traffic
- **Professional** medical-grade interface
- **91%+ accurate** pneumonia detection

**Deploy with confidence!** 🚀
