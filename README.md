# 🫁 **Pneumonia Detection from Chest X-Rays using Deep Learning**

A comprehensive machine learning project that trains a CNN model to detect pneumonia from chest X-ray images and deploys it as a production-ready web application with Docker, Kubernetes, and GPU acceleration.

## 🌟 **Project Overview**

This project demonstrates end-to-end machine learning development:
- **Training**: Custom ResNet18 architecture achieving 91%+ test accuracy
- **Web Application**: Professional Flask-based interface for real-time predictions
- **Production Deployment**: Containerized with Docker, Kubernetes, and auto-scaling
- **GPU Acceleration**: Full CUDA support for optimal performance

## 🎯 **Key Features**

### **Machine Learning**
- **91%+ Test Accuracy** on chest X-ray dataset
- **Custom CNN Architecture** based on ResNet18 with optimized final layers
- **Data Augmentation** to prevent overfitting
- **Early Stopping** with best model selection
- **GPU Acceleration** using PyTorch and CUDA

### **Web Application**
- **Professional Medical-Grade UI** with dark theme
- **Real-time X-ray Analysis** with instant predictions
- **Drag & Drop Interface** for easy image upload
- **Confidence Scores** and probability distributions
- **Responsive Design** for desktop and mobile

### **Production Deployment**
- **Docker Containerization** with multi-stage builds
- **Kubernetes Orchestration** with auto-scaling
- **Health Monitoring** and comprehensive logging
- **Environment Configuration** for dev/prod
- **Load Balancing** and horizontal scaling

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.9+
- PyTorch with CUDA support
- Docker & Docker Compose
- NVIDIA GPU (optional, for acceleration)

### **1. Train the Model**
```bash
# Install dependencies
pip install torch torchvision pillow numpy matplotlib scikit-learn tqdm

# Train the model
python train_model.py

# Model will be saved as 'latest_non_overfit_model.pth'
```

### **2. Run the Web Application**
```bash
# Development mode (with hot reload)
cd xray_flask
docker-compose up xray-classifier-dev

# Visit: http://localhost:5000
```

### **3. Production Deployment**
```bash
# Deploy to production
cd xray_flask
./deploy.sh

# Visit: http://localhost:5000
```

## 🧪 **Model Training**

### **Dataset**
- **Source**: Chest X-ray images (Normal vs Pneumonia)
- **Training**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation**: 16 images (8 Normal, 8 Pneumonia)
- **Testing**: 624 images (234 Normal, 390 Pneumonia)

### **Architecture**
```python
ResNet18 + Custom Final Layers:
├── Dropout(0.7)
├── Linear(512, 512) + ReLU
├── Dropout(0.5)
└── Linear(512, 2)  # Normal vs Pneumonia
```

### **Training Features**
- **Data Augmentation**: Random crops, flips, rotations, color jitter
- **Learning Rate Scheduling**: Cosine annealing with early stopping
- **Regularization**: Dropout, weight decay, gradient accumulation
- **Optimization**: Adam optimizer with mixed precision training

### **Performance Metrics**
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 91%+
- **Test Accuracy**: 91%+
- **Inference Time**: <2 seconds per image

## 🌐 **Web Application**

### **Features**
- **Upload Interface**: Drag & drop or file browser
- **Image Preview**: Real-time X-ray display
- **Instant Analysis**: AI-powered predictions
- **Results Display**: Confidence scores and probabilities
- **Professional UI**: Medical-grade interface

### **API Endpoints**
- **`/`**: Main upload interface
- **`/upload`**: File upload and analysis
- **`/health`**: Application and model status
- **`/predict`**: Alternative prediction endpoint

### **Technology Stack**
- **Backend**: Flask with PyTorch integration
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL, torchvision transforms
- **GPU Support**: Automatic CUDA detection

## 🐳 **Docker & Containerization**

### **Development Environment**
```bash
# Start development with hot reload
docker-compose up xray-classifier-dev

# Features:
# - Volume mounts for live code editing
# - Auto-reload on file changes
# - Development-specific configurations
```

### **Production Environment**
```bash
# Multi-stage production build
docker build -f Dockerfile.prod -t xray-classifier:latest .

# Features:
# - Optimized image size
# - Security hardening
# - Health checks
# - Non-root user
```

### **Docker Compose Services**
- **xray-classifier-dev**: Development with hot reload
- **xray-classifier-prod**: Production deployment
- **Volume Mounts**: Live code synchronization
- **Environment Variables**: Separate dev/prod configs

## ☸️ **Kubernetes Deployment**

### **Auto-scaling Configuration**
```yaml
HorizontalPodAutoscaler:
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilization: 70%
  targetMemoryUtilization: 80%
```

### **Resource Management**
- **CPU**: 250m request, 500m limit
- **Memory**: 512Mi request, 1Gi limit
- **Health Checks**: Liveness and readiness probes
- **Load Balancing**: Service with LoadBalancer type

### **Deployment Commands**
```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check status
kubectl get pods,services,hpa

# Scale manually
kubectl scale deployment xray-classifier --replicas=5
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Development
FLASK_ENV=development
FLASK_DEBUG=1
MODEL_PATH=models/latest_non_overfit_model.pth

# Production
FLASK_ENV=production
FLASK_DEBUG=0
MODEL_PATH=/app/models/latest_non_overfit_model.pth
```

### **Model Requirements**
- **File**: `latest_non_overfit_model.pth`
- **Size**: ~45MB
- **Input**: 224x224 RGB images
- **Output**: 2 classes (Normal, Pneumonia)
- **Framework**: PyTorch 1.9+

## 📊 **Performance & Monitoring**

### **Health Checks**
- **Application Health**: `/health` endpoint
- **Model Status**: Loaded/not_loaded status
- **Device Information**: CUDA/CPU detection
- **Response Time**: <100ms for health checks

### **Scaling Metrics**
- **CPU Utilization**: Target 70%
- **Memory Usage**: Target 80%
- **Response Time**: <2 seconds for predictions
- **Throughput**: Handles multiple concurrent requests

### **Monitoring**
- **Container Health**: Docker health checks
- **Kubernetes Metrics**: HPA monitoring
- **Application Logs**: Comprehensive logging
- **Error Tracking**: Exception handling and reporting

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Model Not Loading**
```bash
# Check model file exists
ls -la models/latest_non_overfit_model.pth

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check container logs
docker logs <container-name>
```

#### **GPU Not Detected**
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

#### **Container Build Issues**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -f Dockerfile.prod .
```

### **Debug Commands**
```bash
# Test model loading
cd xray_flask
python test_model.py

# Check application health
curl http://localhost:5000/health

# View container logs
docker-compose logs -f xray-classifier-dev
```

## 🏗️ **Project Structure**

```
Pneumonia-Classifier/
├── 📁 chest_xray/                 # Dataset
│   ├── train/                     # Training images
│   ├── test/                      # Test images
│   └── val/                       # Validation images
├── 📁 xray_flask/                 # Web application
│   ├── app.py                     # Flask application
│   ├── Dockerfile.dev            # Development container
│   ├── Dockerfile.prod           # Production container
│   ├── docker-compose.yml        # Development environment
│   ├── deploy.sh                 # Production deployment
│   ├── k8s/                      # Kubernetes manifests
│   ├── models/                    # Trained models
│   ├── static/                    # CSS, JS, uploads
│   ├── templates/                 # HTML templates
│   ├── env.development           # Development config
│   ├── env.production            # Production config
│   └── requirements.txt           # Python dependencies
├── 🧠 train_model.py              # Model training script
├── 📊 latest_non_overfit_model.pth # Trained model (45MB)
└── 📖 README.md                   # This file
```

## 🎓 **Learning Outcomes**

This project demonstrates proficiency in:

### **Machine Learning**
- Deep learning with PyTorch
- CNN architecture design
- Data preprocessing and augmentation
- Model training and evaluation
- GPU acceleration and optimization

### **Software Engineering**
- Full-stack web development
- RESTful API design
- Professional UI/UX design
- Error handling and validation
- Testing and debugging

### **DevOps & Production**
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline design
- Environment management
- Monitoring and health checks

### **Performance & Scalability**
- GPU optimization
- Load balancing
- Auto-scaling
- Resource management
- Performance monitoring

## 🚀 **Future Enhancements**

### **Model Improvements**
- [ ] Ensemble methods for higher accuracy
- [ ] Transfer learning with larger datasets
- [ ] Multi-class classification (viral vs bacterial)
- [ ] Attention mechanisms for interpretability

### **Application Features**
- [ ] User authentication and history
- [ ] Batch processing for multiple images
- [ ] Export results to medical reports
- [ ] Integration with medical databases

### **Infrastructure**
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Cloud deployment (AWS, Azure, GCP)
- [ ] Database integration for patient records
- [ ] Advanced monitoring with Prometheus/Grafana

## 📚 **References & Resources**

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **PyTorch**: [Official Documentation](https://pytorch.org/docs/)
- **Flask**: [Web Framework Guide](https://flask.palletsprojects.com/)
- **Docker**: [Container Platform](https://www.docker.com/)
- **Kubernetes**: [Container Orchestration](https://kubernetes.io/)

## 👥 **Contributing**

This is a personal project demonstrating machine learning and software engineering skills. Feel free to:
- Fork the repository
- Submit issues and suggestions
- Contribute improvements
- Use as a learning resource

## 📄 **License**

This project is for educational and portfolio purposes. The trained model and code are provided as-is for demonstration of skills and capabilities.

## 🎉 **Acknowledgments**

- **Dataset Providers**: Medical imaging community
- **Open Source**: PyTorch, Flask, Docker communities
- **Learning Resources**: Online courses and tutorials
- **Hardware**: NVIDIA RTX 3070 for GPU acceleration

---

**Built with ❤️ and ☕ for demonstrating production-ready machine learning applications.**

*This project showcases the ability to take a machine learning concept from research to production deployment, demonstrating skills across the entire ML development lifecycle.*
