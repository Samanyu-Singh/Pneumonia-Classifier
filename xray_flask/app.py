from flask import Flask, render_template, request, jsonify
import os
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'models/latest_non_overfit_model.pth'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
device = None

# Image preprocessing transforms (must match training transforms)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained pneumonia detection model"""
    global model, device
    
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load ResNet18 model
        model = models.resnet18(pretrained=False)
        
        # Modify the final layer to match your training (2 classes: Normal, Pneumonia)
        model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
        # Load trained weights
        if os.path.exists(app.config['MODEL_PATH']):
            model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=device))
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {app.config['MODEL_PATH']}")
            return False
        
        # Set model to evaluation mode
        model.eval()
        model.to(device)
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image_file):
    """Preprocess uploaded image for model input"""
    try:
        # Open and convert to RGB
        image = Image.open(image_file.stream).convert('RGB')
        
        # Apply transforms
        image_tensor = val_test_transform(image).unsqueeze(0)
        
        return image_tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_pneumonia(image_tensor):
    """Make prediction using the loaded model"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            prediction_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction_idx].item()
            
            # Map prediction to label
            prediction = "Pneumonia" if prediction_idx == 1 else "Normal"
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'Normal': probabilities[0][0].item(),
                    'Pneumonia': probabilities[0][1].item()
                }
            }
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page for X-ray upload"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Docker"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        'status': 'healthy', 
        'message': 'X-ray Classifier is running',
        'model_status': model_status,
        'device': str(device) if device else 'unknown'
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload endpoint"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 500
        
        # Preprocess image
        image_tensor = preprocess_image(file)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Make prediction
        result = predict_pneumonia(image_tensor)
        if result is None:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        # Return results
        return jsonify({
            'message': 'Analysis complete',
            'filename': file.filename,
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint (now fully functional)"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Preprocess image
        image_tensor = preprocess_image(file)
        if image_tensor is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        # Make prediction
        result = predict_pneumonia(image_tensor)
        if result is None:
            return jsonify({'error': 'Failed to analyze image'}), 500
        
        return jsonify({
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

# Load model when Flask app starts (not just when run directly)
logger.info("Loading pneumonia detection model...")
if load_model():
    logger.info("Model loaded successfully!")
else:
    logger.error("Failed to load model!")

if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run in debug mode for development
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting X-ray Classifier on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)