#!/usr/bin/env python3
"""
Test script to debug model loading issues
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models

def test_pytorch():
    """Test PyTorch installation"""
    print("🔍 Testing PyTorch...")
    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_model_creation():
    """Test ResNet18 model creation"""
    print("\n🔍 Testing model creation...")
    try:
        model = models.resnet18(pretrained=False)
        print("✅ ResNet18 created successfully")
        
        # Test the custom final layer
        model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        print("✅ Custom final layer added successfully")
        return model
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return None

def test_model_loading(model, model_path):
    """Test loading the trained weights"""
    print(f"\n🔍 Testing model loading from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return False
    
    print(f"✅ Model file exists, size: {os.path.getsize(model_path)} bytes")
    
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        print("✅ Weights loaded successfully")
        
        # Apply weights to model
        model.load_state_dict(state_dict)
        print("✅ Weights applied to model successfully")
        
        # Set to evaluation mode
        model.eval()
        model.to(device)
        print("✅ Model set to evaluation mode")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_inference(model):
    """Test basic inference"""
    print("\n🔍 Testing inference...")
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            
        print(f"✅ Inference successful")
        print(f"Output shape: {output.shape}")
        print(f"Probabilities: Normal={probabilities[0][0]:.4f}, Pneumonia={probabilities[0][1]:.4f}")
        return True
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting model loading tests...\n")
    
    # Test 1: PyTorch
    if not test_pytorch():
        print("❌ PyTorch test failed - cannot continue")
        return
    
    # Test 2: Model creation
    model = test_model_creation()
    if model is None:
        print("❌ Model creation failed - cannot continue")
        return
    
    # Test 3: Model loading
    model_path = "models/latest_non_overfit_model.pth"
    if not test_model_loading(model, model_path):
        print("❌ Model loading failed - cannot continue")
        return
    
    # Test 4: Inference
    if not test_inference(model):
        print("❌ Inference test failed")
        return
    
    print("\n🎉 All tests passed! Model should work in Flask app.")
    print("\nIf you still get errors, check the Flask app logs for specific error messages.")

if __name__ == "__main__":
    main()
