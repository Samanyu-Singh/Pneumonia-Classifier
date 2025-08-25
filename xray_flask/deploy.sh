#!/bin/bash

# X-ray Classifier Deployment Script
set -e

echo "🚀 Deploying X-ray Classifier..."

# Build production image
echo "📦 Building production Docker image..."
docker build -f Dockerfile.prod -t xray-classifier:latest .

# Stop existing container if running
echo "🛑 Stopping existing container..."
docker stop xray-classifier-prod 2>/dev/null || true
docker rm xray-classifier-prod 2>/dev/null || true

# Run production container
echo "🏃 Starting production container..."
docker run -d \
    --name xray-classifier-prod \
    --restart unless-stopped \
    -p 5000:5000 \
    --env-file env.production \
    xray-classifier:latest

# Wait for health check
echo "⏳ Waiting for application to be healthy..."
sleep 10

# Check health
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Deployment successful! Application is running at http://localhost:5000"
    echo "🔍 Health check: http://localhost:5000/health"
else
    echo "❌ Deployment failed! Check logs with: docker logs xray-classifier-prod"
    exit 1
fi

echo "🎉 Deployment complete!"
