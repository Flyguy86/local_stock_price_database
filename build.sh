#!/bin/bash
set -e

echo "🔨 Building shared base image (stock_base:latest)..."
docker build -t stock_base:latest -f Dockerfile.base .

echo ""
echo "🏗️  Building service images..."
docker-compose build --no-cache

echo ""
echo "✅ Build complete!"
echo ""
echo "📊 Image sizes:"
docker images | head -n 1
docker images | grep -E "stock_base|local_stock"

echo ""
echo "💾 Total disk usage:"
docker system df

echo ""
echo "🚀 To start services: docker-compose up -d"
