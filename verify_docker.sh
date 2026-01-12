#!/bin/bash

echo "🔍 Docker Configuration Verification"
echo "===================================="
echo ""

# Check if base image exists
if docker images | grep -q "stock_base"; then
    echo "✅ Base image (stock_base:latest) exists"
    BASE_SIZE=$(docker images stock_base:latest --format "{{.Size}}")
    echo "   Size: $BASE_SIZE"
else
    echo "❌ Base image not found. Run: ./build.sh"
    exit 1
fi

echo ""
echo "📦 Service Images:"
for service in api feature_builder training_service simulation_service optimization; do
    IMG="local_stock_price_database-${service}"
    if docker images | grep -q "$IMG"; then
        SIZE=$(docker images "$IMG" --format "{{.Size}}" | head -n1)
        echo "   ✅ $service: $SIZE"
    else
        echo "   ❌ $service: Not built"
    fi
done

echo ""
echo "🔗 Volume Mounts (from docker-compose.yml):"
grep -A 2 "volumes:" docker-compose.yml | grep "\- \." | head -n 5

echo ""
echo "📁 Dockerfile Status:"
for df in Dockerfile Dockerfile.base Dockerfile.feature Dockerfile.training Dockerfile.simulation Dockerfile.optimize; do
    if [ -f "$df" ]; then
        LINES=$(wc -l < "$df")
        echo "   ✅ $df ($LINES lines)"
    else
        echo "   ❌ $df missing"
    fi
done

echo ""
echo "🎯 Ready to deploy: docker-compose up -d"
