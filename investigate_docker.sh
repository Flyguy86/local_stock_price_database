#!/bin/bash
# Detailed Docker space analysis

echo "=== DOCKER SPACE INVESTIGATION ==="
echo ""

echo "1. Overall Docker disk usage:"
docker system df -v
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "2. Images breakdown:"
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | head -20
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "3. Dangling/unused images:"
docker images -f "dangling=true" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "4. Containers (running and stopped):"
docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Size}}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "5. Volumes:"
docker volume ls
echo ""
docker volume ls -q | xargs docker volume inspect --format '{{ .Name }}: {{ .Mountpoint }}' 2>/dev/null
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "6. Build cache:"
docker buildx du 2>/dev/null || docker builder df 2>/dev/null || echo "Build cache info not available"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "=== WHY IS DOCKER USING SO MUCH SPACE? ==="
echo ""

# Count images
TOTAL_IMAGES=$(docker images -q | wc -l)
DANGLING_IMAGES=$(docker images -qf "dangling=true" | wc -l)
echo "📦 Total Images: $TOTAL_IMAGES"
echo "   └─ Dangling (unused): $DANGLING_IMAGES"
echo ""

# Analyze this project's images
echo "🔍 This project's images:"
docker images | grep -E "local_stock|ray|orchestrator|feature|api|training|simulation|optimization" || echo "   None found with standard names"
echo ""

echo "💡 Common causes of large Docker space usage:"
echo ""
echo "1. MULTIPLE IMAGE VERSIONS (Most common)"
echo "   • Every 'docker-compose up --build' creates NEW images"
echo "   • Old images are kept as <none> (dangling)"
echo "   • Solution: Prune old images"
echo ""
echo "2. LARGE BASE IMAGES"
echo "   • python:3.11 base image: ~900MB"
echo "   • + Ray framework: ~2GB"
echo "   • + ML libraries (numpy, pandas, xgboost, etc): ~1-2GB"
echo "   • Each service image: 2-4GB"
echo "   • × 6 services = 12-24GB total"
echo ""
echo "3. BUILD CACHE LAYERS"
echo "   • Intermediate layers from builds: 2-5GB"
echo "   • Multi-stage builds create more layers"
echo ""
echo "4. CONTAINER LOGS"
echo "   • Long-running containers accumulate logs"
echo "   • Can grow to several GB"
echo ""
echo "=== CLEANUP RECOMMENDATIONS ==="
echo ""
echo "🟢 SAFE (doesn't affect running services):"
echo "   docker image prune -f           # Remove dangling images"
echo "   docker builder prune -f          # Remove build cache"
echo ""
echo "🟡 MODERATE (stops containers, keeps volumes/data):"
echo "   docker compose down              # Stop services"
echo "   docker system prune -f           # Clean all unused resources"
echo "   docker compose up -d             # Restart"
echo ""
echo "🔴 AGGRESSIVE (removes everything, keeps bind mounts):"
echo "   docker compose down -v           # Stop and remove volumes"
echo "   docker system prune -a -f        # Remove all unused images"
echo "   docker volume prune -f           # Remove unused volumes"
echo "   docker compose up --build -d     # Rebuild and restart"
echo ""
echo "⚡ RECOMMENDED FOR YOUR SITUATION:"
echo "   # Stop services"
echo "   docker compose down"
echo ""
echo "   # Remove old images and build cache (saves 5-10GB)"
echo "   docker system prune -a -f"
echo ""
echo "   # Restart with fresh build"
echo "   docker compose up --build -d"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
