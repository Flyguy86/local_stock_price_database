#!/bin/bash
docker-compose up -d --build --force-recreate ray_orchestrator
echo "Ray orchestrator restarted. Checking logs..."
sleep 2
docker-compose logs --tail=20 ray_orchestrator
