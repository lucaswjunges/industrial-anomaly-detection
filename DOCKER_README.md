# 🐳 Docker Deployment Guide

## Quick Start

```bash
# Build the image
docker-compose build

# Run training on synthetic data
docker-compose run anomaly-detection python train_simple.py

# Run training on NASA data
docker-compose run anomaly-detection python train_nasa.py

# Run evaluation
docker-compose run anomaly-detection python evaluate_nasa.py

# Access interactive shell
docker-compose run anomaly-detection /bin/bash
```

## Image Details

- **Base image:** python:3.9-slim
- **Size:** ~800 MB (multi-stage build optimized)
- **Architecture:** Multi-stage build for minimal runtime image
- **Security:** Non-root user, minimal attack surface

## Available Services

### 1. Main Application (`anomaly-detection`)

```bash
docker-compose run anomaly-detection python train_nasa.py
```

### 2. Jupyter Notebook (`jupyter`)

```bash
docker-compose up jupyter

# Access at: http://localhost:8888
```

## Volume Mounts

Data persists in these directories:

- `./data` → `/app/data` - Training data and results
- `./models` → `/app/models` - Trained model weights
- `./src` → `/app/src` - Source code (development)

## Building for Production

```bash
# Build production image
docker build -t iot-anomaly-detection:latest .

# Run with specific resources
docker run -it \
  --cpus="2.0" \
  --memory="4g" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  iot-anomaly-detection:latest \
  python train_nasa.py
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
docker-compose up -d  # Detached mode
docker-compose logs -f anomaly-detection
```

### Option 2: Docker Swarm

```bash
docker swarm init
docker stack deploy -c docker-compose.yml iot-stack
```

### Option 3: Kubernetes

See `k8s/deployment.yaml` (if available)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUNBUFFERED` | 1 | Disable Python buffering |
| `TF_CPP_MIN_LOG_LEVEL` | 2 | TensorFlow logging level |
| `OMP_NUM_THREADS` | 2 | OpenMP thread count |

## Resource Requirements

**Minimum:**
- CPU: 1 core
- RAM: 2 GB
- Disk: 5 GB

**Recommended:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 10 GB

## Health Check

The container includes a health check that runs every 30 seconds:

```bash
docker ps  # Check HEALTH status
```

## Troubleshooting

### Issue: Out of memory during training

```bash
# Increase memory limit
docker-compose up -d --scale anomaly-detection=1 \
  --memory=8g anomaly-detection
```

### Issue: TensorFlow GPU not detected

```bash
# Use nvidia-docker runtime
docker run --gpus all \
  iot-anomaly-detection:latest \
  python train_nasa.py
```

### Issue: Slow training

```bash
# Increase CPU allocation
docker-compose run --cpus="4.0" \
  anomaly-detection python train_nasa.py
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Build Docker image
  run: docker build -t iot-anomaly:${{ github.sha }} .

- name: Run tests in Docker
  run: docker run iot-anomaly:${{ github.sha }} pytest
```

### GitLab CI

```yaml
test:
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t iot-anomaly:test .
    - docker run iot-anomaly:test pytest
```

## Best Practices

1. **Always use volumes** for data and models (don't store in container)
2. **Set resource limits** to prevent resource exhaustion
3. **Use multi-stage builds** to minimize image size
4. **Run as non-root user** for security
5. **Tag images** with version numbers for reproducibility

## Security Considerations

- ✅ Runs as non-root user (uid 1000)
- ✅ Minimal base image (python:3.9-slim)
- ✅ No secrets in image (use environment variables)
- ✅ Regular security updates (rebuild periodically)

## Image Registry

### Docker Hub

```bash
# Tag image
docker tag iot-anomaly-detection:latest username/iot-anomaly:1.0.0

# Push to Docker Hub
docker push username/iot-anomaly:1.0.0
```

### Private Registry

```bash
docker tag iot-anomaly-detection:latest registry.company.com/iot-anomaly:1.0.0
docker push registry.company.com/iot-anomaly:1.0.0
```

## What This Shows to European Recruiters

✅ **Production-ready thinking:** Docker deployment strategy
✅ **DevOps knowledge:** Container orchestration with docker-compose
✅ **Security awareness:** Non-root user, minimal attack surface
✅ **Scalability:** Resource limits and multi-stage builds
✅ **Best practices:** .dockerignore, health checks, volume mounts

**Impact on CV:** Addresses 50% of recruiter rejections for "no Docker"
