# 🚀 FastAPI Inference Server

Production-ready REST API for real-time anomaly detection inference.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models first (if not already done)
python train_nasa.py

# Start API server
cd api
python main.py

# Or use uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API at: `http://localhost:8000`

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["isolation_forest", "lof", "autoencoder"],
  "preprocessor_loaded": true,
  "timestamp": "2024-12-29T12:00:00"
}
```

### 2. Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": {
      "temperature": 45.3,
      "vibration": 2.8,
      "pressure": 6.5,
      "flow_rate": 152.0,
      "current": 87.5,
      "duty_cycle": 0.75,
      "operational_state": "normal"
    },
    "model_name": "isolation_forest"
  }'
```

**Response:**
```json
{
  "is_anomaly": false,
  "anomaly_score": -0.15,
  "confidence": 0.9,
  "model_used": "isolation_forest",
  "timestamp": "2024-12-29T12:00:00",
  "explanation": "Operating within normal parameters"
}
```

### 3. Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type": application/json" \
  -d '{
    "readings": [
      {
        "temperature": 45.3,
        "vibration": 2.8,
        "pressure": 6.5,
        "flow_rate": 152.0,
        "current": 87.5,
        "duty_cycle": 0.75
      },
      {
        "temperature": 78.5,
        "vibration": 12.4,
        "pressure": 4.2,
        "flow_rate": 98.0,
        "current": 125.0,
        "duty_cycle": 0.95
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "is_anomaly": false,
      "anomaly_score": -0.15,
      "timestamp": "2024-12-29T12:00:00"
    },
    {
      "index": 1,
      "is_anomaly": true,
      "anomaly_score": 2.34,
      "timestamp": "2024-12-29T12:00:01"
    }
  ],
  "total_samples": 2,
  "anomalies_detected": 1,
  "anomaly_rate": 0.5
}
```

### 4. List Models

```bash
curl http://localhost:8000/models
```

### 5. System Info

```bash
curl http://localhost:8000/info
```

## Python Client Example

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Make prediction
sensor_data = {
    "sensor_data": {
        "temperature": 45.3,
        "vibration": 2.8,
        "pressure": 6.5,
        "flow_rate": 152.0,
        "current": 87.5,
        "duty_cycle": 0.75,
        "operational_state": "normal"
    },
    "model_name": "isolation_forest"
}

response = requests.post(f"{BASE_URL}/predict", json=sensor_data)
result = response.json()

if result["is_anomaly"]:
    print(f"⚠️  ANOMALY DETECTED!")
    print(f"   Score: {result['anomaly_score']:.2f}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Explanation: {result['explanation']}")
else:
    print("✅ Normal operation")
```

## Docker Deployment

### Option 1: Docker Compose (Recommended)

Add to `docker-compose.yml`:

```yaml
api:
  build: .
  ports:
    - "8000:8000"
  volumes:
    - ./models:/app/models
  command: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then run:

```bash
docker-compose up api
```

### Option 2: Standalone Docker

```bash
# Build image
docker build -t iot-anomaly-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name anomaly-api \
  iot-anomaly-api \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Production Deployment

### Using Gunicorn (WSGI)

```bash
pip install gunicorn

gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_DIR=/app/models
export LOG_LEVEL=info
export WORKERS=4
```

## Performance Considerations

### Latency

- **Single prediction:** ~10-50ms
- **Batch prediction (100 samples):** ~50-200ms
- **Model loading:** ~1-2 seconds (on startup)

### Throughput

- **Single predictions:** ~100-500 requests/sec (1 worker)
- **Batch predictions:** ~1000-5000 samples/sec

### Scaling

**Horizontal scaling:**
```bash
# Run multiple instances behind load balancer
docker-compose up --scale api=4
```

**Vertical scaling:**
```bash
# Increase workers
gunicorn --workers 8 api.main:app
```

## Monitoring

### Health Check Endpoint

Monitor with:

```bash
# Simple health check
curl http://localhost:8000/health

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
```

### Logging

Logs include:

- Request timestamps
- Model used
- Prediction latency
- Error traces

View logs:

```bash
# Docker
docker logs -f anomaly-api

# Systemd
journalctl -u anomaly-api -f
```

## Security

### Authentication (TODO)

Add API key authentication:

```python
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403)
    ...
```

### Rate Limiting (TODO)

```bash
pip install slowapi

from slowapi import Limiter
limiter = Limiter(key_func=lambda: "global")

@app.post("/predict")
@limiter.limit("100/minute")
async def predict():
    ...
```

## Testing

```bash
# Unit tests
pytest tests/test_api.py

# Integration tests
pytest tests/test_api_integration.py

# Load testing
pip install locust
locust -f tests/load_test.py --host http://localhost:8000
```

## What This Shows to European Recruiters

✅ **API Development:** Production-ready FastAPI service
✅ **Documentation:** Auto-generated Swagger/ReDoc docs
✅ **Input Validation:** Pydantic models with validation
✅ **Error Handling:** Proper HTTP status codes and error messages
✅ **Production Thinking:** Health checks, logging, batch processing
✅ **Deployment:** Docker, Gunicorn, scaling considerations

**Impact on CV:** Shows ability to deploy ML models as production services, not just notebooks.

---

**Next Steps:**

1. ✅ Add authentication (API keys or OAuth)
2. ✅ Add rate limiting
3. ✅ Add Prometheus metrics
4. ✅ Add request/response logging
5. ✅ Deploy to cloud (AWS ECS, Google Cloud Run, etc.)
