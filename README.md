# Industrial Anomaly Detection Pipeline

Multivariate anomaly detection for industrial equipment monitoring using real bearing sensor data.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

This project implements an end-to-end machine learning pipeline for detecting anomalies in industrial equipment sensor data. The system is trained on the [NASA IMS Bearing Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) containing real run-to-failure experiments.

**Key features:**
- Three complementary detection algorithms (Isolation Forest, LOF, Autoencoder)
- Feature engineering from raw vibration signals
- Comprehensive test suite (pytest, 80%+ coverage target)
- Production deployment with Docker and FastAPI
- CI/CD pipeline with GitHub Actions

## Quick Start

```bash
# Clone and install
git clone https://github.com/lucaswjunges/industrial-anomaly-detection.git
cd industrial-anomaly-detection
pip install -r requirements.txt

# Train models on NASA bearing data
python train_nasa.py

# Run evaluation
python evaluate_nasa.py

# Start API server
cd api && python main.py
# Access docs at http://localhost:8000/docs
```

## Dataset

The project uses the **NASA IMS Bearing Dataset** from the Prognostics Center of Excellence:
- 4 bearings run until failure under constant load
- Vibration data sampled at 20 kHz
- Multiple failure modes (inner race, outer race, ball)
- Real degradation patterns over time

To download the dataset:
```bash
python -c "from src.data_generation.nasa_bearing_loader import NASABearingLoader; NASABearingLoader().auto_download()"
```

Alternatively, a synthetic data generator is included for testing and comparison.

## Architecture

```
Raw Sensor Data → Preprocessing → Feature Engineering → Models → Predictions
                   (normalize)    (27 features)        (IF/LOF/AE)
```

### Models

**Isolation Forest** - Tree-based anomaly detection
Fast inference (<20ms), works well for global anomalies

**Local Outlier Factor** - Density-based detection
Captures context-dependent anomalies, good for regime changes

**Autoencoder** - Neural network reconstruction
Learns complex multivariate patterns, higher latency

### Features

From 6 raw sensors (temperature, vibration, pressure, flow rate, current, duty cycle):
- Rolling statistics (mean, std, min, max over 5-min windows)
- Temporal derivatives (rate of change)
- Cross-sensor ratios (pressure/flow for cavitation detection)
- Frequency domain features (FFT energy, kurtosis, skewness)

Total: 27 engineered features

## Results

Performance on NASA bearing test set (Test 1, Bearing 1):

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Isolation Forest | 84.2% | 81.5% | 82.8% |
| LOF | 82.7% | 83.1% | 82.9% |
| Autoencoder | 79.3% | 68.4% | 73.4% |

Performance is realistic for industrial data. Synthetic data achieves 95%+ but doesn't reflect real-world complexity.

## Project Structure

```
.
├── src/
│   ├── data_generation/      # Dataset loaders
│   ├── preprocessing/         # Feature engineering
│   ├── models/               # ML models
│   ├── evaluation/           # Metrics
│   └── utils/                # Helpers
├── tests/                    # pytest suite
├── api/                      # FastAPI server
├── examples/                 # Usage examples
├── train_nasa.py            # Training script
├── evaluate_nasa.py         # Evaluation
├── Dockerfile               # Container definition
└── docker-compose.yml       # Orchestration
```

## Testing

```bash
# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Or use the provided script
./run_tests.sh
```

Target: 80%+ code coverage

## Docker Deployment

```bash
# Build
docker-compose build

# Run training
docker-compose run anomaly-detection python train_nasa.py

# Start API
docker-compose up api
```

See [DOCKER.md](DOCKER.md) for detailed deployment options.

## API Usage

Start the FastAPI server:
```bash
cd api
python main.py
```

Make predictions:
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

Response:
```json
{
  "is_anomaly": false,
  "anomaly_score": -0.15,
  "confidence": 0.9,
  "model_used": "isolation_forest",
  "timestamp": "2024-12-29T12:00:00"
}
```

API documentation: http://localhost:8000/docs

## Development

```bash
# Install dev dependencies
pip install black flake8 isort pytest pytest-cov

# Format code
black src/ tests/ api/
isort src/ tests/ api/

# Lint
flake8 src/ tests/ api/

# Run CI pipeline locally
make ci
```

## Makefile Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make train-nasa    # Train on NASA data
make test          # Run tests with coverage
make docker-build  # Build Docker image
make api           # Start API server
make lint          # Code quality checks
```

## Requirements

- Python 3.8+
- TensorFlow 2.0+
- scikit-learn 1.0+
- FastAPI 0.104+
- Docker (optional, for deployment)

See [requirements.txt](requirements.txt) for full list.

## License

MIT License - see [LICENSE](LICENSE) file.

## References

- NASA Prognostics Center Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- Isolation Forest: Liu et al., 2008
- Local Outlier Factor: Breunig et al., 2000
- Autoencoder Anomaly Detection: Various (see technical report)

## Contributing

Pull requests welcome. For major changes, open an issue first to discuss.

Run tests before submitting:
```bash
pytest tests/ -v
```

## Contact

GitHub: [@lucaswjunges](https://github.com/lucaswjunges)
