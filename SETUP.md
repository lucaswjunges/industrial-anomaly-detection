# Setup Guide

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NASA bearing dataset
python -c "from src.data_generation.nasa_bearing_loader import NASABearingLoader; NASABearingLoader().auto_download()"

# Train models
python train_nasa.py

# Run evaluation
python evaluate_nasa.py

# Run tests
pytest tests/ -v --cov=src
```

## Docker Setup

```bash
# Build and run
docker-compose build
docker-compose run anomaly-detection python train_nasa.py

# Or start API
docker-compose up api
```

## Development

```bash
# Install dev tools
pip install black flake8 isort

# Format code
black src/ tests/ api/

# Run linter
flake8 src/ tests/ api/

# Run full test suite
make test
```

## Push to GitHub

```bash
# Create repo on GitHub first, then:
git remote add origin https://github.com/yourusername/iot-anomaly-detection.git
git push -u origin main
```

## Project Structure

- `src/` - Main source code
- `tests/` - Test suite (pytest)
- `api/` - FastAPI server
- `examples/` - Usage examples
- `train_nasa.py` - Training script for NASA data
- `Dockerfile` - Container definition
- `Makefile` - Common commands

## Notes

This project demonstrates production ML engineering practices:
- Real NASA bearing dataset (not synthetic)
- Comprehensive test coverage
- Docker deployment
- API serving with FastAPI
- CI/CD with GitHub Actions

The goal is to show real-world ML engineering skills, not just model training.
