#!/usr/bin/env python3
"""
FastAPI Inference Server for Industrial IoT Anomaly Detection
=============================================================

Production-ready REST API for real-time anomaly detection inference.

Endpoints:
- POST /predict - Anomaly detection on sensor data
- GET /health - Health check
- GET /models - List available models
- GET /info - Model and system information

"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import logging
from datetime import datetime

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from preprocessing.preprocessor import IoTPreprocessor
from models.anomaly_detectors import IsolationForestDetector, LOFDetector, AutoencoderDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Industrial IoT Anomaly Detection API",
    description="Production-grade anomaly detection for bearing equipment monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for loaded models
MODELS = {}
PREPROCESSOR = None
MODEL_DIR = Path(__file__).parent.parent / 'models'

# ========================================
# Pydantic Models (Request/Response)
# ========================================

class SensorReading(BaseModel):
    """Single sensor reading input"""

    temperature: float = Field(..., ge=0, le=150, description="Temperature in °C")
    vibration: float = Field(..., ge=0, le=30, description="Vibration in mm/s RMS")
    pressure: float = Field(..., ge=0, le=20, description="Pressure in bar")
    flow_rate: float = Field(..., ge=0, le=500, description="Flow rate in m³/h")
    current: float = Field(..., ge=0, le=200, description="Current in Amperes")
    duty_cycle: float = Field(..., ge=0, le=1.0, description="Duty cycle (0-1)")
    operational_state: Literal['normal', 'startup', 'high_load', 'maintenance'] = Field(
        default='normal',
        description="Current operational state"
    )

    class Config:
        schema_extra = {
            "example": {
                "temperature": 45.3,
                "vibration": 2.8,
                "pressure": 6.5,
                "flow_rate": 152.0,
                "current": 87.5,
                "duty_cycle": 0.75,
                "operational_state": "normal"
            }
        }

class BatchSensorReadings(BaseModel):
    """Batch of sensor readings"""

    readings: List[SensorReading] = Field(..., min_items=1, max_items=1000)

    class Config:
        schema_extra = {
            "example": {
                "readings": [
                    {
                        "temperature": 45.3,
                        "vibration": 2.8,
                        "pressure": 6.5,
                        "flow_rate": 152.0,
                        "current": 87.5,
                        "duty_cycle": 0.75,
                        "operational_state": "normal"
                    }
                ]
            }
        }

class PredictionRequest(BaseModel):
    """Prediction request with model selection"""

    sensor_data: SensorReading
    model_name: Literal['isolation_forest', 'lof', 'autoencoder', 'ensemble'] = Field(
        default='isolation_forest',
        description="Model to use for prediction"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Custom anomaly threshold (overrides default)"
    )

class PredictionResponse(BaseModel):
    """Prediction response"""

    is_anomaly: bool = Field(..., description="Whether sample is anomalous")
    anomaly_score: float = Field(..., description="Anomaly score (higher = more anomalous)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")

class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    models_loaded: List[str]
    preprocessor_loaded: bool
    timestamp: str

# ========================================
# Model Loading
# ========================================

def load_models():
    """Load all available trained models"""
    global MODELS, PREPROCESSOR

    logger.info("Loading models...")

    try:
        # Try to load NASA models first (preferred)
        preprocessor_path = MODEL_DIR / 'nasa_preprocessor.pkl'
        if preprocessor_path.exists():
            PREPROCESSOR = joblib.load(preprocessor_path)
            logger.info("✅ Loaded NASA preprocessor")

            # Load NASA models
            if (MODEL_DIR / 'nasa_isolation_forest').exists():
                MODELS['isolation_forest'] = IsolationForestDetector.load(
                    MODEL_DIR / 'nasa_isolation_forest'
                )
                logger.info("✅ Loaded NASA Isolation Forest")

            if (MODEL_DIR / 'nasa_lof').exists():
                MODELS['lof'] = LOFDetector.load(MODEL_DIR / 'nasa_lof')
                logger.info("✅ Loaded NASA LOF")

            if (MODEL_DIR / 'nasa_autoencoder').exists():
                MODELS['autoencoder'] = AutoencoderDetector.load(
                    MODEL_DIR / 'nasa_autoencoder'
                )
                logger.info("✅ Loaded NASA Autoencoder")

        # Fallback to synthetic models if NASA not available
        elif (MODEL_DIR / 'preprocessor.pkl').exists():
            PREPROCESSOR = joblib.load(MODEL_DIR / 'preprocessor.pkl')
            logger.info("✅ Loaded synthetic preprocessor")

            if (MODEL_DIR / 'isolation_forest').exists():
                MODELS['isolation_forest'] = IsolationForestDetector.load(
                    MODEL_DIR / 'isolation_forest'
                )
                logger.info("✅ Loaded Isolation Forest")

            if (MODEL_DIR / 'lof').exists():
                MODELS['lof'] = LOFDetector.load(MODEL_DIR / 'lof')
                logger.info("✅ Loaded LOF")

            if (MODEL_DIR / 'autoencoder').exists():
                MODELS['autoencoder'] = AutoencoderDetector.load(
                    MODEL_DIR / 'autoencoder'
                )
                logger.info("✅ Loaded Autoencoder")

        else:
            logger.warning("⚠️  No trained models found. Run train_nasa.py or train_simple.py first.")

        logger.info(f"✅ Loaded {len(MODELS)} models successfully")

    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    load_models()

# ========================================
# API Endpoints
# ========================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Industrial IoT Anomaly Detection API",
        "version": "1.0.0",
        "description": "Real-time anomaly detection for industrial equipment",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "models": "/models",
            "info": "/info"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODELS and PREPROCESSOR else "degraded",
        models_loaded=list(MODELS.keys()),
        preprocessor_loaded=PREPROCESSOR is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/models", tags=["Models"])
async def list_models():
    """List available models"""
    if not MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded. Train models first."
        )

    return {
        "available_models": list(MODELS.keys()),
        "default_model": "isolation_forest",
        "model_details": {
            name: {
                "type": model.__class__.__name__,
                "loaded": True
            }
            for name, model in MODELS.items()
        }
    }

@app.get("/info", tags=["General"])
async def system_info():
    """Get system and model information"""
    return {
        "api_version": "1.0.0",
        "models_loaded": len(MODELS),
        "preprocessor_loaded": PREPROCESSOR is not None,
        "model_directory": str(MODEL_DIR),
        "available_models": list(MODELS.keys()),
        "supported_sensors": [
            "temperature", "vibration", "pressure",
            "flow_rate", "current", "duty_cycle"
        ],
        "operational_states": ["normal", "startup", "high_load", "maintenance"]
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_anomaly(request: PredictionRequest):
    """
    Predict anomaly on single sensor reading

    Returns:
    - is_anomaly: Boolean indicating if sample is anomalous
    - anomaly_score: Numerical score (higher = more anomalous)
    - confidence: Prediction confidence
    - model_used: Which model made the prediction
    """
    if not MODELS or not PREPROCESSOR:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Run training first."
        )

    # Select model
    model_name = request.model_name
    if model_name == 'ensemble':
        # Use all models and average
        if len(MODELS) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No models available for ensemble"
            )
        models_to_use = list(MODELS.values())
        model_names = list(MODELS.keys())
    else:
        if model_name not in MODELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{model_name}' not found. Available: {list(MODELS.keys())}"
            )
        models_to_use = [MODELS[model_name]]
        model_names = [model_name]

    try:
        # Convert sensor reading to DataFrame
        sensor_dict = request.sensor_data.dict()
        df = pd.DataFrame([sensor_dict])

        # Preprocess
        X = PREPROCESSOR.transform(df, regime_column='operational_state')

        # Predict with model(s)
        predictions = []
        scores = []

        for model in models_to_use:
            pred = model.predict(X)[0]
            score = model.decision_function(X)[0]

            predictions.append(pred)
            scores.append(score)

        # Aggregate if ensemble
        if model_name == 'ensemble':
            final_prediction = int(np.mean(predictions) >= 0.5)  # Majority vote
            final_score = float(np.mean(scores))
            model_used = f"ensemble ({', '.join(model_names)})"
            confidence = float(abs(np.mean(predictions) - 0.5) * 2)  # Distance from decision boundary
        else:
            final_prediction = int(predictions[0])
            final_score = float(scores[0])
            model_used = model_names[0]
            confidence = 0.9 if abs(final_score) > 1.0 else 0.6  # Simplified confidence

        # Generate explanation
        if final_prediction == 1:
            if sensor_dict['vibration'] > 5.0:
                explanation = "High vibration detected - possible bearing wear or imbalance"
            elif sensor_dict['temperature'] > 70:
                explanation = "Elevated temperature - possible overheating or friction"
            elif sensor_dict['pressure'] < 4.0:
                explanation = "Low pressure detected - possible leak or cavitation"
            else:
                explanation = "Anomalous pattern detected in multivariate sensor data"
        else:
            explanation = "Operating within normal parameters"

        return PredictionResponse(
            is_anomaly=bool(final_prediction),
            anomaly_score=final_score,
            confidence=confidence,
            model_used=model_used,
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchSensorReadings):
    """
    Predict anomalies on batch of sensor readings

    More efficient than multiple single predictions
    """
    if not MODELS or not PREPROCESSOR:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded"
        )

    try:
        # Convert batch to DataFrame
        sensor_dicts = [reading.dict() for reading in request.readings]
        df = pd.DataFrame(sensor_dicts)

        # Preprocess
        X = PREPROCESSOR.transform(df, regime_column='operational_state')

        # Predict (use isolation forest by default for speed)
        model = MODELS.get('isolation_forest', list(MODELS.values())[0])
        predictions = model.predict(X)
        scores = model.decision_function(X)

        # Format results
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                "index": i,
                "is_anomaly": bool(pred),
                "anomaly_score": float(score),
                "timestamp": datetime.now().isoformat()
            })

        return {
            "predictions": results,
            "total_samples": len(results),
            "anomalies_detected": int(predictions.sum()),
            "anomaly_rate": float(predictions.mean())
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# ========================================
# Error Handlers
# ========================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle value errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

# ========================================
# Main
# ========================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
