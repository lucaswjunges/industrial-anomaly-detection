#!/usr/bin/env python3
"""
Example: Real-time Anomaly Detection Inference
==============================================

This script demonstrates how to use the trained models for real-time inference
on new sensor data. Perfect for showing in interviews!

"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from preprocessing.preprocessor import IoTPreprocessor
from models.anomaly_detectors import IsolationForestDetector, LOFDetector, AutoencoderDetector

def load_trained_models():
    """Load trained models and preprocessor"""
    models_dir = Path(__file__).parent.parent / 'models'

    # Try NASA models first (preferred)
    if (models_dir / 'nasa_preprocessor.pkl').exists():
        print("📥 Loading NASA-trained models...")
        preprocessor = joblib.load(models_dir / 'nasa_preprocessor.pkl')
        iforest = IsolationForestDetector.load(models_dir / 'nasa_isolation_forest')
        lof = LOFDetector.load(models_dir / 'nasa_lof')
        autoencoder = AutoencoderDetector.load(models_dir / 'nasa_autoencoder')
        print("✅ NASA models loaded!")
    else:
        # Fallback to synthetic models
        print("📥 Loading synthetic-trained models...")
        preprocessor = joblib.load(models_dir / 'preprocessor.pkl')
        iforest = IsolationForestDetector.load(models_dir / 'isolation_forest')
        lof = LOFDetector.load(models_dir / 'lof')
        autoencoder = AutoencoderDetector.load(models_dir / 'autoencoder')
        print("✅ Synthetic models loaded!")

    return preprocessor, {'iforest': iforest, 'lof': lof, 'autoencoder': autoencoder}

def predict_single_sample(preprocessor, models, sensor_data):
    """
    Predict anomaly on single sensor reading

    Args:
        preprocessor: Fitted IoTPreprocessor
        models: Dict of trained models
        sensor_data: Dict with sensor readings

    Returns:
        Dict with predictions from all models
    """
    # Convert to DataFrame
    df = pd.DataFrame([sensor_data])

    # Preprocess
    X = preprocessor.transform(df, regime_column='operational_state')

    # Predict with all models
    results = {}

    for name, model in models.items():
        prediction = model.predict(X)[0]
        score = model.decision_function(X)[0]

        results[name] = {
            'is_anomaly': bool(prediction),
            'anomaly_score': float(score),
            'confidence': 'high' if abs(score) > 1.0 else 'medium'
        }

    # Ensemble decision (majority vote)
    votes = [r['is_anomaly'] for r in results.values()]
    results['ensemble'] = {
        'is_anomaly': sum(votes) >= 2,  # At least 2 out of 3
        'agreement': f"{sum(votes)}/3 models agree"
    }

    return results

def main():
    """Example inference workflow"""
    print("=" * 70)
    print("Real-time Anomaly Detection - Inference Example")
    print("=" * 70)
    print()

    # Load models
    preprocessor, models = load_trained_models()

    # Example 1: Normal operation
    print("\n📊 Example 1: NORMAL OPERATION")
    print("-" * 70)

    normal_reading = {
        'temperature': 45.3,
        'vibration': 2.8,
        'pressure': 6.5,
        'flow_rate': 152.0,
        'current': 87.5,
        'duty_cycle': 0.75,
        'operational_state': 'normal'
    }

    print("Sensor readings:")
    for key, value in normal_reading.items():
        print(f"  {key:20s}: {value}")

    print("\nPredictions:")
    results = predict_single_sample(preprocessor, models, normal_reading)

    for model_name, result in results.items():
        if model_name != 'ensemble':
            status = "🚨 ANOMALY" if result['is_anomaly'] else "✅ NORMAL"
            print(f"  {model_name:15s}: {status:15s} (score: {result['anomaly_score']:+.3f}, confidence: {result['confidence']})")

    print(f"\n🎯 ENSEMBLE: {'🚨 ANOMALY DETECTED' if results['ensemble']['is_anomaly'] else '✅ NORMAL OPERATION'}")
    print(f"   {results['ensemble']['agreement']}")

    # Example 2: Anomalous operation (high vibration + temperature)
    print("\n\n📊 Example 2: ANOMALOUS OPERATION (Bearing Failure)")
    print("-" * 70)

    anomaly_reading = {
        'temperature': 78.5,      # HIGH - overheating
        'vibration': 12.4,        # VERY HIGH - bearing wear
        'pressure': 4.2,          # LOW - possible cavitation
        'flow_rate': 98.0,        # LOW - reduced throughput
        'current': 125.0,         # HIGH - increased load
        'duty_cycle': 0.95,       # HIGH - working hard
        'operational_state': 'high_load'
    }

    print("Sensor readings:")
    for key, value in anomaly_reading.items():
        print(f"  {key:20s}: {value}")

    print("\nPredictions:")
    results = predict_single_sample(preprocessor, models, anomaly_reading)

    for model_name, result in results.items():
        if model_name != 'ensemble':
            status = "🚨 ANOMALY" if result['is_anomaly'] else "✅ NORMAL"
            print(f"  {model_name:15s}: {status:15s} (score: {result['anomaly_score']:+.3f}, confidence: {result['confidence']})")

    print(f"\n🎯 ENSEMBLE: {'🚨 ANOMALY DETECTED' if results['ensemble']['is_anomaly'] else '✅ NORMAL OPERATION'}")
    print(f"   {results['ensemble']['agreement']}")

    if results['ensemble']['is_anomaly']:
        print("\n⚠️  RECOMMENDED ACTIONS:")
        if anomaly_reading['vibration'] > 10.0:
            print("   • Check bearing condition - excessive vibration detected")
        if anomaly_reading['temperature'] > 70:
            print("   • Inspect for overheating - temperature above threshold")
        if anomaly_reading['pressure'] < 5.0:
            print("   • Check for leaks or cavitation - low pressure")
        print("   • Schedule maintenance inspection")
        print("   • Reduce load if possible")

    # Example 3: Batch prediction
    print("\n\n📊 Example 3: BATCH PREDICTION (Time Series)")
    print("-" * 70)

    # Simulate degradation over time
    time_series = []
    for i in range(10):
        reading = {
            'temperature': 45.0 + i * 3.0,      # Gradually increasing
            'vibration': 2.5 + i * 0.8,         # Gradually increasing
            'pressure': 6.5 - i * 0.2,          # Gradually decreasing
            'flow_rate': 150.0 - i * 4.0,       # Gradually decreasing
            'current': 85.0 + i * 3.0,          # Gradually increasing
            'duty_cycle': 0.75 + i * 0.02,      # Gradually increasing
            'operational_state': 'normal'
        }
        time_series.append(reading)

    df = pd.DataFrame(time_series)
    X = preprocessor.transform(df, regime_column='operational_state')

    # Use Isolation Forest for speed
    predictions = models['iforest'].predict(X)
    scores = models['iforest'].decision_function(X)

    print("\nTime-series anomaly detection (degradation simulation):")
    print(f"{'Sample':^8} | {'Temp':^7} | {'Vib':^6} | {'Status':^15} | {'Score':^8}")
    print("-" * 70)

    for i, (pred, score) in enumerate(zip(predictions, scores)):
        status = "🚨 ANOMALY" if pred == 1 else "✅ NORMAL"
        temp = time_series[i]['temperature']
        vib = time_series[i]['vibration']
        print(f"{i:^8} | {temp:>7.1f} | {vib:>6.2f} | {status:^15} | {score:>+8.3f}")

    anomaly_onset = np.where(predictions == 1)[0]
    if len(anomaly_onset) > 0:
        print(f"\n⚠️  Anomaly first detected at sample {anomaly_onset[0]}")
        print(f"   Early warning: {anomaly_onset[0]} time steps before failure")

    # Summary
    print("\n\n" + "=" * 70)
    print("✅ INFERENCE EXAMPLES COMPLETE")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("   • Normal operation: All models agree → ✅ NORMAL")
    print("   • Bearing failure: Multiple anomaly indicators → 🚨 ALERT")
    print("   • Time-series: Early detection of degradation pattern")
    print("\n🚀 Use Cases:")
    print("   • Real-time monitoring dashboard")
    print("   • Automated alerting system")
    print("   • Predictive maintenance scheduling")
    print("   • Root cause analysis")
    print("\n📖 For more examples, see:")
    print("   • API examples: api/README.md")
    print("   • Batch processing: evaluate_nasa.py")
    print("   • Model comparison: IMPROVEMENTS_SUMMARY.md")
    print()

if __name__ == '__main__':
    main()
