"""
Simplified evaluation script
"""
import os
import sys
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.insert(0, 'src')

from preprocessing.preprocessor import IoTPreprocessor
from models.anomaly_detectors import (
    IsolationForestDetector,
    LOFDetector,
    AutoencoderDetector
)
from evaluation.evaluator import AnomalyEvaluator

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)

    # Create results directory
    os.makedirs('data/results', exist_ok=True)

    # Load data
    test_df = pd.read_csv('data/processed/test_data.csv')

    # Load preprocessor
    preprocessor = IoTPreprocessor()
    preprocessor.load('models/preprocessor.pkl')

    # Get test features
    X_test = preprocessor.get_feature_matrix(test_df, feature_type='all')
    y_test = test_df['anomaly_label'].values

    print(f"\nTest samples: {len(X_test):,}")
    print(f"Anomalies: {y_test.sum():,} ({y_test.mean():.2%})")

    # Load models
    print("\nLoading models...")
    iforest = IsolationForestDetector()
    iforest.load('models/isolation_forest/model.pkl')

    lof = LOFDetector()
    lof.load('models/lof/model.pkl')

    autoencoder = AutoencoderDetector(input_dim=X_test.shape[1])
    autoencoder.load('models/autoencoder/model')

    # Evaluate
    evaluator = AnomalyEvaluator()
    models = [
        ('Isolation Forest', iforest),
        ('Local Outlier Factor', lof),
        ('Autoencoder', autoencoder)
    ]

    for name, model in models:
        print(f"\nEvaluating {name}...")
        predictions = model.predict(X_test)
        scores = model.score_samples(X_test)

        results = evaluator.evaluate(y_test, predictions, scores, name)

        print(f"  Precision:  {results['classification']['precision']:.3f}")
        print(f"  Recall:     {results['classification']['recall']:.3f}")
        print(f"  F1-Score:   {results['classification']['f1_score']:.3f}")
        print(f"  ROC-AUC:    {results['curve_metrics']['roc_auc']:.3f}")

    # Save results
    evaluator.save_results('data/results/evaluation_results.json')

    # Generate report
    from evaluation.evaluator import generate_evaluation_report
    generate_evaluation_report(evaluator, 'data/results/evaluation_report.txt')

    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - data/results/evaluation_results.json")
    print("  - data/results/evaluation_report.txt")

if __name__ == '__main__':
    main()
