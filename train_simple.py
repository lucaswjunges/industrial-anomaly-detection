"""
Simplified training script that runs from project root
"""
import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from preprocessing.preprocessor import IoTPreprocessor
from models.anomaly_detectors import (
    IsolationForestDetector,
    LOFDetector,
    AutoencoderDetector
)

def main():
    print("="*70)
    print("SIMPLIFIED MODEL TRAINING")
    print("="*70)

    # Create directories
    os.makedirs('models/isolation_forest', exist_ok=True)
    os.makedirs('models/lof', exist_ok=True)
    os.makedirs('models/autoencoder', exist_ok=True)

    # Load data
    print("\nLoading preprocessed data...")
    train_df = pd.read_csv('data/processed/train_data.csv')
    test_df = pd.read_csv('data/processed/test_data.csv')

    # Load preprocessor
    preprocessor = IoTPreprocessor()
    preprocessor.load('models/preprocessor.pkl')

    # Get feature matrices
    train_normal = train_df[train_df['anomaly_label'] == 0]
    X_train = preprocessor.get_feature_matrix(train_normal, feature_type='all')
    X_test = preprocessor.get_feature_matrix(test_df, feature_type='all')
    y_test = test_df['anomaly_label'].values

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")

    # 1. Isolation Forest
    print("\n" + "="*70)
    print("Training Isolation Forest...")
    print("="*70)
    iforest = IsolationForestDetector(contamination=0.02, n_estimators=100)
    iforest.fit(X_train)
    iforest.save('models/isolation_forest/model.pkl')
    print("✓ Isolation Forest saved")

    # 2. LOF
    print("\n" + "="*70)
    print("Training Local Outlier Factor...")
    print("="*70)
    lof = LOFDetector(n_neighbors=20, contamination=0.02)
    lof.fit(X_train)
    lof.save('models/lof/model.pkl')
    print("✓ LOF saved")

    # 3. Autoencoder
    print("\n" + "="*70)
    print("Training Autoencoder...")
    print("="*70)
    autoencoder = AutoencoderDetector(
        input_dim=X_train.shape[1],
        encoding_dim=8,
        hidden_dims=[16, 12]
    )
    autoencoder.fit(X_train, epochs=50, batch_size=64, verbose=1)
    autoencoder.save('models/autoencoder/model')
    print("✓ Autoencoder saved")

    print("\n" + "="*70)
    print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
    print("="*70)

if __name__ == '__main__':
    main()
