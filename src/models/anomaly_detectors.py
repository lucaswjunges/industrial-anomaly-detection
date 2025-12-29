"""
Anomaly Detection Models for Industrial IoT
Implements: Isolation Forest, Local Outlier Factor, Autoencoder
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json
from typing import Tuple, Dict, Optional

class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection.

    Effective for detecting global anomalies by isolating outliers
    through random partitioning.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 100,
        max_samples: int = 256,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest.

        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.contamination = contamination
        self.threshold = None

    def fit(self, X: np.ndarray):
        """Fit Isolation Forest on training data."""
        print(f"Training Isolation Forest on {X.shape[0]:,} samples...")
        self.model.fit(X)
        print("Isolation Forest trained successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns:
            Binary labels: 1 for anomaly, 0 for normal
        """
        predictions = self.model.predict(X)
        # sklearn returns -1 for anomalies, 1 for normal
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Returns:
            Anomaly scores (lower = more anomalous)
        """
        return -self.model.score_samples(X)

    def save(self, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Isolation Forest saved to: {path}")

    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Isolation Forest loaded from: {path}")

class LOFDetector:
    """
    Local Outlier Factor for anomaly detection.

    Identifies local density anomalies by comparing local density
    of a point to the local density of its neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.02,
        novelty: bool = True
    ):
        """
        Initialize LOF.

        Args:
            n_neighbors: Number of neighbors to use
            contamination: Expected proportion of anomalies
            novelty: Use novelty detection (required for prediction)
        """
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty,
            n_jobs=-1
        )
        self.contamination = contamination

    def fit(self, X: np.ndarray):
        """Fit LOF on training data."""
        print(f"Training LOF on {X.shape[0]:,} samples...")
        self.model.fit(X)
        print("LOF trained successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Returns:
            Binary labels: 1 for anomaly, 0 for normal
        """
        predictions = self.model.predict(X)
        # sklearn returns -1 for anomalies, 1 for normal
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.

        Returns:
            Anomaly scores (lower = more anomalous)
        """
        return -self.model.score_samples(X)

    def save(self, path: str):
        """Save model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"LOF saved to: {path}")

    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"LOF loaded from: {path}")

class AutoencoderDetector:
    """
    Autoencoder-based anomaly detection.

    Learns to compress and reconstruct normal patterns.
    Anomalies have high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        hidden_dims: list = [16, 12],
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize Autoencoder.

        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation
            hidden_dims: Hidden layer dimensions [encoder, decoder]
            learning_rate: Learning rate for Adam optimizer
            random_state: Random seed
        """
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.model = None
        self.threshold = None
        self.history = None

        self._build_model()

    def _build_model(self):
        """Build autoencoder architecture."""

        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = encoder_input

        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(x)

        # Decoder
        x = encoded
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        decoded = layers.Dense(self.input_dim, activation='linear')(x)

        # Full autoencoder
        self.model = keras.Model(encoder_input, decoded, name='autoencoder')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

    def fit(
        self,
        X: np.ndarray,
        validation_split: float = 0.1,
        epochs: int = 50,
        batch_size: int = 64,
        verbose: int = 1
    ):
        """
        Train autoencoder on normal data.

        Args:
            X: Training features (normal samples only)
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        print(f"Training Autoencoder on {X.shape[0]:,} samples...")
        print(f"Architecture: {self.input_dim} -> {self.hidden_dims} -> "
              f"{self.encoding_dim} -> {self.hidden_dims[::-1]} -> {self.input_dim}")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        self.history = self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )

        # Set threshold based on training reconstruction error
        train_reconstructions = self.model.predict(X, verbose=0)
        train_errors = np.mean((X - train_reconstructions) ** 2, axis=1)

        # TODO: This is too simplistic - should use validation set for threshold tuning
        # Current approach (99th percentile) is probably too conservative
        self.threshold = np.percentile(train_errors, 99)

        print(f"Autoencoder trained successfully")
        print(f"Reconstruction error threshold: {self.threshold:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels based on reconstruction error.

        Returns:
            Binary labels: 1 for anomaly, 0 for normal
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error as anomaly score.

        Returns:
            Reconstruction errors (higher = more anomalous)
        """
        reconstructions = self.model.predict(X, verbose=0)
        errors = np.mean((X - reconstructions) ** 2, axis=1)
        return errors

    def get_reconstruction(self, X: np.ndarray) -> np.ndarray:
        """Get reconstructed samples."""
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        """Save model and metadata."""
        self.model.save(path + '_model.h5')

        metadata = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate,
            'threshold': float(self.threshold) if self.threshold else None
        }

        with open(path + '_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Autoencoder saved to: {path}")

    def load(self, path: str):
        """Load model and metadata."""
        # Load without compiling to avoid metric deserialization issues
        self.model = keras.models.load_model(path + '_model.h5', compile=False)

        # Recompile with correct metrics
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=[keras.metrics.MeanAbsoluteError()]
        )

        with open(path + '_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_dims = metadata['hidden_dims']
        self.learning_rate = metadata['learning_rate']
        self.threshold = metadata['threshold']

        print(f"Autoencoder loaded from: {path}")

class EnsembleDetector:
    """
    Ensemble anomaly detector combining multiple methods.

    Uses voting or weighted scoring to combine predictions.
    """

    def __init__(
        self,
        detectors: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
        voting_threshold: float = 0.5
    ):
        """
        Initialize ensemble.

        Args:
            detectors: Dictionary of detector instances
            weights: Optional weights for each detector
            voting_threshold: Threshold for ensemble decision
        """
        self.detectors = detectors
        self.weights = weights or {name: 1.0 for name in detectors.keys()}
        self.voting_threshold = voting_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction using weighted voting.

        Returns:
            Binary labels: 1 for anomaly, 0 for normal
        """
        scores = self.score_samples(X)
        return (scores > self.voting_threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute weighted ensemble scores.

        Returns:
            Normalized ensemble scores [0, 1]
        """
        all_scores = {}

        for name, detector in self.detectors.items():
            raw_scores = detector.score_samples(X)

            # Normalize to [0, 1]
            min_score = raw_scores.min()
            max_score = raw_scores.max()

            if max_score > min_score:
                normalized = (raw_scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(raw_scores)

            all_scores[name] = normalized

        # Weighted average
        ensemble_scores = np.zeros(len(X))
        total_weight = sum(self.weights.values())

        for name, scores in all_scores.items():
            ensemble_scores += self.weights[name] * scores

        ensemble_scores /= total_weight

        return ensemble_scores

    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each detector."""
        return {
            name: detector.predict(X)
            for name, detector in self.detectors.items()
        }

def main():
    """Train and evaluate all anomaly detection models."""
    import sys
    import os
    from pathlib import Path

    # Add src directory to path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from preprocessing.preprocessor import IoTPreprocessor

    # Create model directories
    os.makedirs('../../models/isolation_forest', exist_ok=True)
    os.makedirs('../../models/lof', exist_ok=True)
    os.makedirs('../../models/autoencoder', exist_ok=True)

    print("="*60)
    print("ANOMALY DETECTION MODEL TRAINING")
    print("="*60)

    # Load preprocessed data
    train_df = pd.read_csv('../../data/processed/train_data.csv')
    test_df = pd.read_csv('../../data/processed/test_data.csv')

    # Load preprocessor
    preprocessor = IoTPreprocessor()
    preprocessor.load('../../models/preprocessor.pkl')

    # Get feature matrices (normal training data only)
    train_normal = train_df[train_df['anomaly_label'] == 0]
    X_train = preprocessor.get_feature_matrix(train_normal, feature_type='all')
    X_test = preprocessor.get_feature_matrix(test_df, feature_type='all')

    y_test = test_df['anomaly_label'].values

    print(f"\nTraining samples (normal): {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Test anomalies: {y_test.sum():,} ({y_test.mean():.2%})")
    print(f"Feature dimensions: {X_train.shape[1]}")

    # 1. Isolation Forest
    print("\n" + "="*60)
    print("1. ISOLATION FOREST")
    print("="*60)

    iforest = IsolationForestDetector(
        contamination=0.02,
        n_estimators=100,
        max_samples=256
    )
    iforest.fit(X_train)
    iforest.save('../../models/isolation_forest/model.pkl')

    # 2. Local Outlier Factor
    print("\n" + "="*60)
    print("2. LOCAL OUTLIER FACTOR")
    print("="*60)

    lof = LOFDetector(
        n_neighbors=20,
        contamination=0.02,
        novelty=True
    )
    lof.fit(X_train)
    lof.save('../../models/lof/model.pkl')

    # 3. Autoencoder
    print("\n" + "="*60)
    print("3. AUTOENCODER")
    print("="*60)

    autoencoder = AutoencoderDetector(
        input_dim=X_train.shape[1],
        encoding_dim=8,
        hidden_dims=[16, 12],
        learning_rate=0.001
    )
    autoencoder.fit(
        X_train,
        validation_split=0.1,
        epochs=50,
        batch_size=64,
        verbose=1
    )
    autoencoder.save('../../models/autoencoder/model')

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)

    # Quick evaluation
    print("\nQuick evaluation on test set:")

    for name, detector in [
        ('Isolation Forest', iforest),
        ('LOF', lof),
        ('Autoencoder', autoencoder)
    ]:
        predictions = detector.predict(X_test)
        tp = ((predictions == 1) & (y_test == 1)).sum()
        fp = ((predictions == 1) & (y_test == 0)).sum()
        fn = ((predictions == 0) & (y_test == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{name}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Anomalies detected: {predictions.sum()}")

if __name__ == '__main__':
    main()
