#!/usr/bin/env python3
"""
Unit tests for Anomaly Detection Models

Tests cover:
- Model initialization and configuration
- Training (fit) functionality
- Prediction consistency
- Decision function scoring
- Save/load functionality
- Edge cases and error handling

Author: Lucas William Junges
Date: December 2024
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import shutil

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from models.anomaly_detectors import IsolationForestDetector, LOFDetector, AutoencoderDetector


# ========================================
# Fixtures
# ========================================

@pytest.fixture
def sample_training_data():
    """Create sample normal training data"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 27  # Typical feature count after engineering

    # Generate normal data (multivariate Gaussian)
    X = np.random.randn(n_samples, n_features)

    return X


@pytest.fixture
def sample_test_data():
    """Create sample test data with anomalies"""
    np.random.seed(43)
    n_normal = 400
    n_anomaly = 100
    n_features = 27

    # Normal data
    X_normal = np.random.randn(n_normal, n_features)

    # Anomalous data (shifted mean, increased variance)
    X_anomaly = np.random.randn(n_anomaly, n_features) * 3 + 5

    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

    return X, y


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model saving"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir)


# ========================================
# Isolation Forest Tests
# ========================================

class TestIsolationForest:
    """Test suite for IsolationForestDetector"""

    def test_initialization(self):
        """Test IF initializes with correct defaults"""
        model = IsolationForestDetector()

        assert model.contamination == 0.1
        assert model.n_estimators == 100
        assert model.model is None

    def test_custom_params(self):
        """Test IF accepts custom parameters"""
        model = IsolationForestDetector(
            contamination=0.05,
            n_estimators=200,
            random_state=42
        )

        assert model.contamination == 0.05
        assert model.n_estimators == 200

    def test_fit_creates_model(self, sample_training_data):
        """Test that fit creates trained model"""
        model = IsolationForestDetector()
        model.fit(sample_training_data)

        assert model.model is not None

    def test_predict_after_fit(self, sample_training_data, sample_test_data):
        """Test prediction after training"""
        model = IsolationForestDetector(random_state=42)
        model.fit(sample_training_data)

        X_test, y_test = sample_test_data
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})  # Binary predictions

    def test_decision_function(self, sample_training_data, sample_test_data):
        """Test anomaly scoring"""
        model = IsolationForestDetector(random_state=42)
        model.fit(sample_training_data)

        X_test, _ = sample_test_data
        scores = model.decision_function(X_test)

        assert scores.shape == (len(X_test),)
        assert scores.dtype == np.float64

    def test_detects_anomalies(self, sample_training_data, sample_test_data):
        """Test that model actually detects anomalies"""
        model = IsolationForestDetector(contamination=0.2, random_state=42)
        model.fit(sample_training_data)

        X_test, y_test = sample_test_data
        predictions = model.predict(X_test)

        # Should detect at least some anomalies
        assert predictions.sum() > 0
        # Should achieve better than random (>50% recall)
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, predictions)
        assert recall > 0.5

    def test_save_and_load(self, sample_training_data, sample_test_data, temp_model_dir):
        """Test model persistence"""
        # Train and save
        model = IsolationForestDetector(random_state=42)
        model.fit(sample_training_data)

        save_path = temp_model_dir / 'test_iforest'
        model.save(save_path)

        assert (save_path / 'model.pkl').exists()

        # Load and compare
        loaded_model = IsolationForestDetector.load(save_path)

        X_test, _ = sample_test_data
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_reproducibility(self, sample_training_data, sample_test_data):
        """Test that same seed gives same results"""
        X_test, _ = sample_test_data

        model1 = IsolationForestDetector(random_state=42)
        model1.fit(sample_training_data)
        pred1 = model1.predict(X_test)

        model2 = IsolationForestDetector(random_state=42)
        model2.fit(sample_training_data)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


# ========================================
# Local Outlier Factor Tests
# ========================================

class TestLOF:
    """Test suite for LOFDetector"""

    def test_initialization(self):
        """Test LOF initializes with correct defaults"""
        model = LOFDetector()

        assert model.contamination == 0.1
        assert model.n_neighbors == 20
        assert model.model is None

    def test_custom_params(self):
        """Test LOF accepts custom parameters"""
        model = LOFDetector(
            contamination=0.05,
            n_neighbors=30
        )

        assert model.contamination == 0.05
        assert model.n_neighbors == 30

    def test_fit_creates_model(self, sample_training_data):
        """Test that fit creates trained model"""
        model = LOFDetector()
        model.fit(sample_training_data)

        assert model.model is not None

    def test_predict_after_fit(self, sample_training_data, sample_test_data):
        """Test prediction after training"""
        model = LOFDetector()
        model.fit(sample_training_data)

        X_test, y_test = sample_test_data
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})

    def test_decision_function(self, sample_training_data, sample_test_data):
        """Test anomaly scoring"""
        model = LOFDetector()
        model.fit(sample_training_data)

        X_test, _ = sample_test_data
        scores = model.decision_function(X_test)

        assert scores.shape == (len(X_test),)
        assert scores.dtype == np.float64

    def test_detects_anomalies(self, sample_training_data, sample_test_data):
        """Test that model actually detects anomalies"""
        model = LOFDetector(contamination=0.2)
        model.fit(sample_training_data)

        X_test, y_test = sample_test_data
        predictions = model.predict(X_test)

        # Should detect at least some anomalies
        assert predictions.sum() > 0
        # Should achieve better than random
        from sklearn.metrics import recall_score
        recall = recall_score(y_test, predictions)
        assert recall > 0.5

    def test_save_and_load(self, sample_training_data, sample_test_data, temp_model_dir):
        """Test model persistence"""
        # Train and save
        model = LOFDetector()
        model.fit(sample_training_data)

        save_path = temp_model_dir / 'test_lof'
        model.save(save_path)

        assert (save_path / 'model.pkl').exists()

        # Load and compare
        loaded_model = LOFDetector.load(save_path)

        X_test, _ = sample_test_data
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)


# ========================================
# Autoencoder Tests
# ========================================

class TestAutoencoder:
    """Test suite for AutoencoderDetector"""

    def test_initialization(self):
        """Test AE initializes with correct defaults"""
        model = AutoencoderDetector()

        assert model.encoding_dim == 8
        assert model.contamination == 0.1
        assert model.model is None

    def test_custom_params(self):
        """Test AE accepts custom parameters"""
        model = AutoencoderDetector(
            encoding_dim=12,
            contamination=0.05,
            random_state=42
        )

        assert model.encoding_dim == 12
        assert model.contamination == 0.05

    def test_fit_creates_model(self, sample_training_data):
        """Test that fit creates trained model"""
        model = AutoencoderDetector(random_state=42)
        model.fit(sample_training_data, epochs=5, verbose=0)

        assert model.model is not None

    def test_predict_after_fit(self, sample_training_data, sample_test_data):
        """Test prediction after training"""
        model = AutoencoderDetector(random_state=42)
        model.fit(sample_training_data, epochs=5, verbose=0)

        X_test, y_test = sample_test_data
        predictions = model.predict(X_test)

        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})

    def test_decision_function(self, sample_training_data, sample_test_data):
        """Test anomaly scoring based on reconstruction error"""
        model = AutoencoderDetector(random_state=42)
        model.fit(sample_training_data, epochs=5, verbose=0)

        X_test, _ = sample_test_data
        scores = model.decision_function(X_test)

        assert scores.shape == (len(X_test),)
        assert scores.dtype == np.float64
        # Scores should be non-negative (reconstruction errors)
        assert (scores >= 0).all()

    def test_reconstruction(self, sample_training_data):
        """Test that autoencoder can reconstruct input"""
        model = AutoencoderDetector(random_state=42)
        model.fit(sample_training_data, epochs=10, verbose=0)

        # Reconstruct training data
        reconstructed = model.model.predict(sample_training_data, verbose=0)

        # Shape should match
        assert reconstructed.shape == sample_training_data.shape

        # Reconstruction should be reasonably close
        mse = np.mean((sample_training_data - reconstructed)**2)
        assert mse < 1.0  # Reasonable reconstruction error

    def test_save_and_load(self, sample_training_data, sample_test_data, temp_model_dir):
        """Test model persistence"""
        # Train and save
        model = AutoencoderDetector(random_state=42)
        model.fit(sample_training_data, epochs=5, verbose=0)

        save_path = temp_model_dir / 'test_autoencoder'
        model.save(save_path)

        assert (save_path / 'model_model.h5').exists()
        assert (save_path / 'model_metadata.json').exists()

        # Load and compare
        loaded_model = AutoencoderDetector.load(save_path)

        X_test, _ = sample_test_data
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)

        np.testing.assert_array_equal(original_pred, loaded_pred)


# ========================================
# Cross-Model Comparison Tests
# ========================================

def test_all_models_interface_consistency(sample_training_data, sample_test_data):
    """Test that all models follow same interface"""
    X_test, _ = sample_test_data

    models = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        AutoencoderDetector(random_state=42)
    ]

    for model in models:
        # All should support fit
        if isinstance(model, AutoencoderDetector):
            model.fit(sample_training_data, epochs=5, verbose=0)
        else:
            model.fit(sample_training_data)

        # All should support predict
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert set(predictions).issubset({0, 1})

        # All should support decision_function
        scores = model.decision_function(X_test)
        assert scores.shape == (len(X_test),)
        assert scores.dtype == np.float64


# ========================================
# Edge Cases
# ========================================

def test_handles_single_sample(sample_training_data):
    """Test that models handle single sample prediction"""
    models = [
        IsolationForestDetector(random_state=42),
        LOFDetector(),
        AutoencoderDetector(random_state=42)
    ]

    single_sample = sample_training_data[:1]

    for model in models:
        if isinstance(model, AutoencoderDetector):
            model.fit(sample_training_data, epochs=5, verbose=0)
        else:
            model.fit(sample_training_data)

        prediction = model.predict(single_sample)
        assert prediction.shape == (1,)
        assert prediction[0] in {0, 1}


def test_handles_different_feature_dimensions():
    """Test models with different input dimensions"""
    for n_features in [5, 10, 27, 50]:
        X = np.random.randn(500, n_features)

        model = IsolationForestDetector(random_state=42)
        model.fit(X)

        predictions = model.predict(X[:10])
        assert predictions.shape == (10,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
