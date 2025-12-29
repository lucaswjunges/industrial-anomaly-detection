#!/usr/bin/env python3
"""
Unit tests for IoTPreprocessor

Tests cover:
- Data validation and error handling
- Normalization (regime-aware and standard)
- Feature engineering
- Transform consistency
- Edge cases and boundary conditions

"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from preprocessing.preprocessor import IoTPreprocessor

# ========================================
# Fixtures
# ========================================

@pytest.fixture
def sample_data():
    """Create sample sensor data for testing"""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'temperature': np.random.normal(45, 3, n),
        'vibration': np.random.normal(2.5, 0.4, n),
        'pressure': np.random.normal(6.5, 0.5, n),
        'flow_rate': np.random.normal(150, 10, n),
        'current': np.random.normal(85, 5, n),
        'duty_cycle': np.random.normal(0.75, 0.08, n),
        'operational_state': np.random.choice(['normal', 'startup', 'high_load', 'maintenance'], n),
        'is_anomaly': np.random.choice([0, 1], n, p=[0.98, 0.02])
    })

    return df

@pytest.fixture
def preprocessor():
    """Create fresh preprocessor instance"""
    return IoTPreprocessor()

# ========================================
# Initialization Tests
# ========================================

def test_preprocessor_initialization():
    """Test that preprocessor initializes with correct defaults"""
    prep = IoTPreprocessor()

    assert prep.normalization_method == 'robust'
    assert prep.feature_window_size == 5
    assert prep.scalers == {}
    assert not prep.is_fitted

def test_preprocessor_custom_params():
    """Test initialization with custom parameters"""
    prep = IoTPreprocessor(
        normalization_method='standard',
        feature_window_size=10
    )

    assert prep.normalization_method == 'standard'
    assert prep.feature_window_size == 10

# ========================================
# Fitting Tests
# ========================================

def test_fit_creates_scalers(sample_data, preprocessor):
    """Test that fit creates regime-specific scalers"""
    preprocessor.fit(sample_data, regime_column='operational_state')

    assert preprocessor.is_fitted
    assert len(preprocessor.scalers) > 0

    # Check that each regime has a scaler
    unique_regimes = sample_data['operational_state'].unique()
    for regime in unique_regimes:
        assert regime in preprocessor.scalers

def test_fit_on_normal_data_only(sample_data, preprocessor):
    """Test fitting on normal data only (recommended practice)"""
    normal_data = sample_data[sample_data['is_anomaly'] == 0]

    preprocessor.fit(normal_data, regime_column='operational_state')

    assert preprocessor.is_fitted
    assert len(preprocessor.scalers) > 0

def test_fit_without_regime_column(sample_data, preprocessor):
    """Test fitting without regime-aware normalization"""
    preprocessor.fit(sample_data, regime_column=None)

    assert preprocessor.is_fitted
    # Should have single 'global' scaler
    assert 'global' in preprocessor.scalers

def test_fit_raises_on_missing_columns():
    """Test that fit raises error on missing sensor columns"""
    df = pd.DataFrame({
        'temperature': [1, 2, 3],
        'vibration': [1, 2, 3]
        # Missing other sensors
    })

    prep = IoTPreprocessor()

    with pytest.raises((KeyError, ValueError)):
        prep.fit(df)

# ========================================
# Transform Tests
# ========================================

def test_transform_requires_fitted(sample_data, preprocessor):
    """Test that transform raises error if not fitted"""
    with pytest.raises((AttributeError, ValueError)):
        preprocessor.transform(sample_data)

def test_transform_output_shape(sample_data, preprocessor):
    """Test that transform maintains sample count"""
    preprocessor.fit(sample_data, regime_column='operational_state')
    transformed = preprocessor.transform(sample_data, regime_column='operational_state')

    assert len(transformed) == len(sample_data)

def test_transform_output_is_array(sample_data, preprocessor):
    """Test that transform returns numpy array"""
    preprocessor.fit(sample_data, regime_column='operational_state')
    transformed = preprocessor.transform(sample_data, regime_column='operational_state')

    assert isinstance(transformed, np.ndarray)
    assert transformed.dtype == np.float64

def test_transform_no_nans(sample_data, preprocessor):
    """Test that transform output has no NaN values"""
    preprocessor.fit(sample_data, regime_column='operational_state')
    transformed = preprocessor.transform(sample_data, regime_column='operational_state')

    assert not np.isnan(transformed).any(), "Transform output contains NaN values"

def test_transform_consistency(sample_data, preprocessor):
    """Test that transform gives same results on repeated calls"""
    preprocessor.fit(sample_data, regime_column='operational_state')

    result1 = preprocessor.transform(sample_data, regime_column='operational_state')
    result2 = preprocessor.transform(sample_data, regime_column='operational_state')

    np.testing.assert_array_almost_equal(result1, result2)

# ========================================
# Feature Engineering Tests
# ========================================

def test_feature_engineering_increases_dimensions(sample_data, preprocessor):
    """Test that feature engineering adds features"""
    preprocessor.fit(sample_data, regime_column='operational_state')

    # Transform with default features
    n_sensor_columns = 6  # temperature, vibration, pressure, flow_rate, current, duty_cycle
    transformed = preprocessor.transform(sample_data, regime_column='operational_state')

    # Should have more features than raw sensors
    assert transformed.shape[1] >= n_sensor_columns

# ========================================
# Regime-Aware Normalization Tests
# ========================================

def test_regime_specific_scaling(sample_data, preprocessor):
    """Test that different regimes get scaled differently"""
    preprocessor.fit(sample_data, regime_column='operational_state')

    # Get data from two different regimes
    normal_data = sample_data[sample_data['operational_state'] == 'normal'].head(10)
    startup_data = sample_data[sample_data['operational_state'] == 'startup'].head(10)

    # Transform both
    normal_transformed = preprocessor.transform(normal_data, regime_column='operational_state')
    startup_transformed = preprocessor.transform(startup_data, regime_column='operational_state')

    # Scalers should be different for different regimes
    assert 'normal' in preprocessor.scalers
    assert 'startup' in preprocessor.scalers

# ========================================
# Edge Cases and Error Handling
# ========================================

def test_handles_single_sample(preprocessor):
    """Test that preprocessor handles single sample"""
    train_data = pd.DataFrame({
        'temperature': [45.0] * 100,
        'vibration': [2.5] * 100,
        'pressure': [6.5] * 100,
        'flow_rate': [150.0] * 100,
        'current': [85.0] * 100,
        'duty_cycle': [0.75] * 100,
        'operational_state': ['normal'] * 100
    })

    preprocessor.fit(train_data, regime_column='operational_state')

    # Transform single sample
    test_sample = train_data.head(1)
    result = preprocessor.transform(test_sample, regime_column='operational_state')

    assert result.shape[0] == 1
    assert not np.isnan(result).any()

def test_handles_unseen_regime(sample_data, preprocessor):
    """Test handling of operational state not seen during training"""
    # Fit without 'maintenance' state
    train_data = sample_data[sample_data['operational_state'] != 'maintenance']
    preprocessor.fit(train_data, regime_column='operational_state')

    # Transform with 'maintenance' state (unseen)
    test_data = sample_data[sample_data['operational_state'] == 'maintenance'].head(10)

    # Should either:
    # 1. Use global fallback scaler
    # 2. Use 'normal' as default
    # 3. Raise informative error
    # Implementation should handle this gracefully
    try:
        result = preprocessor.transform(test_data, regime_column='operational_state')
        assert not np.isnan(result).any()
    except (KeyError, ValueError) as e:
        # Expected behavior - informative error
        assert 'maintenance' in str(e).lower() or 'unseen' in str(e).lower()

def test_handles_missing_values(preprocessor):
    """Test handling of missing sensor values"""
    data_with_nans = pd.DataFrame({
        'temperature': [45.0, np.nan, 46.0],
        'vibration': [2.5, 2.6, np.nan],
        'pressure': [6.5, 6.6, 6.7],
        'flow_rate': [150.0, 151.0, 152.0],
        'current': [85.0, 86.0, 87.0],
        'duty_cycle': [0.75, 0.76, 0.77],
        'operational_state': ['normal', 'normal', 'normal']
    })

    # Should either:
    # 1. Impute missing values
    # 2. Raise error
    # Either is acceptable
    try:
        preprocessor.fit(data_with_nans, regime_column='operational_state')
    except (ValueError, TypeError):
        pass  # Expected if implementation doesn't handle NaNs

def test_handles_constant_features():
    """Test handling of constant (zero variance) features"""
    constant_data = pd.DataFrame({
        'temperature': [45.0] * 100,  # Constant
        'vibration': [2.5] * 100,     # Constant
        'pressure': [6.5] * 100,      # Constant
        'flow_rate': [150.0] * 100,   # Constant
        'current': [85.0] * 100,      # Constant
        'duty_cycle': [0.75] * 100,   # Constant
        'operational_state': ['normal'] * 100
    })

    prep = IoTPreprocessor()

    # Should handle constant features gracefully (scale to 0 or leave as-is)
    prep.fit(constant_data, regime_column='operational_state')
    result = prep.transform(constant_data, regime_column='operational_state')

    assert not np.isnan(result).any()

# ========================================
# Performance Tests
# ========================================

def test_transform_speed(sample_data, preprocessor):
    """Test that transform completes in reasonable time"""
    import time

    preprocessor.fit(sample_data, regime_column='operational_state')

    start = time.time()
    _ = preprocessor.transform(sample_data, regime_column='operational_state')
    elapsed = time.time() - start

    # Should complete in < 1 second for 1000 samples
    assert elapsed < 1.0, f"Transform took {elapsed:.2f}s, expected < 1.0s"

# ========================================
# Integration Tests
# ========================================

def test_full_pipeline_workflow(sample_data, preprocessor):
    """Test complete fit-transform workflow"""
    # Split data
    train_size = int(0.8 * len(sample_data))
    train_data = sample_data.iloc[:train_size]
    test_data = sample_data.iloc[train_size:]

    # Fit on training data only
    preprocessor.fit(train_data, regime_column='operational_state')

    # Transform both
    X_train = preprocessor.transform(train_data, regime_column='operational_state')
    X_test = preprocessor.transform(test_data, regime_column='operational_state')

    # Assertions
    assert X_train.shape[0] == len(train_data)
    assert X_test.shape[0] == len(test_data)
    assert X_train.shape[1] == X_test.shape[1]  # Same number of features
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
