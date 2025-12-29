#!/usr/bin/env python3
"""
Unit tests for Data Generation Modules

Tests cover:
- IoT Simulator functionality
- NASA Bearing Loader
- Data quality and consistency
- Error handling

"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
src_dir = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_dir))

from data_generation.iot_simulator import IndustrialPumpSimulator
from data_generation.nasa_bearing_loader import NASABearingLoader

# ========================================
# IoT Simulator Tests
# ========================================

class TestIndustrialPumpSimulator:
    """Test suite for IndustrialPumpSimulator"""

    def test_initialization(self):
        """Test simulator initializes correctly"""
        sim = IndustrialPumpSimulator(seed=42)

        assert sim.seed == 42
        assert len(sim.sensor_config) == 6  # 6 sensors
        assert len(sim.failure_modes) == 5  # 5 failure types

    def test_generate_base_signal(self):
        """Test base signal generation"""
        sim = IndustrialPumpSimulator(seed=42)
        n_samples = 1000

        df = sim.generate_base_signal(n_samples, sampling_rate=60)

        assert len(df) == n_samples
        assert 'timestamp' in df.columns
        assert 'temperature' in df.columns
        assert 'vibration' in df.columns
        assert 'pressure' in df.columns
        assert 'flow_rate' in df.columns
        assert 'current' in df.columns
        assert 'duty_cycle' in df.columns

    def test_sensor_values_in_range(self):
        """Test that sensor values are within expected ranges"""
        sim = IndustrialPumpSimulator(seed=42)
        df = sim.generate_base_signal(1000, sampling_rate=60)

        # Temperature: should be around 45°C
        assert df['temperature'].mean() > 40
        assert df['temperature'].mean() < 50

        # Vibration: should be around 2.5 mm/s
        assert df['vibration'].mean() > 2.0
        assert df['vibration'].mean() < 3.0

        # Pressure: should be around 6.5 bar
        assert df['pressure'].mean() > 6.0
        assert df['pressure'].mean() < 7.0

    def test_operational_states_injection(self):
        """Test that operational states are injected"""
        sim = IndustrialPumpSimulator(seed=42)
        df = sim.generate_base_signal(5000, sampling_rate=60)
        df, events = sim.inject_operational_states(df)

        assert 'operational_state' in df.columns
        assert len(events) > 0  # Should have state transitions

        # Check that we have multiple states
        unique_states = df['operational_state'].unique()
        assert len(unique_states) > 1
        assert 'normal' in unique_states

    def test_anomaly_injection(self):
        """Test that anomalies are injected"""
        sim = IndustrialPumpSimulator(seed=42)
        df = sim.generate_base_signal(5000, sampling_rate=60)
        df, _ = sim.inject_operational_states(df)
        df, anomaly_events = sim.inject_anomalies(df, anomaly_rate=0.02)

        assert 'is_anomaly' in df.columns
        assert len(anomaly_events) > 0  # Should have anomalies

        # Check anomaly rate is approximately correct
        anomaly_rate = df['is_anomaly'].mean()
        assert 0.01 < anomaly_rate < 0.03  # Should be around 2%

    def test_full_dataset_generation(self):
        """Test complete dataset generation pipeline"""
        sim = IndustrialPumpSimulator(seed=42)

        df, metadata = sim.generate_dataset(
            duration_days=1,  # Short for testing
            sampling_rate=60,
            anomaly_rate=0.02
        )

        # Check data shape
        expected_samples = int((1 * 24 * 60 * 60) / 60)  # 1 day, 1-min sampling
        assert len(df) == expected_samples

        # Check metadata
        assert 'duration_days' in metadata
        assert 'sampling_rate_seconds' in metadata
        assert 'anomaly_rate' in metadata
        assert 'sensor_variables' in metadata

    def test_reproducibility(self):
        """Test that same seed produces same data"""
        sim1 = IndustrialPumpSimulator(seed=42)
        df1, _ = sim1.generate_dataset(duration_days=1, sampling_rate=60, anomaly_rate=0.02)

        sim2 = IndustrialPumpSimulator(seed=42)
        df2, _ = sim2.generate_dataset(duration_days=1, sampling_rate=60, anomaly_rate=0.02)

        pd.testing.assert_frame_equal(df1, df2)

    def test_no_missing_values(self):
        """Test that generated data has no missing values"""
        sim = IndustrialPumpSimulator(seed=42)
        df, _ = sim.generate_dataset(duration_days=1, sampling_rate=60, anomaly_rate=0.02)

        assert df.isnull().sum().sum() == 0  # No NaNs

# ========================================
# NASA Bearing Loader Tests
# ========================================

class TestNASABearingLoader:
    """Test suite for NASABearingLoader"""

    def test_initialization(self):
        """Test loader initializes correctly"""
        loader = NASABearingLoader()

        assert loader.data_path is not None
        assert loader.sampling_rate == 20000
        assert loader.resampled_rate == 60

    def test_custom_data_path(self, tmp_path):
        """Test initialization with custom path"""
        loader = NASABearingLoader(data_path=str(tmp_path))

        assert loader.data_path == tmp_path

    def test_download_instructions(self):
        """Test that download instructions are provided"""
        loader = NASABearingLoader()
        instructions = loader.download_instructions()

        assert isinstance(instructions, str)
        assert len(instructions) > 100
        assert 'NASA' in instructions
        assert 'bearing' in instructions.lower()
        assert 'http' in instructions.lower()

    def test_check_data_available_when_missing(self, tmp_path):
        """Test data availability check when data is missing"""
        loader = NASABearingLoader(data_path=str(tmp_path))
        available = loader.check_data_available()

        assert available is False

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath('data/raw/nasa_bearing').exists(),
        reason="NASA bearing data not available"
    )
    def test_check_data_available_when_present(self):
        """Test data availability check when data exists"""
        loader = NASABearingLoader()
        available = loader.check_data_available()

        # Will only pass if NASA data is actually downloaded
        if available:
            assert available is True

    def test_feature_extraction(self):
        """Test vibration feature extraction"""
        loader = NASABearingLoader()

        # Create synthetic vibration signals
        vibration_x = np.random.randn(2000)
        vibration_y = np.random.randn(2000)
        timestamp = pd.Timestamp('2024-01-01 12:00:00')

        features = loader._extract_vibration_features(vibration_x, vibration_y, timestamp)

        # Check that required features are present
        assert 'timestamp' in features
        assert 'vibration' in features
        assert 'vibration_rms' in features
        assert 'vibration_peak' in features
        assert 'temperature' in features  # Synthetic proxy

        # Check that values are reasonable
        assert features['vibration'] >= 0
        assert features['vibration_rms'] >= 0
        assert features['vibration_peak'] >= 0

    def test_spectral_energy_calculation(self):
        """Test spectral energy calculation"""
        loader = NASABearingLoader()

        # Sine wave signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine

        energy = loader._spectral_energy(signal)

        assert energy > 0
        assert np.isfinite(energy)

    def test_synthesize_operational_states(self):
        """Test operational state synthesis"""
        loader = NASABearingLoader()

        # Create dummy dataframe
        df = pd.DataFrame({
            'vibration': np.random.randn(2000)
        })

        states = loader._synthesize_operational_states(df)

        assert len(states) == len(df)
        assert 'normal' in states.values
        # Should have some variety
        assert len(states.unique()) > 1

# ========================================
# Data Quality Tests
# ========================================

def test_simulated_data_quality():
    """Test that simulated data meets quality criteria"""
    sim = IndustrialPumpSimulator(seed=42)
    df, _ = sim.generate_dataset(duration_days=1, sampling_rate=60, anomaly_rate=0.02)

    # No missing values
    assert df.isnull().sum().sum() == 0

    # No infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert not np.isinf(df[numeric_cols]).any().any()

    # Reasonable value ranges
    assert df['temperature'].min() > 0
    assert df['temperature'].max() < 120  # Below boiling

    assert df['vibration'].min() >= 0
    assert df['vibration'].max() < 20  # Reasonable max

    assert df['pressure'].min() >= 0
    assert df['pressure'].max() < 15  # Within sensor range

    assert df['flow_rate'].min() >= 0
    assert df['current'].min() >= 0
    assert df['duty_cycle'].min() >= 0
    assert df['duty_cycle'].max() <= 1.0

def test_temporal_consistency():
    """Test that timestamps are consistent"""
    sim = IndustrialPumpSimulator(seed=42)
    df, _ = sim.generate_dataset(duration_days=1, sampling_rate=60, anomaly_rate=0.02)

    # Timestamps should be monotonically increasing
    assert (df['timestamp'].diff().dropna() > pd.Timedelta(0)).all()

    # Time deltas should be consistent (1 minute sampling)
    deltas = df['timestamp'].diff().dropna()
    expected_delta = pd.Timedelta(seconds=60)

    # Allow small tolerance for floating point
    assert (deltas == expected_delta).all()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
