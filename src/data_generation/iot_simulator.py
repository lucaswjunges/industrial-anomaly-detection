"""
Realistic Industrial IoT Sensor Data Simulator
Atlantic Water Operations Ltd. - Pumping Facility

Simulates multivariate time-series data from industrial water pumps
with realistic operational states, anomalies, and environmental factors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import json

class IndustrialPumpSimulator:
    """
    Simulates sensor data from an industrial pumping facility.

    Monitors:
    - Temperature (°C)
    - Vibration (mm/s RMS)
    - Pressure (bar)
    - Flow rate (m³/h)
    - Electrical current (A)
    - Pump duty cycle (%)
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

        # Operational parameters for normal operation
        self.normal_params = {
            'temperature': {'mean': 45.0, 'std': 3.0, 'min': 20, 'max': 85},
            'vibration': {'mean': 2.5, 'std': 0.4, 'min': 0, 'max': 15},
            'pressure': {'mean': 6.5, 'std': 0.5, 'min': 0, 'max': 12},
            'flow_rate': {'mean': 150.0, 'std': 10.0, 'min': 0, 'max': 250},
            'current': {'mean': 85.0, 'std': 5.0, 'min': 0, 'max': 150},
            'duty_cycle': {'mean': 75.0, 'std': 8.0, 'min': 0, 'max': 100}
        }

    def generate_base_signal(
        self,
        n_samples: int,
        sampling_rate: int = 60
    ) -> pd.DataFrame:
        """
        Generate normal operational data with realistic patterns.

        Args:
            n_samples: Number of samples to generate
            sampling_rate: Seconds between samples

        Returns:
            DataFrame with sensor readings
        """
        timestamps = [
            datetime(2024, 1, 1) + timedelta(seconds=i*sampling_rate)
            for i in range(n_samples)
        ]

        data = {'timestamp': timestamps}

        for sensor, params in self.normal_params.items():
            # Base signal with trend and seasonality
            base = params['mean']

            # Daily cycle (24 hour period)
            daily_cycle = 0.1 * params['std'] * np.sin(
                2 * np.pi * np.arange(n_samples) / (24 * 60)
            )

            # Weekly cycle
            weekly_cycle = 0.05 * params['std'] * np.sin(
                2 * np.pi * np.arange(n_samples) / (7 * 24 * 60)
            )

            # Random noise
            noise = np.random.normal(0, params['std'], n_samples)

            # Autocorrelation (smooth transitions)
            signal = base + daily_cycle + weekly_cycle + noise
            signal = self._apply_autocorrelation(signal, alpha=0.7)

            # Clip to physical limits
            signal = np.clip(signal, params['min'], params['max'])

            data[sensor] = signal

        return pd.DataFrame(data)

    def _apply_autocorrelation(
        self,
        signal: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """Apply AR(1) process for smooth transitions."""
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]

        for i in range(1, len(signal)):
            smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * signal[i]

        return smoothed

    def inject_operational_states(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Add realistic operational state transitions.

        States:
        - startup: gradual ramp-up
        - normal: steady-state operation
        - high_load: increased demand
        - maintenance: reduced operation
        - shutdown: gradual ramp-down
        """
        n_samples = len(df)
        states = ['normal'] * n_samples
        events = []

        # Startup periods (beginning of day)
        for day in range(n_samples // (24 * 60) + 1):
            start_idx = day * 24 * 60
            startup_duration = 30  # 30 minutes

            if start_idx < n_samples:
                end_idx = min(start_idx + startup_duration, n_samples)
                states[start_idx:end_idx] = ['startup'] * (end_idx - start_idx)

                # Gradual ramp-up
                ramp = np.linspace(0.3, 1.0, end_idx - start_idx)
                df.loc[start_idx:end_idx-1, 'duty_cycle'] *= ramp
                df.loc[start_idx:end_idx-1, 'flow_rate'] *= ramp
                df.loc[start_idx:end_idx-1, 'current'] *= ramp

                events.append({
                    'timestamp': df.iloc[start_idx]['timestamp'],
                    'type': 'startup',
                    'duration_min': startup_duration
                })

        # High load periods (peak hours)
        for day in range(n_samples // (24 * 60) + 1):
            # Morning peak: 7-9 AM
            peak_start = day * 24 * 60 + 7 * 60
            peak_end = day * 24 * 60 + 9 * 60

            if peak_start < n_samples:
                peak_end = min(peak_end, n_samples)
                states[peak_start:peak_end] = ['high_load'] * (peak_end - peak_start)

                df.loc[peak_start:peak_end-1, 'duty_cycle'] *= 1.15
                df.loc[peak_start:peak_end-1, 'flow_rate'] *= 1.20
                df.loc[peak_start:peak_end-1, 'current'] *= 1.18
                df.loc[peak_start:peak_end-1, 'temperature'] += 5
                df.loc[peak_start:peak_end-1, 'vibration'] *= 1.10

                events.append({
                    'timestamp': df.iloc[peak_start]['timestamp'],
                    'type': 'high_load',
                    'duration_min': peak_end - peak_start
                })

        # Maintenance windows (weekly, late night)
        for week in range(n_samples // (7 * 24 * 60) + 1):
            maint_start = week * 7 * 24 * 60 + 6 * 24 * 60 + 2 * 60  # Sunday 2 AM
            maint_duration = 120  # 2 hours

            if maint_start < n_samples:
                maint_end = min(maint_start + maint_duration, n_samples)
                states[maint_start:maint_end] = ['maintenance'] * (maint_end - maint_start)

                df.loc[maint_start:maint_end-1, 'duty_cycle'] *= 0.5
                df.loc[maint_start:maint_end-1, 'flow_rate'] *= 0.4
                df.loc[maint_start:maint_end-1, 'current'] *= 0.5

                events.append({
                    'timestamp': df.iloc[maint_start]['timestamp'],
                    'type': 'maintenance',
                    'duration_min': maint_duration
                })

        df['operational_state'] = states

        return df, events

    def inject_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_rate: float = 0.02
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Inject realistic anomaly scenarios.

        Anomaly types:
        1. Cavitation: low pressure, high vibration
        2. Bearing wear: increasing vibration, temperature
        3. Seal leak: pressure drop, flow mismatch
        4. Electrical fault: current spikes, temperature rise
        5. Partial blockage: pressure increase, flow decrease
        """
        n_samples = len(df)
        labels = np.zeros(n_samples, dtype=int)
        anomaly_types = ['normal'] * n_samples
        anomaly_events = []

        n_anomalies = int(n_samples * anomaly_rate)

        anomaly_scenarios = [
            self._cavitation_anomaly,
            self._bearing_wear_anomaly,
            self._seal_leak_anomaly,
            self._electrical_fault_anomaly,
            self._blockage_anomaly
        ]

        for _ in range(n_anomalies):
            # Random scenario
            scenario = np.random.choice(anomaly_scenarios)

            # Random start time (avoid first/last 10%)
            start_idx = np.random.randint(
                int(0.1 * n_samples),
                int(0.9 * n_samples)
            )

            # Random duration (30 min to 4 hours)
            duration = np.random.randint(30, 240)
            end_idx = min(start_idx + duration, n_samples)

            # Apply anomaly
            anomaly_name = scenario(df, start_idx, end_idx)

            labels[start_idx:end_idx] = 1
            anomaly_types[start_idx:end_idx] = [anomaly_name] * (end_idx - start_idx)

            anomaly_events.append({
                'timestamp': df.iloc[start_idx]['timestamp'],
                'type': anomaly_name,
                'duration_min': end_idx - start_idx,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

        df['anomaly_label'] = labels
        df['anomaly_type'] = anomaly_types

        return df, anomaly_events

    def _cavitation_anomaly(
        self,
        df: pd.DataFrame,
        start: int,
        end: int
    ) -> str:
        """Simulate cavitation: low pressure, high vibration."""
        df.loc[start:end-1, 'pressure'] *= 0.6
        df.loc[start:end-1, 'vibration'] *= 2.5
        df.loc[start:end-1, 'flow_rate'] *= 0.85
        df.loc[start:end-1, 'temperature'] += 8
        return 'cavitation'

    def _bearing_wear_anomaly(
        self,
        df: pd.DataFrame,
        start: int,
        end: int
    ) -> str:
        """Simulate bearing wear: progressive vibration and temperature increase."""
        duration = end - start
        progression = np.linspace(1.0, 2.8, duration)

        df.loc[start:end-1, 'vibration'] *= progression
        df.loc[start:end-1, 'temperature'] += progression * 10
        df.loc[start:end-1, 'current'] *= (1 + 0.15 * progression / 2.8)
        return 'bearing_wear'

    def _seal_leak_anomaly(
        self,
        df: pd.DataFrame,
        start: int,
        end: int
    ) -> str:
        """Simulate seal leak: pressure drop with flow inconsistency."""
        df.loc[start:end-1, 'pressure'] *= 0.75
        df.loc[start:end-1, 'flow_rate'] *= 0.80
        df.loc[start:end-1, 'duty_cycle'] *= 1.10  # Compensating
        df.loc[start:end-1, 'current'] *= 1.12
        return 'seal_leak'

    def _electrical_fault_anomaly(
        self,
        df: pd.DataFrame,
        start: int,
        end: int
    ) -> str:
        """Simulate electrical fault: current spikes and temperature."""
        df.loc[start:end-1, 'current'] *= 1.35
        df.loc[start:end-1, 'temperature'] += 15
        df.loc[start:end-1, 'vibration'] *= 1.20
        # Add high-frequency noise
        noise = np.random.normal(0, 5, end - start)
        df.loc[start:end-1, 'current'] += noise
        return 'electrical_fault'

    def _blockage_anomaly(
        self,
        df: pd.DataFrame,
        start: int,
        end: int
    ) -> str:
        """Simulate partial blockage: pressure increase, flow decrease."""
        df.loc[start:end-1, 'pressure'] *= 1.40
        df.loc[start:end-1, 'flow_rate'] *= 0.65
        df.loc[start:end-1, 'current'] *= 1.25
        df.loc[start:end-1, 'temperature'] += 10
        df.loc[start:end-1, 'vibration'] *= 1.30
        return 'partial_blockage'

    def generate_dataset(
        self,
        duration_days: int = 30,
        sampling_rate: int = 60,
        anomaly_rate: float = 0.02
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate complete dataset with metadata.

        Args:
            duration_days: Days of data to generate
            sampling_rate: Seconds between samples
            anomaly_rate: Fraction of samples with anomalies

        Returns:
            DataFrame with all sensor data and labels
            Dictionary with metadata and events
        """
        n_samples = duration_days * 24 * 60 * 60 // sampling_rate

        print(f"Generating {n_samples:,} samples ({duration_days} days)...")

        # Generate base signal
        df = self.generate_base_signal(n_samples, sampling_rate)
        print("Base signal generated")

        # Add operational states
        df, operational_events = self.inject_operational_states(df)
        print(f"Injected {len(operational_events)} operational events")

        # Inject anomalies
        df, anomaly_events = self.inject_anomalies(df, anomaly_rate)
        print(f"Injected {len(anomaly_events)} anomalies")

        metadata = {
            'generation_date': datetime.now().isoformat(),
            'duration_days': duration_days,
            'sampling_rate_sec': sampling_rate,
            'n_samples': n_samples,
            'anomaly_rate': anomaly_rate,
            'n_anomalies': len(anomaly_events),
            'operational_events': operational_events,
            'anomaly_events': anomaly_events,
            'sensor_parameters': self.normal_params
        }

        return df, metadata

def main():
    """Generate and save the industrial IoT dataset."""

    simulator = IndustrialPumpSimulator(seed=42)

    # Generate 30 days of data (1-minute sampling)
    df, metadata = simulator.generate_dataset(
        duration_days=30,
        sampling_rate=60,
        anomaly_rate=0.02
    )

    # Save data
    import os
    output_path = '../../data/raw/'
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(output_path + 'sensor_data.csv', index=False)

    with open(output_path + 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nAnomaly distribution:")
    print(df['anomaly_type'].value_counts())
    print(f"\nOperational state distribution:")
    print(df['operational_state'].value_counts())
    print(f"\nSensor statistics (normal operation):")
    normal_data = df[df['anomaly_label'] == 0]
    print(normal_data[['temperature', 'vibration', 'pressure',
                       'flow_rate', 'current', 'duty_cycle']].describe())

    print(f"\nData saved to: {output_path}")

if __name__ == '__main__':
    main()
