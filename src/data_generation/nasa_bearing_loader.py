#!/usr/bin/env python3
"""
NASA IMS Bearing Dataset Loader
=================================

Loads and preprocesses real bearing vibration data from NASA's IMS dataset.
Dataset: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

This replaces synthetic data with REAL industrial sensor data from run-to-failure experiments.

Dataset Details:
- 4 bearings run simultaneously until failure
- Vibration sensors (accelerometers) on each bearing
- Sampling rate: 20 kHz resampled to features
- Multiple test runs with different failure modes
- Real degradation patterns over time

Author: Lucas William Junges
Date: December 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
from scipy import signal
from scipy.stats import kurtosis, skew


class NASABearingLoader:
    """
    Loads and preprocesses NASA IMS Bearing Dataset.

    Converts raw vibration time-series into engineered features suitable
    for anomaly detection models.
    """

    def __init__(self, data_path: str = None):
        """
        Initialize loader.

        Args:
            data_path: Path to NASA bearing dataset directory
                      If None, uses '../data/raw/nasa_bearing/'
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'nasa_bearing'
        else:
            self.data_path = Path(data_path)

        self.sampling_rate = 20000  # Original sampling rate: 20 kHz
        self.resampled_rate = 60    # Resample to 1 sample/min (matching IoT simulator)

    def download_instructions(self) -> str:
        """
        Returns instructions for downloading NASA bearing dataset.

        Returns:
            String with download instructions
        """
        instructions = """
        📥 NASA IMS Bearing Dataset Download Instructions
        ==================================================

        1. Visit NASA Prognostics Center of Excellence:
           https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

        2. Download "IMS Bearing Dataset" (also called "Bearing Data Set")
           Direct link: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#bearing

        3. Extract to: {data_path}

        4. Expected structure:
           {data_path}/
           ├── 1st_test/
           │   ├── 2003.10.22.12.06.24  (bearing 1 data)
           │   ├── 2003.10.22.12.16.24  (bearing 2 data)
           │   └── ...
           ├── 2nd_test/
           └── 3rd_test/

        5. Dataset size: ~1.5 GB (compressed), ~2.8 GB (extracted)

        Alternative: Use wget/curl commands
        ------------------------------------
        cd {data_path}
        wget https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.zip
        unzip IMS.zip

        Or run: python -c "from src.data_generation.nasa_bearing_loader import NASABearingLoader; NASABearingLoader().auto_download()"
        """.format(data_path=self.data_path)

        return instructions

    def auto_download(self) -> bool:
        """
        Attempts to automatically download NASA bearing dataset.

        Returns:
            True if successful, False otherwise
        """
        import urllib.request
        import zipfile

        url = "https://ti.arc.nasa.gov/m/project/prognostic-repository/IMS.zip"
        zip_path = self.data_path.parent / "IMS.zip"

        try:
            print(f"📥 Downloading NASA Bearing Dataset from {url}")
            print(f"   This may take a few minutes (~1.5 GB)...")

            # Create directory
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\r   Progress: {percent}% ({count * block_size / 1e6:.1f} / {total_size / 1e6:.1f} MB)", end='')

            urllib.request.urlretrieve(url, zip_path, reporthook)
            print("\n✅ Download complete!")

            # Extract
            print(f"📦 Extracting to {self.data_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)

            print("✅ Extraction complete!")
            zip_path.unlink()  # Remove zip file

            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n📖 Please download manually:")
            print(self.download_instructions())
            return False

    def check_data_available(self) -> bool:
        """
        Checks if NASA bearing data is available locally.

        Returns:
            True if data exists, False otherwise
        """
        test_dirs = ['1st_test', '2nd_test', '3rd_test']

        if not self.data_path.exists():
            return False

        for test_dir in test_dirs:
            if (self.data_path / test_dir).exists():
                return True

        return False

    def load_test_run(self, test_number: int = 1, bearing_number: int = 1) -> pd.DataFrame:
        """
        Loads a single bearing's run-to-failure data.

        Args:
            test_number: Test run number (1, 2, or 3)
            bearing_number: Bearing number (1-4)

        Returns:
            DataFrame with columns:
            - timestamp: datetime
            - vibration_x: horizontal acceleration (g)
            - vibration_y: vertical acceleration (g)
            - rms: RMS vibration
            - kurtosis: statistical kurtosis
            - is_anomaly: 1 if failure period, 0 otherwise
            - operational_state: synthetic state for compatibility
        """
        test_dir = self.data_path / f'{test_number}th_test' if test_number > 1 else self.data_path / '1st_test'

        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}\n\n{self.download_instructions()}")

        # Get all measurement files sorted by timestamp
        files = sorted(test_dir.glob('*.txt'))

        if len(files) == 0:
            raise FileNotFoundError(f"No data files found in {test_dir}")

        print(f"📊 Loading NASA Bearing Test {test_number}, Bearing {bearing_number}")
        print(f"   Found {len(files)} measurement files")

        features_list = []

        for i, file_path in enumerate(files):
            if i % 100 == 0:
                print(f"   Processing file {i+1}/{len(files)}...", end='\r')

            # Load raw vibration data
            data = np.loadtxt(file_path, delimiter='\t')

            # Extract timestamp from filename (format: 2003.10.22.12.06.24)
            timestamp_str = file_path.stem
            timestamp = pd.to_datetime(timestamp_str, format='%Y.%m.%d.%H.%M.%S')

            # Each bearing has 2 channels (horizontal and vertical)
            bearing_idx = (bearing_number - 1) * 2
            vibration_x = data[:, bearing_idx] if bearing_idx < data.shape[1] else data[:, 0]
            vibration_y = data[:, bearing_idx + 1] if bearing_idx + 1 < data.shape[1] else data[:, 1]

            # Extract features from raw vibration
            features = self._extract_vibration_features(vibration_x, vibration_y, timestamp)
            features_list.append(features)

        print(f"\n✅ Loaded {len(features_list)} measurements")

        # Create DataFrame
        df = pd.DataFrame(features_list)

        # Label anomalies (last 10% of operation = degradation/failure)
        failure_threshold = int(len(df) * 0.90)
        df['is_anomaly'] = 0
        df.loc[failure_threshold:, 'is_anomaly'] = 1

        # Add synthetic operational states for compatibility with existing pipeline
        df['operational_state'] = self._synthesize_operational_states(df)

        print(f"📈 Dataset summary:")
        print(f"   - Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} hours")
        print(f"   - Normal samples: {(df['is_anomaly'] == 0).sum()} ({(df['is_anomaly'] == 0).sum() / len(df) * 100:.1f}%)")
        print(f"   - Anomaly samples: {(df['is_anomaly'] == 1).sum()} ({(df['is_anomaly'] == 1).sum() / len(df) * 100:.1f}%)")

        return df

    def _extract_vibration_features(self, vibration_x: np.ndarray, vibration_y: np.ndarray,
                                   timestamp: pd.Timestamp) -> Dict:
        """
        Extracts features from raw vibration time-series.

        Args:
            vibration_x: Horizontal vibration signal
            vibration_y: Vertical vibration signal
            timestamp: Measurement timestamp

        Returns:
            Dictionary of features matching IoT simulator format
        """
        features = {
            'timestamp': timestamp,

            # Time-domain features
            'vibration': np.sqrt(vibration_x.mean()**2 + vibration_y.mean()**2),  # Overall vibration
            'vibration_rms': np.sqrt(np.mean(vibration_x**2) + np.mean(vibration_y**2)),
            'vibration_peak': max(np.max(np.abs(vibration_x)), np.max(np.abs(vibration_y))),
            'vibration_kurtosis': (kurtosis(vibration_x) + kurtosis(vibration_y)) / 2,
            'vibration_skewness': (skew(vibration_x) + skew(vibration_y)) / 2,

            # Frequency-domain features
            'vibration_spectral_energy': self._spectral_energy(vibration_x) + self._spectral_energy(vibration_y),

            # Synthetic features for compatibility with existing pipeline
            # (bearings don't have these sensors, but we create proxies)
            'temperature': 20 + features_dict['vibration_rms'] * 30,  # Vibration correlates with temperature
            'pressure': 3.5,  # Constant (no pressure sensor in bearing test)
            'flow_rate': 100.0,  # Constant (no flow sensor)
            'current': 10 + features_dict['vibration_rms'] * 5,  # Current increases with vibration/load
            'duty_cycle': 0.75,  # Constant operation
        }

        # Fix self-reference
        features['temperature'] = 20 + features['vibration_rms'] * 30
        features['current'] = 10 + features['vibration_rms'] * 5

        return features

    def _spectral_energy(self, signal_data: np.ndarray) -> float:
        """
        Calculates spectral energy of signal.

        Args:
            signal_data: Time-series signal

        Returns:
            Spectral energy
        """
        # Compute FFT
        fft = np.fft.fft(signal_data)
        psd = np.abs(fft)**2

        # Return total spectral energy
        return np.sum(psd)

    def _synthesize_operational_states(self, df: pd.DataFrame) -> pd.Series:
        """
        Synthesizes operational states for compatibility with existing pipeline.

        NASA bearing data runs continuously, but we add synthetic states
        to match the IoT simulator's operational regime structure.

        Args:
            df: DataFrame with vibration features

        Returns:
            Series with operational states
        """
        n = len(df)
        states = ['normal'] * n

        # Add some state transitions for realism
        # Every ~500 samples, add a brief "high_load" period
        for i in range(0, n, 500):
            end_idx = min(i + 50, n)
            states[i:end_idx] = ['high_load'] * (end_idx - i)

        # Add startup at beginning
        states[:10] = ['startup'] * 10

        # Add maintenance periods occasionally
        for i in range(1000, n, 2000):
            end_idx = min(i + 20, n)
            states[i:end_idx] = ['maintenance'] * (end_idx - i)

        return pd.Series(states, index=df.index)

    def prepare_for_training(self, test_number: int = 1, bearing_number: int = 1,
                            train_split: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads NASA bearing data and splits into train/test.

        Args:
            test_number: NASA test run (1, 2, or 3)
            bearing_number: Bearing number (1-4)
            train_split: Fraction of data for training

        Returns:
            (train_df, test_df) tuple
        """
        # Load data
        df = self.load_test_run(test_number, bearing_number)

        # Split chronologically (important for time-series!)
        split_idx = int(len(df) * train_split)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\n📊 Train/Test Split:")
        print(f"   Train: {len(train_df)} samples ({train_df['is_anomaly'].sum()} anomalies)")
        print(f"   Test:  {len(test_df)} samples ({test_df['is_anomaly'].sum()} anomalies)")

        # Save to disk
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_dir / 'nasa_train_data.csv', index=False)
        test_df.to_csv(output_dir / 'nasa_test_data.csv', index=False)

        print(f"\n💾 Saved to {output_dir}/nasa_*_data.csv")

        return train_df, test_df


def main():
    """
    Example usage: Load NASA bearing data and prepare for training.
    """
    print("=" * 70)
    print("NASA IMS Bearing Dataset Loader")
    print("=" * 70)

    loader = NASABearingLoader()

    # Check if data is available
    if not loader.check_data_available():
        print("\n❌ NASA bearing data not found!")
        print(loader.download_instructions())

        # Offer to auto-download
        response = input("\nAttempt automatic download? (y/n): ")
        if response.lower() == 'y':
            success = loader.auto_download()
            if not success:
                return
        else:
            return

    # Load and prepare data
    print("\n" + "=" * 70)
    print("Loading Test 1, Bearing 1 (outer race failure)")
    print("=" * 70)

    train_df, test_df = loader.prepare_for_training(test_number=1, bearing_number=1)

    print("\n" + "=" * 70)
    print("✅ NASA Bearing Dataset Ready!")
    print("=" * 70)
    print("\n📁 Data files created:")
    print("   - data/processed/nasa_train_data.csv")
    print("   - data/processed/nasa_test_data.csv")
    print("\n🚀 Next step: Run train_simple.py to train models on REAL data")


if __name__ == '__main__':
    main()
