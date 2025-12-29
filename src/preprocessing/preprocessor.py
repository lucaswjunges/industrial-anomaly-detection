"""
Preprocessing Pipeline for Industrial IoT Data
Handles normalization, feature engineering, and regime-based profiling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import json
import pickle
from typing import Dict, Tuple, Optional

class IoTPreprocessor:
    """
    Preprocessing pipeline for multivariate sensor data.

    Features:
    - Regime-based normalization
    - Statistical profiling per operational state
    - Feature engineering (rolling statistics, derivatives)
    - Missing data handling
    """

    def __init__(self, normalization_method: str = 'robust'):
        """
        Initialize preprocessor.

        Args:
            normalization_method: 'standard', 'robust', or 'minmax'
        """
        self.normalization_method = normalization_method
        self.scalers = {}
        self.feature_stats = {}
        self.sensor_columns = [
            'temperature', 'vibration', 'pressure',
            'flow_rate', 'current', 'duty_cycle'
        ]

    def fit(self, df: pd.DataFrame, regime_column: str = 'operational_state'):
        """
        Fit scalers on training data, per operational regime.

        Args:
            df: Training DataFrame
            regime_column: Column defining operational regimes
        """
        print("Fitting preprocessing pipeline...")

        regimes = df[regime_column].unique()

        for regime in regimes:
            regime_data = df[df[regime_column] == regime][self.sensor_columns]

            if self.normalization_method == 'robust':
                scaler = RobustScaler()
            elif self.normalization_method == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown normalization: {self.normalization_method}")

            scaler.fit(regime_data)
            self.scalers[regime] = scaler

            # Compute statistical profile
            self.feature_stats[regime] = {
                'mean': regime_data.mean().to_dict(),
                'std': regime_data.std().to_dict(),
                'q25': regime_data.quantile(0.25).to_dict(),
                'q75': regime_data.quantile(0.75).to_dict(),
                'min': regime_data.min().to_dict(),
                'max': regime_data.max().to_dict()
            }

            print(f"  Regime '{regime}': {len(regime_data)} samples")

        print("Preprocessing pipeline fitted successfully")

    def transform(
        self,
        df: pd.DataFrame,
        regime_column: str = 'operational_state',
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Transform data using fitted scalers.

        Args:
            df: DataFrame to transform
            regime_column: Column defining operational regimes
            add_features: Whether to add engineered features

        Returns:
            Transformed DataFrame
        """
        df_out = df.copy()

        # Normalize per regime
        normalized_values = np.zeros((len(df), len(self.sensor_columns)))

        for regime, scaler in self.scalers.items():
            mask = df[regime_column] == regime
            if mask.sum() > 0:
                regime_data = df.loc[mask, self.sensor_columns]
                normalized_values[mask] = scaler.transform(regime_data)

        # Replace with normalized values
        for i, col in enumerate(self.sensor_columns):
            df_out[f'{col}_norm'] = normalized_values[:, i]

        # Add engineered features
        if add_features:
            df_out = self._add_engineered_features(df_out)

        return df_out

    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based and statistical features."""

        # Rolling statistics (5-minute window = 5 samples)
        window = 5

        for col in self.sensor_columns:
            norm_col = f'{col}_norm'

            # Rolling mean
            df[f'{col}_rolling_mean'] = df[norm_col].rolling(
                window=window, min_periods=1
            ).mean()

            # Rolling std
            df[f'{col}_rolling_std'] = df[norm_col].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            # Rate of change
            df[f'{col}_derivative'] = df[norm_col].diff().fillna(0)

        # Cross-sensor features
        df['temp_vibration_ratio'] = (
            df['temperature_norm'] / (df['vibration_norm'] + 1e-6)
        )
        df['pressure_flow_ratio'] = (
            df['pressure_norm'] / (df['flow_rate_norm'] + 1e-6)
        )
        df['power_proxy'] = df['current_norm'] * df['duty_cycle_norm']

        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        feature_type: str = 'normalized'
    ) -> np.ndarray:
        """
        Extract feature matrix for modeling.

        Args:
            df: Transformed DataFrame
            feature_type: 'normalized', 'engineered', or 'all'

        Returns:
            Feature matrix as numpy array
        """
        if feature_type == 'normalized':
            cols = [f'{c}_norm' for c in self.sensor_columns]
        elif feature_type == 'engineered':
            cols = [c for c in df.columns if (
                '_rolling_' in c or '_derivative' in c or '_ratio' in c or '_proxy' in c
            )]
        elif feature_type == 'all':
            cols = [f'{c}_norm' for c in self.sensor_columns]
            cols += [c for c in df.columns if (
                '_rolling_' in c or '_derivative' in c or '_ratio' in c or '_proxy' in c
            )]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        return df[cols].values

    def detect_regime_violations(
        self,
        df: pd.DataFrame,
        regime_column: str = 'operational_state',
        sigma_threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect statistical violations based on regime profiles.

        Args:
            df: DataFrame with sensor data
            regime_column: Column defining regimes
            sigma_threshold: Number of standard deviations for violation

        Returns:
            Boolean series indicating violations
        """
        violations = pd.Series(False, index=df.index)

        for regime, stats in self.feature_stats.items():
            mask = df[regime_column] == regime

            if mask.sum() == 0:
                continue

            regime_data = df.loc[mask, self.sensor_columns]

            for col in self.sensor_columns:
                mean = stats['mean'][col]
                std = stats['std'][col]

                lower = mean - sigma_threshold * std
                upper = mean + sigma_threshold * std

                col_violations = (
                    (regime_data[col] < lower) | (regime_data[col] > upper)
                )
                violations.loc[mask] |= col_violations.values

        return violations

    def save(self, path: str):
        """Save preprocessor state."""
        state = {
            'normalization_method': self.normalization_method,
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'sensor_columns': self.sensor_columns
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Preprocessor saved to: {path}")

    def load(self, path: str):
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.normalization_method = state['normalization_method']
        self.scalers = state['scalers']
        self.feature_stats = state['feature_stats']
        self.sensor_columns = state['sensor_columns']

        print(f"Preprocessor loaded from: {path}")

def analyze_data_quality(df: pd.DataFrame, sensor_columns: list) -> Dict:
    """
    Analyze data quality metrics.

    Args:
        df: Raw sensor DataFrame
        sensor_columns: List of sensor column names

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'n_samples': len(df),
        'missing_values': {},
        'outliers_iqr': {},
        'value_ranges': {},
        'temporal_gaps': []
    }

    # Missing values
    for col in sensor_columns:
        quality_report['missing_values'][col] = int(df[col].isna().sum())

    # IQR outliers
    for col in sensor_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        quality_report['outliers_iqr'][col] = int(outliers)

    # Value ranges
    for col in sensor_columns:
        quality_report['value_ranges'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }

    # Temporal gaps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diffs = df['timestamp'].diff().dt.total_seconds()
        expected_interval = time_diffs.mode()[0]

        gaps = time_diffs[time_diffs > expected_interval * 2]
        quality_report['temporal_gaps'] = [
            {
                'timestamp': str(df.iloc[i]['timestamp']),
                'gap_seconds': float(time_diffs.iloc[i])
            }
            for i in gaps.index[:10]  # First 10 gaps
        ]

    return quality_report

def main():
    """Demonstration of preprocessing pipeline."""
    import sys
    sys.path.append('../data_generation')

    # Load raw data
    df = pd.read_csv('../../data/raw/sensor_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)

    sensor_cols = ['temperature', 'vibration', 'pressure',
                   'flow_rate', 'current', 'duty_cycle']

    quality_report = analyze_data_quality(df, sensor_cols)

    print(f"\nTotal samples: {quality_report['n_samples']:,}")
    print(f"\nMissing values:")
    for k, v in quality_report['missing_values'].items():
        print(f"  {k}: {v}")

    print(f"\nIQR outliers:")
    for k, v in quality_report['outliers_iqr'].items():
        print(f"  {k}: {v}")

    # Save quality report
    import os
    os.makedirs('../../data/processed', exist_ok=True)
    with open('../../data/processed/quality_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2)

    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)

    # Split train/test (80/20)
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTrain samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")

    # Fit preprocessor on training data (normal samples only)
    preprocessor = IoTPreprocessor(normalization_method='robust')

    train_normal = train_df[train_df['anomaly_label'] == 0]
    preprocessor.fit(train_normal, regime_column='operational_state')

    # Transform both sets
    train_processed = preprocessor.transform(train_df, add_features=True)
    test_processed = preprocessor.transform(test_df, add_features=True)

    # Save processed data
    train_processed.to_csv('../../data/processed/train_data.csv', index=False)
    test_processed.to_csv('../../data/processed/test_data.csv', index=False)

    # Save preprocessor
    os.makedirs('../../models', exist_ok=True)
    preprocessor.save('../../models/preprocessor.pkl')

    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)

    feature_matrix = preprocessor.get_feature_matrix(
        train_processed,
        feature_type='all'
    )
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Features: {feature_matrix.shape[1]}")

    print("\n" + "="*60)
    print("REGIME VIOLATION DETECTION")
    print("="*60)

    violations = preprocessor.detect_regime_violations(
        test_df,
        sigma_threshold=3.0
    )
    print(f"\nStatistical violations detected: {violations.sum()}")
    print(f"Violation rate: {violations.mean():.2%}")

    # Overlap with true anomalies
    if 'anomaly_label' in test_df.columns:
        true_anomalies = test_df['anomaly_label'] == 1
        overlap = (violations & true_anomalies).sum()
        print(f"Overlap with true anomalies: {overlap} / {true_anomalies.sum()}")
        print(f"Precision: {overlap / violations.sum():.2%}")
        print(f"Recall: {overlap / true_anomalies.sum():.2%}")

    print("\nProcessing complete!")

if __name__ == '__main__':
    main()
