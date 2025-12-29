"""
Comprehensive Evaluation Framework for Anomaly Detection
Includes: precision/recall, ROC, cost analysis, operational KPIs
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple
from datetime import timedelta

class AnomalyEvaluator:
    """
    Comprehensive evaluation for anomaly detection systems.

    Metrics:
    - Standard ML metrics (precision, recall, F1, ROC-AUC)
    - Time-to-detection
    - False positive cost analysis
    - Operational reliability KPIs
    """

    def __init__(self):
        self.results = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Comprehensive evaluation of anomaly detector.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            model_name: Name of the model

        Returns:
            Dictionary with all metrics
        """
        results = {
            'model': model_name,
            'classification': self._classification_metrics(y_true, y_pred),
            'curve_metrics': self._curve_metrics(y_true, y_scores),
            'confusion_matrix': self._confusion_matrix(y_true, y_pred)
        }

        self.results[model_name] = results
        return results

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute classification metrics."""
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

    def _curve_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict:
        """Compute ROC and PR curve metrics."""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        return {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'roc_curve': {
                'fpr': fpr.tolist()[:100],  # Subsample for JSON
                'tpr': tpr.tolist()[:100],
                'thresholds': roc_thresholds.tolist()[:100]
            },
            'pr_curve': {
                'precision': precision.tolist()[:100],
                'recall': recall.tolist()[:100],
                'thresholds': pr_thresholds.tolist()[:100]
            }
        }

    def _confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Compute confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        return {
            'matrix': cm.tolist(),
            'normalized': (cm / cm.sum(axis=1, keepdims=True)).tolist()
        }

    def compute_operational_kpis(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        cost_params: Dict[str, float]
    ) -> Dict:
        """
        Compute operational reliability KPIs.

        Args:
            df: DataFrame with timestamps and labels
            predictions: Binary predictions
            cost_params: Cost parameters for financial analysis

        Returns:
            Dictionary with operational KPIs
        """
        y_true = df['anomaly_label'].values

        # Time-to-detection analysis
        ttd_metrics = self._time_to_detection(df, predictions)

        # False alarm analysis
        fp_analysis = self._false_positive_analysis(df, y_true, predictions)

        # Cost analysis
        costs = self._cost_analysis(y_true, predictions, cost_params)

        # Downtime prevention
        downtime = self._downtime_analysis(df, y_true, predictions)

        return {
            'time_to_detection': ttd_metrics,
            'false_positive_analysis': fp_analysis,
            'cost_analysis': costs,
            'downtime_prevention': downtime
        }

    def _time_to_detection(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict:
        """Analyze time-to-detection for anomalies."""
        df = df.copy()
        df['prediction'] = predictions
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        detection_times = []

        # Find anomaly events
        anomaly_starts = []
        in_anomaly = False
        start_idx = 0

        for i, label in enumerate(df['anomaly_label'].values):
            if label == 1 and not in_anomaly:
                in_anomaly = True
                start_idx = i
            elif label == 0 and in_anomaly:
                in_anomaly = False
                anomaly_starts.append((start_idx, i))

        # Calculate detection time for each event
        for start, end in anomaly_starts:
            anomaly_window = df.iloc[start:end]
            detected = anomaly_window[anomaly_window['prediction'] == 1]

            if len(detected) > 0:
                first_detection = detected.iloc[0]
                ttd = (first_detection['timestamp'] - df.iloc[start]['timestamp']).total_seconds() / 60
                detection_times.append(ttd)

        if detection_times:
            return {
                'mean_ttd_minutes': float(np.mean(detection_times)),
                'median_ttd_minutes': float(np.median(detection_times)),
                'max_ttd_minutes': float(np.max(detection_times)),
                'detection_rate': len(detection_times) / len(anomaly_starts),
                'n_events': len(anomaly_starts),
                'n_detected': len(detection_times)
            }
        else:
            return {
                'mean_ttd_minutes': None,
                'median_ttd_minutes': None,
                'max_ttd_minutes': None,
                'detection_rate': 0.0,
                'n_events': len(anomaly_starts),
                'n_detected': 0
            }

    def _false_positive_analysis(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        """Analyze false positive patterns."""
        df = df.copy()
        df['fp'] = (predictions == 1) & (y_true == 0)

        fp_by_state = df.groupby('operational_state')['fp'].sum().to_dict()

        # FP rate per day
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        fp_per_day = df.groupby('date')['fp'].sum()

        return {
            'total_false_positives': int(df['fp'].sum()),
            'fp_rate': float(df['fp'].mean()),
            'fp_by_operational_state': {k: int(v) for k, v in fp_by_state.items()},
            'mean_fp_per_day': float(fp_per_day.mean()),
            'max_fp_per_day': int(fp_per_day.max())
        }

    def _cost_analysis(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        cost_params: Dict[str, float]
    ) -> Dict:
        """
        Financial cost analysis.

        cost_params:
        - false_positive_cost: Cost of investigating false alarm
        - false_negative_cost: Cost of missed anomaly (downtime, damage)
        - true_positive_benefit: Benefit of catching anomaly early
        """
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fp = ((predictions == 1) & (y_true == 0)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()

        fp_cost = fp * cost_params.get('false_positive_cost', 500)
        fn_cost = fn * cost_params.get('false_negative_cost', 10000)
        tp_benefit = tp * cost_params.get('true_positive_benefit', 8000)

        net_value = tp_benefit - fp_cost - fn_cost

        return {
            'false_positive_cost_usd': float(fp_cost),
            'false_negative_cost_usd': float(fn_cost),
            'true_positive_benefit_usd': float(tp_benefit),
            'net_value_usd': float(net_value),
            'roi': float(net_value / (fp_cost + 1)) if fp_cost > 0 else float('inf')
        }

    def _downtime_analysis(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predictions: np.ndarray
    ) -> Dict:
        """Estimate prevented downtime."""
        tp = ((predictions == 1) & (y_true == 1)).sum()
        fn = ((predictions == 0) & (y_true == 1)).sum()

        # Assume each caught anomaly prevents 2 hours downtime
        prevented_hours = tp * 2
        missed_downtime_hours = fn * 2

        # Estimate MTBF (Mean Time Between Failures)
        total_time_hours = len(df) / 60  # 1-min samples
        n_failures = (y_true == 1).sum()

        mtbf = total_time_hours / n_failures if n_failures > 0 else float('inf')

        # Availability calculation
        # Availability = (Total Time - Downtime) / Total Time
        baseline_downtime = n_failures * 2  # 2 hours per failure
        actual_downtime = missed_downtime_hours

        baseline_availability = (total_time_hours - baseline_downtime) / total_time_hours
        improved_availability = (total_time_hours - actual_downtime) / total_time_hours

        return {
            'prevented_downtime_hours': float(prevented_hours),
            'missed_downtime_hours': float(missed_downtime_hours),
            'mtbf_hours': float(mtbf),
            'baseline_availability': float(baseline_availability),
            'improved_availability': float(improved_availability),
            'availability_improvement': float(improved_availability - baseline_availability)
        }

    def compare_models(self) -> pd.DataFrame:
        """Generate comparison table of all evaluated models."""
        comparison_data = []

        for model_name, results in self.results.items():
            row = {
                'Model': model_name,
                'Precision': results['classification']['precision'],
                'Recall': results['classification']['recall'],
                'F1-Score': results['classification']['f1_score'],
                'ROC-AUC': results['curve_metrics']['roc_auc'],
                'PR-AUC': results['curve_metrics']['pr_auc'],
                'FP': results['classification']['false_positives'],
                'FN': results['classification']['false_negatives']
            }
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def save_results(self, path: str):
        """Save all evaluation results."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Evaluation results saved to: {path}")

def generate_evaluation_report(
    evaluator: AnomalyEvaluator,
    output_path: str
):
    """Generate comprehensive evaluation report."""

    report = []
    report.append("="*80)
    report.append("ANOMALY DETECTION EVALUATION REPORT")
    report.append("Atlantic Water Operations Ltd. - Pumping Facility")
    report.append("="*80)
    report.append("")

    # Model comparison
    comparison_df = evaluator.compare_models()
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-"*80)
    report.append(comparison_df.to_string(index=False))
    report.append("")

    # Detailed results per model
    for model_name, results in evaluator.results.items():
        report.append("")
        report.append("="*80)
        report.append(f"DETAILED RESULTS: {model_name.upper()}")
        report.append("="*80)

        # Classification metrics
        clf = results['classification']
        report.append("\nClassification Metrics:")
        report.append(f"  Precision:    {clf['precision']:.4f}")
        report.append(f"  Recall:       {clf['recall']:.4f}")
        report.append(f"  F1-Score:     {clf['f1_score']:.4f}")
        report.append(f"  Specificity:  {clf['specificity']:.4f}")
        report.append(f"  Accuracy:     {clf['accuracy']:.4f}")

        # Confusion matrix
        report.append("\nConfusion Matrix:")
        cm = results['confusion_matrix']['matrix']
        report.append(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
        report.append(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")

        # Curve metrics
        curve = results['curve_metrics']
        report.append("\nCurve Metrics:")
        report.append(f"  ROC-AUC: {curve['roc_auc']:.4f}")
        report.append(f"  PR-AUC:  {curve['pr_auc']:.4f}")

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Evaluation report saved to: {output_path}")

def main():
    """Run comprehensive evaluation."""
    import sys
    import os
    from pathlib import Path

    # Add src directory to path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from preprocessing.preprocessor import IoTPreprocessor
    from models.anomaly_detectors import (
        IsolationForestDetector,
        LOFDetector,
        AutoencoderDetector
    )

    # Create results directory
    os.makedirs('../../data/results', exist_ok=True)

    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    # Load test data
    test_df = pd.read_csv('../../data/processed/test_data.csv')

    # Load preprocessor
    preprocessor = IoTPreprocessor()
    preprocessor.load('../../models/preprocessor.pkl')

    X_test = preprocessor.get_feature_matrix(test_df, feature_type='all')
    y_test = test_df['anomaly_label'].values

    print(f"\nTest samples: {len(X_test):,}")
    print(f"Anomalies: {y_test.sum():,} ({y_test.mean():.2%})")

    # Load models
    iforest = IsolationForestDetector()
    iforest.load('../../models/isolation_forest/model.pkl')

    lof = LOFDetector()
    lof.load('../../models/lof/model.pkl')

    autoencoder = AutoencoderDetector(input_dim=X_test.shape[1])
    autoencoder.load('../../models/autoencoder/model')

    # Evaluate all models
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

        print(f"  Precision: {results['classification']['precision']:.3f}")
        print(f"  Recall:    {results['classification']['recall']:.3f}")
        print(f"  F1-Score:  {results['classification']['f1_score']:.3f}")
        print(f"  ROC-AUC:   {results['curve_metrics']['roc_auc']:.3f}")

    # Operational KPIs
    print("\n" + "="*60)
    print("OPERATIONAL KPI ANALYSIS")
    print("="*60)

    cost_params = {
        'false_positive_cost': 500,      # Investigation cost
        'false_negative_cost': 10000,    # Downtime + damage
        'true_positive_benefit': 8000    # Prevented failure cost
    }

    for name, model in models:
        print(f"\n{name}:")
        predictions = model.predict(X_test)

        kpis = evaluator.compute_operational_kpis(
            test_df,
            predictions,
            cost_params
        )

        # Time to detection
        ttd = kpis['time_to_detection']
        if ttd['mean_ttd_minutes']:
            print(f"  Mean time-to-detection: {ttd['mean_ttd_minutes']:.1f} minutes")
            print(f"  Detection rate: {ttd['detection_rate']:.1%}")

        # Costs
        costs = kpis['cost_analysis']
        print(f"  Net value: ${costs['net_value_usd']:,.0f}")
        print(f"  FP cost: ${costs['false_positive_cost_usd']:,.0f}")
        print(f"  FN cost: ${costs['false_negative_cost_usd']:,.0f}")

        # Downtime
        downtime = kpis['downtime_prevention']
        print(f"  Prevented downtime: {downtime['prevented_downtime_hours']:.1f} hours")
        print(f"  Availability improvement: {downtime['availability_improvement']:.2%}")

    # Save results
    evaluator.save_results('../../data/results/evaluation_results.json')
    generate_evaluation_report(
        evaluator,
        '../../data/results/evaluation_report.txt'
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
