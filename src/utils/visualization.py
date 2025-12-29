"""
Visualization utilities for anomaly detection results
Generates plots for technical report and portfolio
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import List, Dict
import json


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_sensor_profiles(
    df: pd.DataFrame,
    sensor_columns: List[str],
    save_path: str = None
):
    """
    Plot sensor distributions for normal vs anomalous samples.

    Args:
        df: DataFrame with sensor data and anomaly labels
        sensor_columns: List of sensor column names
        save_path: Path to save figure
    """
    n_sensors = len(sensor_columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, sensor in enumerate(sensor_columns):
        ax = axes[idx]

        # Normal samples
        normal_data = df[df['anomaly_label'] == 0][sensor]
        anomaly_data = df[df['anomaly_label'] == 1][sensor]

        ax.hist(normal_data, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        ax.hist(anomaly_data, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)

        ax.set_xlabel(sensor.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Sensor Value Distributions: Normal vs Anomalous Operation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sensor profiles saved to: {save_path}")

    plt.show()


def plot_time_series_with_anomalies(
    df: pd.DataFrame,
    sensor: str = 'vibration',
    start_idx: int = 0,
    duration: int = 1440,
    save_path: str = None
):
    """
    Plot time series with anomaly annotations.

    Args:
        df: DataFrame with timestamp, sensor data, anomaly labels
        sensor: Sensor column to plot
        start_idx: Starting index
        duration: Number of samples to plot
        save_path: Path to save figure
    """
    subset = df.iloc[start_idx:start_idx + duration]

    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot sensor values
    ax.plot(subset['timestamp'], subset[sensor], color='blue', alpha=0.7, linewidth=1)

    # Highlight anomalies
    anomaly_mask = subset['anomaly_label'] == 1
    ax.scatter(
        subset.loc[anomaly_mask, 'timestamp'],
        subset.loc[anomaly_mask, sensor],
        color='red',
        s=50,
        alpha=0.8,
        label='Anomaly',
        zorder=3
    )

    ax.set_xlabel('Timestamp')
    ax.set_ylabel(sensor.replace('_', ' ').title())
    ax.set_title(f'{sensor.title()} Over Time with Anomaly Detection', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrices(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot confusion matrices for all models.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save figure
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, result) in enumerate(results.items()):
        cm = np.array(result['confusion_matrix']['matrix'])

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[idx],
            cbar=False,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )

        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f'{model_name}\n(F1: {result["classification"]["f1_score"]:.3f})')

    plt.suptitle('Confusion Matrices: Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")

    plt.show()


def plot_roc_curves(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot ROC curves for all models.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, result in results.items():
        roc_data = result['curve_metrics']['roc_curve']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        auc_score = result['curve_metrics']['roc_auc']

        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC={auc_score:.3f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves: Model Comparison', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {save_path}")

    plt.show()


def plot_precision_recall_curves(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Plot Precision-Recall curves for all models.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, result in results.items():
        pr_data = result['curve_metrics']['pr_curve']
        precision = pr_data['precision']
        recall = pr_data['recall']
        auc_score = result['curve_metrics']['pr_auc']

        ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AUC={auc_score:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves: Model Comparison', fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curves saved to: {save_path}")

    plt.show()


def plot_model_comparison_bar(
    results: Dict[str, Dict],
    save_path: str = None
):
    """
    Bar chart comparing model performance metrics.

    Args:
        results: Evaluation results dictionary
        save_path: Path to save figure
    """
    models = list(results.keys())
    precision = [results[m]['classification']['precision'] for m in models]
    recall = [results[m]['classification']['recall'] for m in models]
    f1 = [results[m]['classification']['f1_score'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, precision, width, label='Precision', color='#667eea')
    ax.bar(x, recall, width, label='Recall', color='#764ba2')
    ax.bar(x + width, f1, width, label='F1-Score', color='#f093fb')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3, axis='y')

    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison bar chart saved to: {save_path}")

    plt.show()


def generate_all_figures(
    df: pd.DataFrame,
    results_path: str,
    output_dir: str = 'figures'
):
    """
    Generate all figures for technical report and portfolio.

    Args:
        df: Complete dataset DataFrame
        results_path: Path to evaluation results JSON
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load evaluation results
    with open(results_path, 'r') as f:
        results = json.load(f)

    sensor_cols = ['temperature', 'vibration', 'pressure',
                   'flow_rate', 'current', 'duty_cycle']

    print("Generating figures...")

    # 1. Sensor profiles
    plot_sensor_profiles(
        df,
        sensor_cols,
        save_path=f'{output_dir}/sensor_profiles.png'
    )

    # 2. Time series example
    plot_time_series_with_anomalies(
        df,
        sensor='vibration',
        start_idx=10000,
        duration=2880,  # 2 days
        save_path=f'{output_dir}/time_series_anomalies.png'
    )

    # 3. Confusion matrices
    plot_confusion_matrices(
        results,
        save_path=f'{output_dir}/confusion_matrices.png'
    )

    # 4. ROC curves
    plot_roc_curves(
        results,
        save_path=f'{output_dir}/roc_curves.png'
    )

    # 5. PR curves
    plot_precision_recall_curves(
        results,
        save_path=f'{output_dir}/pr_curves.png'
    )

    # 6. Model comparison
    plot_model_comparison_bar(
        results,
        save_path=f'{output_dir}/model_comparison.png'
    )

    print(f"\n✓ All figures generated in: {output_dir}/")


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <data_csv_path> <results_json_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    results_path = sys.argv[2] if len(sys.argv) > 2 else 'data/results/evaluation_results.json'

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    generate_all_figures(df, results_path)
