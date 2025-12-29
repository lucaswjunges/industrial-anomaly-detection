#!/usr/bin/env python3
"""
End-to-End Pipeline Runner
Industrial IoT Anomaly Detection System

Usage:
    python run_pipeline.py --all                  # Run complete pipeline
    python run_pipeline.py --generate             # Generate data only
    python run_pipeline.py --train                # Train models only
    python run_pipeline.py --evaluate             # Evaluate models only
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_command(cmd: str, description: str):
    """Execute a shell command and report status."""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False


def setup_environment():
    """Check and setup Python environment."""
    print("\n" + "="*70)
    print("ENVIRONMENT SETUP")
    print("="*70)

    # Check Python version
    import sys
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")

    # Check required packages
    required = ['numpy', 'pandas', 'sklearn', 'tensorflow']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} not found")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def generate_data():
    """Generate synthetic IoT sensor data."""
    return run_command(
        "python src/data_generation/iot_simulator.py",
        "Data Generation (30 days of sensor data)"
    )


def preprocess_data():
    """Preprocess and engineer features."""
    return run_command(
        "python src/preprocessing/preprocessor.py",
        "Data Preprocessing & Feature Engineering"
    )


def train_models():
    """Train all anomaly detection models."""
    return run_command(
        "python src/models/anomaly_detectors.py",
        "Model Training (Isolation Forest, LOF, Autoencoder)"
    )


def evaluate_models():
    """Evaluate models and generate reports."""
    return run_command(
        "python src/evaluation/evaluator.py",
        "Model Evaluation & KPI Analysis"
    )


def generate_report():
    """Compile LaTeX technical report."""
    report_path = Path("docs/technical_report.tex")

    if not report_path.exists():
        print("✗ LaTeX report not found")
        return False

    print("\n" + "="*70)
    print("STEP: LaTeX Report Compilation")
    print("="*70)
    print("To compile the technical report:")
    print(f"  cd {report_path.parent}")
    print(f"  pdflatex {report_path.name}")
    print(f"  pdflatex {report_path.name}  # Run twice for references")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Industrial IoT Anomaly Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all          Run complete pipeline
  python run_pipeline.py --generate     Generate data only
  python run_pipeline.py --train        Train models only
  python run_pipeline.py --evaluate     Evaluate models only
        """
    )

    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--generate', action='store_true',
                       help='Generate synthetic data')
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess data')
    parser.add_argument('--train', action='store_true',
                       help='Train models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate models')
    parser.add_argument('--report', action='store_true',
                       help='Show report compilation instructions')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Check environment
    if not setup_environment():
        print("\n✗ Environment check failed. Please install dependencies.")
        sys.exit(1)

    # Track success
    success = True

    # Run requested steps
    if args.all or args.generate:
        success &= generate_data()

    if args.all or args.preprocess:
        success &= preprocess_data()

    if args.all or args.train:
        success &= train_models()

    if args.all or args.evaluate:
        success &= evaluate_models()

    if args.all or args.report:
        success &= generate_report()

    # Final summary
    print("\n" + "="*70)
    if success:
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review results in data/results/")
        print("  2. Check evaluation report: data/results/evaluation_report.txt")
        print("  3. Compile technical report: docs/technical_report.pdf")
        print("  4. View portfolio page: index.html")
    else:
        print("✗ PIPELINE COMPLETED WITH ERRORS")
        print("="*70)
        print("\nPlease check error messages above and retry.")
        sys.exit(1)


if __name__ == '__main__':
    main()
