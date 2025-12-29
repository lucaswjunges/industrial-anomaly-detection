#!/bin/bash
# Run pytest with coverage reporting
# Author: Lucas William Junges
# Date: December 2024

set -e  # Exit on error

echo "=========================================="
echo "Running Test Suite with Coverage"
echo "=========================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing test dependencies..."
    pip install pytest pytest-cov coverage
fi

# Run tests with coverage
echo "🧪 Running unit tests..."
python -m pytest tests/ \
    -v \
    --cov=src \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=term:skip-covered \
    --tb=short

echo ""
echo "=========================================="
echo "✅ Test Suite Complete!"
echo "=========================================="
echo ""
echo "📊 Coverage report generated:"
echo "   - Terminal: See above"
echo "   - HTML: htmlcov/index.html"
echo ""
echo "To view HTML report:"
echo "   firefox htmlcov/index.html"
echo "   # or"
echo "   python -m http.server 8000 -d htmlcov"
echo ""
