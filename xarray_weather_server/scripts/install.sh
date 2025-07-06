#!/bin/bash

# XArray Weather Server Installation Script
echo "Installing XArray Weather Server..."

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.8+ is required, found Python $python_version"
    exit 1
fi

echo "✓ Python $python_version found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "Installing xarray_weather_server in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

echo "✓ Installation completed!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  xarray-weather-server"
echo ""
echo "Or run the example:"
echo "  python examples/basic_usage.py" 