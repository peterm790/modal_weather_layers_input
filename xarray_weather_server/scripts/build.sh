#!/bin/bash

# XArray Weather Server Build Script
echo "Building XArray Weather Server package..."

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Create virtual environment for building if it doesn't exist
if [ ! -d "build_venv" ]; then
    echo "Creating build virtual environment..."
    python3 -m venv build_venv
fi

# Activate build environment
source build_venv/bin/activate

# Install build dependencies
echo "Installing build dependencies..."
pip install --upgrade pip build twine

# Build the package
echo "Building package..."
python -m build

echo "✓ Package built successfully!"
echo ""
echo "Generated files:"
ls -la dist/

echo ""
echo "To install locally:"
echo "  pip install dist/xarray_weather_server-1.0.0-py3-none-any.whl"
echo ""
echo "To upload to PyPI (after setting up credentials):"
echo "  twine upload dist/*" 