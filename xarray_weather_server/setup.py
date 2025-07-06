#!/usr/bin/env python3
"""
Setup script for xarray_weather_server package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xarray_weather_server",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="FastAPI server for fetching ERA5 wind data as WeatherLayers-compatible PNGs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/xarray_weather_server",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "xarray>=2023.12.0",
        "numpy>=1.24.3",
        "pillow>=10.1.0",
        "pandas>=2.1.4",
        "zarr>=2.16.1",
        "fsspec>=2023.12.2",
        "gcsfs>=2023.12.2",
    ],
    extras_require={
        "dev": [
            "requests>=2.31.0",
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xarray-weather-server=xarray_weather_server.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "xarray_weather_server": ["*.md", "*.txt"],
    },
) 