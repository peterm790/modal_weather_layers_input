"""
xarray_weather_server - FastAPI server for ERA5 wind data

A FastAPI server that fetches ERA5 wind data from Google Cloud and returns it 
as WeatherLayers-compatible PNG files for deck.gl visualization.

Features:
- Historical wind data from 1950-2024
- Global and regional data queries
- WeatherLayers PNG format output
"""

__version__ = "1.0.0"

from .server import app
from .data_fetcher import fetch_wind_data, create_wind_png
from .utils import parse_date_hour

__all__ = [
    "app",
    "fetch_wind_data", 
    "create_wind_png",
    "parse_date_hour",
] 