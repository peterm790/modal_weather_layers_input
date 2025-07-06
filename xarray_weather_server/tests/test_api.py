"""
Test suite for xarray_weather_server API endpoints.
"""

import pytest
import requests
import json
from datetime import datetime


class TestWeatherAPI:
    """Test class for weather API endpoints."""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for the API."""
        return "http://localhost:8000"
    
    def test_root_endpoint(self, base_url):
        """Test the root endpoint."""
        response = requests.get(f"{base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_endpoint(self, base_url):
        """Test the health check endpoint."""
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_wind_data_endpoint(self, base_url):
        """Test fetching wind data for a specific date."""
        test_date = "24011512"  # Jan 15, 2024 12:00 UTC
        response = requests.get(f"{base_url}/wind/{test_date}")
        
        if response.status_code == 200:
            assert response.headers.get('content-type') == 'image/png'
            assert len(response.content) > 0
        else:
            # Server might not be running or data unavailable
            pytest.skip("Wind data endpoint not available")
    
    def test_regional_wind_data(self, base_url):
        """Test fetching regional wind data."""
        test_date = "24011512"
        params = {
            'lat_min': 30,
            'lat_max': 45,
            'lon_min': -10,
            'lon_max': 40
        }
        response = requests.get(f"{base_url}/wind/{test_date}", params=params)
        
        if response.status_code == 200:
            assert response.headers.get('content-type') == 'image/png'
            assert len(response.content) > 0
        else:
            pytest.skip("Regional wind data endpoint not available")
    
    def test_invalid_date_format(self, base_url):
        """Test invalid date format handling."""
        invalid_date = "invalid"
        response = requests.get(f"{base_url}/wind/{invalid_date}")
        assert response.status_code == 400
    
    def test_invalid_region_bounds(self, base_url):
        """Test invalid region bounds handling."""
        test_date = "24011512"
        params = {
            'lat_min': 100,  # Invalid latitude
            'lat_max': 45,
            'lon_min': -10,
            'lon_max': 40
        }
        response = requests.get(f"{base_url}/wind/{test_date}", params=params)
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__]) 