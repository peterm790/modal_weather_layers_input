"""
Unit tests for utility functions.
"""

import pytest
from datetime import datetime
import numpy as np

from xarray_weather_server.utils import parse_date_hour, validate_region_bounds, format_wind_info


class TestUtils:
    """Test class for utility functions."""
    
    def test_parse_date_hour_valid(self):
        """Test parsing valid date strings."""
        # Test 2024 date
        result = parse_date_hour("24011512")
        expected = datetime(2024, 1, 15, 12)
        assert result == expected
        
        # Test 1999 date
        result = parse_date_hour("99123100")
        expected = datetime(1999, 12, 31, 0)
        assert result == expected
        
        # Test 2000 date
        result = parse_date_hour("00010100")
        expected = datetime(2000, 1, 1, 0)
        assert result == expected
    
    def test_parse_date_hour_invalid(self):
        """Test parsing invalid date strings."""
        with pytest.raises(ValueError):
            parse_date_hour("invalid")
        
        with pytest.raises(ValueError):
            parse_date_hour("1234567")  # Too short
        
        with pytest.raises(ValueError):
            parse_date_hour("123456789")  # Too long
        
        with pytest.raises(ValueError):
            parse_date_hour("24131512")  # Invalid month
    
    def test_validate_region_bounds_valid(self):
        """Test valid region bounds."""
        assert validate_region_bounds(-90, 90, -180, 180) is True
        assert validate_region_bounds(30, 45, -10, 40) is True
        assert validate_region_bounds(0, 0, 0, 0) is True
    
    def test_validate_region_bounds_invalid(self):
        """Test invalid region bounds."""
        # Invalid latitude
        assert validate_region_bounds(-100, 90, -180, 180) is False
        assert validate_region_bounds(-90, 100, -180, 180) is False
        assert validate_region_bounds(45, 30, -180, 180) is False  # lat_min > lat_max
        
        # Invalid longitude
        assert validate_region_bounds(-90, 90, -200, 180) is False
        assert validate_region_bounds(-90, 90, -180, 200) is False
        assert validate_region_bounds(-90, 90, 40, -10) is False  # lon_min > lon_max
    
    def test_format_wind_info(self):
        """Test wind information formatting."""
        # Create sample wind data
        u_wind = np.array([[1.0, 2.0], [3.0, 4.0]])
        v_wind = np.array([[0.5, 1.5], [2.5, 3.5]])
        
        result = format_wind_info(u_wind, v_wind)
        
        assert "shape" in result
        assert "u_range" in result
        assert "v_range" in result
        assert "u_mean" in result
        assert "v_mean" in result
        assert "wind_speed_max" in result
        
        assert result["shape"] == (2, 2)
        assert result["u_range"] == [1.0, 4.0]
        assert result["v_range"] == [0.5, 3.5] 