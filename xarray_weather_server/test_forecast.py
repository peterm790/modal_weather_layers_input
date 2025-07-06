#!/usr/bin/env python3
"""
Test script for the new forecast functionality.
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add the module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'xarray_weather_server'))

from xarray_weather_server.data_fetcher import (
    get_latest_init_time, 
    get_forecast_dataset_info,
    fetch_forecast_wind_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_latest_init_times():
    """Test getting latest init times for forecast datasets."""
    print("Testing latest init times...")
    
    try:
        # Test GFS
        gfs_latest = get_latest_init_time('gfs')
        print(f"GFS latest init time: {gfs_latest}")
        
        # Test GEFS
        gefs_latest = get_latest_init_time('gefs')
        print(f"GEFS latest init time: {gefs_latest}")
        
    except Exception as e:
        print(f"Error getting latest init times: {e}")
        return False
    
    return True


def test_dataset_info():
    """Test getting dataset information."""
    print("\nTesting dataset info...")
    
    try:
        # Test GFS info
        gfs_info = get_forecast_dataset_info('gfs')
        print(f"GFS dataset info: {gfs_info}")
        
        # Test GEFS info
        gefs_info = get_forecast_dataset_info('gefs')
        print(f"GEFS dataset info: {gefs_info}")
        
    except Exception as e:
        print(f"Error getting dataset info: {e}")
        return False
    
    return True


def test_forecast_data_fetch():
    """Test fetching forecast wind data."""
    print("\nTesting forecast data fetch...")
    
    try:
        # Get latest init time for GFS
        latest_init = get_latest_init_time('gfs')
        print(f"Using GFS init time: {latest_init}")
        
        # Test small region to speed up the test
        region = {
            'lat_min': 30,
            'lat_max': 40,
            'lon_min': -80,
            'lon_max': -70
        }
        
        # Fetch GFS data for 24h lead time
        u_wind, v_wind = fetch_forecast_wind_data(
            latest_init, '24h', 'gfs', region=region
        )
        
        print(f"GFS data shape: {u_wind.shape}")
        print(f"U wind range: {u_wind.min():.2f} to {u_wind.max():.2f} m/s")
        print(f"V wind range: {v_wind.min():.2f} to {v_wind.max():.2f} m/s")
        
    except Exception as e:
        print(f"Error fetching forecast data: {e}")
        return False
    
    return True


def test_gefs_ensemble():
    """Test GEFS ensemble functionality."""
    print("\nTesting GEFS ensemble...")
    
    try:
        # Get latest init time for GEFS
        latest_init = get_latest_init_time('gefs')
        print(f"Using GEFS init time: {latest_init}")
        
        # Test small region
        region = {
            'lat_min': 30,
            'lat_max': 40,
            'lon_min': -80,
            'lon_max': -70
        }
        
        # Fetch GEFS ensemble mean
        u_wind_mean, v_wind_mean = fetch_forecast_wind_data(
            latest_init, '24h', 'gefs', ensemble_member=None, region=region
        )
        
        print(f"GEFS ensemble mean shape: {u_wind_mean.shape}")
        
        # Fetch specific ensemble member
        u_wind_member, v_wind_member = fetch_forecast_wind_data(
            latest_init, '24h', 'gefs', ensemble_member=0, region=region
        )
        
        print(f"GEFS ensemble member 0 shape: {u_wind_member.shape}")
        
    except Exception as e:
        print(f"Error testing GEFS ensemble: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Starting forecast functionality tests...")
    
    tests = [
        test_latest_init_times,
        test_dataset_info,
        test_forecast_data_fetch,
        test_gefs_ensemble
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 