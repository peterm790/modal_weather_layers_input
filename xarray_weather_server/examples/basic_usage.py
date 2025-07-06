#!/usr/bin/env python3
"""
Basic usage example for xarray_weather_server.

This example shows how to:
1. Start the server programmatically
2. Make API requests
3. Save wind data PNGs
"""

import asyncio
import requests
import time
from datetime import datetime, timedelta
import uvicorn
from multiprocessing import Process

from xarray_weather_server import app


def start_server():
    """Start the server in a separate process."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


def test_api_requests():
    """Test various API requests."""
    base_url = "http://127.0.0.1:8000"
    
    # Wait for server to start
    time.sleep(5)
    
    print("Testing XArray Weather Server API")
    print("=" * 50)
    
    # Test root endpoint
    print("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✓ Root endpoint working")
            data = response.json()
            print(f"   Server: {data['message']}")
            print(f"   Version: {data['version']}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Root endpoint error: {e}")
    
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
            data = response.json()
            print(f"   Status: {data['status']}")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Health endpoint error: {e}")
    
    print("\n3. Testing wind data endpoint...")
    test_date = "24011512"  # Jan 15, 2024 12:00 UTC
    try:
        print(f"   Fetching wind data for {test_date}...")
        response = requests.get(f"{base_url}/wind/{test_date}")
        if response.status_code == 200:
            print("✓ Wind data endpoint working")
            print(f"   Content type: {response.headers.get('content-type')}")
            print(f"   Content length: {len(response.content)} bytes")
            
            # Save the PNG file
            filename = f"example_{test_date}.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved wind PNG as {filename}")
        else:
            print(f"✗ Wind data endpoint failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"✗ Wind data endpoint error: {e}")
    
    print("\n4. Testing regional wind data...")
    try:
        # Mediterranean region
        params = {
            'lat_min': 30,
            'lat_max': 45,
            'lon_min': -10,
            'lon_max': 40
        }
        response = requests.get(f"{base_url}/wind/{test_date}", params=params)
        if response.status_code == 200:
            print("✓ Regional wind data working")
            print(f"   Content length: {len(response.content)} bytes")
            
            # Save the regional PNG file
            filename = f"example_{test_date}_mediterranean.png"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved regional wind PNG as {filename}")
        else:
            print(f"✗ Regional wind data failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Regional wind data error: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nGenerated files:")
    print(f"- example_{test_date}.png (global wind data)")
    print(f"- example_{test_date}_mediterranean.png (regional wind data)")


def main():
    """Main example function."""
    print("XArray Weather Server - Basic Usage Example")
    print("=" * 60)
    
    # Start server in background
    print("Starting server...")
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        # Test API requests
        test_api_requests()
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.join()


if __name__ == "__main__":
    main() 