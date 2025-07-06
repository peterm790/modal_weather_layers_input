"""
FastAPI server for ERA5 wind data API.
"""

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import logging
import time
import os
import hashlib
import json
from typing import Optional, List, Union

from .data_fetcher import (
    fetch_wind_data, create_wind_png, get_latest_time, get_dataset_info, 
    fetch_wind_data_range, create_wind_data_zip_parallel,
    fetch_forecast_wind_data, fetch_forecast_wind_data_range,
    get_latest_init_time, get_forecast_dataset_info
)
from .utils import parse_date_hour, validate_region_bounds, parse_date_range, generate_hourly_timesteps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_WORKERS = 6  # Adjust based on server CPU cores
COMPRESS_LEVEL = 1  # Fast compression for better performance

# Cache configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
ENABLE_CACHE = True  # Set to False to disable caching

# Ensure cache directory exists
if ENABLE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(start_date, end_date, step_hours, region, dataset_type='era5', init_time=None, ensemble_member=None):
    """Generate a cache key for the request parameters."""
    cache_data = {
        'start_date': start_date,
        'end_date': end_date,
        'step_hours': step_hours,
        'region': region,
        'dataset_type': dataset_type,
        'init_time': init_time.isoformat() if init_time else None,
        'ensemble_member': ensemble_member
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()

def get_cached_data(cache_key):
    """Retrieve cached data if it exists."""
    if not ENABLE_CACHE:
        return None
    
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.zip")
    if os.path.exists(cache_file):
        logger.info(f"CACHE HIT: Using cached data for {cache_key}")
        with open(cache_file, 'rb') as f:
            return f.read()
    return None

def save_cached_data(cache_key, data):
    """Save data to cache."""
    if not ENABLE_CACHE:
        return
    
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.zip")
    try:
        with open(cache_file, 'wb') as f:
            f.write(data)
        logger.info(f"CACHE SAVE: Saved {len(data)/1024:.1f}KB to cache for {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

def parse_lead_time(lead_time_str: str) -> str:
    """Parse and validate lead time string."""
    # Accept formats like: 1h, 24h, 1d, 7d, etc.
    if lead_time_str.endswith('h') or lead_time_str.endswith('d'):
        return lead_time_str
    else:
        # Try to parse as hours
        try:
            hours = int(lead_time_str)
            return f"{hours}h"
        except ValueError:
            raise ValueError(f"Invalid lead time format: {lead_time_str}. Use formats like '1h', '24h', '1d', '7d'")

def generate_lead_time_range(start_hours: int, end_hours: int, step_hours: int) -> List[str]:
    """Generate a list of lead time strings."""
    lead_times = []
    current_hours = start_hours
    
    while current_hours <= end_hours:
        if current_hours % 24 == 0 and current_hours > 0:
            lead_times.append(f"{current_hours // 24}d")
        else:
            lead_times.append(f"{current_hours}h")
        current_hours += step_hours
    
    return lead_times

app = FastAPI(
    title="XArray Weather Server",
    description="Fetch ERA5 historical and GFS/GEFS forecast wind data as WeatherLayers-compatible PNG files",
    version="2.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "XArray Weather Server",
        "version": "2.0.0",
        "datasets": {
            "era5": "Historical reanalysis data (1979-present)",
            "gefs": "GEFS ensemble forecast data (0-35 days, 31 members)"
        },
        "endpoints": {
            "historical": {
                "/wind/{date_hour}": "Get historical wind PNG for specific date (YYMMDDHH format)",
                "/wind/range": "Get historical wind PNGs for date range as ZIP file",
                "/wind/latest": "Get latest available historical wind data"
            },
            "forecast": {
                "/forecast/latest": "Get latest available init_time for GEFS",
                "/forecast/{init_time}/{lead_time}": "Get GEFS forecast wind PNG",
                "/forecast/range": "Get GEFS forecast wind PNGs for lead time range as ZIP"
            },
            "system": {
                "/health": "Health check with dataset status",
                "/cache/clear": "Clear cache (development only)"
            }
        },
        "examples": {
            "historical": {
                "single": "/wind/24011512 (Jan 15, 2024 at 12:00 UTC)",
                "range": "/wind/range?start_date=24011500&end_date=24011523"
            },
            "forecast": {
                "latest_init": "/forecast/latest",
                "single": "/forecast/2024-01-15T00:00:00/7d",
                "ensemble_member": "/forecast/2024-01-15T00:00:00/7d?ensemble_member=5",
                "range": "/forecast/range?init_time=2024-01-15T00:00:00&start_lead_hours=0&end_lead_hours=168&step_hours=6"
            }
        },
        "cache": {
            "enabled": ENABLE_CACHE,
            "directory": CACHE_DIR if ENABLE_CACHE else None
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with dataset status."""
    try:
        era5_info = get_dataset_info()
        gefs_info = get_forecast_dataset_info()
        
        return {
            "status": "healthy",
            "datasets": {
                "era5": era5_info,
                "gefs": gefs_info
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data (development only)."""
    if not ENABLE_CACHE:
        return {"message": "Cache is disabled"}
    
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.zip')]
        for cache_file in cache_files:
            os.remove(os.path.join(CACHE_DIR, cache_file))
        
        logger.info(f"CACHE CLEARED: Removed {len(cache_files)} cached files")
        return {
            "message": f"Cache cleared successfully",
            "files_removed": len(cache_files)
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# Historical ERA5 endpoints (existing)
@app.get("/wind/latest")
async def get_latest_wind():
    """Get wind data for the latest available time."""
    try:
        latest_datetime = get_latest_time()
        
        # Fetch global wind data
        u_wind, v_wind = fetch_wind_data(latest_datetime)
        
        # Create PNG
        png_data = create_wind_png(u_wind, v_wind)
        
        # Generate filename
        filename = latest_datetime.strftime("%y%m%d%H.png")
        
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Failed to get latest wind data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch latest wind data")


@app.get("/wind/range")
async def get_wind_range(
    start_date: str = Query(..., description="Start date in YYMMDDHH format"),
    end_date: str = Query(..., description="End date in YYMMDDHH format"),
    step_hours: int = Query(1, description="Hours between timesteps (default: 1)", ge=1, le=24),
    lat_min: float = Query(-90, description="Minimum latitude", ge=-90, le=90),
    lat_max: float = Query(90, description="Maximum latitude", ge=-90, le=90),
    lon_min: float = Query(-180, description="Minimum longitude", ge=-180, le=180),
    lon_max: float = Query(180, description="Maximum longitude", ge=-180, le=180)
):
    """
    Get wind data PNGs for a date range as a ZIP file.
    
    Args:
        start_date: Start date in YYMMDDHH format
        end_date: End date in YYMMDDHH format
        step_hours: Hours between each timestep (default: 1)
        lat_min: Minimum latitude (default: -90)
        lat_max: Maximum latitude (default: 90)
        lon_min: Minimum longitude (default: -180)
        lon_max: Maximum longitude (default: 180)
    
    Returns:
        ZIP file containing PNG images for each timestep
    """
    request_start_time = time.time()
    
    try:
        logger.info(f"RANGE REQUEST START: {start_date} to {end_date} (step: {step_hours}h)")
        
        # Parse and validate date range
        parse_start = time.time()
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Validate region bounds
        if not validate_region_bounds(lat_min, lat_max, lon_min, lon_max):
            raise HTTPException(status_code=400, detail="Invalid region bounds")
        
        # Generate timesteps
        timesteps = generate_hourly_timesteps(start_dt, end_dt, step_hours)
        parse_time = time.time() - parse_start
        logger.info(f"Parsing & validation: {parse_time:.2f}s ({len(timesteps)} timesteps)")
        
        region = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        # Check cache first
        cache_key = get_cache_key(start_date, end_date, step_hours, region, 'era5')
        cached_data = get_cached_data(cache_key)
        
        if cached_data:
            total_time = time.time() - request_start_time
            logger.info(f"CACHE REQUEST COMPLETED: {total_time:.2f}s total (from cache)")
            logger.info(f"Returning cached ZIP file ({len(cached_data)/1024:.1f}KB)")
            
            zip_filename = f"wind_data_{start_date}_{end_date}.zip"
            return Response(
                content=cached_data,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
        
        # Fetch wind data for all timesteps
        fetch_start = time.time()
        wind_data_list = fetch_wind_data_range(timesteps, region)
        fetch_time = time.time() - fetch_start
        logger.info(f"Data fetching: {fetch_time:.2f}s")
        
        if not wind_data_list:
            raise HTTPException(status_code=404, detail="No wind data found for the specified date range")
        
        # Create ZIP file
        zip_start = time.time()
        zip_data = create_wind_data_zip_parallel(wind_data_list, MAX_WORKERS)
        zip_time = time.time() - zip_start
        logger.info(f"ZIP creation: {zip_time:.2f}s")
        
        # Save to cache
        save_cached_data(cache_key, zip_data)
        
        # Generate filename for the ZIP
        zip_filename = f"wind_data_{start_date}_{end_date}.zip"
        
        total_time = time.time() - request_start_time
        logger.info(f"RANGE REQUEST COMPLETED: {total_time:.2f}s total")
        logger.info(f"Breakdown - Parse: {parse_time:.1f}s, Fetch: {fetch_time:.1f}s, ZIP: {zip_time:.1f}s")
        logger.info(f"Returning ZIP file with {len(wind_data_list)} wind data files ({len(zip_data)/1024:.1f}KB)")
        
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except ValueError as e:
        total_time = time.time() - request_start_time
        logger.error(f"RANGE REQUEST FAILED: {total_time:.2f}s - ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        total_time = time.time() - request_start_time
        logger.error(f"RANGE REQUEST FAILED: {total_time:.2f}s - RuntimeError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        total_time = time.time() - request_start_time
        logger.error(f"RANGE REQUEST FAILED: {total_time:.2f}s - Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/wind/{date_hour}")
async def get_wind_png(
    date_hour: str,
    lat_min: float = -90,
    lat_max: float = 90,
    lon_min: float = -180,
    lon_max: float = 180
):
    """
    Get wind data PNG for a specific date and time.
    
    Args:
        date_hour: Date and hour in YYMMDDHH format (e.g., "24011512" for Jan 15, 2024 12:00 UTC)
        lat_min: Minimum latitude (default: -90)
        lat_max: Maximum latitude (default: 90)
        lon_min: Minimum longitude (default: -180)
        lon_max: Maximum longitude (default: 180)
    
    Returns:
        PNG image with wind data
    """
    try:
        # Parse date
        target_datetime = parse_date_hour(date_hour)
        
        # Validate region bounds
        if not validate_region_bounds(lat_min, lat_max, lon_min, lon_max):
            raise HTTPException(status_code=400, detail="Invalid region bounds")
        
        region = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        # Fetch wind data
        u_wind, v_wind = fetch_wind_data(target_datetime, region)
        
        # Create PNG
        png_data = create_wind_png(u_wind, v_wind)
        
        # Return PNG response
        filename = f"{date_hour}.png"
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# New forecast endpoints
@app.get("/forecast/latest")
async def get_latest_forecast_init_time():
    """
    Get the latest available initialization time for forecast datasets.
    
    Returns:
        JSON with latest init_time and dataset info
    """
    try:
        latest_init = get_latest_init_time()
        dataset_info = get_forecast_dataset_info()
        
        return {
            "dataset_type": "GEFS",
            "latest_init_time": latest_init.isoformat(),
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get latest init time for GEFS: {e}")
        raise HTTPException(status_code=500, detail="Failed to get latest init time for GEFS")


@app.get("/forecast/{init_time}/{lead_time}")
async def get_forecast_wind_png(
    init_time: str,
    lead_time: str,
    ensemble_member: Optional[int] = Query(None, description="GEFS ensemble member (0-30), None for ensemble mean", ge=0, le=30),
    lat_min: float = Query(-90, description="Minimum latitude", ge=-90, le=90),
    lat_max: float = Query(90, description="Maximum latitude", ge=-90, le=90),
    lon_min: float = Query(-180, description="Minimum longitude", ge=-180, le=180),
    lon_max: float = Query(180, description="Maximum longitude", ge=-180, le=180)
):
    """
    Get GEFS forecast wind data PNG for a specific init_time and lead_time.
    
    Args:
        init_time: Initialization time in ISO format (e.g., '2024-01-15T00:00:00')
        lead_time: Lead time (e.g., '24h', '7d')
        ensemble_member: For GEFS, which ensemble member (0-30), None for ensemble mean
        lat_min: Minimum latitude (default: -90)
        lat_max: Maximum latitude (default: 90)
        lon_min: Minimum longitude (default: -180)
        lon_max: Maximum longitude (default: 180)
    
    Returns:
        PNG image with forecast wind data
    """
    try:
        # Parse init_time
        init_dt = datetime.fromisoformat(init_time.replace('Z', '+00:00'))
        
        # Validate and parse lead_time
        lead_time_parsed = parse_lead_time(lead_time)
        
        # Validate region bounds
        if not validate_region_bounds(lat_min, lat_max, lon_min, lon_max):
            raise HTTPException(status_code=400, detail="Invalid region bounds")
        
        region = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        # Fetch forecast wind data
        u_wind, v_wind = fetch_forecast_wind_data(
            init_dt, lead_time_parsed, ensemble_member, region
        )
        
        # Create PNG
        png_data = create_wind_png(u_wind, v_wind)
        
        # Generate filename
        ensemble_suffix = f"_ens{ensemble_member}" if ensemble_member is not None else ""
        filename = f"gefs_{init_dt.strftime('%Y%m%d%H')}_{lead_time}{ensemble_suffix}.png"
        
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process forecast request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/forecast/range")
async def get_forecast_wind_range(
    init_time: str = Query(..., description="Initialization time in ISO format (e.g., '2024-01-15T00:00:00')"),
    start_lead_hours: int = Query(0, description="Start lead time in hours", ge=0),
    end_lead_hours: int = Query(168, description="End lead time in hours", ge=0),
    step_hours: int = Query(6, description="Hours between lead times (default: 6)", ge=1),
    ensemble_member: Optional[int] = Query(None, description="GEFS ensemble member (0-30), None for ensemble mean", ge=0, le=30),
    lat_min: float = Query(-90, description="Minimum latitude", ge=-90, le=90),
    lat_max: float = Query(90, description="Maximum latitude", ge=-90, le=90),
    lon_min: float = Query(-180, description="Minimum longitude", ge=-180, le=180),
    lon_max: float = Query(180, description="Maximum longitude", ge=-180, le=180)
):
    """
    Get GEFS forecast wind data PNGs for a range of lead times as a ZIP file.
    
    Args:
        init_time: Initialization time in ISO format
        start_lead_hours: Start lead time in hours (default: 0)
        end_lead_hours: End lead time in hours (default: 168)
        step_hours: Hours between each lead time (default: 6)
        ensemble_member: For GEFS, which ensemble member (0-30), None for ensemble mean
        lat_min: Minimum latitude (default: -90)
        lat_max: Maximum latitude (default: 90)
        lon_min: Minimum longitude (default: -180)
        lon_max: Maximum longitude (default: 180)
    
    Returns:
        ZIP file containing PNG images for each lead time
    """
    request_start_time = time.time()
    
    try:
        logger.info(f"FORECAST RANGE REQUEST START: GEFS init: {init_time}, lead: {start_lead_hours}-{end_lead_hours}h (step: {step_hours}h)")
        
        # Parse init_time
        init_dt = datetime.fromisoformat(init_time.replace('Z', '+00:00'))
        
        # Validate region bounds
        if not validate_region_bounds(lat_min, lat_max, lon_min, lon_max):
            raise HTTPException(status_code=400, detail="Invalid region bounds")
        
        # Generate lead times
        lead_times = generate_lead_time_range(start_lead_hours, end_lead_hours, step_hours)
        logger.info(f"Generated {len(lead_times)} lead times: {lead_times[:5]}..." if len(lead_times) > 5 else f"Generated {len(lead_times)} lead times: {lead_times}")
        
        region = {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        }
        
        # Check cache first
        cache_key = get_cache_key(
            f"{start_lead_hours}h", f"{end_lead_hours}h", step_hours, region, 
            'gefs', init_dt, ensemble_member
        )
        cached_data = get_cached_data(cache_key)
        
        if cached_data:
            total_time = time.time() - request_start_time
            logger.info(f"FORECAST CACHE REQUEST COMPLETED: {total_time:.2f}s total (from cache)")
            logger.info(f"Returning cached ZIP file ({len(cached_data)/1024:.1f}KB)")
            
            ensemble_suffix = f"_ens{ensemble_member}" if ensemble_member is not None else ""
            zip_filename = f"gefs_forecast_{init_dt.strftime('%Y%m%d%H')}_{start_lead_hours}h-{end_lead_hours}h{ensemble_suffix}.zip"
            return Response(
                content=cached_data,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
        
        # Fetch forecast wind data for all lead times
        fetch_start = time.time()
        wind_data_list = fetch_forecast_wind_data_range(
            init_dt, lead_times, ensemble_member, region
        )
        fetch_time = time.time() - fetch_start
        logger.info(f"Forecast data fetching: {fetch_time:.2f}s")
        
        if not wind_data_list:
            raise HTTPException(status_code=404, detail="No forecast wind data found for the specified parameters")
        
        # Convert to format expected by ZIP creation function
        # (lead_time_str, u_wind, v_wind) -> (datetime, u_wind, v_wind) with lead_time as identifier
        zip_data_list = []
        for lead_time_str, u_wind, v_wind in wind_data_list:
            # Create a pseudo-datetime using lead_time as identifier
            # This is just for filename generation in the ZIP
            pseudo_dt = init_dt + timedelta(hours=int(lead_time_str.replace('h', '').replace('d', '')) * (24 if 'd' in lead_time_str else 1))
            zip_data_list.append((pseudo_dt, u_wind, v_wind))
        
        # Create ZIP file
        zip_start = time.time()
        zip_data = create_wind_data_zip_parallel(zip_data_list, MAX_WORKERS)
        zip_time = time.time() - zip_start
        logger.info(f"ZIP creation: {zip_time:.2f}s")
        
        # Save to cache
        save_cached_data(cache_key, zip_data)
        
        # Generate filename for the ZIP
        ensemble_suffix = f"_ens{ensemble_member}" if ensemble_member is not None else ""
        zip_filename = f"gefs_forecast_{init_dt.strftime('%Y%m%d%H')}_{start_lead_hours}h-{end_lead_hours}h{ensemble_suffix}.zip"
        
        total_time = time.time() - request_start_time
        logger.info(f"FORECAST RANGE REQUEST COMPLETED: {total_time:.2f}s total")
        logger.info(f"Breakdown - Fetch: {fetch_time:.1f}s, ZIP: {zip_time:.1f}s")
        logger.info(f"Returning ZIP file with {len(wind_data_list)} forecast wind files ({len(zip_data)/1024:.1f}KB)")
        
        return Response(
            content=zip_data,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except ValueError as e:
        total_time = time.time() - request_start_time
        logger.error(f"FORECAST RANGE REQUEST FAILED: {total_time:.2f}s - ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        total_time = time.time() - request_start_time
        logger.error(f"FORECAST RANGE REQUEST FAILED: {total_time:.2f}s - RuntimeError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        total_time = time.time() - request_start_time
        logger.error(f"FORECAST RANGE REQUEST FAILED: {total_time:.2f}s - Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") 