"""
Utility functions for xarray_weather_server.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, List


def parse_date_hour(date_hour: str) -> datetime:
    """
    Parse YYMMDDHH format to datetime.
    
    Args:
        date_hour: Date string in YYMMDDHH format
        
    Returns:
        datetime: Parsed datetime object
        
    Raises:
        ValueError: If date format is invalid
    """
    if not re.match(r'^\d{8}$', date_hour):
        raise ValueError("Date format must be YYMMDDHH (8 digits)")
    
    try:
        # Parse YYMMDDHH
        year = int(date_hour[:2])
        month = int(date_hour[2:4])
        day = int(date_hour[4:6])
        hour = int(date_hour[6:8])
        
        # Convert 2-digit year to 4-digit (assuming 50-99 = 1950-1999, 00-49 = 2000-2049)
        if year >= 50:
            year += 1900
        else:
            year += 2000
        
        return datetime(year, month, day, hour)
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")


def parse_date_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """
    Parse date range from YYMMDDHH format strings.
    
    Args:
        start_date: Start date in YYMMDDHH format
        end_date: End date in YYMMDDHH format
        
    Returns:
        tuple: (start_datetime, end_datetime)
        
    Raises:
        ValueError: If date formats are invalid or end_date is before start_date
    """
    start_dt = parse_date_hour(start_date)
    end_dt = parse_date_hour(end_date)
    
    if end_dt < start_dt:
        raise ValueError("End date must be after start date")
    
    return start_dt, end_dt


def generate_hourly_timesteps(start_dt: datetime, end_dt: datetime, step_hours: int = 1) -> List[datetime]:
    """
    Generate hourly timesteps between start and end dates.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime  
        step_hours: Hours between each timestep (default: 1)
        
    Returns:
        List[datetime]: List of datetime objects for each timestep
        
    Raises:
        ValueError: If step_hours is not positive or range is too large
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    
    timesteps = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        timesteps.append(current_dt)
        current_dt += timedelta(hours=step_hours)
    
    # Limit to prevent extremely large requests
    if len(timesteps) > 480:  # 20 days at hourly intervals
        raise ValueError("Date range too large. Maximum 480 timesteps (20 days at hourly intervals)")
    
    return timesteps


def format_datetime_for_filename(dt: datetime) -> str:
    """
    Format datetime for use in filenames.
    
    Args:
        dt: Datetime object
        
    Returns:
        str: Formatted string suitable for filenames (YYMMDDHH format)
    """
    return dt.strftime("%y%m%d%H")


def validate_region_bounds(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> bool:
    """
    Validate geographic region bounds.
    
    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude  
        lon_min: Minimum longitude
        lon_max: Maximum longitude
        
    Returns:
        bool: True if bounds are valid
    """
    if not (-90 <= lat_min <= lat_max <= 90):
        return False
    if not (-180 <= lon_min <= lon_max <= 180):
        return False
    return True


def format_wind_info(u_wind, v_wind) -> dict:
    """
    Format wind data information for logging/debugging.
    
    Args:
        u_wind: U component wind data
        v_wind: V component wind data
        
    Returns:
        dict: Wind data statistics
    """
    import numpy as np
    
    return {
        "shape": u_wind.shape,
        "u_range": [float(np.min(u_wind)), float(np.max(u_wind))],
        "v_range": [float(np.min(v_wind)), float(np.max(v_wind))],
        "u_mean": float(np.mean(u_wind)),
        "v_mean": float(np.mean(v_wind)),
        "wind_speed_max": float(np.max(np.sqrt(u_wind**2 + v_wind**2))),
    } 