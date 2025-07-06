"""
ERA5 data fetching and PNG creation module.
"""

from tkinter import S
import xarray as xr
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import io
import logging
from typing import Optional, Tuple, List, Union
import pandas as pd
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .utils import format_wind_info, format_datetime_for_filename

logger = logging.getLogger(__name__)


def get_era5_dataset():
    """Load the ERA5 dataset."""
    logger.info("Loading ERA5 dataset from Google Cloud...")
    dataset = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
        storage_options=dict(token='anon'),
    )
    dataset = dataset.rename({'latitude':'lat', 'longitude':'lon'})
    dataset = dataset[['10m_u_component_of_wind', '10m_v_component_of_wind']]
    
    # Fix longitude coordinates
    dataset.coords['lon'] = (dataset.coords['lon'] + 180) % 360 - 180
    dataset = dataset.sortby(dataset.lon)
    
    logger.info("ERA5 dataset loaded successfully")
    return dataset


def get_gefs_dataset():
    """Load the GEFS dataset."""
    logger.info("Loading GEFS dataset from dynamical.org...")
    gefs_url = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr?email=petermarsh790@gmail.com"
    dataset = xr.open_zarr(gefs_url)
    logger.info("GEFS dataset loaded successfully")
    return dataset


def get_latest_init_time() -> datetime:
    """Get the latest available init_time from GEFS dataset."""
    try:
        ds = get_gefs_dataset()
        latest_init = ds.init_time.max().values
        return pd.to_datetime(latest_init).to_pydatetime()
    except Exception as e:
        logger.error(f"Failed to get latest init time for GEFS: {e}")
        raise RuntimeError("Failed to get latest init time for GEFS")


def fetch_forecast_wind_data(
    init_time: datetime,
    lead_time: Union[str, timedelta],
    ensemble_member: Optional[int] = None,
    region: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch GEFS forecast wind data for a specific init_time and lead_time."""
    if region is None:
        region = {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180}
    
    try:
        ds = get_gefs_dataset()
        
        # Convert lead_time to timedelta if string
        if isinstance(lead_time, str):
            if lead_time.endswith('h'):
                lead_td = timedelta(hours=int(lead_time[:-1]))
            elif lead_time.endswith('d'):
                lead_td = timedelta(days=int(lead_time[:-1]))
            else:
                raise ValueError(f"Invalid lead time format: {lead_time}")
        else:
            lead_td = lead_time
        
        ds_selected = ds.sel(init_time=init_time, method='nearest')
        ds_selected = ds_selected.sel(lead_time=lead_td, method='nearest')
        
        # Handle ensemble member selection
        if ensemble_member is not None:
            if ensemble_member < 0 or ensemble_member > 30:
                raise ValueError(f"Invalid ensemble member: {ensemble_member}. Must be 0-30")
            ds_selected = ds_selected.isel(ensemble_member=ensemble_member)
        else:
            ds_selected = ds_selected.mean(dim='ensemble_member')
        
        # Select spatial region
        ds_subset = ds_selected.sel(
            latitude=slice(region['lat_max'], region['lat_min']),
            longitude=slice(region['lon_min'], region['lon_max'])
        )
        
        # Extract wind components
        u_wind = ds_subset['wind_u_10m'].values
        v_wind = ds_subset['wind_v_10m'].values
        
        return u_wind, v_wind
        
    except Exception as e:
        logger.error(f"Failed to fetch GEFS wind data: {e}")
        raise RuntimeError("GEFS wind data not available for requested parameters")


def fetch_forecast_wind_data_range(
    init_time: datetime,
    lead_times: Union[List[str], List[timedelta], List[Union[str, timedelta]]],
    ensemble_member: Optional[int] = None,
    region: Optional[dict] = None
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Fetch GEFS forecast wind data for multiple lead times efficiently."""
    if region is None:
        region = {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180}
    
    try:
        ds = get_gefs_dataset()
        
        # Convert lead_times to timedeltas
        lead_timedeltas = []
        lead_time_strs = []
        for lt in lead_times:
            if isinstance(lt, str):
                if lt.endswith('h'):
                    lead_td = timedelta(hours=int(lt[:-1]))
                elif lt.endswith('d'):
                    lead_td = timedelta(days=int(lt[:-1]))
                else:
                    raise ValueError(f"Invalid lead time format: {lt}")
                lead_time_strs.append(lt)
            else:
                lead_td = lt
                lead_time_strs.append(f"{int(lt.total_seconds()/3600)}h")
            lead_timedeltas.append(lead_td)
        
        # Select init_time and spatial region
        ds_init = ds.sel(init_time=init_time, method='nearest')
        ds_subset = ds_init.sel(
            latitude=slice(region['lat_max'], region['lat_min']),
            longitude=slice(region['lon_min'], region['lon_max'])
        )
        
        # Handle ensemble member selection
        if ensemble_member is not None:
            if ensemble_member < 0 or ensemble_member > 30:
                raise ValueError(f"Invalid ensemble member: {ensemble_member}")
            ds_subset = ds_subset.isel(ensemble_member=ensemble_member)
        else:
            ds_subset = ds_subset.mean(dim='ensemble_member')
        
        results = []
        for lt_str, lead_td in zip(lead_time_strs, lead_timedeltas):
            try:
                ds_lead = ds_subset.sel(lead_time=lead_td, method='nearest')
                u_wind = ds_lead['wind_u_10m'].values
                v_wind = ds_lead['wind_v_10m'].values
                results.append((lt_str, u_wind, v_wind))
            except Exception as e:
                logger.warning(f"Failed to extract GEFS data for {lt_str}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No forecast wind data could be fetched")
        
        logger.info(f"Successfully fetched GEFS wind data for {len(results)}/{len(lead_times)} lead times")
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch GEFS wind data range: {e}")
        raise RuntimeError("GEFS wind data not available for requested lead times")


def get_forecast_dataset_info() -> dict:
    """Get information about the loaded GEFS dataset."""
    try:
        ds = get_gefs_dataset()
        return {
            "status": "loaded",
            "dataset_type": "GEFS",
            "init_time_range": {
                "start": str(ds.init_time.min().values),
                "end": str(ds.init_time.max().values)
            },
            "lead_time_range": {
                "start": str(ds.lead_time.min().values),
                "end": str(ds.lead_time.max().values)
            },
            "ensemble_members": len(ds.ensemble_member),
            "variables": list(ds.data_vars.keys()),
            "dimensions": dict(ds.dims)
        }
    except Exception as e:
        return {
            "status": "error",
            "dataset_type": "GEFS",
            "error": str(e)
        }


def fetch_wind_data(target_datetime: datetime, region: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch wind data for a specific datetime.
    
    Args:
        target_datetime: Target date and time
        region: Optional region bounds {lat_min, lat_max, lon_min, lon_max}
    
    Returns:
        tuple: (u_wind, v_wind) as numpy arrays
        
    Raises:
        RuntimeError: If data cannot be fetched
    """
    ds = get_era5_dataset()
    
    # Default to global coverage if no region specified
    if region is None:
        region = {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180}
    
    try:
        # Select region and time
        ds_subset = ds.sel(
            lat=slice(region['lat_max'], region['lat_min']),  # Note: ERA5 lat is descending
            lon=slice(region['lon_min'], region['lon_max'])
        )
        
        # Select specific time (closest available)
        ds_time = ds_subset.sel(time=target_datetime, method='nearest')
        
        # Extract wind components
        u_wind = ds_time['10m_u_component_of_wind'].values
        v_wind = ds_time['10m_v_component_of_wind'].values
        
        # Log wind data info
        wind_info = format_wind_info(u_wind, v_wind)
        logger.info(f"Fetched wind data for {target_datetime}")
        logger.info(f"Data shape: {wind_info['shape']}")
        logger.info(f"U wind range: {wind_info['u_range'][0]:.2f} to {wind_info['u_range'][1]:.2f} m/s")
        logger.info(f"V wind range: {wind_info['v_range'][0]:.2f} to {wind_info['v_range'][1]:.2f} m/s")
        
        return u_wind, v_wind
        
    except Exception as e:
        logger.error(f"Failed to fetch wind data: {e}")
        raise RuntimeError(f"Wind data not available for {target_datetime}")


def fetch_wind_data_range(timesteps: List[datetime], region: Optional[dict] = None) -> List[Tuple[datetime, np.ndarray, np.ndarray]]:
    """Fetch wind data for multiple timesteps efficiently."""
    if region is None:
        region = {'lat_min': -90, 'lat_max': 90, 'lon_min': -180, 'lon_max': 180}
    
    ds = get_era5_dataset()
    
    try:
        # Select spatial region
        ds_subset = ds.sel(
            lat=slice(region['lat_max'], region['lat_min']),
            lon=slice(region['lon_min'], region['lon_max'])
        )
        
        # Select time range (min to max of requested timesteps)
        min_time = min(timesteps)
        max_time = max(timesteps)
        
        ds_time_range = ds_subset.sel(time=slice(min_time, max_time))
        ds_time_range = ds_time_range[['10m_u_component_of_wind', '10m_v_component_of_wind']].load()
        
        # Extract data for each requested timestep
        results = []
        u_wind_data = ds_time_range['10m_u_component_of_wind']
        v_wind_data = ds_time_range['10m_v_component_of_wind']
        
        for target_datetime in timesteps:
            try:
                ds_closest = u_wind_data.sel(time=target_datetime, method='nearest')
                u_wind = ds_closest.values
                v_wind = v_wind_data.sel(time=target_datetime, method='nearest').values
                results.append((target_datetime, u_wind, v_wind))
            except Exception as e:
                logger.warning(f"Failed to extract data for {target_datetime}: {e}")
                continue
        
        if not results:
            raise RuntimeError("No wind data could be fetched for any of the requested timesteps")
        
        logger.info(f"Successfully fetched wind data for {len(results)}/{len(timesteps)} timesteps")
        return results
        
    except Exception as e:
        logger.error(f"Failed to fetch wind data range: {e}")
        raise RuntimeError(f"Wind data not available for requested timesteps")


def create_wind_png_optimized(u_wind: np.ndarray, v_wind: np.ndarray, compress_level: int = 1) -> bytes:
    """
    Create WeatherLayers-compatible PNG from wind data with optimizations.
    
    Args:
        u_wind: U component wind data
        v_wind: V component wind data
        compress_level: PNG compression level (0-9, lower is faster)
    
    Returns:
        bytes: PNG image data
        
    Raises:
        RuntimeError: If PNG creation fails
    """
    try:
        # Ensure data is finite and handle NaN values
        u_wind = np.nan_to_num(u_wind, nan=0.0, copy=False)  # Avoid copying if possible
        v_wind = np.nan_to_num(v_wind, nan=0.0, copy=False)
        
        # Pre-allocate the output array for better memory efficiency
        height, width = u_wind.shape
        wind_data = np.empty((height, width, 4), dtype=np.uint8)
        
        # Convert to WeatherLayers format (0-255 with 128 offset) - in-place operations
        wind_data[:, :, 0] = np.clip(u_wind + 128, 0, 255)  # Red: U component
        wind_data[:, :, 1] = np.clip(v_wind + 128, 0, 255)  # Green: V component
        
        # Blue channel: wind speed for reference
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        wind_data[:, :, 2] = np.clip(wind_speed * 5 + 128, 0, 255)  # Blue: Wind speed (scaled)
        
        # Alpha channel: full opacity
        wind_data[:, :, 3] = 255
        
        # Create PIL image
        img = Image.fromarray(wind_data, mode='RGBA')
        
        # Save to bytes with optimized compression
        img_bytes = io.BytesIO()
        img.save(
            img_bytes, 
            format='PNG', 
            optimize=True,
            compress_level=compress_level  # Lower = faster, higher = smaller
        )
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to create PNG: {e}")
        raise RuntimeError("Failed to generate wind PNG")


def create_png_worker(args: Tuple[datetime, np.ndarray, np.ndarray, str]) -> Tuple[str, bytes]:
    """
    Worker function for parallel PNG creation.
    
    Args:
        args: Tuple of (datetime, u_wind, v_wind, filename)
        
    Returns:
        Tuple of (filename, png_data)
    """
    dt, u_wind, v_wind, filename = args
    
    try:
        start_time = time.time()
        png_data = create_wind_png_optimized(u_wind, v_wind, compress_level=0)  # Fastest compression
        elapsed = time.time() - start_time
        
        logger.info(f"Created {filename} in {elapsed:.2f}s ({len(png_data)/1024:.1f}KB)")
        return filename, png_data
        
    except Exception as e:
        logger.error(f"Failed to create PNG for {filename}: {e}")
        raise


def create_wind_data_zip_parallel(wind_data_list: List[Tuple[datetime, np.ndarray, np.ndarray]], max_workers: int = 4) -> bytes:
    """
    Create a ZIP file containing PNG files for multiple timesteps using parallel processing.
    
    Args:
        wind_data_list: List of (datetime, u_wind, v_wind) tuples
        max_workers: Maximum number of worker threads for PNG creation
        
    Returns:
        bytes: ZIP file containing PNG files
        
    Raises:
        RuntimeError: If ZIP creation fails
    """
    try:
        start_time = time.time()
        logger.info(f"Creating ZIP with {len(wind_data_list)} files using {max_workers} workers...")
        
        # Prepare arguments for parallel processing
        png_args = []
        for dt, u_wind, v_wind in wind_data_list:
            filename = f"{format_datetime_for_filename(dt)}.png"
            png_args.append((dt, u_wind, v_wind, filename))
        
        # Create PNGs in parallel
        png_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PNG creation tasks
            future_to_filename = {
                executor.submit(create_png_worker, args): args[3] 
                for args in png_args
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_filename):
                filename = future_to_filename[future]
                try:
                    result_filename, png_data = future.result()
                    png_results[result_filename] = png_data
                except Exception as e:
                    logger.warning(f"Failed to create PNG for {filename}: {e}")
                    continue
        
        png_creation_time = time.time() - start_time
        logger.info(f"PNG creation completed in {png_creation_time:.2f}s")
        
        # Create ZIP file with minimal compression for speed
        zip_start_time = time.time()
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED, compresslevel=1) as zip_file:
            for filename in sorted(png_results.keys()):  # Sort for consistent ordering
                zip_file.writestr(filename, png_results[filename])
        
        zip_buffer.seek(0)
        zip_data = zip_buffer.getvalue()
        
        zip_creation_time = time.time() - zip_start_time
        total_time = time.time() - start_time
        
        logger.info(f"ZIP creation completed in {zip_creation_time:.2f}s")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Created ZIP archive with {len(png_results)} PNG files, size: {len(zip_data)/1024:.1f}KB")
        
        return zip_data
        
    except Exception as e:
        logger.error(f"Failed to create ZIP archive: {e}")
        raise RuntimeError("Failed to generate wind data ZIP file")


# Keep the old functions for backward compatibility
def create_wind_png(u_wind: np.ndarray, v_wind: np.ndarray) -> bytes:
    """Legacy function - use create_wind_png_optimized for better performance."""
    return create_wind_png_optimized(u_wind, v_wind, compress_level=6)


def create_wind_data_zip(wind_data_list: List[Tuple[datetime, np.ndarray, np.ndarray]]) -> bytes:
    """Legacy function - use create_wind_data_zip_parallel for better performance."""
    return create_wind_data_zip_parallel(wind_data_list, max_workers=4)


def get_latest_time() -> datetime:
    """Get the latest available time from the ERA5 dataset."""
    ds = get_era5_dataset()
    latest_time = ds.time.max().values
    return pd.to_datetime(latest_time).to_pydatetime()


def get_dataset_info() -> dict:
    """Get information about the loaded dataset."""
    try:
        ds = get_era5_dataset()
        return {
            "status": "loaded",
            "time_range": {
                "start": str(ds.time.min().values),
                "end": str(ds.time.max().values)
            },
            "spatial_resolution": "0.25°",
            "variables": list(ds.data_vars.keys()),
            "dimensions": dict(ds.dims)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        } 