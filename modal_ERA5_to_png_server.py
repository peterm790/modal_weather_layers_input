import modal
import xarray as xr
import numpy as np
from PIL import Image
from datetime import datetime
import io

app = modal.App("era5-wind-png")

image = modal.Image.debian_slim().pip_install(
    "xarray", "numpy", "pillow", "pandas", "zarr", "fsspec", "gcsfs"
)

@app.function(cpu=2, memory=4096, timeout=600, image=image)
def get_wind_png(date_hour: str):
    """Get ERA5 wind PNG for YYYYMMDDHH format date."""
    
    # Parse date
    year = int(date_hour[:4])
    month = int(date_hour[4:6])
    day = int(date_hour[6:8])
    hour = int(date_hour[8:10])
    target_datetime = datetime(year, month, day, hour)
    
    # Load ERA5 data
    ds = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
        storage_options=dict(token='anon'),
    )
    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    
    # Get wind data
    ds_time = ds.sel(time=target_datetime, method='nearest')
    u_wind = ds_time['10m_u_component_of_wind'].values
    v_wind = ds_time['10m_v_component_of_wind'].values
    
    # Create PNG
    u_wind = np.nan_to_num(u_wind, nan=0.0)
    v_wind = np.nan_to_num(v_wind, nan=0.0)
    
    height, width = u_wind.shape
    wind_data = np.empty((height, width, 4), dtype=np.uint8)
    
    # WeatherLayers format
    wind_data[:, :, 0] = np.clip(u_wind + 128, 0, 255)  # Red: U
    wind_data[:, :, 1] = np.clip(v_wind + 128, 0, 255)  # Green: V
    wind_data[:, :, 2] = np.clip(np.sqrt(u_wind**2 + v_wind**2) * 5 + 128, 0, 255)  # Blue: speed
    wind_data[:, :, 3] = 255  # Alpha
    
    img = Image.fromarray(wind_data)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

@app.function(image=image)
@modal.fastapi_endpoint(method="GET", docs=True)
def api_get_wind_png(date_hour: str):
    """Web API endpoint."""
    try:
        png_data = get_wind_png.remote(date_hour)
        from fastapi import Response
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=wind_{date_hour}.png"}
        )
    except Exception as e:
        from fastapi import Response
        return Response(content=f"Error: {str(e)}", media_type="text/plain", status_code=400)

@app.local_entrypoint()
def main():
    png_data = get_wind_png.remote("2024011512")
    with open("wind_2024011512.png", "wb") as f:
        f.write(png_data)
    print(f"Generated wind PNG: {len(png_data)} bytes")
