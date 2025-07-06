# This modal cronjob runs everyday at 930am to generate inputs for www.petesforecast.com . All time steps are run in parallel. 

import fsspec
import s3fs

import xarray as xr
import rioxarray
import numpy as np

import io
from PIL import Image

import json
import os

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cmocean as cm

import modal

app = modal.App("petesforecast_generate_wind_layers")

image = modal.Image.debian_slim().pip_install(
    "fsspec", "xarray", "rioxarray", "numpy", "pillow", "matplotlib", "cmocean", "s3fs", "zarr", "cfgrib", "kerchunk", "aiohttp"
)

@app.function(cpu=2, memory=4096, timeout=600, image=image, secrets=[modal.Secret.from_name("aws-credentials")])
def generate_wind_layers(inputs):
    ref_url, valid_time_step, file_name = inputs
    # Get AWS credentials from environment variables
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    s3 = s3fs.S3FileSystem(
        key=aws_access_key_id,
        secret=aws_secret_access_key,
        client_kwargs={'region_name': 'af-south-1'}
    )

    fs_ = fsspec.filesystem(
        "reference",
        fo=ref_url,
        target_protocol='s3',
        target_options={'anon': False}
    )

    m = fs_.get_mapper("")
    ds = xr.open_dataset(m, engine="zarr", backend_kwargs=dict(consolidated=False))

    # Adjust longitude to -180 to 180 range
    ds['longitude'] = (ds['longitude'] + 180) % 360 - 180
    ds = ds.sortby('longitude')

    # Select wind components for the specified time step
    u10 = ds.u10.isel(valid_time=valid_time_step)
    v10 = ds.v10.isel(valid_time=valid_time_step)

    # Generate the file name prefix
    output_path = f"s3://peterm790/petesforecast/wind/{ds.time.dt.strftime('%Y%m%d').item()}/"
    file_prefix = f"{ds.time.dt.strftime('%Y%m%d').item()}_{str(ds.time.dt.hour.item()).zfill(2)}_{valid_time_step}"
    full_file_name = f"{file_prefix}_{file_name}"

    def y2lat(y, height):
        lat = (360 / np.pi) * np.arctan(np.exp(((180 - (y / height) * 360) * np.pi) / 180)) - 90
        return ((-lat + 90) / 180) * height

    def interpolate(p, v, width):
        x, y = p
        x = x % width
        y = np.clip(y, 0, v.shape[0] - 1)
        
        x1, x2 = int(np.floor(x)), int(np.ceil(x)) % width
        y1, y2 = int(np.floor(y)), int(np.ceil(y))

        fx, fy = x - x1, y - y1

        c00, c10 = v[y1, x1], v[y1, x2]
        c01, c11 = v[y2, x1], v[y2, x2]

        return (1-fx)*(1-fy)*c00 + fx*(1-fy)*c10 + (1-fx)*fy*c01 + fx*fy*c11

    def generate_wind_png(wind_u, wind_v):
        width, height = wind_u.shape[1], wind_u.shape[0]
        image = Image.new('RGB', (width, height))
        pixels = image.load()

        ugrd_min, ugrd_max = wind_u.min().item(), wind_u.max().item()
        vgrd_min, vgrd_max = wind_v.min().item(), wind_v.max().item()

        for y in range(height):
            for x in range(width):
                lat = y2lat(y, height)
                u = interpolate([x, lat], wind_u.values, width)
                v = interpolate([x, lat], wind_v.values, width)

                r = int(np.clip(255 * (u - ugrd_min) / (ugrd_max - ugrd_min), 0, 255))
                g = int(np.clip(255 * (v - vgrd_min) / (vgrd_max - vgrd_min), 0, 255))
                b = 0

                pixels[x, y] = (r, g, b)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    # Generate PNG
    png_path = os.path.join(output_path, f"{full_file_name}.png")
    png_data = generate_wind_png(u10, v10)
    with s3.open(png_path, 'wb') as f:
        f.write(png_data)

    # Generate JSON
    json_data = {
        "source": "http://nomads.ncep.noaa.gov",
        "width": u10.shape[1],
        "height": u10.shape[0],
        "uMin": float(u10.min()),
        "uMax": float(u10.max()),
        "vMin": float(v10.min()),
        "vMax": float(v10.max()),
    }

    json_path = os.path.join(output_path, f"{full_file_name}.json")
    with s3.open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Generate GeoTIFF
    da = np.sqrt(u10**2 + v10**2)
    vmin, vmax = da.min().item(), da.max().item()
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cm.cm.speed)
    rgb_array = sm.to_rgba(da, bytes=True)[:, :, :3]

    ds_rgb = xr.DataArray(rgb_array, dims=['latitude', 'longitude', 'band'],
                          coords={'latitude': da.latitude, 'longitude': da.longitude, 'band': ['red', 'green', 'blue']})
    ds_rgb = ds_rgb.transpose('band', 'latitude', 'longitude')
    ds_rgb = ds_rgb.rio.write_crs('EPSG:4326')
    ds_rgb = ds_rgb.rio.reproject('EPSG:4326')
    ds_rgb = ds_rgb.astype(np.uint8)

    tif_path = os.path.join(output_path, f"{full_file_name}.tif")

    with s3.open(tif_path, "wb") as f:
        ds_rgb.rio.to_raster(f, driver='GTiff')

    print(f"Generated {png_path}, {json_path}, and {tif_path}")


@app.function(image=image, cpu=1, memory=500, timeout=600, schedule=modal.Cron("30 7 * * *"))
def scheduled_main():
    ref_url = "s3://lambdagfsreferencestack-gfsreference01a4696a-1lywfe3wpr52o/references/latest.json"
    file_name = 'ws'
    inputs = []
    for valid_time_step in range(0, 128):
        inputs.append([ref_url, valid_time_step, file_name])

    for result in generate_wind_layers.map(inputs):
        print(result)