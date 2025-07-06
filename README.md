# ERA5 Wind PNG Modal Function

Minimal Modal function that generates WeatherLayers-compatible PNG images from ERA5 wind data.

## How it Works

- **Input**: Date/time as `YYYYMMDDHH` format (e.g., `2024011512` = Jan 15, 2024 12:00 UTC)
- **Output**: PNG with wind data encoded as Red=U component, Green=V component, Blue=wind speed
- **Data**: ERA5 global wind data from Google Cloud (0.25° resolution)

## Setup & Deploy

```bash
# Install dependencies
micromamba env create -f environment.yml
micromamba activate era5-modal

# Test locally
modal run modal_ERA5_to_png_server.py

# Deploy to production
modal deploy modal_ERA5_to_png_server.py
```

## API Usage

### cURL Example
```bash
curl "https://your-modal-url.com/api_get_wind_png?date_hour=2024011512" \
  -o wind.png
```

### React Example
```javascript
const fetchWindPNG = async (dateHour) => {
  const response = await fetch(
    `https://your-modal-url.com/api_get_wind_png?date_hour=${dateHour}`
  );
  
  if (response.ok) {
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    return url; // Use as <img src={url} />
  }
  throw new Error('Failed to fetch wind data');
};

// Usage
const windImageUrl = await fetchWindPNG('2024011512');
```

## Date Format

- `YYYYMMDDHH` - 4-digit year, 2-digit month, 2-digit day, 2-digit hour (UTC)
- Examples: `2024011512` (Jan 15, 2024 12:00 UTC), `2024123100` (Dec 31, 2024 00:00 UTC) 