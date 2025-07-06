#!/usr/bin/env python3
"""
Development server with auto-reload enabled.
"""

import uvicorn
from xarray_weather_server.server import app

if __name__ == "__main__":
    print("🚀 Starting XArray Weather Server in DEVELOPMENT mode")
    print("📂 Auto-reload: ENABLED")
    print("🌐 URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🔄 Changes to Python files will automatically restart the server")
    print("-" * 60)
    
    uvicorn.run(
        "xarray_weather_server.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 