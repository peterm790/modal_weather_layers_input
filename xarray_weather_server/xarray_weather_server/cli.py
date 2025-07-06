"""
Command-line interface for xarray_weather_server.
"""

import argparse
import uvicorn
import logging
from .server import app


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="XArray Weather Server - ERA5 wind data API"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Log level (default: info)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"Starting XArray Weather Server on {args.host}:{args.port}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")
    
    # Start the server
    uvicorn.run(
        "xarray_weather_server.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main() 