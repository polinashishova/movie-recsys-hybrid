"""
Script for running the API application.

Usage example:
cd project
python scripts/run_api.py
"""

import argparse
import sys
from pathlib import Path
from mrh.config import APP_HOST, APP_PORT

import uvicorn


def main():
    """
    Run UVicorn server for the Movie RecSys API.
    
    Supports command line arguments for configuring host, port,
    development mode, and number of workers.
    """
    
    parser = argparse.ArgumentParser(description="Run movie recommendations API server")
    parser.add_argument("--host", default=APP_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=APP_PORT, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development only!)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (>=1 for production)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    display_host = "localhost" if args.host in ["0.0.0.0", ""] else args.host

    print(f"Starting server at http://{display_host}:{args.port}")
    print(f"Swagger UI:       http://{display_host}:{args.port}/docs")
    print(f"Health Check:     http://{display_host}:{args.port}/health")
    print(f"Reload mode:      {'ON' if args.reload else 'OFF'}")

    uvicorn.run(
        "mrh.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()