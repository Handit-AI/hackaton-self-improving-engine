#!/bin/bash
set -e

# Log startup
echo "Starting application..."

# Run the application
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}

