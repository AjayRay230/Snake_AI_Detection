#!/bin/bash

# Exit on error
set -e

echo "Starting Snake AI Detection API..."

# Run FastAPI using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 7860 --workers 1
