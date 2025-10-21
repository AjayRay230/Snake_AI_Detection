#!/bin/bash
# =============================
# start.sh for Snake AI API
# =============================
# Run FastAPI with Gunicorn using 1 worker to avoid OOM
# Bind to Render's $PORT environment variable

exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker api.main:app --bind 0.0.0.0:$PORT
