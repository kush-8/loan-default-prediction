#!/bin-bash

# Activate virtual environment
source venv/Scripts/activate 

echo "--- Launching the FastAPI Server ---"
uvicorn src.app:app --host 0.0.0.0 --port 8000