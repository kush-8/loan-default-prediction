#!/bin/bash
echo "--- Running batch prediction ---"

# Activate virtual environment
source venv/Scripts/activate 

python src/predict.py

echo "--- Prediction complete. See results.csv ---"