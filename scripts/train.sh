#!/bin/bash
echo "--- Running the training pipeline ---"

# Activate virtual environment 
source venv/Scripts/activate 

python src/train.py

echo "--- Training complete ---"