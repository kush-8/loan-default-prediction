#!/bin/bash
echo "--- Starting the Full MLOps Pipeline ---"

# Activate virtual environment
source venv/Scripts/activate 

# Step 1: Ingest Data
echo "\n--- Step 1: Ingesting Data ---"
python src/data_ingestion.py

# Step 2: Train Model
echo "\n--- Step 2: Running the Training Pipeline ---"
python src/train.py

# Step 3: Make Batch Predictions
echo "\n--- Step 3: Running Batch Predictions ---"
python src/predict.py

# Step 4: Generate Model Explanations
echo "\n--- Step 4: Generating Model Explanations ---"
python src/explain.py

echo "\n--- Full MLOps Pipeline Complete ---"