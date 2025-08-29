# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install only the lean requirements for the API
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only the essential directories for the application
COPY src/ ./src
COPY config/ ./config
COPY models/ ./models

# Copy the specific data files needed by the API
COPY data/raw/application_train.csv ./data/raw/application_train.csv
COPY data/raw/HomeCredit_columns_description.csv ./data/raw/HomeCredit_columns_description.csv

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application when the container starts
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]