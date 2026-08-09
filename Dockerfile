# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set build metadata
LABEL maintainer="kush-8"
LABEL description="Loan Default Risk Prediction API"
LABEL version="1.1.0"

# Security: run as non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --no-create-home appuser

# Set the working directory
WORKDIR /app

# Copy and install only the lean requirements for the API
# This layer is cached unless requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only the essential directories for the application
# (training data is NOT included — model is pre-trained and serialised)
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]