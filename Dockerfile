# Use a slim Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install torch-scatter and torch-geometric with the correct PyTorch version
# Replace 'cpu' with your CUDA version if needed (e.g., 'cu118' for CUDA 11.8)
RUN pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
RUN pip install torch-sparse
RUN pip install torch-geometric==2.1.0

# Copy application code and models
COPY . /app

# Expose port
EXPOSE 5000

# Install Gunicorn
RUN pip install gunicorn

# Start the application with Gunicorn
CMD ["gunicorn", "smartAD:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "360"]