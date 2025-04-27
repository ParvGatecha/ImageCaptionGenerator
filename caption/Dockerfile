# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set Hugging Face cache directory (optional, useful in K8s with shared volumes)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
# Newer versions use HF_HOME
ENV HF_HOME=/app/.cache/huggingface 

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any are needed, e.g., for OpenCV)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     apt-get clean

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Consider using --no-cache-dir to reduce image size, but it slows down builds
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create cache directory and set permissions (if needed)
# RUN mkdir -p /app/.cache/huggingface && \
#     chown -R <user>:<group> /app/.cache/huggingface # Replace <user>:<group> if running as non-root

# Copy the rest of the application code
COPY ./app /app/app
# Copy tests if you want to run them in the container (optional)
# COPY ./tests /app/tests

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use the 'standard' uvicorn install for performance
# Use 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]