# Base image with Python
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy your training script and requirements
COPY train.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the entry point for training
CMD ["python", "train.py"]
