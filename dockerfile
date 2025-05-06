# Base image with Python
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy your training script and requirements
COPY train.py .
COPY train.csv .
COPY requirements.txt .
COPY test.csv .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt


# Define the entry point for training
CMD ["python", "train.py"]