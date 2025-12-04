FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

CMD ["python", "main.py"]
