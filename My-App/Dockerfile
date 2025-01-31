FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-python-dev \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies, including dlib
RUN pip install dlib

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install remaining Python dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy application code
COPY app/ ./app

# Expose application port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]


# #FROM python:3.9-slim
# FROM datalorax/dlib:latest

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Install dependencies
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     gcc \
# #     libpq-dev && \
# #     apt-get clean && rm -rf /var/lib/apt/lists/*

# # # Install system dependencies for dlib and other tools
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     build-essential \
# #     cmake \
# #     libopenblas-dev \
# #     liblapack-dev \
# #     libx11-dev \
# #     libgtk2.0-dev \
# #     libboost-python-dev \
# #     libboost-all-dev \
# #     && apt-get clean \
# #     && rm -rf /var/lib/apt/lists/*

# # Upgrade pip to the latest version
# RUN pip install --upgrade pip

# # Set working directory
# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt
# #
# #RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100 -r requirements.txt


# # Copy app code
# COPY app/ ./app

# CMD ["pytest", "--maxfail=3", "--disable-warnings", "-v"]

# # Expose port
# EXPOSE 8000

# # Run the app
# CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]

