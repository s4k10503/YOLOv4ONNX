# Use a lighter base image with only CUDA runtime
FROM nvidia/cuda:11.0-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install essential packages and Python
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python-headless \
    opencv-python \
    scipy \
    onnxruntime-gpu

# Copy the setup script and main application
COPY setup.py /app/
COPY main.py /app/
COPY utils/ /app/utils/

# Run the setup script to download the ONNX model
RUN python3 setup.py

# Run app.py when the container launches
CMD ["python3", "main.py"]
