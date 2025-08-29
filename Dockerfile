# Use an official Python runtime as a parent image
FROM python:3.12

# Install system dependencies, including build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Set compiler environment variables for CMake
ENV CC=gcc
ENV CXX=g++

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We install torch separately to ensure we get the CPU version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly verify kokoro-tts installation
RUN python -c "import kokoro; print('kokoro from requirements.txt successfully imported')"

# Copy the content of the local src directory to the working directory
COPY . . 

# Download nltk data
RUN python -c "import nltk; nltk.download('punkt')"

# Specify the port number the container should expose
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
