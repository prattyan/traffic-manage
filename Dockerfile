# Use an official lightweight Python image
FROM python:3.9-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV and GLib
# (This is crucial for computer vision apps to avoid "libGL.so missing" errors)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow separately to avoid version conflicts
RUN pip install tensorflow

# Copy the rest of the application code
COPY . .

# Expose the port that Dash runs on (default is 8050)
EXPOSE 8050

# Command to run the application
# (Note: Check if the main file is 'app.py' or 'main.py'. I assumed 'app.py' below)
CMD ["python", "main.py"]