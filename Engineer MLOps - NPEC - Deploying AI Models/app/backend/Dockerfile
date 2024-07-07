# Use a base image with Python installed, specify the version for compatibility
# FROM python:3.10.11-slim as builder
FROM python:3.10.11-slim
# Set the working directory inside the image
WORKDIR /app/backend/ATHENA
# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libc6-dev \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \     
    libglib2.0-0 \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*
# Install Poetry at a fixed version to ensure consistent builds
# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/backend/ATHENA/
 
# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ENV PATH="/root/.local/bin:$PATH"

# Start a new build stage to minimize the final image size
# FROM python:3.10.11-slim as runtime
# Set the working directory inside the image
# WORKDIR /app/backend/ATHENA

# ENV PATH="/root/.local/bin:$PATH"

# Copy installed packages and the application code from the builder stage
# COPY --from=builder /app /app
# Copy the application code to the working directory, excluding the data directory
COPY . /app/backend

# Ensure the application code is executable
RUN chmod -R 755 /app
# Inform Docker that the container listens on the specified port at runtime
EXPOSE 3983
# Define the command to run your application using uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3983"]
# CMD ['fastapi', 'run', 'app/app.py', '--port', '80']
# Build and run commands
# docker build -t athena-backend .
# docker run -p 8000:8000 athena-backend