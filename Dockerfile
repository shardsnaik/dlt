# FROM python:3.10-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     DEBIAN_FRONTEND=noninteractive

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     gcc \
#     g++ \
#     wget \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*


# # Copy requirements first (for better caching)
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# # Copy the application code
# COPY medical_bot_main_file.py .
# COPY medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf .

# # Copy the model file (if it exists locally)
# # COPY model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf /app/model/

# # Alternative: Download model during build (uncomment if needed)
# # RUN wget -O /app/model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf \
# #     "YOUR_MODEL_DOWNLOAD_URL_HERE"

# # Create a non-root user for security
# RUN groupadd -r appuser && useradd -r -g appuser appuser
# RUN chown -R appuser:appuser /app
# USER appuser

# # Expose the port
# EXPOSE 8000

# # Command to run the application
# # CMD ["python" ,"medical_bot_main_file.py"]
# CMD ["uvicorn", "medical_bot_main_file:app", "--host", "0.0.0.0", "--port", "8000"]


# For Local running 

# # Use Python 3.12 slim image
# FROM python:3.12-slim

# # Set working directory inside container
# WORKDIR /app

# # Install system dependencies (for llama.cpp)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     python3-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Install llama-cpp-python with CPU support
# RUN CMAKE_ARGS="-DLLAMA_CUBLAS=OFF" pip install llama-cpp-python

# # Copy application files
# COPY medical_bot_main_file.py .
# COPY medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf .

# # Expose FastAPI port
# EXPOSE 8000

# # Run FastAPI with uvicorn
# CMD ["uvicorn", "medical_bot_main_file:app", "--host", "0.0.0.0", "--port", "8000"]


# For AWS Lambda running 
# Use AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for llama-cpp-python
RUN dnf install -y gcc gcc-c++ git cmake make wget unzip \
    && dnf clean all

# Copy requirements first for caching
COPY requirements.txt  .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "uvicorn[standard]" fastapi

# Install llama-cpp-python from PyPI (Linux build)
RUN pip install llama-cpp-python

# Copy application code and model
COPY medical_bot_main_file.py ${LAMBDA_TASK_ROOT}/
COPY medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf ${LAMBDA_TASK_ROOT}/

# Set environment variables
ENV MODEL_PATH=${LAMBDA_TASK_ROOT}/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf

# Add Mangum for AWS Lambda
RUN pip install mangum

# Set Lambda handler (FastAPI app inside medical_bot_main_file.py)
CMD ["medical_bot_main_file.app"]
