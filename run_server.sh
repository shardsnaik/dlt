#!/bin/bash
set -e

echo "Starting Medical Bot API..."
echo "Model loading... This may take a moment."

# Check if model file exists
if [ ! -f "/app/model/medgemma-4b-it-finnetunned-merged_new_for_cpu_q5_k_m.gguf" ]; then
    echo "ERROR: Model file not found!"
    exit 1
fi

# Start the application with proper logging
exec python -m uvicorn medical_bot_main_file:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log