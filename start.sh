#!/bin/bash

# Set the environment variable from the HF Secret
# This is crucial for your Gemini calls to work
export GOOGLE_API_KEY=$GOOGLE_API_KEY

# Start the 3 backend FastAPI services in the background
# The '&' symbol runs them as background processes.
echo "Starting Orchestrator on port 8000..."
uvicorn sync_async_routing_API:app --host 0.0.0.0 --port 8000 &

gunicorn -k uvicorn.workers.UvicornWorker sync_path_microservice:app \
  --workers 2 \
  --threads 2 \
  --bind 0.0.0.0:8001 &


echo "Starting Async Service on port 8002..."
# Using the filename from your log: async_microservice.py
uvicorn async_microservice:app --host 0.0.0.0 --port 8002 &


# Wait a bit for services to start
sleep 15

echo "Running load test..."
python load_test_sync.py  # your 25-request script

# Start the Streamlit app in the foreground
# This is the main process that will keep the container running.
# Hugging Face will route traffic to this port (8501).
echo "Starting Streamlit UI on port 8501..."
streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0

