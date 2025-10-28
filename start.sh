#!/bin/bash

# Set the environment variable from the HF Secret
# This is crucial for your Gemini calls to work
export GOOGLE_API_KEY=$GOOGLE_API_KEY

# Start the 3 backend FastAPI services in the background
# The '&' symbol runs them as background processes.
echo "Starting Orchestrator on port 8000..."
uvicorn sync_async_routing_API:app --host 0.0.0.0 --port 8000 &

echo "Starting Sync Service on port 8001..."
# Using the filename from your log: sync_path_microservice.py
uvicorn sync_path_microservice:app --host 0.0.0.0 --port 8001 &


echo "Starting Async Service on port 8002..."
# Using the filename from your log: async_microservice.py
uvicorn async_microservice:app --host 0.0.0.0 --port 8002 &

# --- ROBUST WAIT SCRIPT ---
# Instead of a fixed 'sleep 10', we will actively wait for each
# service to be online and responding to requests.

echo "Waiting for services to come online..."

# Function to wait for a port to be ready
wait_for_port() {
  local port=$1
  local service_name=$2
  echo -n "Waiting for $service_name (port $port)..."
  
  # We use curl to poll the service.
  # We loop until curl returns a successful status (exit code 0).
  # '--silent' hides progress
  # '--head' just gets headers, not the full page
  # '--fail' makes curl exit with an error code on HTTP errors (like 404, 503)
  while ! curl -s --head --fail "http://localhost:$port" > /dev/null; do
    echo -n "."
    sleep 1
  done
  echo " OK!"
}

# Wait for all three services
wait_for_port 8000 "Orchestrator"
wait_for_port 8001 "Sync Service"
wait_for_port 8002 "Async Service"

echo "All backend services are up!"
# --- END OF WAIT SCRIPT ---


# The 'sleep 10' is no longer needed because we have the robust wait script above.
# sleep 10


# This load test is what's generating your errors.
# You probably don't want to run this every time your app starts in production.
# I've commented it out. You can re-enable it for testing.
# echo "Running load test..."
# python load_test_sync.py


# Start the Streamlit app in the foreground
# This is the main process that will keep the container running.
# Hugging Face will route traffic to this port (8501).
echo "Starting Streamlit UI on port 8501..."
streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
