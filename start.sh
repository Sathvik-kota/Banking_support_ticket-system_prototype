#!/bin/bash

# Set the environment variable from the HF Secret
export GOOGLE_API_KEY=$GOOGLE_API_KEY

# Start the 3 backend FastAPI services in the background
echo "Starting Orchestrator on port 8000..."
uvicorn sync_async_routing_API:app --host 0.0.0.0 --port 8000 &

echo "Starting Sync Service on port 8001..."
uvicorn sync_path_microservice:app --host 0.0.0.0 --port 8001 &

echo "Starting Async Service on port 8002..."
uvicorn async_microservice:app --host 0.0.0.0 --port 8002 &

# --- PYTHON-BASED WAIT SCRIPT (no curl needed) ---
echo "Waiting for services to come online..."

# Create a Python script to check if ports are ready
cat > /tmp/wait_for_ports.py << 'PYTHON_SCRIPT'
import socket
import time
import sys

def wait_for_port(port, service_name, timeout=120):
    """Wait for a port to be ready."""
    print(f"Waiting for {service_name} (port {port})...", end='', flush=True)
    start_time = time.time()
    
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print(" OK!")
                return True
            
            if time.time() - start_time > timeout:
                print(f" TIMEOUT after {timeout}s")
                return False
                
            print(".", end='', flush=True)
            time.sleep(1)
            
        except Exception as e:
            print(".", end='', flush=True)
            time.sleep(1)

# Wait for all three services
wait_for_port(8000, "Orchestrator")
wait_for_port(8001, "Sync Service")
wait_for_port(8002, "Async Service")

print("All backend services are up!")
PYTHON_SCRIPT

# Run the Python wait script
python /tmp/wait_for_ports.py

# Start the Streamlit app in the foreground
echo "Starting Streamlit UI on port 8501..."
streamlit run app.py --server.port 8501 --server.headless true --server.add
