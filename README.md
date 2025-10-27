---
title: Streamlit FastAPI Orchestrator
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# Streamlit FastAPI Orchestrator

This Space demonstrates a Streamlit-based user interface that orchestrates three FastAPI microservices for text generation and API-based inferencing within a Docker container.

## Features

- Interactive Streamlit UI on port 8501
- Orchestrates three FastAPI services via background processes
- API endpoints for synchronous and asynchronous text generation
- Modular startup via `start.sh` script
- Designed for Hugging Face Docker Spaces deployment

## Usage

1. Clone this repository or fork your own Space
2. Add environment variables required by your services (e.g., `GOOGLE_API_KEY`)
3. Start all services via `start.sh`
4. Interact using the Streamlit web interface

## File Structure

- `app.py` : Streamlit frontend
- `sync_async_routing_API.py`, `sync_path_microservice.py`, `async_microservice.py` : FastAPI backends
- `start.sh` : Container startup script
- `requirements.txt` : Python dependencies
- `Dockerfile` : Container build recipe
