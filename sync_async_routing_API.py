from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="Ticket Routing Service (Orchestrator)")

# ---------- Ticket Model ----------
class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# ---------- Service URLs ----------
SYNC_SERVICE_URL = "http://localhost:8001/sync_ticket"    # Sync service API
ASYNC_SERVICE_URL = "http://localhost:8002/async_ticket"  # Async service API
ASYNC_RESULT_URL = "http://localhost:8002/result"         # Async service RESULT API

# ---------- Routing API ----------
@app.post("/ticket")
def route_ticket(ticket: Ticket):
    """
    Receives a ticket from frontend (Streamlit) and routes
    to Sync or Async service based on severity
    """
    try:
        if ticket.severity.lower() == "high":
            # route to Sync Service
            response = requests.post(SYNC_SERVICE_URL, json=ticket.dict())
        else:
            # route to Async Service
            response = requests.post(ASYNC_SERVICE_URL, json=ticket.dict())
        
        response.raise_for_status() # Raise an exception for 4xx/5xx errors
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to microservice: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# --- !!! NEW ENDPOINT TO FIX THE 404 ERROR !!! ---
@app.get("/result/{ticket_id}")
def get_async_result(ticket_id: str):
    """
    Forwards the result request from the frontend to the async microservice.
    """
    try:
        # Construct the full URL to the async service's result endpoint
        url = f"{ASYNC_RESULT_URL}/{ticket_id}"
        
        response = requests.get(url)
        response.raise_for_status() # Check for errors from the async service
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to async result service: {e}")
        # If the async service can't be reached
        raise HTTPException(status_code=503, detail="Async service unavailable")
# --- NEW ENDPOINT TO CLEAR MEMORY IN BOTH SERVICES ---
@app.post("/clear_memory")
def clear_all_memory():
    """Calls the clear_memory endpoint on both sync and async services."""
    sync_url = f"{SYNC_SERVICE_URL}/clear_memory"
    async_url = f"{ASYNC_SERVICE_URL}/clear_memory"
    results = {}

    # Try clearing sync memory
    try:
        print(f"Attempting to clear sync memory at {sync_url}")
        sync_resp = requests.post(sync_url, timeout=5) # Add timeout
        sync_resp.raise_for_status()
        results["sync_clear"] = sync_resp.json()
        print("Sync memory clear request successful.")
    except Exception as e:
        print(f"Error clearing sync memory: {e}")
        results["sync_clear"] = {"status": "error", "detail": str(e)}

    # Try clearing async memory
    try:
        print(f"Attempting to clear async memory at {async_url}")
        async_resp = requests.post(async_url, timeout=5) # Add timeout
        async_resp.raise_for_status()
        results["async_clear"] = async_resp.json()
        print("Async memory clear request successful.")
    except Exception as e:
        print(f"Error clearing async memory: {e}")
        results["async_clear"] = {"status": "error", "detail": str(e)}

    return results
# --- END NEW ENDPOINT ---


