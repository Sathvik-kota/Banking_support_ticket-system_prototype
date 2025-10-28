from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
