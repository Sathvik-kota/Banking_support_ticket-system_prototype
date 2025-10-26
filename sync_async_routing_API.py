from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Ticket Routing Service")

# ---------- Ticket Model ----------
class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# ---------- Service URLs ----------
SYNC_SERVICE_URL = "http://localhost:8001/sync_ticket"    # Sync service API
ASYNC_SERVICE_URL = "http://localhost:8002/async_ticket"  # Async service API

# ---------- Routing API ----------
@app.post("/ticket")
def route_ticket(ticket: Ticket):
    """
    Receives a ticket from frontend (Streamlit) and routes
    to Sync or Async service based on severity
    """
    if ticket.severity.lower() == "high":
        # route to Sync Service
        response = requests.post(SYNC_SERVICE_URL, json=ticket.dict())
    else:
        # route to Async Service
        response = requests.post(ASYNC_SERVICE_URL, json=ticket.dict())
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"status": "error", "detail": "Service unavailable"}
