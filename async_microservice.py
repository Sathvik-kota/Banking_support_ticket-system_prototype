from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import google.generativeai as genai  # Import Google Gemini
import os
import json
from uuid import uuid4

# --- Gemini API Initialization ---
try:
    # Google's SDK automatically looks for the GOOGLE_API_KEY env variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    
    # Set up the model
    generation_config = {
        "temperature": 0.3,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json", # Force JSON output
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", # Use the correct, stable model name
        generation_config=generation_config,
    )
    print("Google Gemini client initialized successfully (Async Service).")
    
except Exception as e:
    print(f"Error initializing Google Gemini client (Async Service): {e}")
    model = None
# --- End of Gemini Initialization ---


app = FastAPI(title="Async Ticket Service (Queue + Workers) - GEMINI MODE")

# In-memory queue and result store
ticket_queue = asyncio.Queue()
results_store = {}

# Ticket model
class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# Prompt builder
def create_prompt(ticket: Ticket) -> str:
    return f"""
You are an expert banking support assistant.
Classify this ticket into:
1. AI Code Patch
2. Vibe Workflow
Return a single, valid JSON object with 'decision', 'reason', and 'next_actions' (as a list of strings).
Do not return markdown (```json ... ```), just the raw JSON object.

Ticket:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}
"""

# Worker function
async def worker(worker_id: int):
    print(f"Worker {worker_id} starting...")
    if not model:
        print(f"Worker {worker_id}: Gemini client not initialized. Worker stopping.")
        return

    while True:
        try:
            ticket_id, ticket = await ticket_queue.get()
            print(f"Worker {worker_id} processing ticket {ticket_id} (GEMINI MODE)")
            
            # Store initial status
            results_store[ticket_id] = {"status": "processing"}

            try:
                prompt = create_prompt(ticket)
                
                # Run the blocking genai.generate_content call in a separate thread
                # This is CRITICAL for an async worker.
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                
                result_json = json.loads(response.text)
                
                # Store the final, successful result
                results_store[ticket_id] = {"status": "completed", "result": result_json}
                
            except Exception as e:
                print(f"Worker {worker_id} error processing {ticket_id}: {e}")
                # Store the error result
                results_store[ticket_id] = {"status": "error", "detail": str(e)}
            
            finally:
                ticket_queue.task_done()
                
        except Exception as e:
            # Error getting from queue?
            print(f"Worker {worker_id} critical error: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    # Start 3 async workers on app startup
    print("Starting 3 async workers (GEMINI MODE)...")
    for i in range(3):
        asyncio.create_task(worker(i))

# Submit ticket (non-blocking)
@app.post("/async_ticket")
async def async_ticket(ticket: Ticket):
    ticket_id = str(uuid4())
    await ticket_queue.put((ticket_id, ticket))
    results_store[ticket_id] = {"status": "queued"}
    return {"ticket_id": ticket_id, "status": "queued"}

# Get ticket result
@app.get("/result/{ticket_id}")
async def get_result(ticket_id: str):
    result = results_store.get(ticket_id, {"status": "pending"})
    return result

