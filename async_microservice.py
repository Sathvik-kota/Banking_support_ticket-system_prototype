from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI  # Import modern async client
import os
import json
from uuid import uuid4

# Initialize the Async OpenAI client
try:
    client = AsyncOpenAI()
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
except Exception as e:
    print(f"Error initializing AsyncOpenAI client: {e}")
    client = None

app = FastAPI(title="Async Ticket Service (Queue + Workers)")

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
Ticket:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}
"""

# Worker function
async def worker(worker_id: int):
    print(f"Worker {worker_id} starting...")
    if not client:
        print(f"Worker {worker_id}: OpenAI client not initialized. Worker stopping.")
        return

    while True:
        try:
            ticket_id, ticket = await ticket_queue.get()
            print(f"Worker {worker_id} processing ticket {ticket_id}")
            
            # Store initial status
            results_store[ticket_id] = {"status": "processing"}

            try:
                prompt = create_prompt(ticket)
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                result_json = json.loads(response.choices[0].message.content)
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
    print("Starting 3 workers...")
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
