# ---------- async_service.py ----------
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import openai
import os
import json
from uuid import uuid4

openai.api_key = os.getenv("OPENAI_API_KEY")

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
Return JSON with 'decision', 'reason', and 'next_actions'.
Ticket:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}
"""

# Worker function
async def worker(worker_id: int):
    while True:
        ticket_id, ticket = await ticket_queue.get()
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": create_prompt(ticket)}],
                temperature=0.3,
            )
            results_store[ticket_id] = json.loads(response.choices[0].message.content)
        except Exception as e:
            results_store[ticket_id] = {"status": "error", "detail": str(e)}
        ticket_queue.task_done()

# Start 3 async workers
for i in range(3):
    asyncio.create_task(worker(i))

# Submit ticket (non-blocking)
@app.post("/async_ticket")
async def async_ticket(ticket: Ticket):
    ticket_id = str(uuid4())
    await ticket_queue.put((ticket_id, ticket))
    return {"ticket_id": ticket_id, "status": "queued"}

# Get ticket result
@app.get("/result/{ticket_id}")
async def get_result(ticket_id: str):
    return results_store.get(ticket_id, {"status": "pending"})
