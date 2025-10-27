# ---------- sync_service.py ----------
from fastapi import FastAPI
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Sync Ticket Service (Threaded)")

# Thread pool for parallel execution
executor = ThreadPoolExecutor(max_workers=3)  # 3 threads for demo

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

# Call OpenAI
def classify_ticket(ticket: Ticket):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": create_prompt(ticket)}],
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# API endpoint
@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    future = executor.submit(classify_ticket, ticket)
    return future.result()
