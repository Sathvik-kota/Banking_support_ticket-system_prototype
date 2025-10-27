from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AuthenticationError  # Import AuthenticationError
import os
import json

# Initialize the OpenAI client
# It automatically reads the OPENAI_API_KEY from the environment
try:
    client = OpenAI()
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

app = FastAPI(title="Sync Ticket Service (Threaded)")

# Thread pool for parallel execution
executor = ThreadPoolExecutor(max_workers=3)

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

# Call OpenAI
def classify_ticket(ticket: Ticket):
    if not client:
        print("!!! Error: classify_ticket called but OpenAI client is not initialized.")
        return {"status": "error", "detail": "OpenAI client not initialized."}
        
    try:
        prompt = create_prompt(ticket)
        print(f"--- Sending prompt to OpenAI for sync ticket ---")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"} # Enforce JSON output
        )
        print("--- Received response from OpenAI ---")
        result_json = json.loads(response.choices[0].message.content)
        return result_json

    except AuthenticationError as e:
        # This is the most likely error. It will now be logged clearly.
        print(f"!!! OpenAI Authentication Error: {e}")
        return {"status": "error", "detail": f"OpenAI Authentication Error. Check your API key. Error: {e}"}
        
    except Exception as e:
        # Log any other unexpected error
        print(f"!!! Unexpected Error in classify_ticket: {e}")
        return {"status": "error", "detail": str(e)}

# API endpoint
@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    print(f"Received sync ticket: {ticket.summary}")
    # Run the blocking I/O call in a separate thread
    future = executor.submit(classify_ticket, ticket)
    result = future.result()
    
    if result.get("status") == "error":
        print(f"--- Error processing sync ticket: {result.get('detail')} ---")
        raise HTTPException(status_code=500, detail=result.get("detail"))
    
    print("--- Successfully processed sync ticket ---")
    return result
