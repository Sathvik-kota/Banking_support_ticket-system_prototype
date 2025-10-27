from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai  # Import Google Gemini
import os
import json
import time

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
        model_name="gemini-2.5-flash",
        generation_config=generation_config,
    )
    print("Google Gemini client initialized successfully.")
    
except Exception as e:
    print(f"Error initializing Google Gemini client: {e}")
    model = None
# --- End of Gemini Initialization ---


app = FastAPI(title="Sync Ticket Service (Threaded) - GEMINI MODE")

# Thread pool for parallel execution
executor = ThreadPoolExecutor(max_workers=3)

# Ticket model
class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# Prompt builder
def create_prompt(ticket: Ticket) -> str:
    # This prompt is updated to specifically ask for the JSON structure
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

# Call Gemini
def classify_ticket(ticket: Ticket):
    if not model:
        print("!!! Error: classify_ticket called but Gemini model is not initialized.")
        return {"status": "error", "detail": "Gemini model not initialized."}
        
    try:
        prompt = create_prompt(ticket)
        print(f"--- Sending prompt to Gemini for sync ticket ---")
        
        response = model.generate_content(prompt)
        
        print("--- Received response from Gemini ---")
        # Extract the text and parse it as JSON
        # We wrap this in a try-block in case the model
        # doesn't return valid JSON despite our instructions.
        result_json = json.loads(response.text)
        return result_json

    except Exception as e:
        # Log any unexpected error
        print(f"!!! Unexpected Error in classify_ticket (Gemini): {e}")
        # Log the raw response if available, as it might not be JSON
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"--- Raw Gemini Response: {response.text} ---")
        return {"status": "error", "detail": str(e)}

# API endpoint
@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    print(f"Received sync ticket (GEMINI MODE): {ticket.summary}")
    # Run the blocking I/O call in a separate thread
    future = executor.submit(classify_ticket, ticket)
    result = future.result()
    
    if result.get("status") == "error":
        print(f"--- Error processing sync ticket: {result.get('detail')} ---")
        raise HTTPException(status_code=500, detail=result.get("detail"))
    
    print("--- Successfully processed sync ticket (GEMINI) ---")
    return result

