import google.generativeai as genai
import google.api_core.exceptions
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import os
import json
import time
from sentence_transformers import SentenceTransformer, util
import torch
import asyncio
# Removed: import asyncio.to_thread # Use this for running blocking code in async
from uuid import uuid4

# --- RAG Memory (Global for the service) ---
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load SentenceTransformer model: {e}")
    embed_model = None

memory_store = []  # In-memory store for RAG
# -------------------------------------------

# --- Gemini Configuration ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    print("Google Gemini client initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize Gemini client: {e}")
    genai = None

# Define the exact JSON structure we want Gemini to return
class TicketResponse(BaseModel):
    decision: str = Field(description="The classification category.")
    reason: str = Field(description="A brief reason for the decision.")
    next_actions: list[str] = Field(description="A list of next actions.")

TICKET_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING"},
        "reason": {"type": "STRING"},
        "next_actions": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    },
    "required": ["decision", "reason", "next_actions"]
}
# -----------------------------

app = FastAPI(title="Async Ticket Service (RAG + Gemini)")

class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# In-memory queue and result store
ticket_queue = asyncio.Queue()
results_store = {}

# --- RAG Functions (must be sync, will be called in a thread) ---
def add_to_memory(ticket_text, response_json):
    if not embed_model:
        print("No embed model, skipping add_to_memory.")
        return
    try:
        # Note: encode() is a blocking CPU-bound operation
        embedding = embed_model.encode(ticket_text, convert_to_tensor=True)
        memory_store.append({
            "text": ticket_text,
            "embedding": embedding,
            "response": response_json  # Store the full JSON string
        })
        print(f"Added to async memory. Memory size: {len(memory_store)}")
    except Exception as e:
        print(f"Error adding to memory: {e}")

# --- UPDATED retrieve_context function ---
def retrieve_context(query_text, top_k=2):
    if not embed_model or not memory_store:
        print("No memory or embed model, returning empty context.")
        return "No relevant past cases found."
    
    try:
        # Encode the query
        query_emb = embed_model.encode(query_text, convert_to_tensor=True)
        
        # Calculate similarities
        sims = [util.cos_sim(query_emb, item["embedding"]).item() for item in memory_store]
        
        # Log the raw scores for debugging
        print(f"Raw similarity scores for '{query_text}': {sims}")

        # Get ALL indices sorted by similarity (not just top_k)
        all_indices_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

        # Filter FIRST, then take top_k from filtered results
        # This ensures we only consider truly relevant cases
        relevant_indices = [
            i for i in all_indices_sorted
            if sims[i] >= 0.70 and sims[i] < 0.99  # Strict similarity threshold
        ][:top_k]  # Take only top_k AFTER filtering

        if not relevant_indices:
            print(f"No context found above 90% similarity threshold. Best score was: {max(sims) if sims else 'N/A'}")
            return "No relevant past cases found."

        # Build context string with similarity scores for transparency
        context_parts = []
        for i in relevant_indices:
            context_parts.append(
                f"Past Ticket (similarity: {sims[i]:.2f}): {memory_store[i]['text']}\n"
                f"Past Response: {memory_store[i]['response']}"
            )
        
        context = "\n\n".join(context_parts)
        print(f"Retrieved {len(relevant_indices)} relevant context(s) for async prompt")
        return context
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context."
# --- END UPDATED ---

# --- THIS IS THE NEW, BETTER PROMPT ---
def build_rag_prompt(ticket: Ticket, context: str) -> str:
    return f"""
You are an expert banking support assistant. Your job is to classify a new ticket.
You must choose one of three categories:
1.  AI Code Patch: Select this for technical bugs, API errors, code-related problems, or system failures.
2.  Vibe Workflow: Select this for standard customer requests (e.g., "unblock my card," "payment failed," "reset password," or general banking inquiries).
3.  Unknown: Select this for random, vague, or irrelevant tickets (e.g., messages like "hi", "hello", or non-descriptive/empty queries).

Use the following past cases as context if relevant:
---
{context}
---
Important Instructions:
- If the retrieval context is irrelevant or noisy, ignore it and focus only on the provided ticket information.
- Do NOT guess if any information is missing or unclear.
- If information is insufficient, respond with the category "Unknown" with a clear reason.

Now classify this new ticket. Return only the valid JSON response.
New Ticket:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}
"""
# ----------------------------------------

async def classify_ticket_with_gemini_async(ticket: Ticket):
    if not genai:
        print("Worker error: Gemini client not initialized.")
        return {"error": "Gemini client not initialized"}, "Gemini client not initialized", 0.0 # Return 3 values

    try:
        # 1. Retrieve context (blocking, run in thread)
        context_str = await asyncio.to_thread(retrieve_context, ticket.summary)
        
        # 2. Build the prompt
        prompt = build_rag_prompt(ticket, context_str)
        
        # 3. Call Gemini (blocking, run in thread)
        def gemini_call():
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TICKET_SCHEMA
                )
            )
            return response.text

        # --- TIMER FIX: Start timer *just* before the API call thread ---
        start_time = time.time()
            
        result_json_str = await asyncio.to_thread(gemini_call)
        
        # --- TIMER FIX: End timer *immediately* after the API call thread ---
        processing_time = time.time() - start_time
        print(f"Gemini API processing time (async): {processing_time:.2f}s")
        
        # 4. Parse the JSON
        result_data = json.loads(result_json_str)
        
        # 5. Add to memory *after* (blocking, run in thread)
        await asyncio.to_thread(add_to_memory, ticket.summary, result_json_str)
        
        # Add the *correct* processing time and context to the result
        result_data["processing_time"] = processing_time
        result_data["retrieved_context"] = context_str
        
        # --- FIX: Return 3 values on success ---
        return result_data, None, processing_time 

    except Exception as e:
        print(f"!!! Unexpected Error in async classify_ticket (Gemini): {e}")
        # Return 3 values on error
        return {"error": str(e)}, str(e), 0.0


# Worker function
async def worker(worker_id: int):
    print(f"Worker {worker_id} starting...")
    if not genai or not embed_model:
        print(f"Worker {worker_id}: Client not initialized. Worker stopping.")
        return

    while True:
        try:
            ticket_id, ticket = await ticket_queue.get()
            print(f"Worker {worker_id} processing ticket {ticket_id}: {ticket.summary}")
            
            results_store[ticket_id] = {"status": "processing"}

            try:
                # Unpack the tuple (now always 3 values)
                result_data, error_detail, processing_time = await classify_ticket_with_gemini_async(ticket) 
                
                if error_detail:
                    results_store[ticket_id] = {"status": "error", "detail": error_detail}
                else:
                    results_store[ticket_id] = {"status": "completed", "result": result_data}
                
            except Exception as e:
                print(f"Worker {worker_id} error processing {ticket_id}: {e}")
                results_store[ticket_id] = {"status": "error", "detail": str(e)}
            
            finally:
                ticket_queue.task_done()
                
        except Exception as e:
            print(f"Worker {worker_id} critical error: {e}")
            await asyncio.sleep(1) # Prevent tight loop on critical error


@app.on_event("startup")
async def startup_event():
    print("Starting 3 workers...")
    for i in range(3):
        asyncio.create_task(worker(i))

# Submit ticket (non-blocking)
@app.post("/async_ticket")
async def async_ticket(ticket: Ticket):
    ticket_id = str(uuid4())
    await ticket_queue.put((ticket_id, ticket))
    results_store[ticket_id] = {"status": "queued"}

    # --- POLLING LOOP: check every 2 seconds (you can tune this) ---
    for _ in range(5):  # Try up to 5 times → total 10 seconds
        await asyncio.sleep(2)
        current_status = results_store.get(ticket_id, {}).get("status")
        print(f"⏳ Polling... current status = {current_status}")
        
        if current_status == "completed":
            return {
                "ticket_id": ticket_id,
                "status": "completed",
                "result": results_store[ticket_id]["result"]
            }
        elif current_status == "error":
            return {
                "ticket_id": ticket_id,
                "status": "error",
                "detail": results_store[ticket_id].get("detail", "Unknown error")
            }

    # If still not done after polling
    return {
        "ticket_id": ticket_id,
        "status": "processing",
        "message": "Still processing, please check /result/{ticket_id} later."
    }


# Get ticket result
@app.get("/result/{ticket_id}")
async def get_result(ticket_id: str):
    result = results_store.get(ticket_id, {"status": "pending"})
    return result

