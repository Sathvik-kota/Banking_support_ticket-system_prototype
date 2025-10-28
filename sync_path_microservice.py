import google.generativeai as genai
import google.api_core.exceptions
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import json
import time
from sentence_transformers import SentenceTransformer, util
import torch

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

app = FastAPI(title="Sync Ticket Service (RAG + Gemini)")

class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# --- RAG Functions ---
def add_to_memory(ticket_text, response_json):
    if not embed_model:
        print("No embed model, skipping add_to_memory.")
        return
    try:
        embedding = embed_model.encode(ticket_text, convert_to_tensor=True)
        memory_store.append({
            "text": ticket_text,
            "embedding": embedding,
            "response": response_json  # Store the full JSON string
        })
        print(f"Added to sync memory. Memory size is now: {len(memory_store)}") # Log sync memory
    except Exception as e:
        print(f"Error adding to memory: {e}")

# --- UPDATED retrieve_context function (Matches Async version) ---
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
        print(f"Retrieved {len(relevant_indices)} relevant context(s) for sync prompt") # Log sync context
        return context
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context."
# --- END UPDATED ---


# --- UPDATED build_rag_prompt (Matches Async version) ---
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
# --- END UPDATED ---

def classify_ticket_with_gemini(ticket: Ticket):
    if not genai:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")
    if not embed_model:
        # Don't crash if embed model failed, just proceed without RAG
        print("WARNING: Embed model not available, proceeding without RAG.")
        context_str = "RAG model not available."
    else:
        # 1. Retrieve context first
        context_str = retrieve_context(ticket.summary)
    
    # 2. Build the prompt
    prompt = build_rag_prompt(ticket, context_str)
    
    # 3. Call Gemini
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # --- TIMER FIX: Start timer *just* before the API call ---
        start_time = time.time()
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TICKET_SCHEMA
            )
        )
        
        # --- TIMER FIX: End timer *immediately* after the API call ---
        processing_time = time.time() - start_time
        print(f"Gemini API processing time (sync): {processing_time:.2f}s") # Log sync time
        
        # 4. Parse the JSON
        # The response.text *is* the JSON string
        result_json_str = response.text
        result_data = json.loads(result_json_str)
        
        # 5. Add to memory *after* a successful classification
        add_to_memory(ticket.summary, result_json_str)
        
        # Add the *correct* processing time and context to the result
        result_data["processing_time"] = processing_time
        result_data["retrieved_context"] = context_str
        
        return result_data

    except google.api_core.exceptions.NotFound as e:
        print(f"!!! Model not found error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini Model not found. Check model name.")
    except Exception as e:
        print(f"!!! Unexpected Error in classify_ticket (Gemini): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    print(f"Received sync ticket (GEMINI RAG MODE): {ticket.summary}")
    
    try:
        # The processing time is now correctly calculated *inside* this function
        result_data = classify_ticket_with_gemini(ticket)
        
        return result_data

    except HTTPException as e:
        # Re-raise the exception if it's one we already created
        raise e
    except Exception as e:
        print(f"--- Error processing sync ticket: {e} ---")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
@app.post("/clear_memory")
def clear_sync_memory():
    global memory_store
    print(f"Clearing sync memory (current size: {len(memory_store)}).")
    memory_store = []
    return {"status": "Sync memory cleared"}

