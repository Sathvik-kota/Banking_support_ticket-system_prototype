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
        print(f"Added to memory. Memory size is now: {len(memory_store)}")
    except Exception as e:
        print(f"Error adding to memory: {e}")

def retrieve_context(query_text, top_k=2):
    if not embed_model or not memory_store:
        print("No memory or embed model, returning empty context.")
        return "No relevant past cases found."
    
    try:
        query_emb = embed_model.encode(query_text, convert_to_tensor=True)
        
        # Calculate cosine similarities
        sims = [util.cos_sim(query_emb, item["embedding"]).item() for item in memory_store]
        
        # Get the indices of the top_k most similar items
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]

        # Filter out self-similarity (score > 0.99)
        # This prevents a ticket from matching itself if submitted twice
        relevant_indices = [i for i in top_indices if sims[i] < 0.99]

        if not relevant_indices:
            return "No relevant past cases found."

        # Build context string
        context = "\n\n".join([
            f"Past Ticket: {memory_store[i]['text']}\nPast Response: {memory_store[i]['response']}" 
            for i in relevant_indices
        ])
        print(f"Retrieved context for prompt: {context}")
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context."

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

def classify_ticket_with_gemini(ticket: Ticket):
    if not genai:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")

    # 1. Retrieve context first
    context_str = retrieve_context(ticket.summary)
    
    # 2. Build the prompt
    prompt = build_rag_prompt(ticket, context_str)
    
    # 3. Call Gemini
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TICKET_SCHEMA
            )
        )
        
        # 4. Parse the JSON
        # The response.text *is* the JSON string
        result_json_str = response.text
        result_data = json.loads(result_json_str)
        
        # 5. Add to memory *after* a successful classification
        add_to_memory(ticket.summary, result_json_str)
        
        return result_data, context_str

    except google.api_core.exceptions.NotFound as e:
        print(f"!!! Model not found error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini Model not found. Check model name.")
    except Exception as e:
        print(f"!!! Unexpected Error in classify_ticket (Gemini): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    print(f"Received sync ticket (GEMINI RAG MODE): {ticket.summary}")
    
    start_time = time.time()
    
    try:
        result_data, retrieved_context = classify_ticket_with_gemini(ticket)
        
        processing_time = time.time() - start_time
        
        # Add the new fields to the response
        result_data["processing_time"] = processing_time
        result_data["retrieved_context"] = retrieved_context
        
        return result_data

    except HTTPException as e:
        # Re-raise the exception if it's one we already created
        raise e
    except Exception as e:
        print(f"--- Error processing sync ticket: {e} ---")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

