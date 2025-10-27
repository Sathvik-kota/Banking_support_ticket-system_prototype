import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import time
from sentence_transformers import SentenceTransformer, util
import torch

# Configure the Gemini client
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    print("Google Gemini client initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Gemini client: {e}")
    genai = None

# ADD RAG MODEL + MEMORY
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    memory_store = []  # List of {"text": "...", "embedding": tensor, "response": "..."}
    print("SentenceTransformer model loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    embed_model = None

app = FastAPI(title="Sync Ticket Service (Gemini RAG)")

class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# RAG HELPER FUNCTIONS
def add_to_memory(ticket_text, response_text):
    if not embed_model:
        print("Embed model not loaded, skipping memory add.")
        return
    try:
        embedding = embed_model.encode(ticket_text, convert_to_tensor=True)
        memory_store.append({"text": ticket_text, "embedding": embedding, "response": response_text})
        print(f"Added to memory. New memory size: {len(memory_store)}")
    except Exception as e:
        print(f"Error adding to memory: {e}")

# --- MODIFIED: retrieve_context function ---
def retrieve_context(query_text, top_k=2):
    if not memory_store or not embed_model:
        print("No memory or embed model, returning empty context.")
        return ""
    try:
        query_emb = embed_model.encode(query_text, convert_to_tensor=True)
        sims = [util.cos_sim(query_emb, item["embedding"]).item() for item in memory_store]
        
        # --- FIX: Removed the "sim > 0.5" filter ---
        # This now just finds the top_k indices, regardless of score.
        # This is better for a prototype so you always see what it's retrieving.
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        
        if not top_indices:
            print("No context found (memory store was empty).")
            return ""
            
        print(f"Top similarity scores found: {[sims[i] for i in top_indices]}")
        context = "\n\n".join([f"Past Ticket: {memory_store[i]['text']}\nResponse: {memory_store[i]['response']}" for i in top_indices])
        print(f"Retrieved context: {context}")
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""
# --- END MODIFIED ---

def create_rag_prompt(ticket: Ticket):
    """Creates the Gemini prompt and returns the prompt AND the context."""
    context = retrieve_context(ticket.summary)
    ticket_text = f"Channel: {ticket.channel}, Severity: {ticket.severity}, Summary: {ticket.summary}"
    
    prompt = f"""
You are an expert banking support assistant.

Use the following past cases as context if relevant:
---
{context if context else "No relevant past cases found."}
---

Now classify this new ticket into:
1. AI Code Patch
2. Vibe Workflow

Return a single, valid JSON object with 'decision', 'reason', and 'next_actions' (as a list of strings).

New Ticket:
{ticket_text}
"""
    return prompt, (context if context else "No relevant past cases found.")

@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    if not genai or not embed_model:
        raise HTTPException(status_code=503, detail="AI service not initialized")

    print(f"Received sync ticket (GEMINI RAG MODE): {ticket.summary}")
    
    try:
        prompt, retrieved_context = create_rag_prompt(ticket)
        print("--- Sending prompt to Gemini for sync ticket ---")
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        start_time = time.perf_counter()
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        
        processing_time = time.perf_counter() - start_time
        print(f"Gemini sync processing time: {processing_time:.2f}s")
        
        result_json = json.loads(response.text)
        result_json["processing_time"] = processing_time
        result_json["retrieved_context"] = retrieved_context
        
        add_to_memory(ticket.summary, response.text)
        
        return result_json

    except Exception as e:
        error_msg = str(e)
        print(f"!!! Unexpected Error in classify_ticket (Gemini): {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

