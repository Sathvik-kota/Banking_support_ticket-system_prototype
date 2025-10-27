import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import asyncio
import time
from sentence_transformers import SentenceTransformer, util
import torch
from uuid import uuid4
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

app = FastAPI(title="Async Ticket Service (Gemini RAG)")

ticket_queue = asyncio.Queue()
results_store = {}

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

# --- MODIFIED: retrieve_context function (with fix) ---
def retrieve_context(query_text, top_k=2):
    if not memory_store or not embed_model:
        print("No memory or embed model, returning empty context.")
        return ""
    try:
        query_emb = embed_model.encode(query_text, convert_to_tensor=True)
        sims = [util.cos_sim(query_emb, item["embedding"]).item() for item in memory_store]
        
        # --- FIX: Removed the "sim > 0.5" filter ---
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        
        if not top_indices or sims[top_indices[0]] == 0.0:
            print("No context found (memory store was empty or no similarity).")
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
3. General / Non-Issue — casual greetings, unclear, or unrelated messages.


Return a single, valid JSON object with 'decision', 'reason', and 'next_actions' (as a list of strings).

New Ticket:
{ticket_text}
"""
    return prompt, (context if context else "No relevant past cases found.")

# Worker function
async def worker(worker_id: int):
    print(f"Worker {worker_id} starting...")
    if not genai or not embed_model:
        print(f"Worker {worker_id}: AI services not initialized. Worker stopping.")
        return

    # Use the correct, stable model name
    model = genai.GenerativeModel('gemini-2.5-flash')

    while True:
        try:
            ticket_id, ticket = await ticket_queue.get()
            print(f"Worker {worker_id} processing ticket {ticket_id}: {ticket.summary}")
            
            results_store[ticket_id] = {"status": "processing"}

            try:
                prompt, retrieved_context = create_rag_prompt(ticket)
                
                start_time = time.perf_counter()
                
                # Run the blocking generate_content call in a separate thread
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        response_mime_type="application/json",
                    )
                )
                
                processing_time = time.perf_counter() - start_time
                print(f"Worker {worker_id} Gemini processing time: {processing_time:.2f}s")

                result_json = json.loads(response.text)
                result_json["processing_time"] = processing_time
                result_json["retrieved_context"] = retrieved_context
                
                results_store[ticket_id] = {"status": "completed", "result": result_json}
                
                # Run blocking add_to_memory in a thread
                await asyncio.to_thread(add_to_memory, ticket.summary, response.text)
                
            except Exception as e:
                error_msg = str(e)
                print(f"Worker {worker_id} error processing {ticket_id}: {error_msg}")
                results_store[ticket_id] = {"status": "error", "detail": error_msg}
            
            finally:
                ticket_queue.task_done()
                
        except Exception as e:
            print(f"Worker {worker_id} critical error: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    print("Starting 3 workers...")
    for i in range(3):
        asyncio.create_task(worker(i))

@app.post("/async_ticket")
async def async_ticket(ticket: Ticket):
    ticket_id = str(uuid4())
    await ticket_queue.put((ticket_id, ticket))
    results_store[ticket_id] = {"status": "queued"}
    return {"ticket_id": ticket_id, "status": "queued"}

@app.get("/result/{ticket_id}")
async def get_result(ticket_id: str):
    result = results_store.get(ticket_id, {"status": "pending"})
    return result

