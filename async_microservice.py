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

memory_store = []  # In-memory store for RAG - STARTS EMPTY, NO EXAMPLES
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
    """Add a ticket and its response to memory, with noise filtering."""
    if not embed_model:
        print("No embed model, skipping add_to_memory.")
        return
    
    # Skip storing vague/empty tickets - NOISE FILTER
    ticket_lower = ticket_text.strip().lower()
    if (len(ticket_text.strip()) < 10 or 
        ticket_lower in ['hi', 'hello', 'hey', 'test', 'hi there', 'hello there'] or
        ticket_text.startswith("Example:")):  # Explicitly block "Example:" entries
        print(f"Skipping storage of vague/example ticket: '{ticket_text}'")
        return
        
    try:
        embedding = embed_model.encode(ticket_text, convert_to_tensor=True)
        memory_store.append({
            "text": ticket_text,
            "embedding": embedding,
            "response": response_json
        })
        print(f"Added to memory. Memory size is now: {len(memory_store)}")
    except Exception as e:
        print(f"Error adding to memory: {e}")

def retrieve_context(query_text, top_k=2):
    """Retrieve relevant past tickets based on semantic similarity."""
    if not embed_model or not memory_store:
        print("No memory or embed model, returning empty context.")
        return "No relevant past cases found."
    
    try:
        # Encode the query
        query_emb = embed_model.encode(query_text, convert_to_tensor=True)
        
        # Filter out example/demo tickets BEFORE similarity calculation
        filtered_store = [
            item for item in memory_store 
            if not item["text"].startswith("Example:")
        ]
        
        if not filtered_store:
            print("No valid memories after filtering examples.")
            return "No relevant past cases found."
        
        # Calculate similarities
        sims = [util.cos_sim(query_emb, item["embedding"]).item() for item in filtered_store]
        
        # Log the raw scores for debugging
        print(f"Raw similarity scores for '{query_text}': {[f'{s:.3f}' for s in sims]}")

        # Get ALL indices sorted by similarity
        all_indices_sorted = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

        # Filter FIRST by threshold (>=0.90 and <0.99), then take top_k
        relevant_indices = [
            i for i in all_indices_sorted
            if sims[i] >= 0.90 and sims[i] < 0.99
        ][:top_k]

        if not relevant_indices:
            best_score = max(sims) if sims else 0.0
            print(f"No context found above 90% similarity. Best score: {best_score:.3f}")
            return "No relevant past cases found."

        # Build context string with similarity scores for transparency
        context_parts = []
        for i in relevant_indices:
            context_parts.append(
                f"Past Ticket (similarity: {sims[i]:.2f}): {filtered_store[i]['text']}\n"
                f"Past Response: {filtered_store[i]['response']}"
            )
        
        context = "\n\n".join(context_parts)
        print(f"Retrieved {len(relevant_indices)} relevant context(s) for prompt")
        return context
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context."

def build_rag_prompt(ticket: Ticket, context: str) -> str:
    """Build the prompt for Gemini with RAG context."""
    return f"""You are an expert banking support assistant. Your job is to classify a new ticket.
You must choose one of three categories:
1. AI Code Patch: Select this for technical bugs, API errors, code-related problems, or system failures.
2. Vibe Workflow: Select this for standard customer requests (e.g., "unblock my card," "payment failed," "reset password," or general banking inquiries).
3. Unknown: Select this for random, vague, or irrelevant tickets (e.g., messages like "hi", "hello", or non-descriptive/empty queries).

Use the following past cases as context ONLY if they are relevant:
---
{context}
---

CRITICAL INSTRUCTIONS:
- If the retrieval context is irrelevant or noisy, IGNORE IT completely and focus ONLY on the current ticket.
- Do NOT guess if information is missing or unclear.
- If the ticket summary is vague, incomplete, or just a greeting, classify as "Unknown".
- Base your decision primarily on the NEW TICKET information, not the past context.

Now classify this new ticket. Return only valid JSON.

New Ticket:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}
"""

def classify_ticket_with_gemini(ticket: Ticket):
    """Classify a ticket using Gemini with RAG context."""
    if not genai:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    if not embed_model:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    
    try:
        # 1. Retrieve context from memory
        context_str = retrieve_context(ticket.summary)
        
        # 2. Build the prompt with context
        prompt = build_rag_prompt(ticket, context_str)
        
        # 3. Call Gemini API
        start_time = time.time()
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=TICKET_SCHEMA
            )
        )
        processing_time = time.time() - start_time
        print(f"Gemini API processing time: {processing_time:.2f}s")
        
        # 4. Parse the response
        result_json_str = response.text
        result_data = json.loads(result_json_str)
        
        # 5. Add to memory for future queries
        add_to_memory(ticket.summary, result_json_str)
        
        # 6. Add metadata to response
        result_data["processing_time"] = processing_time
        result_data["retrieved_context"] = context_str
        
        return result_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Gemini response: {str(e)}")
    except google.api_core.exceptions.GoogleAPIError as e:
        print(f"Gemini API error: {e}")
        raise HTTPException(status_code=503, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in classify_ticket: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

# --- API Endpoints ---
@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    """Synchronous endpoint for ticket classification with RAG."""
    print(f"Received sync ticket (GEMINI RAG MODE): {ticket.summary}")
    result = classify_ticket_with_gemini(ticket)
    return result

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gemini_initialized": genai is not None,
        "embedding_model_initialized": embed_model is not None,
        "memory_size": len(memory_store)
    }

@app.get("/debug/memory")
def debug_memory():
    """Debug endpoint to inspect memory store."""
    return {
        "memory_size": len(memory_store),
        "entries": [
            {
                "text": item["text"],
                "response_preview": item["response"][:150] + "..." if len(item["response"]) > 150 else item["response"]
            }
            for item in memory_store
        ]
    }

@app.post("/clear_memory")
def clear_memory():
    """Clear all entries from memory store."""
    global memory_store
    old_size = len(memory_store)
    memory_store = []
    return {"message": f"Memory cleared. Removed {old_size} entries."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
