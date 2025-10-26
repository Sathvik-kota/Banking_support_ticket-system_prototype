from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

# ---------- Set your OpenAI API key ----------
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly for testing

app = FastAPI(title="Sync Ticket Service")

# ---------- Ticket Model ----------
class Ticket(BaseModel):
    channel: str
    severity: str
    summary: str

# ---------- Prompt Function ----------
def create_prompt(ticket: Ticket) -> str:
    return f"""
You are an expert banking support assistant.

Classify incoming support tickets into:
1. AI-generated code remediation → requires writing or fixing code
2. Vibe-script troubleshooting workflow → requires following a scripted procedure

Analyze the ticket and provide:
- decision: "AI Code Patch" or "Vibe Workflow"
- reasoning (1–2 sentences)
- 2–3 next actions

Ticket Details:
Channel: {ticket.channel}
Severity: {ticket.severity}
Summary: {ticket.summary}

Respond in JSON format like this:
{{
  "decision": "<AI Code Patch or Vibe Workflow>",
  "reason": "<short reasoning>",
  "next_actions": ["<action 1>", "<action 2>", "<action 3>"]
}}
"""

# ---------- Sync Endpoint ----------
@app.post("/sync_ticket")
def sync_ticket(ticket: Ticket):
    prompt = create_prompt(ticket)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        result_text = response.choices[0].message.content
        # Convert string response to JSON
        import json
        result_json = json.loads(result_text)
        return result_json
    except Exception as e:
        return {"status": "error", "detail": str(e)}
