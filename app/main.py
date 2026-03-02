from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.bart_guard import BartGuard, REFUSAL
from app.llm_client import build_client
from app.post_guard import check_input, check_output, DistressLevel
from app.prompts import SYSTEM_PROMPT
from app.session_manager import SessionStore

load_dotenv()

app = FastAPI(title="PACE")

guard = BartGuard()

try:
    llm_client, LLM_MODEL = build_client()
except RuntimeError as e:
    raise SystemExit(f"[PACE] LLM configuration error: {e}") from e

session_store = SessionStore(llm_client, LLM_MODEL)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    situation: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    refused: bool
    guard_label: str
    guard_confidence: float
    # extra field so the frontend can optionally show a different UI for crises
    distress_level: str = "none"


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
def app_js():
    return FileResponse(WEB_DIR / "app.js")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    situation = (req.situation or "").strip()
    if not situation:
        return ChatResponse(
            response="Please describe the situation you observe (one or two sentences is enough).",
            refused=False,
            guard_label="IN_DOMAIN_COACHING",
            guard_confidence=1.0,
        )

    
    refuse, res = guard.should_refuse(situation)
    print(f"[BART GUARD] label='{res.label}' confidence={res.confidence:.2f}")
    if refuse:
        return ChatResponse(
            response=REFUSAL,
            refused=True,
            guard_label=res.label,
            guard_confidence=res.confidence,
        )

    
    backstop = check_input(situation)

    session = session_store.get_session(req.session_id)
    derived_context = session.derived_context

    user_payload = f"""
    Parent situation:
    {situation}

    Write a supportive response the parent can read and use.
    """

    if backstop.triggered and backstop.regenerate:
        
        messages = [
            {"role": "system", "content": backstop.override_system},
            {"role": "user", "content": user_payload},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(derived_context=derived_context)},
            {"role": "user", "content": user_payload},
        ]

    
    try:
        output = llm_client.chat(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.25,
            max_tokens=180,
        )
    except Exception as e:
        print(f"[LLM ERROR] ({LLM_MODEL}): {repr(e)}")
        output = "There has been an error, try again in a while!"

   
    output_check = check_output(output)
    if output_check.triggered:
        output = output_check.static_response

    
    session_store.add_message(req.session_id, "user", situation)
    session_store.add_message(req.session_id, "assistant", output.strip())
    session_store.update_derived_context(req.session_id)

    return ChatResponse(
        response=output.strip(),
        refused=False,
        guard_label=res.label,
        guard_confidence=res.confidence,
        distress_level=backstop.level.value,
    )


def run():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)