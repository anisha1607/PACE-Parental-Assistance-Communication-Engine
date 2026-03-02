import time
from typing import Dict, List
from app.llm_client import BaseLLMClient
from app.prompts import CONTEXT_DERIVATION_PROMPT

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.derived_context: str = "No previous context."
        self.last_accessed = time.time()

class SessionStore:
    def __init__(self, llm_client: BaseLLMClient, model: str):
        self.sessions: Dict[str, Session] = {}
        self.llm = llm_client
        self.model = model

    def get_session(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id)
        session = self.sessions[session_id]
        session.last_accessed = time.time()
        return session

    def update_derived_context(self, session_id: str):
        # Skip context updates for eval sessions — saves one LLM call per test case
        if session_id.startswith("eval_"):
            return
        session = self.get_session(session_id)
        if not session.messages:
            return

        # Prepare a history string for the LLM
        history = ""
        for msg in session.messages[-6:]: # Look at last 6 messages
            role = "User" if msg["role"] == "user" else "PACE"
            history += f"{role}: {msg['content']}\n"

        prompt = CONTEXT_DERIVATION_PROMPT.format(
            current_context=session.derived_context,
            history=history
        )

        try:
            new_context = self.llm.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )
            session.derived_context = new_context.strip()
        except Exception as e:
            print(f"Error updating context for session {session_id}: {e}")

    def add_message(self, session_id: str, role: str, content: str):
        session = self.get_session(session_id)
        session.messages.append({"role": role, "content": content})
        # Keep history manageable
        if len(session.messages) > 20:
            session.messages = session.messages[-20:]