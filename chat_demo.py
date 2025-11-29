# chat_demo.py
import os
import json
import requests

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")

# ---------- Models / Ollama ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# IMPORTANT: set this to the model you actually pulled.
# e.g. "llama3.2:1b" or "llama3.1:3b"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

# ---------- System prompt (persona + rules) ----------
SYSTEM_PROMPT = """
You are CeylonTrip, an AI travel assistant specialized ONLY in Sri Lanka.

RULES:
- You MUST answer ONLY questions related to travel in Sri Lanka.
- If the user asks about another country (India, Thailand, etc.), reply:
  "I can only answer questions about traveling in Sri Lanka."
- Use the provided CONTEXT when possible (destinations, routes, tips).
- Do NOT invent live prices, real-time schedules, or current weather.
- If something is not covered in the context, say you are not sure and suggest
  what the traveler can check locally (guesthouses, official sites, operators).
- Keep itineraries geographically sensible and mention approximate travel times when relevant.
- Prefer clear bullet points and day-by-day plans when the user asks for itineraries.
- If the user message is small talk (e.g. "ok", "thanks", "hi"), respond briefly and naturally
  without giving extra Sri Lanka information.
""".strip()


# ---------- Helpers to load index + embedder ----------
_index = None
_meta = None
_embedder = None


def load_index_and_meta():
    global _index, _meta
    if _index is not None and _meta is not None:
        return _index, _meta

    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("Index not found. Run `python build_index.py` first.")
    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        _meta = json.load(f)
    return _index, _meta


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# ---------- Small-talk detection ----------
def is_small_talk(message: str) -> bool:
    message = message.lower().strip()
    smalltalk_phrases = {
        "ok", "okay", "kk", "k",
        "thanks", "thank you", "tnx", "thx",
        "great", "nice", "cool", "awesome",
        "hi", "hello", "hey",
        "good", "good job", "well done",
        "bye", "goodbye", "see you"
    }
    return message in smalltalk_phrases


def small_talk_reply(message: str) -> str:
    msg = message.lower().strip()
    if msg in {"thanks", "thank you", "tnx", "thx"}:
        return "You’re welcome! If you want, I can help you plan more Sri Lanka trips 😊"
    if msg in {"hi", "hello", "hey"}:
        return "Hi! I’m CeylonTrip. Ask me anything about traveling in Sri Lanka 🌴"
    if msg in {"bye", "goodbye", "see you"}:
        return "Bye! Hope you have an amazing trip in Sri Lanka someday 🇱🇰"
    # default generic
    return "Got it! Whenever you’re ready, ask me about Sri Lanka travel plans 😊"


# ---------- Retrieval ----------
def retrieve(query: str, top_k: int = 5):
    index, meta = load_index_and_meta()
    model = get_embedder()

    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k=min(top_k, len(meta)))

    chunks = []
    for idx in I[0]:
        idx_int = int(idx)
        if 0 <= idx_int < len(meta):
            chunks.append(meta[idx_int]["text"])
    return chunks


# ---------- Call Ollama ----------
def call_ollama(messages):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    # Default Ollama chat format
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    return data.get("content") or data.get("response") or str(data)


# ---------- Build answer ----------
def answer_question(user_question: str) -> str:
    # 1) Small talk: answer naturally, no RAG
    if is_small_talk(user_question):
        return small_talk_reply(user_question)

    # 2) Retrieve Sri Lanka context
    context_chunks = retrieve(user_question, top_k=5)

    # If no context found, likely not about Sri Lanka or too vague
    if not context_chunks:
        return "I can only help with travel questions related to Sri Lanka."

    context_text = "\n\n---\n\n".join(context_chunks)

    user_block = f"""
Use the following CONTEXT about Sri Lanka to answer the QUESTION.
If the CONTEXT is insufficient, say you are not sure and explain what the traveler
should check locally (e.g. with accommodation, official sites, or operators).

CONTEXT:
{context_text}

QUESTION:
{user_question}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

    reply = call_ollama(messages)
    return reply


# ---------- CLI loop ----------
def main():
    print("CeylonTrip – Sri Lanka Travel Assistant (demo)")
    print("Type your question (empty line to exit).")

    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            print("Bye!")
            break

        print("\nThinking...")
        try:
            ans = answer_question(q)
        except Exception as e:
            ans = f"[Error] {e}"

        print("\nCeylonTrip:", ans)


if __name__ == "__main__":
    main()
