# streamlit_app.py
import os
import json
import requests

import streamlit as st
import faiss
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

# Make sure this matches a model you actually pulled, e.g. "llama3.2:1b" or "llama3.1:3b"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

# ---------- System prompt ----------
SYSTEM_PROMPT = """
You are CeylonTrip, an AI travel assistant specialized ONLY in Sri Lanka.

RULES:
- You MUST answer ONLY questions related to travel in Sri Lanka.
- If the user asks about another country, reply:
  "I can only answer questions about traveling in Sri Lanka."
- Use the provided CONTEXT when possible (destinations, routes, tips).
- Do NOT invent live prices, real-time schedules, or current weather.
- If something is not covered in the context, say you are not sure and suggest
  what the traveler can check locally (guesthouses, official sites, operators).
- Keep itineraries geographically sensible and mention approximate travel times.
- Prefer clear bullet points and day-by-day plans for itineraries.
- If the user message is small talk (e.g. "ok", "thanks", "hi"), respond briefly
  and naturally without giving extra Sri Lanka information.
""".strip()


# ---------- Small-talk helpers ----------
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
        return "You’re welcome! If you like, I can help you plan more Sri Lanka trips 🥥🌴"
    if msg in {"hi", "hello", "hey"}:
        return "Ayubowan! 🙏 I’m CeylonTrip. Ask me anything about traveling in Sri Lanka 🇱🇰"
    if msg in {"bye", "goodbye", "see you"}:
        return "Goodbye! Hope you have a beautiful journey in Sri Lanka one day 🐘🏝️"
    return "Got it! Whenever you’re ready, ask me about Sri Lanka travel plans 🌴"


# ---------- Cached resources ----------
@st.cache_resource
def load_index_and_meta():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("Index not found. Run `python build_index.py` first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


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
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    return data.get("content") or data.get("response") or str(data)


# ---------- Build answer ----------
def answer_question(user_question: str) -> str:
    # Small talk → no RAG
    if is_small_talk(user_question):
        return small_talk_reply(user_question)

    context_chunks = retrieve(user_question, top_k=5)

    if not context_chunks:
        return "I can only help with travel questions related to Sri Lanka 🇱🇰."

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


# ---------- Streamlit basic config ----------
st.set_page_config(
    page_title="CeylonTrip – Sri Lanka Travel Assistant",
    page_icon="🥥",
    layout="centered",
)

# ---------- Custom tropical styling ----------
TROPICAL_CSS = """
<style>
:root {
  --ceylon-primary: #0f766e;   /* deep teal like Indian Ocean */
  --ceylon-accent:  #f97316;   /* sunset orange */
  --ceylon-soft:    #ecfdf5;   /* soft green */
  --ceylon-sand:    #fef9c3;   /* beach sand */
}

/* Make the whole app clean white with a soft tropical gradient in the center */
[data-testid="stAppViewContainer"] {
  background-color: #ffffff;
}

/* Center column width & add a subtle card feel */
.main-block {
  max-width: 780px;
  margin: 0 auto;
  padding: 1.5rem 1.5rem 2.5rem 1.5rem;
  border-radius: 1.5rem;
  background: linear-gradient(145deg, #ffffff 0%, #fef9c3 40%, #ecfdf5 100%);
  box-shadow: 0 18px 45px rgba(15, 118, 110, 0.08);
  border: 1px solid rgba(15, 118, 110, 0.08);
}

/* Sidebar with tropical gradient */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fff7ed 0%, #ecfeff 40%, #ecfdf3 100%);
  border-right: 1px solid rgba(15, 118, 110, 0.10);
}

/* Sidebar text tweaks */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
  color: #0f766e !important;
}

/* Chat messages: center them and give subtle rounded bubbles */
[data-testid="stChatMessage"] {
  max-width: 760px;
  margin-left: auto;
  margin-right: auto;
}

/* User and assistant bubbles */
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
  border-radius: 1.2rem;
  padding: 0.75rem 1rem;
}

/* Make assistant messages feel like cool ocean water */
[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) div[data-testid="stMarkdownContainer"] {
  background: rgba(239, 246, 255, 0.9);
  border: 1px solid rgba(59, 130, 246, 0.12);
}

/* User messages: warm sand */
[data-testid="stChatMessage"]:has(div[aria-label="user"]) div[data-testid="stMarkdownContainer"] {
  background: rgba(254, 243, 199, 0.9);
  border: 1px solid rgba(234, 179, 8, 0.18);
}

/* Make chat input a bit rounded and airy */
textarea {
  border-radius: 1rem !important;
}

/* Tiny tropical badge under the title */
.ceylon-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  background-color: rgba(15, 118, 110, 0.1);
  color: #0f766e;
  font-size: 0.78rem;
  font-weight: 600;
}

/* "Thinking" tropical emoji animation (we'll show it inside spinner text) */
.tropical-thinking {
  display: inline-block;
  animation: floaty 1.8s ease-in-out infinite;
  margin-right: 0.25rem;
}

@keyframes floaty {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-4px); }
  100% { transform: translateY(0px); }
}
</style>
"""
st.markdown(TROPICAL_CSS, unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🇱🇰 CeylonTrip")
    st.markdown(
        """
**Sri Lanka RAG Travel Assistant**

- 🌴 _Local LLM via Ollama_
- 🐘 Curated Sri Lanka knowledge:
  - 🏙️ Cities & hill country
  - 🏝️ Beaches & surf spots
  - 🚂 Scenic routes & travel times
  - 🍛 Culture, food & practical tips

_No live prices or real-time schedules – think of me as a smart guidebook._
        """
    )
    st.markdown("---")
    st.markdown("**Try asking:**")
    st.markdown(
        """
- ✈️ *“I have 7 days in August, I like nature and beaches. Plan a trip.”*  
- 🌊 *“Best surf spots in July in Sri Lanka?”*  
- 🚂 *“How to combine Kandy, Ella and the south coast?”*  
        """
    )

# ---------- Main content ----------
st.markdown(
    """
<div class="main-block">
  <h1 style="margin-bottom:0.25rem;">🌴 CeylonTrip – Sri Lanka AI Travel Assistant</h1>
  <div class="ceylon-badge">
    🥥 Sri Lanka only · Itineraries · Routes · Tips
  </div>
  <p style="margin-top:0.75rem; font-size:0.92rem;">
    Ask me anything about traveling in Sri Lanka – I’ll help you stitch together beaches, hill country,
    cultural spots, and wildlife into a smooth itinerary.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")  # a bit of spacing

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New user input
user_input = st.chat_input("Ask about Sri Lanka travel… 🌴")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Tropical thinking spinner
        spinner_text = (
            '<span class="tropical-thinking">🌴</span>'
            "Brewing your Ceylon travel plan…"
        )
        with st.spinner(spinner_text):
            try:
                reply = answer_question(user_input)
            except Exception as e:
                reply = f"Sorry, something went wrong: `{e}`"
            st.markdown(reply, unsafe_allow_html=False)

    st.session_state.messages.append({"role": "assistant", "content": reply})
