# 🌴 CeylonTrip – Sri Lanka RAG Travel Assistant

CeylonTrip is a **Sri Lanka–specialized travel chatbot** built on top of a **local LLM** (Llama via Ollama) and a **Retrieval-Augmented Generation (RAG)** pipeline.

It uses a curated knowledge base of Sri Lanka travel data (destinations, routes, tips) and exposes it via a clean **Streamlit** web UI and a simple **CLI** demo.

---

## ✨ Features

- 🇱🇰 **Sri Lanka–only assistant**  
  Refuses to answer about other countries; focuses on Sri Lanka itineraries, routes, and tips.

- 🧠 **RAG pipeline**  
  - CSV + Markdown knowledge base (`destinations.csv`, `routes.csv`, `tips.md`)  
  - Semantic embeddings with `sentence-transformers`  
  - Vector search with **FAISS**

- 🗺️ **Travel expertise**
  - Suggested multi-day itineraries (e.g. “7 days in August, nature + beaches”)  
  - Route and travel-time guidance (e.g. “Kandy → Ella → South Coast”)  
  - Seasonal advice, temple etiquette, safety & practical tips

- 💻 **Two interfaces**
  - `chat_demo.py` – terminal/CLI chat  
  - `streamlit_app.py` – tropical-themed web chat UI

- 🥥 **Tropical UI**
  - Streamlit app styled with a Sri Lanka vibe (white background + teal/orange/beach colors)  
  - Emojis/icons representing Sri Lanka: 🌴 🥥 🐘 🚂 🏝️

---

## 🧱 Tech Stack

- **LLM & Inference**
  - [Ollama](https://ollama.com/) for local model serving
  - Llama 3–family small models (e.g. `llama3.2:1b` or `llama3.1:3b`)

- **RAG**
  - `sentence-transformers` (`all-MiniLM-L6-v2`) for text embeddings
  - `faiss-cpu` for vector similarity search
  - Custom CSV/Markdown data:
    - `data/destinations.csv`
    - `data/routes.csv`
    - `data/tips.md`

- **App**
  - Python 3
  - [Streamlit](https://streamlit.io/) for the web UI
  - `requests`, `pandas`, `numpy`

---

## 🧬 Architecture Overview

1. **Knowledge base**  
   - `destinations.csv` – destinations (region, type, best months, highlights, vibe, etc.)
   - `routes.csv` – point-to-point connections with transport notes and approximate durations
   - `tips.md` – general travel tips (seasons, etiquette, safety, etc.)

2. **Index building (`build_index.py`)**
   - Loads the CSV/Markdown files
   - Converts rows/sections into text chunks
   - Encodes chunks using `all-MiniLM-L6-v2` into dense vectors
   - Normalizes vectors and builds a **FAISS** index
   - Saves:
     - `data/index/faiss.index`
     - `data/index/meta.json`

3. **Retrieval**
   - User query → embedded to a vector
   - FAISS returns top-K most similar chunks
   - Chunks are concatenated into a `CONTEXT` block

4. **Generation**
   - A **strict system prompt** ensures:
     - Sri Lanka–only answers
     - No live prices / real-time schedules
     - Responsible travel behavior
   - Prompt = _system prompt_ + `CONTEXT` + _user question_
   - Sent to Ollama `/api/chat` endpoint with the chosen Llama model
   - Response is rendered in CLI or Streamlit

---

## 📂 Project Structure

```text
Ceylontrip_bot/
├── data/
│   ├── destinations.csv      # curated Sri Lanka destinations
│   ├── routes.csv            # between-city route info
│   ├── tips.md               # markdown with general travel tips
│   └── index/
│       ├── faiss.index       # FAISS vector index (generated)
│       └── meta.json         # metadata about chunks (generated)
├── build_index.py            # build RAG index from CSV/MD
├── chat_demo.py              # CLI demo chatbot
├── streamlit_app.py          # Streamlit web app
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python** 3.9+  
- **Ollama** installed (Windows/macOS/Linux)  
- Enough disk + RAM for the chosen model (1B/3B recommended for low-resource machines)

Install Ollama:  
👉 see official instructions at [ollama.com](https://ollama.com/)

### 2. Clone & set up environment

```bash
git clone <your-repo-url>.git
cd Ceylontrip_bot

# Create virtual environment
python -m venv venv
# Windows:
venv\Scriptsctivate
# macOS/Linux:
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

`requirements.txt` includes (or should include):

```txt
faiss-cpu
sentence-transformers
pandas
numpy
requests
streamlit
```

### 3. Pull a model with Ollama

From any terminal:

```bash
# Example small model
ollama pull llama3.2:1b
# or
ollama pull llama3.1:3b
```

You can check installed models with:

```bash
ollama list
```

> ⚠️ Make sure the model you set in the code (e.g. `llama3.2:1b`) matches what `ollama list` shows.

### 4. Build the RAG index

From inside your virtual environment:

```bash
python build_index.py
```

This will:

- Read `destinations.csv`, `routes.csv`, `tips.md`
- Create embeddings & FAISS index
- Output to `data/index/faiss.index` and `data/index/meta.json`

Run this again if you modify the CSV/MD content.

---

## 💬 Running the Chatbot

### Option A – CLI demo

```bash
# from repo root, with venv activated
python chat_demo.py
```

You should see:

```text
CeylonTrip – Sri Lanka Travel Assistant (demo)
Type your question (empty line to exit).
```

Examples:

```text
You: I have 7 days in August; I like nature and beaches. How should I plan my trip in Sri Lanka?
You: Best surf spots in July?
You: How to combine Kandy, Ella, and Mirissa in 6 days?
```

The bot will:

- Retrieve relevant info from the index
- Generate an answer using the LLM
- Refuse questions about non–Sri Lanka destinations

Simple replies like `ok`, `thank you`, `hi` will trigger **short, natural small-talk responses** instead of full itineraries.

---

### Option B – Streamlit web app

```bash
# from repo root, with venv activated
streamlit run streamlit_app.py
```

This launches a web UI (typically at `http://localhost:8501`) with:

- 🌴 Tropical-themed header and card
- 🥥 Sidebar explaining the assistant
- 💬 Chat interface with bubbles and Sri Lanka-flavored icons
- 🌴 “Thinking” spinner while the LLM prepares the answer

Use it as a conversational planner for Sri Lanka trips.

---

## ⚙️ Configuration

### Change model

In both `chat_demo.py` and `streamlit_app.py`:

```python
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
```

You can either:

- Set an environment variable:

  ```bash
  # Windows (cmd)
  set OLLAMA_MODEL=llama3.1:3b

  # PowerShell
  $env:OLLAMA_MODEL="llama3.1:3b"
  ```

- Or hard-code the model name, as long as it exists in `ollama list`.

### Modify system behavior

The main behavior is controlled by:

- `SYSTEM_PROMPT` in both files
- `is_small_talk` / `small_talk_reply`
- The logic that returns  
  `"I can only help with travel questions related to Sri Lanka 🇱🇰."`  
  when retrieval returns no relevant context

You can tweak the tone (more formal, more casual), the language, or add more rules.

---




## 👤 Author

Anjalika Wijesiri
