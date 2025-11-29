# build_index.py
import os
import re
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")
os.makedirs(INDEX_DIR, exist_ok=True)

DEST_PATH = os.path.join(DATA_DIR, "destinations.csv")
ROUTES_PATH = os.path.join(DATA_DIR, "routes.csv")
TIPS_PATH = os.path.join(DATA_DIR, "tips.md")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_destinations(path: str):
    df = pd.read_csv(path)
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"[DESTINATION] {row['name']}\n"
            f"Region: {row['region']}\n"
            f"Types: {row['types']}\n"
            f"Best months: {row['best_months']}\n"
            f"Recommended days: {row['recommended_days']}\n"
            f"Highlights: {row['highlights']}\n"
            f"Vibe: {row['vibe']}\n"
            f"Details: {row['description']}"
        )
        chunks.append(
            {
                "id": f"dest_{slug(str(row['name']))}",
                "source": "destinations",
                "text": text,
            }
        )
    return chunks


def load_routes(path: str):
    df = pd.read_csv(path)
    chunks = []
    for _, row in df.iterrows():
        text = (
            f"[ROUTE] {row['from']} → {row['to']}\n"
            f"Transport: {row['transport']}\n"
            f"Approx time: {row['hours_min']}–{row['hours_max']} hours\n"
            f"Scenic: {row['scenic']}\n"
            f"Notes: {row['notes']}"
        )
        chunks.append(
            {
                "id": f"route_{slug(str(row['from']))}_{slug(str(row['to']))}",
                "source": "routes",
                "text": text,
            }
        )
    return chunks


def load_tips(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by "## " to make sections
    sections = content.split("\n## ")
    chunks = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue

        # Re-add "## " to all except possibly first
        if i == 0:
            title = "General travel tips"
            text = sec
        else:
            if "\n" in sec:
                title_line, body = sec.split("\n", 1)
            else:
                title_line, body = sec, ""
            title = title_line.strip("# ").strip()
            text = f"## {title}\n{body}"

        chunks.append(
            {
                "id": f"tips_{i:02d}",
                "source": "tips",
                "text": f"[TIPS] {title}\n{text}",
            }
        )
    return chunks


def build_corpus():
    corpus = []
    if os.path.exists(DEST_PATH):
        corpus.extend(load_destinations(DEST_PATH))
    if os.path.exists(ROUTES_PATH):
        corpus.extend(load_routes(ROUTES_PATH))
    if os.path.exists(TIPS_PATH):
        corpus.extend(load_tips(TIPS_PATH))

    if not corpus:
        raise RuntimeError(
            "No data found. Make sure destinations.csv, routes.csv, tips.md are in data/."
        )
    return corpus


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def main():
    print("Loading corpus...")
    corpus = build_corpus()
    texts = [c["text"] for c in corpus]
    print(f"Total chunks: {len(texts)}")

    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Encoding embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype("float32")
    embeddings = normalize(embeddings)

    print("Building FAISS index (cosine similarity via dot product)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)

    print(f"Saving metadata to {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("✅ Done building RAG index.")


if __name__ == "__main__":
    main()
