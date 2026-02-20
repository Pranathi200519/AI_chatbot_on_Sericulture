# app.py
import json
import requests
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import sys

# -----------------------------
# CONFIG / PATHS
# -----------------------------
# FAISS index + metadata expected under sericulture_cache/ (adjust if yours differs)
FAISS_INDEX_PATH = Path("sericulture_cache/faiss_index.bin")
FAISS_META_PATH = Path("sericulture_cache/faiss_meta.json")

# Use your trained model directory name
EMBEDDING_MODEL_DIR = "trained_embed_model"

# Ollama settings (if you use Ollama); adjust if needed
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL =  "gemma2:2b"


# -----------------------------
# LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sericulture-backend")

# -----------------------------
# VERIFY & LOAD FAISS + META + EMBEDDER
# -----------------------------
log.info("🚀 Loading FAISS index and metadata...")

if not FAISS_META_PATH.exists():
    log.error("Metadata file not found: %s", FAISS_META_PATH)
    log.error("Make sure you have built the FAISS index and saved faiss_meta.json to sericulture_cache/")
    raise SystemExit(1)

with FAISS_META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

if not FAISS_INDEX_PATH.exists():
    log.error("FAISS index file not found: %s", FAISS_INDEX_PATH)
    log.error("Make sure you ran the script that creates the FAISS index (build_faiss_index.py / build script).")
    raise SystemExit(1)

try:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
except Exception:
    log.exception("Failed to read FAISS index from %s", FAISS_INDEX_PATH)
    raise

# Load your trained SentenceTransformer (directory created when you saved the trained model)
if Path(EMBEDDING_MODEL_DIR).exists():
    log.info("Load pretrained SentenceTransformer: %s", EMBEDDING_MODEL_DIR)
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_DIR)
    except Exception:
        log.exception("Failed to load trained embed model from %s", EMBEDDING_MODEL_DIR)
        raise
else:
    log.error("Trained embed model folder not found: %s", EMBEDDING_MODEL_DIR)
    log.error("If you want, update EMBEDDING_MODEL_DIR in app.py or put the trained model folder here.")
    raise SystemExit(1)

log.info("✅ FAISS and metadata loaded successfully!")

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="Sericulture Chatbot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Sericulture backend running"}


def search_faiss(query: str, top_k: int = 3):
    # Encode with your trained embedding model
    emb = embedding_model.encode([query], convert_to_numpy=True)
    if emb.ndim == 1:
        emb = np.expand_dims(emb, 0)
    # Ensure dtype float32 for faiss
    distances, ids = index.search(emb.astype("float32"), top_k)

    results = []
    for i, nid in enumerate(ids[0]):
        idx = str(int(nid))
        if idx == "-1" or idx not in meta:
            continue
        item = meta[idx]
        results.append({
            "id": idx,
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "category": item.get("category", ""),
            "text": item.get("answer", ""),
            "distance": float(distances[0][i])
        })
    return results


def call_ollama(question: str, context_text: str):
    """
    Safely call Ollama if you have it running. Adjust OLLAMA_URL/OLLAMA_MODEL above as needed.
    If you don't use Ollama, you may replace this function with another LLM call or a simple fallback.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Context:\n{context_text}\n\nAnswer the question:\n{question}",
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if res.status_code != 200:
            log.error("Ollama returned status %s: %s", res.status_code, res.text)
            return f"❌ Ollama error: status {res.status_code}"
        data = res.json()

        # Handle variants of response shapes
        for k in ("response", "text", "result"):
            if isinstance(data, dict) and k in data:
                return data[k]

        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            c = data["choices"][0]
            return c.get("message", c.get("content", c.get("text", ""))) or ""

        return str(data)

    except Exception as exc:
        log.exception("Ollama call failed")
        return f"❌ Ollama exception: {exc}"


@app.post("/chat")
async def chat_api(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return {"answer": "⚠️ Please enter a question.", "sources": []}

    docs = search_faiss(query, top_k=4)
    context_text = "\n".join(f"- {d['text']}" for d in docs)

    answer = call_ollama(query, context_text)

    return {"answer": answer, "sources": docs}


if __name__ == "__main__":
    import uvicorn
    log.info("\n🚀 Sericulture Chatbot Backend Running!")
    log.info("👉 Backend test URL: http://127.0.0.1:8020")
    log.info("✔ Frontend UI: http://127.0.0.1:5500/index.html")
    # NOTE: if port already used -> change this number (8020 -> 8011 etc.)
    uvicorn.run(app, host="127.0.0.1", port=8020)
