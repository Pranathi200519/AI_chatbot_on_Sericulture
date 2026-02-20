# preprocess_data.py
import json
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uuid

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_FILES = [
    "dataset.json",
    "dataset.jsonl",
    "Answers.json",     # optional — will be auto-detected if exists
]

OUTPUT_CLEAN_DATA = "clean_dataset.json"
CACHE_DIR = Path("sericulture_cache")
CACHE_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = CACHE_DIR / "faiss_index.bin"
FAISS_META_PATH = CACHE_DIR / "faiss_meta.json"

EMBED_MODEL = "all-MiniLM-L6-v2"


# ------------------------------------------------------------
# 1. Load multiple dataset files safely
# ------------------------------------------------------------
def load_file(path):
    if not Path(path).exists():
        return []

    if path.endswith(".json"):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except:
            return []

    if path.endswith(".jsonl"):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        res = []
        for ln in lines:
            try:
                res.append(json.loads(ln))
            except:
                pass
        return res

    return []


def normalize_item(item):
    """Fix item structure → always return dict(question, answer, category)."""

    q = (
        item.get("question")
        or item.get("Query")
        or item.get("prompt")
        or ""
    )

    a = (
        item.get("answer")
        or item.get("response")
        or item.get("output")
        or ""
    )

    cat = item.get("category") or item.get("tag") or "general"

    return {
        "question": q.strip(),
        "answer": a.strip(),
        "category": str(cat).strip()
    }


# ------------------------------------------------------------
# 2. Combine + Clean + Deduplicate
# ------------------------------------------------------------
def preprocess_all():
    print("\n🔍 Loading dataset files...")
    all_data_raw = []

    for f in DATA_FILES:
        res = load_file(f)
        if len(res) > 0:
            print(f"✔ Loaded {len(res)} items from {f}")
        all_data_raw.extend(res)

    if len(all_data_raw) == 0:
        print("❌ No data found. Make sure dataset.json or dataset.jsonl exists.")
        return

    print(f"\n🔧 Normalizing {len(all_data_raw)} items...")

    clean = []
    seen = set()

    for item in all_data_raw:
        norm = normalize_item(item)

        # Remove empty
        if not norm["question"] or not norm["answer"]:
            continue

        key = norm["question"].lower().strip() + "||" + norm["answer"].lower().strip()
        if key in seen:
            continue
        seen.add(key)

        clean.append(norm)

    print(f"✔ Clean dataset size: {len(clean)} items")

    Path(OUTPUT_CLEAN_DATA).write_text(json.dumps(clean, indent=2), encoding="utf-8")
    print(f"📁 Saved cleaned dataset → {OUTPUT_CLEAN_DATA}")

    return clean


# ------------------------------------------------------------
# 3. Build FAISS index from cleaned dataset
# ------------------------------------------------------------
def build_faiss(clean_data):
    print("\n⚙ Building FAISS index...")

    model = SentenceTransformer(EMBED_MODEL)

    texts = [item["answer"] for item in clean_data]
    embeddings = model.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"📦 Saved FAISS index → {FAISS_INDEX_PATH}")

    # Save metadata
    meta = {str(i): clean_data[i] for i in range(len(clean_data))}
    FAISS_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"📁 Saved metadata → {FAISS_META_PATH}")

    print("\n🎉 FAISS rebuild complete!")
    print("Your backend is now ready for better search + cleaner results.\n")


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    cleaned = preprocess_all()

    if cleaned:
        build_faiss(cleaned)
