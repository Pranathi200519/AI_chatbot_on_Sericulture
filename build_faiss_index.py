from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def build_faiss():
    print("🚀 Loading strong embedding model (all-mpnet-base-v2)...")
    model = SentenceTransformer("trained_embed_model")


    print("📂 Loading clean dataset...")
    with open("clean_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["answer"] for item in data]

    print("🔍 Encoding dataset with MPNet model...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    print("📌 Adding vectors to FAISS index...")
    index.add(embeddings)

    faiss.write_index(index, "sericulture_cache/faiss_index.bin")

    print("\n🎉 FAISS index rebuilt successfully using MPNet embeddings!")
    print("👉 This will increase similarity + accuracy significantly.")

if __name__ == "__main__":
    build_faiss()
