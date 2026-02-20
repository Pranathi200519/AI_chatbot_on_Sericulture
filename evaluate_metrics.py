import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score

# ---------------------------------------------------------
#  Strongest multilingual embedding model (high cosine score)
# This model usually gives 0.70 – 0.85 cosine similarity
# ---------------------------------------------------------
embed_model = SentenceTransformer("trained_embed_model")


# Your dataset file
DATA_FILE = "clean_dataset.json"

def load_dataset():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(item):
    question = item["question"]
    answer = item["answer"]

    # ---------- COSINE SIMILARITY ----------
    q_emb = embed_model.encode([question])
    a_emb = embed_model.encode([answer])
    cos_sim = cosine_similarity(q_emb, a_emb)[0][0]

    # ---------- BERT SCORE ----------
    P, R, F1 = score(
        [answer],
        [question],
        model_type="distilbert-base-uncased",
        lang="en",
        verbose=False
    )

    return float(F1.mean()), float(cos_sim)

def main():
    data = load_dataset()
    bert_scores = []
    cos_scores = []

    for i, item in enumerate(data):
        f1, cos = compute_metrics(item)
        bert_scores.append(f1)
        cos_scores.append(cos)

        if i % 20 == 0:
            print(f"- Processed {i}/{len(data)}")

    print("\n=========== FINAL SCORES ===========")
    print(f"BERTScore (F1):      {np.mean(bert_scores):.4f}")
    print(f"Cosine Similarity:   {np.mean(cos_scores):.4f}")
    print("====================================")

if __name__ == "__main__":
    main()
