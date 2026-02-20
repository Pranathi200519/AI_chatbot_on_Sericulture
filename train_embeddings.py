from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import json

train_file = "train_pairs.json"

# Load training data
examples = []
with open(train_file, "r", encoding="utf-8") as f:
    pairs = json.load(f)
    for p in pairs:
        examples.append(InputExample(texts=[p["sentence1"], p["sentence2"]], label=1.0))

# Model to fine-tune
model = SentenceTransformer("all-MiniLM-L6-v2")

# DataLoader
loader = DataLoader(examples, shuffle=True, batch_size=16)

# Loss function
train_loss = losses.CosineSimilarityLoss(model)

# Train
model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=3,
    warmup_steps=50,
    output_path="trained_embed_model"
)

print("Training complete. Model saved in trained_embed_model/")




