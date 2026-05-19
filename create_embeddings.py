import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load documents
with open("backend/data/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

documents = [d["text"] for d in docs]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(documents, normalize_embeddings=True)

# Save embeddings
np.save("backend/data/embeddings.npy", embeddings)

print("✅ Embeddings saved successfully!")