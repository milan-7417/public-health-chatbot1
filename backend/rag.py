import os
import json
import requests
import numpy as np
from dotenv import load_dotenv
from groq import Groq

from backend.memory import get_history

load_dotenv()

# ==============================
# API KEYS
# ==============================
HF_API_KEY = os.getenv("HF_API_KEY")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# LOAD DOCUMENTS
# ==============================
DOC_PATH = "backend/data/docs.json"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [item["text"] for item in data]

# ==============================
# HF EMBEDDING FUNCTION
# ==============================
def get_embedding(text):
    API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})

        if response.status_code != 200:
            print("HF API error:", response.text)
            return None

        result = response.json()

        # FIX: extract embedding
        if isinstance(result, list):
            return result[0]
        else:
            return None

    except Exception as e:
        print("Embedding error:", e)
        return None


# ==============================
# COSINE SIMILARITY
# ==============================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==============================
# 🔥 PRECOMPUTE DOCUMENT EMBEDDINGS
# ==============================
print("🔄 Generating document embeddings...")

doc_embeddings = []

for doc in documents:
    emb = get_embedding(doc)
    if emb is not None:
        doc_embeddings.append(emb)
    else:
        doc_embeddings.append([0] * 384)  # fallback

print("✅ Embeddings ready")


# ==============================
# RETRIEVE CONTEXT
# ==============================
def retrieve_context(query, k=3):

    query_embedding = get_embedding(query)

    if query_embedding is None:
        return "No context available."

    scores = []

    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((score, documents[i]))

    # sort by similarity
    scores.sort(key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scores[:k]]

    return "\n".join(top_docs)


# ==============================
# CALL GROQ LLM
# ==============================
def call_llm(prompt):
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content


# ==============================
# MAIN RAG
# ==============================
def rag_answer(query, language="en"):

    context = retrieve_context(query)
    history = get_history()

    prompt = f"""
You are a helpful Public Health AI assistant.

Context:
{context}

Conversation History:
{history}

User Question:
{query}

Answer clearly and accurately.
"""

    return call_llm(prompt)