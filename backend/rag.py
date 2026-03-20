import os
import json
import requests
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

    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    return response.json()

# ==============================
# COSINE SIMILARITY
# ==============================
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# RETRIEVE CONTEXT
# ==============================
def retrieve_context(query, k=3):
    query_embedding = get_embedding(query)

    scores = []

    for doc in documents:
        doc_embedding = get_embedding(doc)
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, doc))

    # sort by similarity
    scores.sort(reverse=True)

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

History:
{history}

Question:
{query}

Answer clearly.
"""

    return call_llm(prompt)