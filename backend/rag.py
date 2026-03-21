import numpy as np
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from backend.memory import get_history

load_dotenv()

# =========================
# 🔹 GROQ LLM SETUP
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# 🔹 LOAD DOCUMENTS
# =========================
with open("backend/data/docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# =========================
# 🔹 MODEL (LAZY LOAD)
# =========================
embed_model = None

def get_model():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("BAAI/bge-small-en")
    return embed_model

# =========================
# 🔹 PRECOMPUTE EMBEDDINGS (FAST 🚀)
# =========================
doc_embeddings = None

def load_doc_embeddings():
    global doc_embeddings
    if doc_embeddings is None:
        model = get_model()
        doc_embeddings = model.encode(documents)
    return doc_embeddings

# =========================
# 🔹 RETRIEVE CONTEXT
# =========================
def retrieve_context(query, k=3):

    model = get_model()
    doc_embeddings = load_doc_embeddings()

    query_vec = model.encode(query)

    # cosine similarity using dot product (fast)
    scores = np.dot(doc_embeddings, query_vec)

    top_k_idx = np.argsort(scores)[-k:]

    context = ""
    for idx in top_k_idx:
        context += documents[idx] + "\n"

    return context

# =========================
# 🔹 LLM CALL (GROQ)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=400
    )

    return response.choices[0].message.content

# =========================
# 🔹 RAG MAIN FUNCTION
# =========================
def rag_answer(query):

    context = retrieve_context(query)

    history = get_history()

    prompt = f"""
You are a professional medical AI assistant.

Use the given context to answer the user question accurately.

If answer is not in context, still give a helpful general medical answer.

Context:
{context}

Conversation History:
{history}

User Question:
{query}

Give clear, short, and medically correct answer.
"""

    return call_llm(prompt)