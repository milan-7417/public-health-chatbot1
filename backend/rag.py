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
# 🔹 LOAD DOCUMENTS (FIXED)
# =========================
with open("backend/data/docs.json", "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

# ✅ Convert dict → string safely
documents = []
for doc in raw_docs:
    if isinstance(doc, dict):
        documents.append(doc.get("text", ""))
    else:
        documents.append(str(doc))

# =========================
# 🔹 MODEL (LAZY LOAD)
# =========================
embed_model = None

def get_model():
    global embed_model
    if embed_model is None:
        print("Loading embedding model...")
        embed_model = SentenceTransformer("BAAI/bge-small-en")
    return embed_model

# =========================
# 🔹 PRECOMPUTE EMBEDDINGS
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

    # cosine similarity
    scores = np.dot(doc_embeddings, query_vec)

    top_k_idx = np.argsort(scores)[-k:]

    context = ""
    for idx in top_k_idx:
        context += documents[idx] + "\n"

    return context

# =========================
# 🔹 LLM CALL (SAFE)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional medical assistant. Give detailed, clear, and structured answers. Do NOT mention 'based on context' or conversation history."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=700   
    )

    return response.choices[0].message.content

# =========================
# 🔹 RAG MAIN FUNCTION
# =========================

def rag_answer(query):

    context = retrieve_context(query)
    history = get_history()

    prompt = f"""
You are a medical assistant.

Answer the question in a detailed and structured way.

Include:
- Definition
- Causes
- Symptoms
- Treatment
- Prevention (if applicable)

Do NOT say "based on context" or mention conversation history.

Context:
{context}

Conversation History:
{history}

Question:
{query}
"""

    return call_llm(prompt)

