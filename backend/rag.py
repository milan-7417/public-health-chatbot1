import faiss
import numpy as np
import os
import json
from dotenv import load_dotenv
from groq import Groq

from backend.memory import get_history

# Load env
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==============================
# LOAD FAISS INDEX (lightweight)
# ==============================
VECTOR_PATH = "backend/vector_db/faiss.index"
index = faiss.read_index(VECTOR_PATH)

# ==============================
# LOAD DOCUMENTS (JSON instead of NPY)
# ==============================
DOC_PATH = "backend/data/docs.json"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = [item["text"] for item in data]

# ==============================
# LAZY LOAD EMBEDDING MODEL
# ==============================
embed_model = None

def get_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("sentence-transformers/BAAI/bge-small-en")
    return embed_model


# ==============================
# RETRIEVE CONTEXT
# ==============================
def retrieve_context(query, k=3):
    model = get_model()

    query_vector = model.encode([query]).astype("float32")
    D, I = index.search(query_vector, k)

    context = ""
    for idx in I[0]:
        if idx < len(documents):
            context += documents[idx] + "\n"

    return context


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
# MAIN RAG FUNCTION
# ==============================
def rag_answer(query, language="en"):

    context = retrieve_context(query)
    history = get_history()

    prompt = f"""
You are a helpful Public Health AI assistant.

Use the provided context to answer.

Context:
{context}

Conversation history:
{history}

User question:
{query}

Answer clearly and accurately.
"""

    answer = call_llm(prompt)
    return answer