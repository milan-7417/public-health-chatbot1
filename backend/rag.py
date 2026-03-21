import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from groq import Groq

from backend.memory import get_history

# ==============================
# 🔐 ENV
# ==============================
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

# ==============================
# 📂 LOAD DOCUMENTS (JSON)
# ==============================
DOC_PATH = "backend/data/docs.json"

if os.path.exists(DOC_PATH):
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)
else:
    print("❌ docs.json not found")
    documents = []

# ==============================
# 🔥 HF EMBEDDING API
# ==============================
HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def get_embedding(text):
    try:
        response = requests.post(
            HF_URL,
            headers=headers,
            json={"inputs": [text]},
            timeout=30
        )

        result = response.json()

        if isinstance(result, dict) and "error" in result:
            print("HF error:", result)
            return None

        return np.array(result[0], dtype="float32")

    except Exception as e:
        print("Embedding error:", e)
        return None


# ==============================
# 🔍 SIMPLE RETRIEVAL (NO FAISS)
# ==============================
def retrieve_context(query, k=3):

    if not documents:
        return ""

    query_vec = get_embedding(query)
    if query_vec is None:
        return ""

    scores = []

    for doc in documents:
        doc_vec = get_embedding(doc)

        if doc_vec is None:
            continue

        # cosine similarity
        score = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )

        scores.append((score, doc))

    # sort by similarity
    scores.sort(reverse=True, key=lambda x: x[0])

    top_docs = [doc for _, doc in scores[:k]]

    return "\n".join(top_docs)


# ==============================
# 🧠 LLM
# ==============================
def call_llm(prompt):

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print("LLM error:", e)
        return "⚠️ Error generating response"


# ==============================
# 🚀 RAG PIPELINE
# ==============================
def rag_answer(query, language="en"):

    try:
        context = retrieve_context(query)
        history = get_history()

        prompt = f"""
You are a helpful Public Health AI assistant.

Context:
{context}

Conversation history:
{history}

User question:
{query}

Answer clearly.
"""

        answer = call_llm(prompt)
        return answer

    except Exception as e:
        print("RAG error:", e)
        return "⚠️ Error generating response"