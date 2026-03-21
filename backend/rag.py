import faiss
import numpy as np
import os
import requests
from dotenv import load_dotenv
from groq import Groq

from backend.memory import get_history

# ==============================
# 🔐 Load environment variables
# ==============================
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ==============================
# 🤖 Groq client
# ==============================
client = Groq(api_key=GROQ_API_KEY)

# ==============================
# 📂 Load FAISS index
# ==============================
VECTOR_PATH = "backend/vector_db/faiss.index"
DOC_PATH = "backend/vector_db/documents.npy"

index = faiss.read_index(VECTOR_PATH)
documents = np.load(DOC_PATH, allow_pickle=True)

# ==============================
# 🔥 HuggingFace Embedding API
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
            json={
                "inputs": [text]   # ✅ IMPORTANT FIX
            },
            timeout=30
        )

        result = response.json()

        # ❌ API error handling
        if isinstance(result, dict) and "error" in result:
            print("HF API error:", result)
            return None

        # ✅ Convert to numpy
        embedding = np.array(result[0], dtype="float32")

        return embedding

    except Exception as e:
        print("Embedding error:", e)
        return None


# ==============================
# 🔍 Retrieve context from FAISS
# ==============================
def retrieve_context(query, k=3):

    query_vector = get_embedding(query)

    if query_vector is None:
        return ""

    query_vector = np.expand_dims(query_vector, axis=0)

    D, I = index.search(query_vector, k)

    context = ""
    for idx in I[0]:
        if idx < len(documents):
            context += documents[idx] + "\n"

    return context


# ==============================
# 🧠 Call Groq LLM
# ==============================
def call_llm(prompt):

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print("LLM error:", e)
        return "⚠️ Error generating response"


# ==============================
# 🚀 RAG Pipeline
# ==============================
def rag_answer(query, language="en"):

    try:
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

    except Exception as e:
        print("RAG error:", e)
        return "⚠️ Error generating response"