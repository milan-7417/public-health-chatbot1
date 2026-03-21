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
    raw_docs = json.load(f)

documents = []
for doc in raw_docs:
    if isinstance(doc, dict):
        documents.append(doc.get("text", ""))
    else:
        documents.append(str(doc))

# =========================
# 🔹 LOAD MODEL ON STARTUP (FAST)
# =========================
print("🚀 Loading embedding model once...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

# =========================
# 🔹 PRECOMPUTE EMBEDDINGS (VERY FAST AFTER THIS)
# =========================
print("🚀 Computing document embeddings...")
doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# 🔹 RETRIEVE CONTEXT (OPTIMIZED)
# =========================
def retrieve_context(query, k=3):

    query_vec = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # cosine similarity (fast)
    scores = np.dot(doc_embeddings, query_vec)

    # top-k
    top_k_idx = np.argsort(scores)[-k:][::-1]

    context = "\n".join([documents[i] for i in top_k_idx])

    return context


# =========================
# 🔹 LLM CALL (FAST + CLEAN)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional medical assistant. "
                    "Give detailed, structured answers with headings. "
                    "Do NOT mention context or history."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=600
    )

    return response.choices[0].message.content


# =========================
# 🔹 RAG MAIN FUNCTION
# =========================
def rag_answer(query):

    context = retrieve_context(query)
    history = get_history()

    prompt = f"""
Answer the following medical question in a clear and structured format.

Include:
- Definition
- Causes
- Symptoms
- Treatment
- Prevention

Question:
{query}

Conversation History:
{history}

Context:
{context}
"""

    return call_llm(prompt)





