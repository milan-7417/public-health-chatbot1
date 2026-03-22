import numpy as np
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from backend.memory import get_history
from backend.translation import translate   #  ADDED

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
# 🔹 PRECOMPUTE EMBEDDINGS
# =========================
print("🚀 Computing document embeddings...")
doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# 🔹 RETRIEVE CONTEXT
# =========================
def retrieve_context(query, k=3):

    query_vec = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(doc_embeddings, query_vec)
    top_k_idx = np.argsort(scores)[-k:][::-1]

    context = "\n".join([documents[i] for i in top_k_idx])

    return context


# =========================
# 🔹 LLM CALL
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
        max_tokens=700   # ✅ slightly increased
    )

    return response.choices[0].message.content


# =========================
# 🔹 RAG MAIN FUNCTION (FIXED)
# =========================
def rag_answer(query, language="en"):

    # 🔹 Translate
    try:
        if language == "hi":
            query_en = translate(query, "hin_Deva", "eng_Latn")
        elif language == "or":
            query_en = translate(query, "ory_Orya", "eng_Latn")
        else:
            query_en = query
    except:
        query_en = query

    # 🔹 Context (LIMITED)
    context = retrieve_context(query_en, k=2)
    context = context[:600]

    # 🔹 Language
    if language == "hi":
        lang_instruction = "Answer in Hindi."
    elif language == "or":
        lang_instruction = "Answer in Odia."
    else:
        lang_instruction = "Answer in English."

    # 🔹 SHORT PROMPT
    prompt = f"""
{lang_instruction}

Question: {query_en}

Context: {context}

Give:
- Definition
- Symptoms
- Treatment
- Prevention
"""

    # HARD LIMIT
    if len(prompt) > 2000:
        prompt = prompt[:2000]

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Please try again"

    if len(answer) > 1000:
        answer = answer[:1000]

    return answer