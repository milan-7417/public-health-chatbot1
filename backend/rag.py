import numpy as np
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from backend.memory import get_history
from backend.translation import translate

load_dotenv()

# =========================
# 🔹 GROQ SETUP
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# 🔹 LOAD DOCS
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
# 🔹 MODEL LOAD
# =========================
print("🚀 Loading model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

print("🚀 Encoding docs...")
doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# 🔹 RETRIEVE CONTEXT
# =========================
def retrieve_context(query, k=2):

    query_vec = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(doc_embeddings, query_vec)
    top_k_idx = np.argsort(scores)[-k:][::-1]

    return "\n".join([documents[i] for i in top_k_idx])


# =========================
# 🔹 INTENT DETECTION
# =========================
def detect_intent(query):
    q = query.lower()

    if "what is" in q or "define" in q:
        return "definition"
    elif "symptom" in q:
        return "symptoms"
    elif "treat" in q or "cure" in q:
        return "treatment"
    elif "prevent" in q:
        return "prevention"
    else:
        return "general"


# =========================
# 🔹 LLM CALL (CLEAN OUTPUT)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Give accurate answers. "
                    "Do not use random symbols. "
                    "Do not generate irrelevant diseases. "
                    "Keep answer clean and readable."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=350
    )

    return response.choices[0].message.content


# =========================
# 🔹 RAG MAIN FUNCTION
# =========================
def rag_answer(query, language="en"):

    # 🔹 CONTEXT MEMORY FIX
    try:
        history = get_history()

        if isinstance(history, list) and len(history) > 0:
            last_q = history[-1].get("question", "")

            if len(query.split()) <= 4:
                query = f"{last_q} {query}"

    except:
        pass

    # 🔹 TRANSLATION FIX (SAFE)
    try:
        if language == "hi":
            query_en = translate(query, "hin_Deva", "eng_Latn")
        else:
            query_en = query
    except:
        query_en = query

    # 🔹 INTENT
    intent = detect_intent(query_en)

    # 🔹 CONTEXT
    context = retrieve_context(query_en)
    context = context[:600]

    # 🔹 LANGUAGE CONTROL
    if language == "hi":
        lang_instruction = "Answer in simple Hindi."
    else:
        lang_instruction = "Answer in English."

    #  SMART PROMPT
    if intent == "definition":
        instruction = "Give only definition in 2-3 lines."

    elif intent == "symptoms":
        instruction = "List only symptoms."

    elif intent == "treatment":
        instruction = "Give treatment only."

    elif intent == "prevention":
        instruction = "Give prevention steps."

    else:
        instruction = "Give short structured answer."

    prompt = f"""
{lang_instruction}

{instruction}

Disease or Topic: {query_en}

Context: {context}
"""

    #  LIMIT PROMPT
    if len(prompt) > 2000:
        prompt = prompt[:2000]

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("Error:", e)
        return "⚠️ Try again"

    #  CLEAN OUTPUT (REMOVE GARBAGE)
    answer = answer.replace("**", "")
    answer = answer.replace("#", "")
    answer = answer.strip()

    return answer