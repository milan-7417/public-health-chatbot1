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
# 🔹 LOAD MODEL ONCE
# =========================
print("🚀 Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

# =========================
# 🔹 PRECOMPUTE EMBEDDINGS
# =========================
print("🚀 Computing embeddings...")
doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# 🔹 RETRIEVE CONTEXT (STRICT)
# =========================
def retrieve_context(query, k=2):

    query_vec = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(doc_embeddings, query_vec)
    top_k_idx = np.argsort(scores)[-k:][::-1]

    context = "\n".join([documents[i] for i in top_k_idx])

    return context[:400]   # tighter control


# =========================
# 🔹 INTENT DETECTION
# =========================
def detect_intent(query):
    q = query.lower()

    if "what is" in q or "define" in q:
        return "definition"

    elif "symptom" in q:
        return "symptoms"

    elif "cure" in q or "treat" in q or "treatment" in q:
        return "treatment"

    elif "prevent" in q:
        return "prevention"

    else:
        return "general"


# =========================
# 🔹 LLM CALL (SAFE)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Answer ONLY what is asked. "
                    "Do not change disease/topic. "
                    "Do not add unrelated diseases. "
                    "Do not mention context/history."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=450   # ✅ medium answer
    )

    return response.choices[0].message.content


# =========================
# 🔹 MAIN FUNCTION
# =========================
def rag_answer(query, language="en"):

    # =====================
    # 🔹 TRANSLATE INPUT
    # =====================
    try:
        if language == "hi":
            query_en = translate(query, "hin_Deva", "eng_Latn")
        elif language == "or":
            query_en = translate(query, "ory_Orya", "eng_Latn")
        else:
            query_en = query
    except:
        query_en = query

    # =====================
    # 🔹 GET HISTORY
    # =====================
    history = get_history()

    # =====================
    # 🔹 FOLLOW-UP FIX (IMPORTANT)
    # =====================
    if isinstance(history, list) and len(history) > 0:
        last_q = history[-1].get("question", "")

        # Only for short queries like "how to cure it"
        if len(query_en.split()) <= 4:
            query_en = f"{last_q} {query_en}"

    # =====================
    # 🔹 INTENT
    # =====================
    intent = detect_intent(query_en)

    # =====================
    # 🔹 CONTEXT
    # =====================
    context = retrieve_context(query_en)

    # =====================
    # 🔹 PROMPT (SMART + CLEAN)
    # =====================
    if intent == "definition":
        instruction = "Explain in 4-5 lines."

    elif intent == "symptoms":
        instruction = "List key symptoms clearly."

    elif intent == "treatment":
        instruction = "Explain treatment directly. No definition."

    elif intent == "prevention":
        instruction = "Give prevention steps."

    else:
        instruction = "Give a helpful answer."

    prompt = f"""
{instruction}

Disease/Topic: {query_en}

Use ONLY this context:
{context}
"""

    # =====================
    # 🔹 TOKEN SAFETY
    # =====================
    if len(prompt) > 1800:
        prompt = prompt[:1800]

    # =====================
    # 🔹 LLM CALL
    # =====================
    try:
        answer_en = call_llm(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Please try again"

    # =====================
    # 🔹 RESPONSE CONTROL (MEDIUM LENGTH)
    # =====================
    words = answer_en.split()
    if len(words) > 150:
        answer_en = " ".join(words[:150])

    # =====================
    # 🔹 TRANSLATE BACK
    # =====================
    try:
        if language == "hi":
            answer = translate(answer_en, "eng_Latn", "hin_Deva")
            answer = answer.replace("मैं एक चिकित्सा सहायक हूँ", "")

        elif language == "or":
            answer = translate(answer_en, "eng_Latn", "ory_Orya")

        else:
            answer = answer_en

    except:
        answer = answer_en

    return answer