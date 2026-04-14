import numpy as np
import json
import os
import re
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from backend.memory import get_history
from backend.translation import translate

load_dotenv()

# =========================
# 🔹 PATH FIX (FOR EXE)
# =========================
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# =========================
# 🔹 GROQ SETUP
# =========================
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================
# 🔹 LOAD DOCUMENTS
# =========================
with open(resource_path("backend/data/docs.json"), "r", encoding="utf-8") as f:
    raw_docs = json.load(f)

documents = []
for doc in raw_docs:
    if isinstance(doc, dict):
        documents.append(doc.get("text", ""))
    else:
        documents.append(str(doc))

# =========================
# 🔹 LOAD MODEL (ONLY FOR QUERY)
# =========================
print("🚀 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 🔥 faster model

# =========================
# 🔹 LOAD PRECOMPUTED EMBEDDINGS (FAST)
# =========================
print("⚡ Loading precomputed embeddings...")
doc_embeddings = np.load(resource_path("backend/data/embeddings.npy"))

# =========================
# 🔹 EXTRACT DISEASE
# =========================
def extract_disease(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    stopwords = [
        "what", "is", "how", "to", "cure", "treat", "it",
        "the", "a", "an", "of", "and", "for"
    ]

    words = text.split()

    for word in words:
        if word not in stopwords and len(word) > 3:
            return word

    return ""

# =========================
# 🔹 RETRIEVE CONTEXT (SMART)
# =========================
def retrieve_context(query, k=3):

    query_vec = embed_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = np.dot(doc_embeddings, query_vec)

    disease = extract_disease(query)

    filtered_idx = [
        i for i in range(len(documents))
        if disease and disease in documents[i].lower()
    ]

    if filtered_idx:
        ranked = sorted(filtered_idx, key=lambda i: scores[i], reverse=True)
        top_k_idx = ranked[:k]
    else:
        top_k_idx = np.argsort(scores)[-k:][::-1]

    context = "\n".join([documents[i] for i in top_k_idx])

    return context[:500]

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
# 🔹 LLM CALL
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Answer ONLY what is asked. "
                    "STRICTLY stay on the given disease. "
                    "Do not mix diseases. "
                    "Do not hallucinate."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content

# =========================
# 🔹 MAIN FUNCTION
# =========================
def rag_answer(query, language="en"):

    # 🔹 TRANSLATE
    try:
        if language == "hi":
            query_en = translate(query, "hin_Deva", "eng_Latn")
        elif language == "or":
            query_en = translate(query, "ory_Orya", "eng_Latn")
        else:
            query_en = query
    except:
        query_en = query

    history = get_history()

    # =====================
    # 🔥 FOLLOW-UP FIX
    # =====================
    if isinstance(history, list) and len(history) > 0:
        last_q = history[-1].get("question", "")
        last_disease = extract_disease(last_q)

        if re.search(r"\bit\b", query_en.lower()):
            if last_disease:
                query_en = re.sub(
                    r"\bit\b",
                    last_disease,
                    query_en,
                    flags=re.IGNORECASE
                )

        elif len(query_en.split()) <= 4 and last_disease:
            query_en = f"{last_disease} {query_en}"

    # =====================
    # 🔹 INTENT
    # =====================
    intent = detect_intent(query_en)

    # =====================
    # 🔹 CONTEXT
    # =====================
    context = retrieve_context(query_en)

    # =====================
    # 🔹 PROMPT
    # =====================
    if intent == "definition":
        instruction = "Explain clearly in 4-5 lines."
    elif intent == "symptoms":
        instruction = "List symptoms clearly."
    elif intent == "treatment":
        instruction = "Give treatment steps only."
    elif intent == "prevention":
        instruction = "Give prevention steps."
    else:
        instruction = "Give helpful medical answer."

    prompt = f"""
{instruction}

Disease: {extract_disease(query_en)}

Question: {query_en}

Use ONLY this context:
{context}
"""

    if len(prompt) > 1800:
        prompt = prompt[:1800]

    try:
        answer_en = call_llm(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Please try again"

    # =====================
    # 🔹 RESPONSE LIMIT
    # =====================
    words = answer_en.split()
    if len(words) > 200:
        answer_en = " ".join(words[:200])

    # =====================
    # 🔹 TRANSLATE BACK
    # =====================
    try:
        if language == "hi":
            answer = translate(answer_en, "eng_Latn", "hin_Deva")
        elif language == "or":
            answer = translate(answer_en, "eng_Latn", "ory_Orya")
        else:
            answer = answer_en
    except:
        answer = answer_en

    return answer