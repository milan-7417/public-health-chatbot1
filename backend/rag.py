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
# 🔹 LLM CALL
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Give correct and relevant answers only. "
                    "Do not add unrelated diseases. "
                    "Keep answer clean and structured."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.6,
        max_tokens=500
    )

    return response.choices[0].message.content


# =========================
# 🔹 RAG MAIN FUNCTION
# =========================
def rag_answer(query, language="en"):

    # =========================
    # 🔹 STEP 1: CONTEXT FIX
    # =========================
    try:
        history = get_history()

        if isinstance(history, list) and len(history) > 0:
            last_q = history[-1].get("question", "")

            if len(query.split()) <= 4 and "it" in query.lower():
                query = f"{last_q} {query}"

    except:
        pass

    # =========================
    # 🔹 STEP 2: TRANSLATE TO ENGLISH
    # =========================
    try:
        if language == "hi":
            query_en = translate(query, "hin_Deva", "eng_Latn")
        elif language == "or":
            query_en = translate(query, "ory_Orya", "eng_Latn")
        else:
            query_en = query
    except:
        query_en = query

    # =========================
    # 🔹 STEP 3: INTENT
    # =========================
    intent = detect_intent(query_en)

    # =========================
    # 🔹 STEP 4: CONTEXT
    # =========================
    context = retrieve_context(query_en)
    context = context[:700]

    # =========================
    # 🔹 STEP 5: PROMPT
    # =========================
    if intent == "definition":
        instruction = "Explain clearly."

    elif intent == "symptoms":
        instruction = "List symptoms with short explanation."

    elif intent == "treatment":
        instruction = "Explain treatment and cure."

    elif intent == "prevention":
        instruction = "Give prevention steps clearly."

    else:
        instruction = """Give a structured answer:
- Definition
- Symptoms
- Treatment
- Prevention"""

    prompt = f"""
{instruction}

Question: {query_en}

Context: {context}
"""

    if len(prompt) > 2500:
        prompt = prompt[:2500]

    # =========================
    # 🔹 STEP 6: LLM CALL
    # =========================
    try:
        answer_en = call_llm(prompt)
    except Exception as e:
        print("Error:", e)
        return "⚠️ Please try again"

    # =========================
    # 🔹 STEP 7: TRANSLATE BACK
    # =========================
    try:
        if language == "hi":
            answer = translate(answer_en, "eng_Latn", "hin_Deva")
        elif language == "or":
            answer = translate(answer_en, "eng_Latn", "ory_Orya")
        else:
            answer = answer_en
    except:
        answer = answer_en

    # =========================
    # 🔹 CLEAN OUTPUT
    # =========================
    answer = answer.replace("**", "").replace("#", "").strip()

    return answer