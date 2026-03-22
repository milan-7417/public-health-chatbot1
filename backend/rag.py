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
print("🚀 Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

print("🚀 Encoding documents...")
doc_embeddings = embed_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# =========================
# 🔹 RETRIEVAL
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

    if any(x in q for x in ["what is", "define", "meaning"]):
        return "definition"

    elif any(x in q for x in ["symptom", "sign"]):
        return "symptoms"

    elif any(x in q for x in ["treat", "cure", "medicine"]):
        return "treatment"

    elif any(x in q for x in ["prevent", "precaution"]):
        return "prevention"

    else:
        return "general"


# =========================
# 🔹 LLM CALL (BALANCED)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. "
                    "Give accurate, relevant and moderately detailed answers. "
                    "Avoid unnecessary repetition."
                )
            },
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.6,
        max_tokens=400   # ✅ balanced (no token error + good detail)
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

            # If short query → attach context
            if len(query.split()) <= 4:
                query = f"{last_q} {query}"

    except:
        pass

    # =========================
    # 🔹 STEP 2: TRANSLATE
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
    # 🔹 STEP 4: RETRIEVE CONTEXT
    # =========================
    context = retrieve_context(query_en, k=2)
    context = context[:700]   # ✅ controlled

    # =========================
    # 🔹 STEP 5: LANGUAGE
    # =========================
    if language == "hi":
        lang_instruction = "Answer in Hindi."
    elif language == "or":
        lang_instruction = "Answer in Odia."
    else:
        lang_instruction = "Answer in English."

    # =========================
    #  STEP 6: SMART PROMPT
    # =========================
    if intent == "definition":
        instruction = "Give a clear definition with 2-3 lines explanation."

    elif intent == "symptoms":
        instruction = "List important symptoms with short explanation."

    elif intent == "treatment":
        instruction = "Explain treatment and cure clearly."

    elif intent == "prevention":
        instruction = "Give prevention steps clearly."

    else:
        instruction = """Give a structured answer:
- Definition
- Symptoms
- Treatment
- Prevention"""

    prompt = f"""
{lang_instruction}

{instruction}

Question: {query_en}

Context: {context}
"""

    # =========================
    #  SAFETY LIMIT (NO TOKEN ERROR)
    # =========================
    if len(prompt) > 2500:
        prompt = prompt[:2500]

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Please try again later"

    # =========================
    #  OUTPUT CONTROL
    # =========================
    if len(answer) > 1200:
        answer = answer[:1200]

    return answer