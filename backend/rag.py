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
# 🔹 LOAD MODEL
# =========================
print("🚀 Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

print("🚀 Computing embeddings...")
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
# 🔹 DETECT USER INTENT 
# =========================
def detect_intent(query):
    q = query.lower()

    if any(word in q for word in ["what is", "define", "meaning"]):
        return "definition"

    elif any(word in q for word in ["symptom", "sign"]):
        return "symptoms"

    elif any(word in q for word in ["treat", "cure", "medicine"]):
        return "treatment"

    elif any(word in q for word in ["prevent", "precaution"]):
        return "prevention"

    else:
        return "general"


# =========================
# 🔹 LLM CALL (OPTIMIZED)
# =========================
def call_llm(prompt):

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a medical assistant. Give direct and relevant answers only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=250
    )

    return response.choices[0].message.content


# =========================
# 🔹 RAG MAIN FUNCTION
# =========================
def rag_answer(query, language="en"):

    # 🔹 CONTEXT FIX (FOLLOW-UP)
    try:
        history = get_history()
        if isinstance(history, list) and len(history) > 0:
            last_q = history[-1].get("question", "")
            if len(query.split()) <= 4:
                query = f"{last_q} {query}"
    except:
        pass

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

    # 🔹 DETECT INTENT 
    intent = detect_intent(query_en)

    # 🔹 CONTEXT
    context = retrieve_context(query_en)
    context = context[:500]

    # 🔹 LANGUAGE
    if language == "hi":
        lang_instruction = "Answer in Hindi."
    elif language == "or":
        lang_instruction = "Answer in Odia."
    else:
        lang_instruction = "Answer in English."

    # =========================
    #  SMART PROMPT BASED ON INTENT
    # =========================
    if intent == "definition":
        instruction = "Give only a clear definition."

    elif intent == "symptoms":
        instruction = "List only symptoms clearly."

    elif intent == "treatment":
        instruction = "Give only treatment and cure."

    elif intent == "prevention":
        instruction = "Give only prevention steps."

    else:
        instruction = """Give:
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

    #  LIMIT PROMPT
    if len(prompt) > 1800:
        prompt = prompt[:1800]

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("LLM Error:", e)
        return "⚠️ Try again"

    #  LIMIT OUTPUT (WhatsApp safe)
    if len(answer) > 900:
        answer = answer[:900]

    return answer