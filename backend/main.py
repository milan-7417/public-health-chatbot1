from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.rag import rag_answer, call_llm
from backend.outbreak import get_outbreak_alerts
from backend.translation import translate

from pypdf import PdfReader
from twilio.twiml.messaging_response import MessagingResponse

import os
from dotenv import load_dotenv

load_dotenv()

uploaded_report_text = ""

app = FastAPI()

# ✅ Serve frontend (IMPORTANT for HuggingFace)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# ✅ CORS (safe for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# CHAT
# =========================
@app.post("/chat")
def chat(data: dict):

    query = data.get("query")
    language = data.get("language", "en")

    if not query:
        return {"answer": "⚠️ Empty query"}

    global uploaded_report_text

    try:
        # 🔹 Translate to English
        if language == "hi":
            query = translate(query, "hin_Deva", "eng_Latn")

        elif language == "or":
            query = translate(query, "ory_Orya", "eng_Latn")

        # 🔹 Use RAG (IMPORTANT FIX)
        answer = rag_answer(query)

        # 🔹 Translate back
        if language == "hi":
            answer = translate(answer, "eng_Latn", "hin_Deva")

        elif language == "or":
            answer = translate(answer, "eng_Latn", "ory_Orya")

    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error generating response"

    return {"answer": answer}


# =========================
# ALERTS
# =========================
@app.get("/alerts")
def alerts():
    return {"alerts": get_outbreak_alerts()}


# =========================
# REPORT ANALYSIS
# =========================
@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...)):

    global uploaded_report_text

    try:
        reader = PdfReader(file.file)
    except:
        return {"analysis": "⚠️ Unable to read PDF"}

    text = ""

    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        except:
            continue

    if not text.strip():
        return {"analysis": "⚠️ No readable text found"}

    uploaded_report_text = text[:4000]

    prompt = f"""
Analyze this medical report:

{text[:2000]}

Give:
1. Key findings
2. Abnormal values
3. What it means
4. Basic health advice
"""

    answer = call_llm(prompt)

    return {"analysis": answer}


# =========================
# REPORT CHAT
# =========================
@app.post("/report-chat")
def report_chat(data: dict):

    query = data.get("query")
    language = data.get("language", "en")

    global uploaded_report_text

    if not uploaded_report_text:
        return {"answer": "⚠️ Please upload a medical report first."}

    try:
        if language == "hi":
            query = translate(query, "hin_Deva", "eng_Latn")

        elif language == "or":
            query = translate(query, "ory_Orya", "eng_Latn")

        prompt = f"""
You are a medical assistant.

Medical Report:
{uploaded_report_text}

User Question:
{query}

Answer ONLY based on report.
"""

        answer = call_llm(prompt)

        if language == "hi":
            answer = translate(answer, "eng_Latn", "hin_Deva")

        elif language == "or":
            answer = translate(answer, "eng_Latn", "ory_Orya")

    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error generating response"

    return {"answer": answer}


# =========================
# DELETE REPORT
# =========================
@app.post("/delete-report")
def delete_report():
    global uploaded_report_text
    uploaded_report_text = ""
    return {"message": "Report deleted"}


# =========================
# WHATSAPP BOT
# =========================
@app.post("/whatsapp")
async def whatsapp_reply(request: Request):

    form = await request.form()
    incoming_msg = form.get("Body")

    try:
        # 🔥 Use RAG instead of raw LLM (IMPORTANT FIX)
        answer = rag_answer(incoming_msg)

    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error processing request"

    response = MessagingResponse()
    response.message(answer)

    return str(response)