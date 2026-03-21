from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.rag import rag_answer, call_llm
from backend.outbreak import get_outbreak_alerts
from backend.translation import translate
from backend.memory import add_to_history, get_history

from pypdf import PdfReader
from twilio.twiml.messaging_response import MessagingResponse
from langdetect import detect

import os
from dotenv import load_dotenv

load_dotenv()

uploaded_report_text = ""

app = FastAPI()

# =========================
# ✅ STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# =========================
# ✅ CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 🔥 CHAT (FIXED)
# =========================
@app.post("/chat")
def chat(data: dict):

    query = data.get("query")
    language = data.get("language", "en")

    if not query:
        return {"answer": "⚠️ Empty query"}

    try:
        original_query = query

        # 🔥 REMOVE OLD TRANSLATION (rag.py handles it)

        # 🔥 CONTEXT FIX
        history = get_history()
        if len(query.split()) <= 4:
            query = f"{query} (context: {history})"

        # ✅ PASS LANGUAGE (IMPORTANT FIX)
        answer = rag_answer(query, language)

        # 🔥 SAVE HISTORY
        add_to_history(original_query, answer)

    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error generating response"

    return {"answer": answer}


# =========================
# 🔔 ALERTS
# =========================
@app.get("/alerts")
def alerts():
    return {"alerts": get_outbreak_alerts()}


# =========================
# 📄 REPORT ANALYSIS
# =========================
@app.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...), language: str = "en"):

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

    # 🔥 Language instruction
    lang_instruction = "Answer in English"
    if language == "hi":
        lang_instruction = "Answer in Hindi"
    elif language == "or":
        lang_instruction = "Answer in Odia"

    prompt = f"""
You are a professional medical expert.

Analyze the following medical report carefully:

{text[:2000]}

Provide response in structured format:
1. Key Findings
2. Abnormal Values
3. What it means
4. Health Advice

IMPORTANT: {lang_instruction}
"""

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error analyzing report"

    return {"analysis": answer}


# =========================
# 📊 REPORT CHAT (KEEP ONLY ONE)
# =========================
@app.post("/report-chat")
def report_chat(data: dict):

    query = data.get("query")
    language = data.get("language", "en")

    global uploaded_report_text

    if not uploaded_report_text:
        return {"answer": "⚠️ Please upload a medical report first."}

    if not query:
        return {"answer": "⚠️ Empty question"}

    lang_instruction = "Answer in English"
    if language == "hi":
        lang_instruction = "Answer in Hindi"
    elif language == "or":
        lang_instruction = "Answer in Odia"

    prompt = f"""
You are a medical assistant.

Medical Report:
{uploaded_report_text}

User Question:
{query}

Give a clear and accurate answer based ONLY on the report.

IMPORTANT: {lang_instruction}
"""

    try:
        answer = call_llm(prompt)
    except Exception as e:
        print("Error:", e)
        answer = "⚠️ Error generating response"

    return {"answer": answer}


# =========================
# 🗑 DELETE REPORT
# =========================
@app.post("/delete-report")
def delete_report():
    global uploaded_report_text
    uploaded_report_text = ""
    return {"message": "Report deleted"}


# =========================
# 📱 WHATSAPP BOT (FIXED)
# =========================
from fastapi.responses import Response
@app.post("/whatsapp")
async def whatsapp_reply(request: Request):

    form = await request.form()
    incoming_msg = form.get("Body")

    print("📩 Incoming:", incoming_msg)

    if not incoming_msg:
        incoming_msg = "Hello"

    try:
        # 🔹 SIMPLE TEST (FIRST VERIFY)
        answer = rag_answer(incoming_msg)

        if not answer or answer.strip() == "":
            answer = "⚠️ No response generated"

    except Exception as e:
        print("❌ Error:", e)
        answer = "⚠️ Server error"

    print("📤 Reply:", answer)

    # 🔥 IMPORTANT: Twilio response
    twilio_response = MessagingResponse()
    twilio_response.message(answer)

    xml_response = str(twilio_response)

    print("📦 XML Sent:", xml_response)

    # 🔥 CRITICAL FIX
    return Response(
        content=xml_response,
        media_type="application/xml"
    )