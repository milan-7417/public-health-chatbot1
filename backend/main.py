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
import sys
import os

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# =========================
# ✅ STATIC FILES
# =========================
app.mount("/", StaticFiles(directory=resource_path("frontend"), html=True), name="frontend")


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
# CHAT 
# =========================
@app.post("/chat")
def chat(data: dict):

    query = data.get("query")
    language = data.get("language", "en")

    if not query:
        return {"answer": "⚠️ Empty query"}

    try:
        original_query = query

        

        #  CONTEXT 
        history = get_history()
        if len(query.split()) <= 4:
            query = f"{query} (context: {history})"

        # ✅ PASS LANGUAGE (IMPORTANT FIX)
        answer = rag_answer(query, language)

        # SAVE HISTORY
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

    #  Language instruction
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
from fastapi import Request, Response
from twilio.twiml.messaging_response import MessagingResponse

@app.post("/whatsapp")
async def whatsapp_reply(request: Request):

    try:
        # 🔹 Get incoming data from Twilio
        form = await request.form()
        incoming_msg = form.get("Body", "").strip()

        print("📩 Incoming:", incoming_msg)

        #  Handle empty message
        if not incoming_msg:
            incoming_msg = "Hello"

        #  BASIC SMART HANDLING (VERY IMPORTANT)
        msg_lower = incoming_msg.lower()

        if msg_lower in ["hi", "hello", "hey"]:
            answer = (
                "Hello 👋\n\n"
                "I am your Public Health AI Assistant.\n"
                "You can ask about diseases, symptoms, treatment, or reports.\n\n"
                "Example:\n👉 What is malaria?"
            )

        elif len(incoming_msg.split()) <= 2:
            answer = "Please ask a clear health-related question 😊"

        else:
            #  MAIN RAG CALL
            answer = rag_answer(incoming_msg)

            #  Safety fallback
            if not answer or answer.strip() == "":
                answer = "⚠️ I couldn't generate a response. Please try again."

            #  LIMIT LENGTH (VERY IMPORTANT FOR WHATSAPP)
            if len(answer) > 1200:
                answer = answer[:1200] + "..."

        print("📤 Reply:", answer)

        # Twilio response
        twilio_response = MessagingResponse()
        twilio_response.message(answer)

        xml_response = str(twilio_response)

        print("📦 XML Sent:", xml_response)

        return Response(
            content=xml_response,
            media_type="application/xml"
        )

    except Exception as e:
        print("❌ Error:", e)

        fallback_response = MessagingResponse()
        fallback_response.message("⚠️ Server error. Please try again later.")

        return Response(
            content=str(fallback_response),
            media_type="application/xml"
        )