---
title: Public Health Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: backend/main.py
pinned: false
---
# 🩺 Public Health AI Chatbot

An intelligent multilingual AI-powered health assistant that provides information about diseases, medical reports, and public health alerts.

Built using FastAPI + RAG + Local Embeddings + LLM + WhatsApp integration.

🚀 Features
💬 Chatbot

Ask questions about diseases, vaccines, treatments

Context-aware responses using RAG

Maintains conversation flow

🌍 Multilingual Support

Supports:

English 🇬🇧

Hindi 🇮🇳

Odia 🇮🇳

Automatic translation (query + response)

📄 Medical Report Analyzer

Upload PDF medical reports

Extracts and analyzes:

Key findings

Abnormal values

Meaning

Health advice

Ask follow-up questions about report

📢 Outbreak Alerts

Live health news & outbreak alerts

Auto-refresh every 60 seconds

📱 WhatsApp Integration

Chat with bot directly via WhatsApp

Works using Twilio Sandbox

🧠 Architecture
User → Frontend → FastAPI Backend → RAG → LLM → Response
🔹 RAG (Retrieval Augmented Generation)

Uses docs.json instead of vector DB

Embeddings generated using:

BAAI/bge-small-en

Retrieves top relevant context using cosine similarity

📂 Project Structure
public-health-chatbot/
│
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── rag.py                   # RAG logic 
│   ├── memory.py                # Chat memory
│   ├── outbreak.py              # Alerts API
│   ├── translation.py           # Multilingual support
│   ├── convert_pdf_to_json.py   # Convert PDFs → docs.json
│   └── data/
│       └── docs.json            # Knowledge base
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
├── Dockerfile
├── README.md
└── .env
⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/public-health-chatbot.git
cd public-health-chatbot
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Add Environment Variables

Create .env file:

GROQ_API_KEY=your_groq_api_key
4️⃣ Run Locally
uvicorn backend.main:app --reload

Open:
👉 http://127.0.0.1:8000

🧾 Convert PDFs to Knowledge Base

Use this script:

python backend/convert_pdf_to_json.py

This converts medical PDFs into:

backend/vector_db/docs.json
🐳 Docker Deployment
Build Image
docker build -t health-chatbot .
Run Container
docker run -p 7860:7860 health-chatbot
☁️ Deploy on Hugging Face Spaces

Create Space (SDK: Docker or FastAPI)

Upload project files

Add environment variables:

GROQ_API_KEY

App will auto-deploy

📱 WhatsApp Integration (Twilio)
Steps:

Go to Twilio Sandbox

Join sandbox via WhatsApp

Set webhook:

https://your-app-url/whatsapp

Start chatting 🚀

🔥 Tech Stack

Backend: FastAPI

Frontend: HTML, CSS, JavaScript

LLM: Groq (LLaMA 3)

Embeddings: SentenceTransformers (bge-small-en)

Translation: IndicTrans

PDF Parsing: PyPDF

Messaging: Twilio WhatsApp API

⚡ Key Highlights

❌ No FAISS / Vector DB

✅ Lightweight & fast (docs.json based)

✅ Multilingual AI system

✅ Real-time health alerts

✅ WhatsApp chatbot ready

⚠️ Disclaimer

This AI assistant provides general health information only.
For medical diagnosis or treatment, consult a certified doctor.

👨‍💻 Author

Milan Kumar
Shoaib Ahmad

⭐ Future Improvements

FAISS integration for faster retrieval

Voice assistant support

User authentication

Personalized health tracking

💡 Contribution

Feel free to fork and improve this!