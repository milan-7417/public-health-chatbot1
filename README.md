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

An intelligent **AI-powered Public Health Assistant** that provides medical guidance, report analysis, outbreak alerts, and multilingual conversational support using **RAG (Retrieval-Augmented Generation)** and **LLMs**.

---

## 🚀 Features

### 💬 Smart AI Chatbot

* Context-aware responses using **RAG + FAISS**
* Powered by **Groq LLM (Llama 3.1)**
* Maintains conversational history

### 🌍 Multilingual Support

* Supports **English, Hindi, Odia**
* Real-time translation pipeline

### 📄 Medical Report Analysis

* Upload PDF reports
* Extract:

  * Key findings
  * Abnormal values
  * Health advice

### 🦠 Outbreak Alerts

* Fetches latest health alerts
* Displays real-time disease updates

### 📱 WhatsApp Integration

* Chat directly via WhatsApp using **Twilio API**

### 🖥️ Web Interface

* Clean UI using **HTML, CSS, JavaScript**
* Sidebar chat history
* Alerts section

---

## 🧠 Tech Stack

### 🔹 Backend

* FastAPI
* FAISS (Vector Search)
* Groq API (LLM)
* PyPDF (Report parsing)

### 🔹 Frontend

* HTML
* CSS
* JavaScript

### 🔹 AI/ML

* BAAI/bge-small-en (Embeddings)
* RAG Architecture

### 🔹 Deployment

* Hugging Face Spaces (Docker)
* GitHub (Version Control)

---

## 🏗️ Project Structure

```
public-health-chatbot/
│
├── backend/
│   ├── main.py
│   ├── rag.py
│   ├── memory.py
│   ├── outbreak.py
│   ├── translation.py
│   └── vector_db/
│       ├── faiss.index
│       └── documents.npy
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/public-health-chatbot.git
cd public-health-chatbot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_api_key
```

### 5️⃣ Run Application

```bash
uvicorn backend.main:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

## ☁️ Deployment (HuggingFace Spaces)

### Steps:

1. Create a **Docker Space**
2. Push project files
3. Add environment variables
4. Auto-deploy 🚀

Live URL:

```
https://your-space-name.hf.space
```

---

## 📲 WhatsApp Integration

* Uses **Twilio WhatsApp API**
* Configure webhook:

```
https://your-space-name.hf.space/whatsapp
```

---

## 🧠 How It Works

1. User query → translated to English
2. FAISS retrieves relevant context
3. LLM generates response
4. Response translated back
5. Returned to user

---

## ⚡ Optimizations

* Lazy model loading
* Memory-efficient FAISS usage
* Reduced embedding size
* Fast inference using Groq

---

## 🔮 Future Enhancements

* Voice input support 🎤
* Image-based diagnosis 📷
* Advanced medical datasets
* User authentication
* Mobile app integration

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork and submit PRs.

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Milan Kumar & Shoaib Ahmad**

---

## ⭐ Acknowledgements

* Hugging Face
* Groq
* FastAPI
* FAISS
* Twilio

---

## 💡 Note

> This project is for **educational purposes only** and does not replace professional medical advice.

---
