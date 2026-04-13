# =========================
# 🔹 CHAT MEMORY (FIXED)
# =========================

chat_history = []
MAX_HISTORY = 8


# =========================
# 🔹 ADD TO HISTORY
# =========================
def add_to_history(question, answer):

    chat_history.append({
        "question": question,
        "answer": answer
    })

    # Keep only last N messages
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)


# =========================
# 🔹 GET RAW HISTORY (FOR RAG)
# =========================
def get_history():
    return chat_history   


# =========================
# 🔹 GET FORMATTED HISTORY (OPTIONAL for UI)
# =========================
def get_history_text():

    history_text = ""

    for msg in chat_history:
        history_text += f"User: {msg['question']}\n"
        history_text += f"Bot: {msg['answer']}\n"

    return history_text