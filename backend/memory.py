chat_history = []
MAX_HISTORY = 8


def add_to_history(user, bot):

    chat_history.append({
        "user": user,
        "bot": bot
    })

    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)


def get_history():

    history_text = ""

    for msg in chat_history:
        history_text += f"User: {msg['user']}\n"
        history_text += f"Bot: {msg['bot']}\n"

    return history_text