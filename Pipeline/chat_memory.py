from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def update_chat_history(query: str, answer: str):
    """Add a turn to history and trim to last 6 turns (12 messages)."""
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
    
    # Keep only last 6 turns
    if len(chat_history) > 12:
        del chat_history[:-12]

def get_chat_history():
    return chat_history

def clear_chat_history():
    chat_history.clear()