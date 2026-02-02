import os
import json
from datetime import datetime
from src.config.settings import FEEDBACK_FILE

def save_feedback(query: str, answer: str, rating: str, user_id: str = None):
    """
    Saves user feedback to a JSON file.
    user_id is optional (defaults to None) to support anonymous feedback.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id if user_id else "anonymous",
        "question": query,
        "answer": answer,
        "rating": rating
    }

    data = []
    # Check if file exists and is not empty
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = [] # Reset if file is corrupted

    data.append(entry)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return True