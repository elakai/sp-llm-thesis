# src/core/feedback.py
import os
import json
from datetime import datetime
from src.config.settings import FEEDBACK_FILE

def save_feedback(query: str, answer: str, rating: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "answer": answer,
        "rating": rating
    }

    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

    data.append(entry)

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)