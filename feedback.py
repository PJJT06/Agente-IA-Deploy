import json
from datetime import datetime

def save_feedback(question, answer, rating):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating
    }

    with open("feedback.json", "a") as f:
        f.write(json.dumps(entry) + "\n")
