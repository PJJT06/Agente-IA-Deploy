import json
from datetime import datetime

def log_metrics(query, context, response_time):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response_time": response_time,
        "empty_context": (context is None or len(context.strip()) == 0),
        "context_used": context
    }

    with open("metrics.json", "a") as f:
        f.write(json.dumps(entry) + "\n")
