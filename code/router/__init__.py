"""Message Notification Router - WhatsApp triage for HackerRank Orchestrate.

Layers:
    dataset.py     CSV loading and joins (stdlib only)
    evidence.py    TF-IDF retrieval over message_history.csv
    features.py    pattern layer - how much this user values this sender
    content.py     content layer - urgency and risk of this message
    modalities.py  text / image / voice branches
    fusion.py      weighted blend plus guardrail overrides
    writer.py      contract-validated output.csv
"""

__version__ = "0.1.0"
