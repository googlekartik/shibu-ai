"""Intent descriptions used by the explainability panel and docs."""
INTENTS = {
    "GREETING": "User greeted the bot",
    "GOODBYE": "User said goodbye",
    "THANKS": "User thanked the bot",
    "HELP": "User asked for help / capabilities",
    "ABOUT_BOT": "User asked about the chatbot itself",
    "PYTHON": "Question about Python",
    "AI": "Question about artificial intelligence",
    "MACHINE_LEARNING": "Question about machine learning",
    "MATHEMATICS": "Mathematical calculation (SymPy)",
    "WEB_SEARCH": "Current/external information (web search)",
    "PDF_GENERATION": "PDF export requested",
    "UNKNOWN": "No rule matched with sufficient confidence",
}


def describe(intent: str) -> str:
    return INTENTS.get(intent, intent.replace("_", " ").title())
