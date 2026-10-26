import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.context import ConversationContext


def test_context_stores_history():
    ctx = ConversationContext()
    ctx.add_exchange("What is Python?", "Python is...", "PYTHON", "python", 0.95, topic="python")
    assert ctx.last_intent == "PYTHON"
    assert ctx.last_topic == "python"
    assert len(ctx.conversation_history) == 2
