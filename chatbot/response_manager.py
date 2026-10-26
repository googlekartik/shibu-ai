"""Formats the final response payload sent to the UI / API."""
from chatbot.confidence import classify


def format_response(*, text, intent, rule_id, confidence, tool="rule_engine",
                    action="Answer from rule base", extra=None, processing_ms=0.0):
    payload = {
        "response": text,
        "intent": intent,
        "rule_id": rule_id,
        "confidence": confidence,
        "confidence_label": classify(confidence),
        "tool": tool,
        "action": action,
        "processing_ms": round(processing_ms, 1),
    }
    if extra:
        payload.update(extra)
    return payload
