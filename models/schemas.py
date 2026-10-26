"""Simple request/response shapes (no heavy validation library needed)."""
from dataclasses import dataclass, field


@dataclass
class ChatRequest:
    message: str


@dataclass
class ChatResponse:
    response: str
    intent: str
    rule_id: str
    confidence: float
    confidence_label: str
    tool: str
    action: str
    processing_ms: float = 0.0
    extra: dict = field(default_factory=dict)
