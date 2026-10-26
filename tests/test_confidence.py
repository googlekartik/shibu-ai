import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.confidence import calculate_confidence, classify
from chatbot.rule_engine import match_rule


def test_confidence_range():
    for text in ["hi", "What is Python?", "xyz unknown thing"]:
        c = match_rule(text)["confidence"]
        assert 0.0 <= c <= 1.0


def test_classify():
    assert classify(0.9) == "HIGH"
    assert classify(0.5) == "MEDIUM"
    assert classify(0.2) == "LOW"
    assert calculate_confidence(exact=True) == 0.98
