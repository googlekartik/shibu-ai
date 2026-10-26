import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.patterns import RULES
from chatbot.rule_engine import match_rule
from services.math_service import MathService

svc = MathService()


def test_rule_count_grew():
    assert len(RULES) >= 75


def test_new_topics():
    cases = {
        "What is Java?": "JAVA",
        "What is JavaScript?": "JAVASCRIPT",
        "What is Git?": "GIT_GITHUB",
        "What is an operating system?": "OPERATING_SYSTEM",
        "What is cybersecurity?": "CYBERSECURITY",
        "What is cloud computing?": "CLOUD_COMPUTING",
        "What is blockchain?": "BLOCKCHAIN",
        "What is SQL?": "SQL",
        "How do I prepare for an interview?": "INTERVIEW_PREP",
    }
    for question, intent in cases.items():
        assert match_rule(question)["intent"] == intent, question


def test_conversions():
    assert svc.calculate("convert 5 km to miles")["answer"] == "3.106864"
    assert svc.calculate("convert 100 celsius to fahrenheit")["answer"] == "212"
    assert svc.calculate("convert 10 kg to pounds")["answer"].startswith("22.046")
