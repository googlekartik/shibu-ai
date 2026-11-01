"""Content-processing / visual-presentation pipeline (2026-11-01).

Tests the ten modules through their public functions and through
the orchestrator, plus the /api/chat integration: maths and
greetings must stay plain, a long structured answer must gain
blocks without changing its response text, and no citation may
be invented.
"""
import os

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-presentation-tests.db")
os.environ.setdefault("RAG_BACKEND", "sqlite")

from services.citation_manager import (normalize_sources, parse_sources_from_text,
                                       validate_citations)
from services.content_extraction_engine import clean_passage, split_sections
from services.content_intent_detector import detect_intent, is_simple_request
from services.content_processing_engine import process_presentation
from services.content_structuring_engine import structure
from services.fact_verification_engine import verify
from services.intelligent_content_analyzer import analyze
from services.interactive_component_renderer import build_quiz
from services.response_quality_controller import validate
from services.visual_response_generator import generate_visuals


LONG_EXPLAIN = (
    "Machine learning is a field of artificial intelligence that learns patterns from data. "
    "Machine learning is used in 80% of search systems according to one survey, and it is used "
    "in many products. For example, machine learning powers recommendations. "
    "Key points:\n- Machine learning learns from examples.\n- Machine learning improves with more data.\n"
    "In conclusion, machine learning helps computers learn from data without explicit rules."
)

COMPARE_TEXT = (
    "Python is a general-purpose programming language. Python is easy to read and is used for web and data work. "
    "Java is a compiled programming language. Java is used for large enterprise systems and Android apps. "
    "Python has simple syntax. Java has strict typing."
)


def test_detect_compare_entities_and_level():
    intent = detect_intent("Compare Python vs Java for beginners")
    assert intent["content_type"] == "compare"
    assert intent["entities"] == ["Python", "Java"]
    assert intent["level"] == "beginner"


def test_detect_quiz_maths_and_simple():
    assert detect_intent("Give me a quiz on machine learning")["content_type"] == "quiz"
    assert detect_intent("Calculate 25% of 800")["content_type"] == "maths"
    assert is_simple_request(detect_intent("hi"), "Hello!")
    assert detect_intent("Explain machine learning for my exam")["level"] == "exam"


def test_citations_never_invented():
    sources = normalize_sources([{"source_name": "Doc A", "type": "document"}])
    assert sources[0]["citation"] == "S1"
    blocks = validate_citations([{"kind": "paragraph", "text": "Claim [S1] and invented [S9]."}], sources)
    assert "[S1]" in blocks[0]["text"]
    assert "[S9]" not in blocks[0]["text"]
    web = parse_sources_from_text(
        "Here's what I found:\n\nSources (from the web):\n1. Example Title\nhttps://example.com/x\n")
    assert web and web[0]["url"] == "https://example.com/x" and web[0]["citation"] == "W1"


def test_extraction_strips_boilerplate_and_splits_sections():
    cleaned = clean_passage("<p>Real fact.</p>\nSubscribe to our newsletter\nAccept cookies")
    assert "Real fact" in cleaned and "Subscribe" not in cleaned
    sections = split_sections("## Types\nMachine learning has types.\n## Uses\nIt is used widely.")
    assert [s["heading"] for s in sections] == ["Types", "Uses"]


def test_analyze_definitions_statistics_and_contradictions():
    passages = [
        {"text": "Machine learning is a field of artificial intelligence.", "source_name": "A"},
        {"text": "Accuracy was 80% in the first study.", "source_name": "A"},
        {"text": "Accuracy was 90% in the second study.", "source_name": "B"},
    ]
    result = analyze(passages, "what is machine learning", content_type="explain")
    assert result["definitions"], "definition sentence was not recognised"
    assert len(result["statistics"]) == 2
    # Same figure pattern, different values, different sources: flagged, not merged.
    assert result["contradictions"], "conflicting figures were not flagged"


def test_verify_corroborated_single_and_unverified():
    analysis = {"facts": [{"text": "Shared fact.", "source_name": "A"},
                          {"text": "Shared fact.", "source_name": "B"},
                          {"text": "Only here.", "source_name": "A"}]}
    out = verify(analysis, [{"citation": "S1"}, {"citation": "S2"}])
    statuses = [c["status"] for c in out["claims"]]
    assert "corroborated" in statuses and "single_source" in statuses
    bare = verify({"facts": [{"text": "No source fact."}]}, [])
    assert bare["claims"][0]["status"] == "unverified"


def test_structure_compare_table_and_steps():
    analysis = analyze([{"text": s, "source_name": "A"} for s in COMPARE_TEXT.split(". ") if s],
                       "compare Python vs Java")
    blocks = structure(COMPARE_TEXT, analysis,
                       detect_intent("Compare Python vs Java"))
    kinds = [b["kind"] for b in blocks]
    assert "table" in kinds
    table = next(b for b in blocks if b["kind"] == "table")
    assert table["headers"][1] == "Python" and table["headers"][2] == "Java"
    steps_text = "How to train a model:\n1. Collect data.\n2. Train the model.\n3. Evaluate the model.\n" + ("More detail. " * 20)
    step_blocks = structure(steps_text, analyze([], "how to train"), detect_intent("How to train a model step by step"))
    assert any(b["kind"] == "steps" for b in step_blocks)


def test_visuals_chart_flowchart_and_no_fake_gallery():
    blocks = [{"kind": "table", "headers": ["Figure"], "rows": [["Accuracy 80%", "A"], ["Accuracy 90%", "B"]]},
              {"kind": "steps", "items": ["Collect data", "Train model", "Evaluate model"]}]
    visuals = generate_visuals(blocks, detect_intent("Show statistics for accuracy"))
    assert any(v["kind"] == "chart" for v in visuals)
    visuals2 = generate_visuals(blocks, detect_intent("How to train a model step by step"))
    assert any(v["kind"] == "flowchart" for v in visuals2)
    assert not any(v["kind"] == "gallery" for v in visuals + visuals2)


def test_quiz_only_when_requested_and_grounded():
    analysis = analyze([{"text": "Machine learning is a field of artificial intelligence.", "source_name": "A"},
                        {"text": "Deep learning is a subset of machine learning.", "source_name": "A"}],
                       "machine learning")
    assert build_quiz(analysis, detect_intent("Explain machine learning")) is None
    quiz = build_quiz(analysis, detect_intent("Quiz me on machine learning"))
    assert quiz and quiz["questions"]
    question = quiz["questions"][0]
    assert question["options"][question["answer_index"]] in question["explanation"]


def test_quality_controller_keeps_simple_plain_and_drops_empty():
    assert validate([{"kind": "paragraph", "text": "Hi"}], "Hi",
                    detect_intent("hi"), []) is None
    assert validate([{"kind": "paragraph", "text": "  "},
                     {"kind": "paragraph", "text": "Real answer."}],
                    "Real answer.", detect_intent("Explain machine learning"), []) is not None


def test_orchestrator_compare_with_sources():
    payload = {"response": COMPARE_TEXT + " " + ("Python and Java are both widely used. " * 12),
               "intent": "WEB_SEARCH", "rule_id": "web_search", "confidence": 0.9,
               "sources": [{"citation": "W1", "source_name": "Example", "url": "https://example.com", "type": "web"}]}
    presentation = process_presentation("Compare Python vs Java", payload)
    assert presentation is not None
    assert presentation["content_type"] == "compare"
    assert any(b["kind"] == "table" for b in presentation["blocks"])
    assert presentation["citations"][0]["citation"] == "W1"
    assert presentation["verification"]["summary"]


def test_orchestrator_raw_request_stays_plain():
    payload = {"response": LONG_EXPLAIN, "intent": "WEB_SEARCH", "rule_id": "web_search", "confidence": 0.9}
    assert process_presentation("Explain machine learning, raw output please", payload) is None


def _client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_chat_maths_and_greeting_have_no_presentation():
    client = _client()
    maths = client.post("/api/chat", json={"message": "Calculate 25% of 800"}).get_json()
    assert "200" in maths["response"] and "presentation" not in maths
    greeting = client.post("/api/chat", json={"message": "hello"}).get_json()
    assert "presentation" not in greeting


def test_chat_running_total_still_works_with_pipeline():
    client = _client()
    first = client.post("/api/chat", json={"message": "564+2674+679+45467+549"}).get_json()
    assert "49933" in first["response"]
    second = client.post("/api/chat", json={"message": "add 967464 in previous answer"}).get_json()
    assert "1017397" in second["response"]
