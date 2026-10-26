import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.rule_engine import match_rule
from chatbot.router import _phrase_search_results
from services.web_answer import summarise_web_results, format_web_answer
from utils.text_processing import extract_key_sentences, clean_text
from chatbot.context import ConversationContext as ContextStore
from chatbot.router import process_single_message


def ask(text):
    ctx = ContextStore()
    return process_single_message(text, ctx)["response"]


def test_science_social_gk_maths():
    assert "scientific method" in match_rule("What is science?")["response"]
    assert "Newton" in match_rule("What are Newtons laws?")["response"]
    assert "Photosynthesis" in match_rule("What is photosynthesis?")["response"]
    assert "15 August 1947" in match_rule("When did India get independence?")["response"]
    assert "New Delhi" in match_rule("What is the capital of India?")["response"]
    assert "Jupiter" in match_rule("What is the largest planet?")["response"]
    assert "Mount Everest" in match_rule("What is the tallest mountain?")["response"]
    assert "Algebra" in match_rule("What is algebra?")["response"]
    assert "Pythagoras" in match_rule("What is geometry?")["response"]


def test_reasoning_english_hindi_conversation():
    assert "arithmetic" in ask("Next number in series 2, 4, 6, 8").lower() or "Next term: 10" in ask("Next number in series 2, 4, 6, 8")
    assert "joyful" in ask("Synonym of happy")
    assert "sad" in ask("Antonym of happy")
    assert "parts of speech" in match_rule("What are parts of speech?")["response"].lower()
    assert "Sandhi" in match_rule("sandhi kya hai")["response"] or "sandhi" in match_rule("sandhi")["response"].lower()
    assert "story" in ask("Tell me a story").lower() or "student" in ask("Tell me a story")


def test_text_processing_web_answer():
    assert clean_text("<p>  Hello   world </p>") == "Hello world"
    sents = extract_key_sentences(
        ["Photosynthesis is how plants make food using sunlight. Cats are cute animals."],
        "what is photosynthesis")
    assert any("Photosynthesis" in s for s in sents)
    results = [
        {"title": "Photosynthesis explained", "url": "https://example.com/photo",
         "snippet": "Photosynthesis is how green plants make food using sunlight, water and carbon dioxide.", "source": "DuckDuckGo"},
        {"title": "Other", "url": "https://example.com/other", "snippet": "Unrelated cats.", "source": "DuckDuckGo"},
    ]
    summary = summarise_web_results("what is photosynthesis", results)
    assert "Photosynthesis" in summary["summary"]
    text = _phrase_search_results("what is photosynthesis", results)
    assert "Here's what I found on the web" in text
    assert "https://example.com/photo" in text
    assert "not from my rule database" in text


def test_web_fallback_uses_search_when_rule_is_definition(monkeypatch):
    # Mock the search service used by the router so no internet is needed.
    import chatbot.router as router
    class FakeSearch:
        def search(self, q):
            return [{"title": "Quantum computing", "url": "https://example.com/qc",
                     "snippet": "Quantum computing uses qubits, which can be in superposition, to solve some problems faster than classical computers.", "source": "DuckDuckGo"}]
    monkeypatch.setattr(router, "search_service", FakeSearch())
    ctx = ContextStore()
    out = process_single_message("What is quantum computing?", ctx)
    assert out["intent"] == "WEB_SEARCH"
    assert "Quantum computing" in out["response"]
    assert "https://example.com/qc" in out["response"]
