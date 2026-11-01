"""Regression tests for the 17 live-test defects found on 2026-11-02
(browser test-and-fix loop, live site at commit e81fefa5).

Each test is named for its defect. No live network, no real keys:
web results are fabricated, provider calls are mocked, and the
Flask tests set the session directly (as the app itself stores it).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

os.environ.setdefault("SHIBU_SQLITE_PATH", "/tmp/shibu-live-defects-tests.db")
os.environ.setdefault("RAG_BACKEND", "sqlite")

from chatbot.context import ConversationContext
from chatbot.router import process_message
import chatbot.router as router
from services.code_service import generate_code, generate_code_for
from services.content_processing_engine import process_presentation
from services.daily_task_service import _friendly_error, build_daily_task_payload
from services.math_service import MathService
from services.meta_service import MetaService
from services.web_answer import filter_results, format_web_answer


def _ctx():
    return ConversationContext()


def _flatten_blocks(blocks):
    out = []
    for block in blocks or []:
        for value in block.values():
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, list):
                        out.extend(str(x) for x in item)
    return "\n".join(out)


# ---------------------------------------------------------------- D1
def test_d1_rail_upload_wired_to_same_picker_as_composer():
    js_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "static", "js", "app.js")
    js = open(js_path, encoding="utf-8").read()
    assert "function openFilePicker()" in js
    assert 'document.getElementById("attachBtn").onclick = openFilePicker;' in js
    assert 'document.getElementById("railUpload").onclick = openFilePicker;' in js


# ---------------------------------------------------------------- D2
def test_d2_howto_web_answer_drops_film_and_book_pollution():
    results = [
        {"title": "How to Make a Killing (2025 film) - Box Office", "url": "http://x/film",
         "snippet": "The film grossed $12 million at the box office. Starring actors. Budget $5 million."},
        {"title": "How to make tea", "url": "http://x/tea",
         "snippet": "Boil water. Pour the boiling water over the tea leaves. Let it steep for 3 minutes. Remove the leaves and serve."},
    ]
    kept = filter_results("how to make tea", results)
    assert len(kept) == 1 and kept[0]["title"] == "How to make tea"
    answer = format_web_answer("how to make tea", results)
    assert "Killing" not in answer and "box office" not in answer.lower()
    assert "Boil water" in answer or "Pour the boiling water" in answer


# ---------------------------------------------------------------- D3
def test_d3_meta_parses_each_plausible_response_shape():
    shapes = [
        {"choices": [{"message": {"content": "hello from choices"}}]},
        {"choices": [{"text": "hello from text"}]},
        {"output": [{"content": [{"type": "text", "text": "hello from output"}]}]},
        {"output_text": "hello from output_text"},
        {"content": [{"type": "text", "text": "hello from content"}]},
        {"text": "hello from top level"},
        {"answer": "hello from answer"},
    ]
    for data in shapes:
        assert MetaService._extract_text(data).startswith("hello from")
    assert MetaService._extract_text({"choices": []}) is None
    assert MetaService._extract_text({"weird": {"nested": [1, 2]}}) is None
    assert MetaService._extract_text({"choices": [{"message": {"content": ""}}]}) is None


def test_d3_meta_unparseable_response_is_a_clean_error_without_payload(monkeypatch):
    import requests

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": {"unexpected": "shape", "secret": "xyz"}}}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    svc = MetaService(api_key="k", model="m")
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    # Round 3: safe diagnostics only — HTTP status + top-level keys,
    # never payload values.
    assert err.startswith("The Meta-compatible API returned a response in an unexpected format")
    assert "HTTP 200" in err and "response keys: choices" in err
    assert "secret" not in err and "xyz" not in err


def test_d3_meta_failure_falls_through_to_openai_then_hf(monkeypatch):
    from services.model_manager import ModelManager
    monkeypatch.setenv("META_API_KEY", "meta-key")
    monkeypatch.setenv("META_MODEL", "meta-model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_FORCE_LOCAL", raising=False)

    class _Meta:
        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            return None, "The Meta-compatible API returned a response in an unexpected format."
    class _OpenAI:
        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            return "OpenAI draft answer", None
    monkeypatch.setattr("services.meta_service.MetaService", _Meta)
    monkeypatch.setattr("services.openai_service.OpenAIService", _OpenAI)
    manager = ModelManager()
    assert manager.mode == "meta"
    text, err = manager.generate([{"role": "user", "content": "plan my day"}])
    assert text == "OpenAI draft answer" and err is None

    # All providers fail: the original Meta error is returned, never
    # mislabelled as another provider's failure.
    class _FailOpenAI:
        def __init__(self, *a, **k):
            pass

        def generate(self, *a, **k):
            return None, "OpenAI returned HTTP 429"
    monkeypatch.setattr("services.openai_service.OpenAIService", _FailOpenAI)
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setattr(ModelManager, "_generate_api",
                        lambda self, *a, **k: (None, "HF failed"))
    manager = ModelManager()
    text, err = manager.generate([{"role": "user", "content": "plan my day"}])
    assert text is None and "Meta-compatible" in err


def test_d3_fallback_template_never_discloses_provider_error(monkeypatch):
    assert "Meta" not in _friendly_error(
        "The Meta-compatible API returned a response in an unexpected format.")
    assert _friendly_error("The Meta-compatible API returned a response in an unexpected format.") == \
        "the language model could not draft this"

    class _StubManager:
        mode = "meta"
        model_id = "meta-model"

        def generate(self, *a, **k):
            return None, "The Meta-compatible API returned a response in an unexpected format."
    monkeypatch.setattr("services.daily_task_service.ModelManager", _StubManager)
    payload, _needs_web = build_daily_task_payload("Plan my day with gym and study", _ctx(), 0.0)
    assert payload is not None
    assert "Meta" not in payload["response"]
    assert "unexpected format" not in payload["response"]
    # Round 3: the template is delivered cleanly — no mention of
    # the language model, no failure disclosure at all.
    for forbidden in ("basic template", "could not draft", "not available",
                      "language model", "Pinecone", "400", "/api/", "X-Pinecone"):
        assert forbidden not in payload["response"]
    assert "Your day plan:" in payload["response"]


# ---------------------------------------------------------------- D4
def test_d4_explain_machine_learning_simply_has_multiple_blocks():
    payload = process_message("Explain machine learning simply", _ctx())
    assert "Key points" in payload["response"]
    assert "patterns" in payload["response"].lower()
    assert "data" in payload["response"].lower()
    presentation = process_presentation("Explain machine learning simply", payload, _ctx())
    assert presentation is not None
    assert len(presentation["blocks"]) > 1


# ---------------------------------------------------------------- D5
def test_d5_web_news_drops_irrelevant_result():
    results = [
        {"title": "Air New Zealand announces new route", "url": "http://x/nz",
         "snippet": "Air New Zealand news: new flights and travel updates."},
        {"title": "AI beats humans at protein folding", "url": "http://x/ai",
         "snippet": "New AI model predicts protein structures with high accuracy."},
    ]
    kept = filter_results("Search latest AI news", results)
    assert len(kept) == 1 and "protein" in kept[0]["title"]
    answer = format_web_answer("Search latest AI news", results)
    assert "Air New Zealand" not in answer
    assert "protein" in answer


# ---------------------------------------------------------------- D7
@pytest.fixture()
def flask_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIBU_SQLITE_PATH", str(tmp_path / "defects.db"))
    monkeypatch.setenv("RAG_BACKEND", "sqlite")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_d7_dev_panel_chat_get_returns_json_403_not_bare_405(flask_client, monkeypatch):
    monkeypatch.setenv("DEV_PANEL", "1")
    res = flask_client.get("/api/chat")
    assert res.status_code == 403
    assert "Developer panel mode" in res.get_json()["error"]
    res = flask_client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 403
    assert "Developer panel mode" in res.get_json()["error"]
    # Panel-only guard: without DEV_PANEL, GET is the plain 405.
    monkeypatch.delenv("DEV_PANEL", raising=False)
    assert flask_client.get("/api/chat").status_code == 405


# ---------------------------------------------------------------- D8
def _signed_in_at_ask_goal(client):
    with client.session_transaction() as sess:
        sess["google_user_id"] = "google#tester1"
        sess["google_email"] = "tester@example.com"
        sess["google_display_name"] = "Tester"
        sess["profile_avatar_id"] = "avatar-1"
        sess["onboarding_completed"] = False
        sess["onboarding_stage"] = "ask_goal"


def test_d8_quick_action_click_at_ask_goal_is_the_goal_not_a_query(flask_client, monkeypatch):
    monkeypatch.delenv("DEV_PANEL", raising=False)
    _signed_in_at_ask_goal(flask_client)
    res = flask_client.post("/api/chat", json={"message": "Ask a question"})
    data = res.get_json()
    assert res.status_code == 200
    assert data["rule_id"] == "onboarding_goal"
    assert "ask me anything" in data["response"].lower()
    assert "Stupid Question" not in data["response"]
    with flask_client.session_transaction() as sess:
        assert sess.get("onboarding_completed") is True


def test_d8_typed_custom_goal_still_falls_through(flask_client, monkeypatch):
    monkeypatch.delenv("DEV_PANEL", raising=False)
    _signed_in_at_ask_goal(flask_client)
    res = flask_client.post("/api/chat", json={"message": "What is Python?"})
    data = res.get_json()
    assert "Python" in data["response"]
    assert data["rule_id"] != "onboarding_goal"


# ---------------------------------------------------------------- D9
def test_d9_memory_store_then_recall():
    ctx = _ctx()
    process_message("remember that my favourite colour is blue", ctx)
    answer = process_message("what is my favourite colour?", ctx)
    assert "blue" in answer["response"].lower()
    assert "black of this chat background" not in answer["response"]


def test_d9_memory_recall_miss_says_not_remembered():
    answer = process_message("what is my favourite colour?", _ctx())
    assert "haven't saved anything" in answer["response"].lower()
    assert "black of this chat background" not in answer["response"]


def test_d9_memory_forget_then_recall_says_not_remembered():
    ctx = _ctx()
    process_message("remember that my favourite colour is blue", ctx)
    process_message("forget my favourite colour", ctx)
    answer = process_message("what is my favourite colour?", ctx)
    assert "From what you asked me to remember" not in answer["response"]
    assert "haven't saved anything" in answer["response"].lower()


# ---------------------------------------------------------------- D10
def test_d10_shorter_is_strictly_shorter():
    ctx = _ctx()
    first = process_message("What is photosynthesis?", ctx)
    shorter = process_message("make it shorter", ctx)
    assert len(shorter["response"]) < len(first["response"])


def test_d10_simpler_is_a_rewording_not_the_same_sentence():
    ctx = _ctx()
    first = process_message("What is photosynthesis?", ctx)
    simpler = process_message("explain that more simply", ctx)
    assert "simple" in simpler["response"].lower()
    assert simpler["response"] != first["response"]
    first_sentence = first["response"].split(".")[0].strip()
    core = simpler["response"].replace("In simple terms:", "").strip().rstrip(".")
    assert core != first_sentence
    assert "sunlight" in simpler["response"].lower()


# ---------------------------------------------------------------- D11
def test_d11_python_reverse_string_template_with_article():
    code, _expl = generate_code("Write Python code to reverse a string and explain it")
    assert code is not None and "s[::-1]" in code


def test_d11_javascript_templates_for_the_same_task_set():
    for task, marker in [("reverse string", "reverseString"),
                         ("factorial", "factorial"),
                         ("fibonacci", "fibonacci"),
                         ("sort", "sort("),
                         ("prime", "isPrime"),
                         ("even odd", "number % 2"),
                         ("palindrome", "isPalindrome")]:
        code, expl = generate_code_for(task, "javascript")
        assert code is not None and marker in code, task
        assert "JavaScript" in expl


def test_d11_followup_switch_to_javascript_keeps_the_task():
    ctx = _ctx()
    first = process_message("Write Python code to reverse a string and explain it", ctx)
    assert "```python" in first["response"]
    assert "s[::-1]" in first["response"]
    second = process_message("now write the same in JavaScript", ctx)
    assert "```javascript" in second["response"]
    assert "reverseString" in second["response"]


# ---------------------------------------------------------------- D12
def test_d12_compound_interest_annuity_exact_example():
    result = MathService().calculate(
        "I save 500 rupees a month at 12 percent annual interest compounded monthly. "
        "How much do I have after 2 years?")
    assert result["success"]
    expected = 500 * (((1 + 0.12 / 12) ** 24 - 1) / (0.12 / 12))
    assert abs(float(result["answer"]) - expected) < 0.01
    assert abs(float(result["answer"]) - 13487) < 1
    assert "r =" in result["calculation"] or "monthly rate" in result["calculation"]


def test_d12_annuity_second_parameter_set_and_end_to_end():
    result = MathService().calculate(
        "I deposit 1000 rupees per month at 6 percent annual interest compounded monthly for 5 years. What do I get?")
    assert result["success"]
    expected = 1000 * (((1 + 0.06 / 12) ** 60 - 1) / (0.06 / 12))
    assert abs(float(result["answer"]) - expected) < 0.01
    payload = process_message(
        "I save 500 rupees a month at 12 percent annual interest compounded monthly. "
        "How much do I have after 2 years?", _ctx())
    assert "13486" in payload["response"]


# ---------------------------------------------------------------- D14
def test_d14_python_javascript_comparison_has_real_rows():
    payload = process_message("Compare Python and JavaScript for web development in a table", _ctx())
    text = payload["response"]
    assert "Django" in text and "Flask" in text
    assert "React" in text and "Node" in text
    assert "dynamically typed" in text.lower()
    assert "browser" in text.lower()
    assert "won't guess" not in text


def test_d14_quiz_on_the_comparison_quizzes_the_comparison():
    ctx = _ctx()
    process_message("Compare Python and JavaScript for web development in a table", ctx)
    quiz = process_message("give me a quiz on that comparison", ctx)
    assert "Django" in quiz["response"]
    assert "JavaScript" in quiz["response"]
    assert "grammar" not in quiz["response"].lower()
    assert "comparative" not in quiz["response"].lower()


# ---------------------------------------------------------------- D15
def test_d15_typo_tolerance_resolves_photosynthesis(monkeypatch):
    monkeypatch.setattr(router.search_service, "search", lambda query: [])
    monkeypatch.setattr(router, "_llm_answer", lambda *a, **k: (None, None))
    payload = process_message("wht is fotosintesis", _ctx(), web_search_enabled=False)
    assert "photosynthesis" in payload["response"].lower()
    assert "I'm not sure" not in payload["response"]


def test_d15_gibberish_still_falls_back(monkeypatch):
    monkeypatch.setattr(router.search_service, "search", lambda query: [])
    monkeypatch.setattr(router, "_llm_answer", lambda *a, **k: (None, None))
    payload = process_message("xqzt blorp zzzqq", _ctx(), web_search_enabled=False)
    assert "I'm not sure" in payload["response"]


# ---------------------------------------------------------------- D17
def test_d17_multipart_explain_facts_question_in_order():
    payload = process_message(
        "Explain black holes to a 10 year old, then give 3 fun facts, "
        "then ask me one question about it", _ctx())
    text = payload["response"]
    assert "black hole" in text.lower()
    assert "light" in text.lower() or "gravity" in text.lower()
    assert text.count("Fun fact") == 3
    assert "dictionary" not in text.lower()
    assert text.rstrip().endswith("?")
    assert text.index("Fun fact 1") < text.index("Fun fact 3") < text.rindex("?")


def test_d17_standalone_fun_facts_use_the_bank():
    ctx = _ctx()
    process_message("What is photosynthesis?", ctx)
    payload = process_message("give me 2 fun facts", ctx)
    assert payload["response"].count("Fun fact") == 2
    assert "oxygen" in payload["response"].lower()


# ---------------------------------------------------------------- D16
def _signed_in_completed(client):
    with client.session_transaction() as sess:
        sess["google_user_id"] = "google#doctester"
        sess["google_email"] = "doc@example.com"
        sess["google_display_name"] = "Doc Tester"
        sess["profile_avatar_id"] = "avatar-1"
        sess["onboarding_completed"] = True


class _NoEvidenceRAG:
    def rag_answer(self, user_id, question, context=None, **kwargs):
        return {"answer": "", "sources": [], "backend": "sqlite",
                "evidence_found": False, "error": ""}


def test_d16_no_evidence_message_depends_on_whether_documents_exist(flask_client, monkeypatch):
    import app as app_module
    from services.sqlite_store import SQLiteStore
    monkeypatch.delenv("DEV_PANEL", raising=False)
    monkeypatch.setattr(app_module, "_rag_service", lambda: _NoEvidenceRAG())

    # No documents: upload wording, never an internal API path.
    _signed_in_completed(flask_client)
    res = flask_client.post("/api/chat", json={"message": "What does my document say about refunds?"})
    text = res.get_json()["response"]
    assert "not uploaded any documents" in text
    assert "/api/rag/ingest" not in text

    # Has documents: rephrasing wording, no upload instruction.
    monkeypatch.setattr(SQLiteStore, "list_documents",
                        lambda self, user_id: [{"document_id": "d1", "source_name": "notes.pdf"}])
    res = flask_client.post("/api/chat", json={"message": "What does my document say about refunds?"})
    text = res.get_json()["response"]
    assert "none of them cover" in text
    assert "/api/rag/ingest" not in text
    assert "Upload a document" not in text


# ---------------------------------------------------------------- D-C6
def test_dc6_template_week_plan_has_no_quiz_block(monkeypatch):
    class _StubManager:
        mode = "meta"
        model_id = "meta-model"

        def generate(self, *a, **k):
            return None, "The Meta-compatible API returned a response in an unexpected format."
    monkeypatch.setattr("services.daily_task_service.ModelManager", _StubManager)
    payload, _needs_web = build_daily_task_payload(
        "Plan my week for the gym and study for an exam", _ctx(), 0.0)
    assert payload is not None and "Template" in payload["tool"]
    presentation = process_presentation(
        "Plan my week for the gym and study for an exam", payload, _ctx())
    assert presentation is not None
    assert not [b for b in presentation["blocks"] if b.get("kind") == "quiz"]


def test_dc6_genuine_quiz_request_still_gets_a_quiz_block():
    payload = {"response": ("Machine learning is a field of artificial intelligence that learns patterns from data. "
                            "Machine learning is used in 80% of search systems. "
                            "For example, machine learning powers recommendations.\n"
                            "Key points:\n- Machine learning learns from examples.\n"
                            "- Machine learning improves with more data."),
               "intent": "UNKNOWN", "rule_id": "definition", "confidence": 0.3}
    presentation = process_presentation("Give me a quiz on machine learning", payload, _ctx())
    assert presentation is not None
    assert [b for b in presentation["blocks"] if b.get("kind") == "quiz"]


# ---------------------------------------------------------------- D-C3
def test_dc3_running_total_uses_subtracted_for_subtraction():
    ctx = _ctx()
    process_message("564+2674+679+45467+549", ctx)
    subtracted = process_message("subtract 8000 from previous answer", ctx)
    assert "41933" in subtracted["response"]
    assert "subtracted from" in subtracted["response"]
    assert "added to" not in subtracted["response"]
    added = process_message("add 100 to previous answer", ctx)
    assert "added to" in added["response"]


# ---------------------------------------------------------------- D-Setup
def test_dsetup_announcement_bar_hidden_when_text_empty():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css = open(os.path.join(base, "static", "css", "style.css"), encoding="utf-8").read()
    assert ".announcement.hidden" in css
    assert css.index(".announcement {") < css.index(".announcement.hidden")
    js = open(os.path.join(base, "static", "js", "app.js"), encoding="utf-8").read()
    assert 'if (!text) return;' in js
    html = open(os.path.join(base, "templates", "index.html"), encoding="utf-8").read()
    assert 'id="announcement" class="announcement hidden"' in html
