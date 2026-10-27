import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.context import ConversationContext, ContextStore
from chatbot.router import process_message
import chatbot.router as router


def _ctx():
    return ConversationContext()


def test_classify_self_driving_car():
    # Acceptance 1: exact example from the brief.
    p = process_message("Classify a self-driving car environment.", _ctx())
    assert p["intent"] == "ENVIRONMENT_ANALYSIS"
    assert "environment_analysis" in p
    props = {x["property"]: x["classification"] for x in p["environment_analysis"]["properties"]}
    assert props == {"Observability": "Partially Observable",
                     "Predictability": "Stochastic",
                     "Task Structure": "Sequential",
                     "Environment Dynamics": "Dynamic"}
    assert "Partially Observable" in p["response"]


def test_classify_chess_and_own_system_are_distinct():
    chess = process_message("Classify a chess environment.", _ctx())
    props = {x["property"]: x["classification"] for x in chess["environment_analysis"]["properties"]}
    assert props["Observability"] == "Fully Observable"
    assert props["Predictability"] == "Deterministic"
    own = process_message("Classify Shibu AI's environment.", _ctx())
    assert own["environment_analysis"]["scenario"].startswith("Shibu AI")
    assert own["environment_analysis"]["scenario"] != chess["environment_analysis"]["scenario"]


def test_undetermined_when_information_missing():
    p = process_message(
        "Classify the environment of a robot that cannot see everything and faces random obstacles.",
        _ctx())
    props = {x["property"]: x["classification"] for x in p["environment_analysis"]["properties"]}
    assert props["Observability"] == "Partially Observable"
    assert props["Predictability"] == "Stochastic"
    # No evidence about task structure / dynamics in that sentence.
    assert props["Task Structure"] == "Undetermined"
    assert props["Environment Dynamics"] == "Undetermined"
    assert "No information was given" in p["response"]


def test_explain_it_more_simply_uses_previous_answer():
    # Acceptance 2: "it" = the previous answer.
    ctx = _ctx()
    first = process_message("What is Python?", ctx)
    simpler = process_message("Explain it more simply.", ctx)
    assert "simple" in simpler["response"].lower()
    assert first["response"].split(".")[0] in simpler["response"]


def test_another_example_retains_previous_topic():
    # Acceptance 3.
    ctx = _ctx()
    process_message("What is Python?", ctx)
    p = process_message("Give me another example.", ctx)
    assert "python" in p["response"].lower()
    assert "Another example" in p["response"]


def test_unrelated_question_gets_no_forced_table():
    # Acceptance 4.
    p = process_message("What is the capital of India?", _ctx())
    assert "environment_analysis" not in p
    assert "New Delhi" in p["response"]


def test_new_chat_does_not_mix_previous_context():
    # Acceptance 5: separate ContextStore sessions are isolated.
    store = ContextStore()
    a = store.get("session-a")
    process_message("What is Python?", a)
    b = store.get("session-b")  # a genuinely new chat
    p = process_message("Explain it more simply.", b)
    assert p["intent"] == "CLARIFY"  # asks, does not import Python from session-a


def test_ambiguous_followup_asks_instead_of_guessing():
    # Acceptance 6.
    ctx = _ctx()
    process_message("What is Python?", ctx)
    p = process_message("What about the second one?", ctx)
    assert p["intent"] == "CLARIFY"
    assert "guessing" in p["response"].lower() or "which" in p["response"].lower()


def test_long_conversation_rolling_summary():
    # Acceptance 7.
    ctx = _ctx()
    for q in ("What is Python?", "What is AI?", "What is machine learning?"):
        process_message(q, ctx)
    assert ctx.summary
    assert "machine learning" in ctx.summary.lower()
    assert len(ctx.summary) <= 500
    assert len(ctx.recent_pairs()) == 3


def test_warehouse_robot_not_answered_as_ecosystem():
    p = process_message("Tell me about a robot navigating a warehouse environment", _ctx())
    assert p["intent"] == "ENVIRONMENT_ANALYSIS"
    assert "ecosystem" not in p["response"].lower()
    assert p["environment_analysis"]["scenario"].startswith("A robot navigating")


def test_followup_chain_uses_original_answer():
    ctx = _ctx()
    process_message("What is Python?", ctx)
    process_message("Explain it more simply.", ctx)
    p = process_message("Give me another example.", ctx)
    # Second sentence of the real Python answer, not the simpler wrapper.
    assert "readable syntax" in p["response"]


def test_analyzer_failure_still_shows_normal_answer(monkeypatch):
    # Acceptance 8: simulated analyzer failure must not hide the answer.
    def _boom(_text):
        raise RuntimeError("simulated analyzer failure")
    monkeypatch.setattr(router, "analyze_environment", _boom)
    p = process_message("Classify a self-driving car environment.", _ctx())
    assert p["response"]  # a normal (rule) answer, not a crash / empty reply
    assert "environment_analysis" not in p
