import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.context import ConversationContext
from chatbot.router import process_message


def test_screenshot_sequence_adds_to_previous_answer():
    # Screenshot: 564+2674+679+45467+549 = 49933, then
    # "add 967464 in privious answer" must give 1017397, not a web search.
    ctx = ConversationContext()
    first = process_message("564+2674+679+45467+549", ctx)
    assert "49933" in first["response"]
    assert ctx.active_task.startswith("Running calculation")
    assert ctx.last_numeric_value == 49933
    second = process_message("add 967464 in privious answer", ctx)
    assert second["intent"] == "MATH_FOLLOWUP"
    assert "1017397" in second["response"]
    assert "Answer: 1017397" in second["response"]
    assert second["intent"] != "WEB_SEARCH"
    assert ctx.last_numeric_value == 1017397


def test_chained_operations_and_bare_continuation():
    ctx = ConversationContext()
    process_message("100 + 50", ctx)  # 150
    assert process_message("add 50 in previous answer", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "200"
    assert process_message("add 10", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "210"
    assert process_message("subtract 10 from previous answer", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "200"
    assert process_message("multiply previous answer by 2", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "400"
    assert process_message("previous answer + 5", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "405"
    assert process_message("divide previous answer by 5", ctx)["response"].split("Answer: ")[1].split("\n")[0] == "81"


def test_no_previous_numeric_answer_asks_not_web_searches():
    ctx = ConversationContext()
    p = process_message("add 5 in previous answer", ctx)
    assert p["intent"] == "CLARIFY"
    assert "previous" in p["response"].lower()


def test_previous_non_numeric_answer_asks_not_web_searches():
    ctx = ConversationContext()
    process_message("What is the capital of India?", ctx)
    p = process_message("add 5 in previous answer", ctx)
    assert p["intent"] == "CLARIFY"
    assert p["intent"] != "WEB_SEARCH"


def test_new_topic_replaces_task_memory():
    ctx = ConversationContext()
    process_message("100 + 50", ctx)
    assert ctx.last_numeric_value == 150
    process_message("What is Python?", ctx)
    # A new topic must not leave a stale number to add to.
    assert ctx.last_numeric_value is None
    assert ctx.active_task == "Discussing python"


def test_what_was_previous_answer_shows_it():
    ctx = ConversationContext()
    process_message("What is Python?", ctx)
    p = process_message("What was my previous answer?", ctx)
    assert "Python is" in p["response"]


def test_history_exposes_task_memory():
    ctx = ConversationContext()
    process_message("10 + 20", ctx)
    d = ctx.to_dict()
    assert d["active_task"] == "Running calculation — last result 30"
    assert d["last_numeric_answer"] == "30"
