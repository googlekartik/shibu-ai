import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.context import ConversationContext
from chatbot.router import process_message
from services.code_service import generate_code, debug_code, compare
from services.code_runner import run_python_code
from services.document_service import summarize, answer_from_document
from services.file_generation_service import generate_pptx, generate_docx, generate_xlsx


def _ctx():
    return ConversationContext()


def test_preferences_saved_and_simple_tone():
    ctx = _ctx()
    out = process_message("set tone to simple", ctx)
    assert ctx.preferences["tone"] == "simple"
    out = process_message("What is Python?", ctx)
    assert "Simple version" in out["response"]


def test_clarifying_question_not_guess():
    out = process_message("help me", _ctx())
    assert out["intent"] == "CLARIFY"
    assert "tell me a little more" in out["response"]


def test_multi_step():
    out = process_message("What is Python? then calculate 25% of 800", _ctx())
    assert out["intent"] == "MULTI_STEP"
    assert "200" in out["response"]


def test_step_by_step_maths():
    out = process_message("Solve step by step 25 + 50", _ctx())
    assert "Step by step" in out["response"]
    assert "75" in out["response"]


def test_code_generation_and_debug():
    code, _ = generate_code("Write Python code for factorial")
    assert "def factorial" in code
    assert "syntax error" in debug_code("def foo(:\n    pass").lower()
    assert "Python" in compare("Python", "Java")


def test_safe_python_runner():
    ok = run_python_code("print(2 + 3)")
    assert ok["success"] and ok["output"].strip() == "5"
    bad = run_python_code("import os\nprint(os.listdir('.'))")
    assert not bad["success"]


def test_document_summary_and_qa(tmp_path):
    text = ("Python is a programming language. Python is used for web development and data science. "
            "Python was created by Guido van Rossum. It is easy to learn and very popular.")
    assert "Python" in summarize(text)
    assert "web development" in answer_from_document("What is Python used for?", text)


def test_file_generation(tmp_path):
    out = str(tmp_path)
    p, f = generate_pptx("Test Topic", ["Point one", "Point two"], out)
    assert os.path.getsize(p) > 0 and f.endswith(".pptx")
    p, f = generate_docx("Test Doc", "Hello world.", out)
    assert os.path.getsize(p) > 0 and f.endswith(".docx")
    p, f = generate_xlsx("Test Sheet", ["Item", "Value"], [["A", 10], ["B", 20]], out)
    assert os.path.getsize(p) > 0 and f.endswith(".xlsx")
