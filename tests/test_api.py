import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as app_module


def _client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health():
    r = _client().get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_chat_greeting():
    r = _client().post("/api/chat", json={"message": "Hey"})
    assert r.status_code == 200
    assert r.get_json()["intent"] == "GREETING"


def test_chat_empty_rejected():
    r = _client().post("/api/chat", json={"message": "   "})
    assert r.status_code == 400


def test_calculate():
    r = _client().post("/api/calculate", json={"expression": "25% of 800"})
    assert r.status_code == 200
    assert r.get_json()["answer"] == "200"
