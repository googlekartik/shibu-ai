"""Muse-style frontend + anime characters tests (2026): the five
characters, the new layout markers, code-block rendering markers,
identity wording, the Gemini provider chain, in-chat character
choice, and legacy avatar-id aliases. No network, no keys: the
Gemini provider is exercised only with a mocked HTTP response, and
is inactive whenever its env key is absent."""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
from services.model_manager import ModelManager
from services.gemini_service import GeminiService

EXPECTED = [
    ("avatar-1", "Monkey D. Luffy", "avatars/avatar-1.svg"),
    ("avatar-2", "Levi Ackerman", "avatars/avatar-2.svg"),
    ("avatar-3", "Satoru Gojo", "avatars/avatar-3.svg"),
    ("avatar-4", "Itachi Uchiha", "avatars/avatar-4.svg"),
    ("avatar-5", "Eren Yeager", "avatars/avatar-5.svg"),
]


def _logged_client(user_id, avatar_id=None, completed=False):
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = user_id
        sess["google_display_name"] = "Test User"
        if avatar_id is not None:
            sess["profile_avatar_id"] = avatar_id
        if completed:
            sess["onboarding_completed"] = True
    return client


def test_characters_ids_names_and_files():
    assert [(a["id"], a["name"], a["file"]) for a in app_module.AVATARS] == EXPECTED
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _id, _name, _file in EXPECTED:
        path = os.path.join(base, "static", _file)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert content.strip().startswith("<svg")
        assert os.path.getsize(path) > 100


def test_character_assets_serve_and_avatars_api():
    client = app_module.app.test_client()
    for _id, _name, _file in EXPECTED:
        res = client.get("/static/" + _file)
        assert res.status_code == 200
        assert b"<svg" in res.data
    data = client.get("/api/avatars").get_json()
    assert [a["name"] for a in data["avatars"]] == [name for _i, name, _f in EXPECTED]
    assert [a["id"] for a in data["avatars"]] == [i for i, _n, _f in EXPECTED]


def test_layout_markers_in_homepage_css_and_js():
    client = app_module.app.test_client()
    page = client.get("/").data
    for marker in (b'id="rail"', b'id="railChat"', b'id="railHistory"',
                   b'id="railUpload"', b'id="railCharacter"',
                   b'id="characterPanel"', b'id="panelAvatar"',
                   b'id="panelName"', b'id="panelStatus"',
                   b'id="panelHistoryList"', b'id="composer"',
                   b'id="attachBtn"', b'id="sendBtn"'):
        assert marker in page, marker
    css = client.get("/static/css/style.css").data
    for marker in (b".rail", b".character-panel", b".code-block",
                   b".character-cards", b".msg.user"):
        assert marker in css, marker
    js = client.get("/static/js/app.js").data
    for marker in (b"renderCodeBlocks", b"code-copy-btn",
                   b"navigator.clipboard", b"showCharacterCards",
                   b"character-card", b"updateCharacterPanel",
                   b"panelHistoryList"):
        assert marker in js, marker


def test_identity_names_character_and_never_claims_muse():
    client = _logged_client("google#test-muse-identity-1", "avatar-2", completed=True)
    res = client.post("/api/chat", json={"message": "who are you"})
    text = res.get_json()["response"]
    assert "Levi Ackerman" in text
    assert "Shibu AI" in text
    assert "Muse" not in text
    assert "Meta AI" not in text
    res = client.post("/api/chat", json={"message": "are you human"})
    text = res.get_json()["response"]
    assert "Levi Ackerman" in text
    assert "not human" in text.lower()
    # Anonymous chat keeps the plain Shibu AI wording.
    anon = app_module.app.test_client()
    text = anon.post("/api/chat", json={"message": "who are you"}).get_json()["response"]
    assert "Shibu AI" in text
    assert "Levi Ackerman" not in text
    assert "Muse" not in text


def test_profile_reports_character_name():
    client = _logged_client("google#test-muse-identity-2", "avatar-3", completed=True)
    me = client.get("/api/me").get_json()
    assert me["avatar_id"] == "avatar-3"
    assert me["character_name"] == "Satoru Gojo"
    assert me["avatar_url"].endswith("avatars/avatar-3.svg")


def test_gemini_provider_inactive_without_key(monkeypatch):
    for var in ("GEMINI_API_KEY", "HF_TOKEN",
                "HF_FORCE_LOCAL", "HF_USE_LOCAL"):
        monkeypatch.delenv(var, raising=False)
    svc = GeminiService()
    assert svc.enabled is False
    text, err = svc.generate([{"role": "user", "content": "hi"}])
    assert text is None
    assert "GEMINI_API_KEY" in err
    assert ModelManager().mode != "gemini"


def test_gemini_provider_mocked_success_and_manager_mode(monkeypatch):
    for var in ("HF_TOKEN", "HF_FORCE_LOCAL", "HF_USE_LOCAL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "Hello from Gemini."}],
                                                "role": "model"},
                                    "finishReason": "STOP"}]}

    with mock.patch("requests.post", return_value=_Resp()) as post:
        svc = GeminiService()
        assert svc.enabled is True
        text, err = svc.generate([{"role": "user", "content": "hi"}])
        assert text == "Hello from Gemini."
        assert err is None
        assert post.call_args[0][0] == ("https://generativelanguage.googleapis.com/v1beta"
                                        "/models/gemini-2.5-flash:generateContent")
        assert post.call_args[1]["headers"]["x-goog-api-key"] == "test-key"
        mm = ModelManager()
        assert mm.mode == "gemini"
        assert mm.model_id == "gemini-2.5-flash"
        assert mm.status()["loaded"] is True
        text, err = mm.generate([{"role": "user", "content": "hi"}])
        assert text == "Hello from Gemini."


def test_gemini_provider_mocked_error_text(monkeypatch):
    class _Resp:
        status_code = 403
        text = "forbidden"

        def json(self):
            return {"error": {"message": "bad key"}}

    with mock.patch("requests.post", return_value=_Resp()):
        svc = GeminiService(api_key="zzkey", model="m")
        text, err = svc.generate([{"role": "user", "content": "hi"}])
        assert text is None
        assert "403" in err
        assert "zzkey" not in err  # the key value is never echoed


def test_in_chat_character_choice_flow():
    client = _logged_client("google#test-muse-onboarding-1")
    res = client.post("/api/chat", json={"message": "hello"})
    data = res.get_json()
    assert data.get("needs_avatar") is True
    assert len(data.get("avatars", [])) == 5
    res = client.post("/api/profile/avatar", json={"avatar_id": "avatar-3"})
    assert res.status_code == 200
    assert res.get_json()["avatar_id"] == "avatar-3"
    status = client.get("/api/onboarding/status").get_json()
    assert status["needs_welcome"] is True
    assert status["welcome_message"] == "Hi! Welcome! What should I call you?"


def test_legacy_avatar_ids_are_aliases():
    client = _logged_client("google#test-muse-onboarding-2")
    res = client.post("/api/profile/avatar", json={"avatar_id": "zara"})
    assert res.status_code == 200
    assert res.get_json()["avatar_id"] == "avatar-3"
    assert res.get_json()["character_name"] == "Satoru Gojo"
    # A profile saved under a legacy id still resolves on read.
    legacy = _logged_client("google#test-muse-onboarding-3", "maya", completed=True)
    me = legacy.get("/api/me").get_json()
    assert me["avatar_id"] == "avatar-1"
    assert me["character_name"] == "Monkey D. Luffy"
