"""Onboarding + avatar tests (2026): profile is user-scoped, avatar
choices are validated, first-time welcome runs once, and returning
users skip it. No database, no Google, no network: the Flask session
is set directly, which is exactly how the app stores a profile."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


def _client_with_login():
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["google_user_id"] = "google#test-onboarding"
        sess["google_display_name"] = "Test User"
    return client


def test_avatars_are_five_unique_and_files_exist():
    assert len(app_module.AVATARS) == 5
    assert len({a["id"] for a in app_module.AVATARS}) == 5
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for a in app_module.AVATARS:
        path = os.path.join(base, "static", a["file"])
        assert os.path.exists(path), f"missing avatar asset {a['file']}"
        assert os.path.getsize(path) > 100


def test_profile_anonymous_and_avatar_requires_login():
    client = app_module.app.test_client()
    assert client.get("/api/profile").get_json()["logged_in"] is False
    res = client.post("/api/profile/avatar", json={"avatar_id": "maya"})
    assert res.status_code == 401


def test_avatar_validation_and_save():
    client = _client_with_login()
    res = client.post("/api/profile/avatar", json={"avatar_id": "nope"})
    assert res.status_code == 400
    res = client.post("/api/profile/avatar", json={"avatar_id": "zara"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["avatar_id"] == "zara"
    assert data["avatar_url"].endswith("avatars/avatar-3.svg")
    # Changing avatar later must not restart onboarding for a
    # completed user: complete it, change avatar, stay completed.
    with client.session_transaction() as sess:
        sess["onboarding_completed"] = True
    res = client.post("/api/profile/avatar", json={"avatar_id": "luna"})
    assert res.get_json()["onboarding_completed"] is True


def test_welcome_flow_ask_name_then_goal_then_normal_chat():
    client = _client_with_login()
    client.post("/api/profile/avatar", json={"avatar_id": "maya"})
    status = client.get("/api/onboarding/status").get_json()
    assert status["needs_welcome"] is True
    assert status["welcome_message"] == "Hi! Welcome! What should I call you?"
    # Name step: saved, goal question + quick actions returned.
    res = client.post("/api/chat", json={"message": "Kartik"})
    data = res.get_json()
    assert "Nice to meet you, Kartik!" in data["response"]
    assert data["quick_actions"] == app_module.QUICK_ACTIONS
    assert client.get("/api/profile").get_json()["display_name"] == "Kartik"
    # Goal step: the next message is a normal chat request, and
    # onboarding is now complete, so a later chat is not onboarding.
    res = client.post("/api/chat", json={"message": "What is Python?"})
    assert "ONBOARDING" not in res.get_json().get("rule_id", "")
    assert client.get("/api/profile").get_json()["onboarding_completed"] is True
    status = client.get("/api/onboarding/status").get_json()
    assert status["needs_welcome"] is False


def test_returning_user_skips_welcome_and_avatar_page_requires_login():
    client = app_module.app.test_client()
    assert client.get("/avatar").status_code == 302  # redirect to /login
    logged = _client_with_login()
    with logged.session_transaction() as sess:
        sess["profile_avatar_id"] = "maya"
        sess["onboarding_completed"] = True
    assert logged.get("/api/onboarding/status").get_json()["needs_welcome"] is False
    # Anonymous chat is untouched by onboarding.
    anon = app_module.app.test_client()
    res = anon.post("/api/chat", json={"message": "What is Python?"})
    assert res.status_code == 200
    assert res.get_json()["response"]


def test_login_and_avatar_pages_render():
    client = app_module.app.test_client()
    assert b"Welcome to Shibu AI" in client.get("/login").data
    logged = _client_with_login()
    page = logged.get("/avatar").data
    assert b"Choose your avatar" in page
    for name in (b"Maya", b"Aria", b"Zara", b"Luna", b"Nia"):
        assert name in page
    index = client.get("/").data
    assert b"Conversations" in index
    assert b"userAvatar" in index
