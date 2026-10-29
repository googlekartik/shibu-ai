"""Steps 5-7 + Supabase tests (no real AWS, no real Supabase/OpenAI)."""

import pytest

from services.cache_service import (cached_answer_lookup, cached_answer_store,
                                    looks_personal, memory_clear, normalize_query)
from services.retry import _is_transient, retry_on_transient
from services.usage import _pct, get_usage, is_admin


def test_normalize_and_personal_gate():
    assert normalize_query("  What is Python?  ") == "what is python"
    assert normalize_query("What is Python?") == normalize_query("what is python")
    assert looks_personal("what is my password") is True
    assert looks_personal("what is python") is False


def test_memory_cache_roundtrip():
    memory_clear()
    cached_answer_store(None, "what is python", "What is Python?",
                        "Python is a language.", [{"title": "P", "url": "https://example.com"}])
    hit = cached_answer_lookup(None, "what is python")
    assert hit["summary"].startswith("Python is")
    assert hit["cached"] is True
    assert cached_answer_lookup(None, "something else") is None
    memory_clear()


def test_retry_only_transient():
    calls = {"n": 0}

    @retry_on_transient(max_attempts=3, base_delay=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("ProvisionedThroughputExceededException")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3

    calls2 = {"n": 0}

    @retry_on_transient(max_attempts=3, base_delay=0)
    def permanent():
        calls2["n"] += 1
        raise ValueError("ValidationError: bad input")

    with pytest.raises(ValueError):
        permanent()
    assert calls2["n"] == 1  # permanent errors are not retried
    assert _is_transient(RuntimeError("ThrottlingException")) is True


class _FakeDynamo:
    def __init__(self):
        self.tables = {
            "shibu-users": {"TableStatus": "ACTIVE",
                            "ProvisionedThroughput": {"ReadCapacityUnits": 2, "WriteCapacityUnits": 2},
                            "GlobalSecondaryIndexes": [], "TableSizeBytes": 1000, "ItemCount": 1},
            "shibu-conversations": {"TableStatus": "ACTIVE",
                                    "ProvisionedThroughput": {"ReadCapacityUnits": 3, "WriteCapacityUnits": 3},
                                    "GlobalSecondaryIndexes": [{"ProvisionedThroughput": {"ReadCapacityUnits": 3, "WriteCapacityUnits": 3}}],
                                    "TableSizeBytes": 2000, "ItemCount": 2},
            "shibu-messages": {"TableStatus": "ACTIVE",
                               "ProvisionedThroughput": {"ReadCapacityUnits": 8, "WriteCapacityUnits": 8},
                               "GlobalSecondaryIndexes": [{"ProvisionedThroughput": {"ReadCapacityUnits": 2, "WriteCapacityUnits": 2}}],
                               "TableSizeBytes": 3000, "ItemCount": 10},
            "shibu-cached-answers": {"TableStatus": "ACTIVE",
                                     "ProvisionedThroughput": {"ReadCapacityUnits": 3, "WriteCapacityUnits": 3},
                                     "GlobalSecondaryIndexes": [], "TableSizeBytes": 4000, "ItemCount": 3},
        }

    def describe_table(self, TableName):
        return {"Table": self.tables[TableName]}


def test_usage_totals_and_80pct_warning():
    assert _pct(20, 25) == 80.0
    usage = get_usage(region="ap-south-1", dynamodb_client=_FakeDynamo(),
                      cloudwatch_client=None)
    # 2 + (3+3) + (8+2) + 3 = 21 RCU, same for WCU by construction.
    assert usage["totals"]["provisioned_rcu"] == 21
    assert usage["totals"]["provisioned_wcu"] == 21
    assert usage["totals"]["rcu_pct_of_free"] == 84.0
    assert any("80%" in w for w in usage["warnings"])
    assert "Approximate" in usage["note"]


def test_is_admin_requires_env(monkeypatch):
    monkeypatch.delenv("ADMIN_GOOGLE_SUB", raising=False)
    monkeypatch.delenv("ADMIN_USER_ID", raising=False)
    assert is_admin("google#123") is False
    monkeypatch.setenv("ADMIN_GOOGLE_SUB", "123")
    assert is_admin("google#123") is True
    assert is_admin("google#999") is False


def test_supabase_sync_marks_synced_with_mocked_index(monkeypatch):
    import services.supabase_search as sup

    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    from tests.test_dynamodb_store_2026 import _make_store

    with moto.mock_aws():
        store = _make_store()
        store.get_or_create_user("google#1")
        store.set_consent("google#1", True)
        conv = store.create_conversation("google#1", title="chat")
        saved = store.save_exchange("google#1", conv["conversation_id"], "q", "a good answer")
        store.approve_for_learning("google#1", conv["conversation_id"], saved["bot"]["message_id"])
        assert store.list_learning_pending()["items"]
        monkeypatch.setattr(sup, "index_good_answer", lambda message, api_key=None: True)
        result = sup.sync_learning_pending(store)
        assert result["indexed"] == 1 and result["failed"] == 0
        assert store.list_learning_pending()["items"] == []


def test_supabase_search_returns_empty_without_config(monkeypatch):
    import services.supabase_search as sup

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert sup.search_good_answers(None, "google#1", "python") == []
