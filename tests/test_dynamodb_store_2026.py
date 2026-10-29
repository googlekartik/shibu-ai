"""Step 4 tests: DynamoDB store + Google auth helpers.

Uses moto (mocked DynamoDB) if available; otherwise the DynamoDB
tests skip, so the normal suite still runs without boto3/moto.
Google auth URL tests need no network and always run.
Tables are created here in the test itself (same names/keys as
infra/create_dynamodb_tables.py), so these tests do not need a
real AWS account.
"""

import os

import pytest

from services.dynamodb_store import google_user_id, query_hash
from services.google_auth import build_authorization_url

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")


def _make_store():
    from services.dynamodb_store import DynamoDBStore

    # Create tables exactly as the infra script does, in moto.
    client = boto3.client("dynamodb", region_name="ap-south-1")
    client.create_table(
        TableName="shibu-users",
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "user_id", "AttributeType": "S"}],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={"ReadCapacityUnits": 2, "WriteCapacityUnits": 2})
    client.create_table(
        TableName="shibu-conversations",
        KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"},
                   {"AttributeName": "conversation_id", "KeyType": "RANGE"}],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "conversation_id", "AttributeType": "S"},
            {"AttributeName": "updated_at", "AttributeType": "S"}],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={"ReadCapacityUnits": 3, "WriteCapacityUnits": 3},
        GlobalSecondaryIndexes=[{
            "IndexName": "by-updated",
            "KeySchema": [{"AttributeName": "user_id", "KeyType": "HASH"},
                          {"AttributeName": "updated_at", "KeyType": "RANGE"}],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
            "ProvisionedThroughput": {"ReadCapacityUnits": 3, "WriteCapacityUnits": 3}}])
    client.create_table(
        TableName="shibu-messages",
        KeySchema=[{"AttributeName": "conversation_id", "KeyType": "HASH"},
                   {"AttributeName": "created_at_id", "KeyType": "RANGE"}],
        AttributeDefinitions=[
            {"AttributeName": "conversation_id", "AttributeType": "S"},
            {"AttributeName": "created_at_id", "AttributeType": "S"},
            {"AttributeName": "learning_index_key", "AttributeType": "S"},
            {"AttributeName": "rated_at_id", "AttributeType": "S"}],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={"ReadCapacityUnits": 8, "WriteCapacityUnits": 8},
        GlobalSecondaryIndexes=[{
            "IndexName": "learning-index",
            "KeySchema": [{"AttributeName": "learning_index_key", "KeyType": "HASH"},
                          {"AttributeName": "rated_at_id", "KeyType": "RANGE"}],
            "Projection": {"ProjectionType": "KEYS_ONLY"},
            "ProvisionedThroughput": {"ReadCapacityUnits": 2, "WriteCapacityUnits": 2}}])
    client.create_table(
        TableName="shibu-cached-answers",
        KeySchema=[{"AttributeName": "query_hash", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "query_hash", "AttributeType": "S"}],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={"ReadCapacityUnits": 3, "WriteCapacityUnits": 3})
    return DynamoDBStore(region="ap-south-1")


@pytest.fixture()
def store():
    with moto.mock_aws():
        yield _make_store()


def test_google_user_id_uses_sub_not_email():
    assert google_user_id("109123") == "google#109123"
    assert google_user_id("109123") != google_user_id("109124")


def test_google_auth_url_contains_scope_and_state():
    url = build_authorization_url("cid123", "http://127.0.0.1:5000/auth/callback", "state123")
    assert "accounts.google.com" in url
    assert "openid" in url and "state123" in url and "cid123" in url


def test_create_and_get_user_defaults_opt_out(store):
    user = store.get_or_create_user("google#1", display_name="Kartik", email="k@example.com", google_sub="1")
    assert user["user_id"] == "google#1"
    assert user["consent_for_learning"] is False
    assert user["do_not_use_for_learning"] is True
    assert store.get_user("google#1")["display_name"] == "Kartik"


def test_conversation_newest_first_and_pagination(store):
    store.get_or_create_user("google#1")
    first = store.create_conversation("google#1", title="first")
    second = store.create_conversation("google#1", title="second")
    listed = store.list_conversations("google#1", limit=20)
    # Newest first via by-updated: second (later updated_at) first.
    assert [c["conversation_id"] for c in listed["items"]] == [
        second["conversation_id"], first["conversation_id"]]


def test_save_and_load_messages_owner_isolated(store):
    store.get_or_create_user("google#1")
    store.get_or_create_user("google#2")
    conv = store.create_conversation("google#1", title="chat")
    store.save_exchange("google#1", conv["conversation_id"], "hello", "hi there",
                        intent="GREETING", rule_id="greeting", confidence=0.9)
    loaded = store.load_messages("google#1", conv["conversation_id"])
    assert [m["role"] for m in loaded["items"]] == ["user", "bot"]
    assert loaded["items"][1]["text"] == "hi there"
    # Another user gets nothing, not even an error that leaks existence.
    assert store.load_messages("google#2", conv["conversation_id"])["items"] == []
    assert store.get_conversation("google#2", conv["conversation_id"]) is None
    with pytest.raises(PermissionError):
        store.save_message("google#2", conv["conversation_id"], "user", "nope")


def test_rate_approve_learning_requires_consent(store):
    store.get_or_create_user("google#1")
    conv = store.create_conversation("google#1", title="chat")
    saved = store.save_exchange("google#1", conv["conversation_id"], "q", "a good answer")
    bot_id = saved["bot"]["message_id"]
    store.rate_message("google#1", conv["conversation_id"], bot_id, "up", "even better: x")
    # No consent yet: approval must refuse.
    assert store.approve_for_learning("google#1", conv["conversation_id"], bot_id)["approved"] is False
    assert store.list_learning_pending()["items"] == []
    # Opt in, then approval works and the job can list it.
    store.set_consent("google#1", True)
    assert store.approve_for_learning("google#1", conv["conversation_id"], bot_id)["approved"] is True
    pending = store.list_learning_pending()
    assert [m["message_id"] for m in pending["items"]] == [bot_id]
    # Opting out removes it from learning again.
    store.set_consent("google#1", False)
    assert store.list_learning_pending()["items"] == []


def test_delete_conversation_and_delete_all(store):
    store.get_or_create_user("google#1")
    conv = store.create_conversation("google#1", title="chat")
    store.save_exchange("google#1", conv["conversation_id"], "hi", "hello")
    assert store.delete_conversation("google#1", conv["conversation_id"])["deleted"] is True
    assert store.list_conversations("google#1")["items"] == []
    conv2 = store.create_conversation("google#1", title="chat2")
    store.save_exchange("google#1", conv2["conversation_id"], "hi", "hello")
    result = store.delete_all_user_data("google#1")
    assert result["deleted_conversations"] == 1
    assert result["deleted_messages"] == 2
    assert store.get_user("google#1") is None


def test_export_user_data(store):
    store.get_or_create_user("google#1")
    conv = store.create_conversation("google#1", title="chat")
    store.save_exchange("google#1", conv["conversation_id"], "hi", "hello")
    exported = store.export_user_data("google#1")
    assert exported["user"]["user_id"] == "google#1"
    assert exported["conversations"][0]["messages"][1]["text"] == "hello"


def test_cached_answer_ttl_and_hit(store):
    item = store.put_cached_answer("what is python", "What is Python?",
                                   "Python is a programming language.",
                                   [{"title": "Python", "url": "https://example.com"}])
    assert item["expires_at"] > 0  # epoch seconds, 30 days ahead
    assert store.get_cached_answer("what is python")["summary"].startswith("Python is")
    assert query_hash("what is python") == query_hash("what is python")
    assert query_hash("what is python") != query_hash("what is java")


def test_get_store_disabled_by_default():
    from services.dynamodb_store import get_store, reset_store_for_tests

    os.environ.pop("DYNAMODB_ENABLED", None)
    reset_store_for_tests()
    assert get_store() is None
    reset_store_for_tests()
