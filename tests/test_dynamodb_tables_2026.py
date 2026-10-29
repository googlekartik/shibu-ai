"""Step 2/3 tests: DynamoDB table specs only. No AWS call, no boto3 needed.

These tests check the approved design, not a live account:
4 tables, correct keys, the 2 GSIs, TTL only on cached-answers,
and total provisioned capacity inside the 25 RCU / 25 WCU free allowance.
Live creation is tested separately with a mocked client (moto) and,
by Kartik, by running infra/create_dynamodb_tables.py once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))

from create_dynamodb_tables import TABLE_SPECS, table_name, total_capacity


def _by_suffix(suffix):
    for spec in TABLE_SPECS:
        if spec["suffix"] == suffix:
            return spec
    raise AssertionError(f"missing table spec: {suffix}")


def test_four_tables():
    assert [s["suffix"] for s in TABLE_SPECS] == [
        "users", "conversations", "messages", "cached-answers",
    ]


def test_table_names_use_shibu_prefix():
    assert table_name("shibu", "users") == "shibu-users"
    assert table_name("shibu", "cached-answers") == "shibu-cached-answers"


def test_capacity_inside_free_allowance():
    reads, writes = total_capacity()
    assert (reads, writes) == (21, 21)
    assert reads <= 25 and writes <= 25


def test_conversations_newest_first_gsi():
    spec = _by_suffix("conversations")
    assert spec["key_schema"][0]["AttributeName"] == "user_id"
    assert spec["key_schema"][1]["AttributeName"] == "conversation_id"
    gsi = spec["gsis"][0]
    assert gsi["IndexName"] == "by-updated"
    assert gsi["KeySchema"][1]["AttributeName"] == "updated_at"


def test_messages_learning_gsi_is_sparse_design():
    # Learning GSI keys are learning_index_key / rated_at_id; only
    # approved-for-learning messages carry learning_index_key, so
    # the index is sparse by construction (checked here by key names).
    spec = _by_suffix("messages")
    assert spec["key_schema"][0]["AttributeName"] == "conversation_id"
    assert spec["key_schema"][1]["AttributeName"] == "created_at_id"
    gsi = spec["gsis"][0]
    assert gsi["IndexName"] == "learning-index"
    assert gsi["KeySchema"][0]["AttributeName"] == "learning_index_key"
    assert gsi["Projection"]["ProjectionType"] == "KEYS_ONLY"


def test_ttl_only_on_cached_answers():
    for spec in TABLE_SPECS:
        if spec["suffix"] == "cached-answers":
            assert spec["ttl_attribute"] == "expires_at"
        else:
            assert spec["ttl_attribute"] is None


def test_users_key_is_google_id_ready():
    # user_id is the sole key; Step 4 will store "google#<sub>" in it.
    spec = _by_suffix("users")
    assert spec["key_schema"] == [{"AttributeName": "user_id", "KeyType": "HASH"}]
