"""Step 4: DynamoDB data-access layer for Shibu AI.

Main store, not a cache. Every function takes ``user_id`` as its first
argument and that ID must come from the server-side session after
Google login (``google#<sub>``) — never from a request body, query
string, or anything the browser sends. Functions that touch a
conversation first fetch that conversation and check its owner;
a mismatch returns None / False and leaks nothing, not even whether
the conversation exists.

Tables (created by infra/create_dynamodb_tables.py, ap-south-1):
  shibu-users         PK user_id
  shibu-conversations  PK user_id, SK conversation_id, GSI by-updated
  shibu-messages       PK conversation_id, SK created_at_id,
                       GSI learning-index (sparse)
  shibu-cached-answers PK query_hash, TTL on expires_at (shared, no user data)

The store is optional: app.py only creates it when
DYNAMODB_ENABLED=1 (or "true"/"yes"). Without that, or without boto3,
or if DynamoDB is unreachable, get_store() returns None and the app
keeps working in-memory exactly as before. A DynamoDB failure inside
a chat request must never break the chat response — callers catch it.

No secret is stored here. Credentials come from boto3's normal chain
(environment variables, ~/.aws/credentials, or an IAM role).
For local testing, DYNAMODB_ENDPOINT can point at DynamoDB Local.

Privacy rules enforced here, not just in the UI:
  * Learning is allowed only if consent_for_learning is True AND
    do_not_use_for_learning is False. Defaults are False / True
    (opt-in learning, nobody is in by default).
  * Cached web answers are shared and never store a user_id.
    A query that looks personal must never be cached — callers
    decide that before calling put_cached_answer().
  * delete_all_user_data() deletes Supabase search rows first (if
    Supabase is configured), then messages, conversations, then the
    user item, and reports whether the search deletion succeeded.
    If search deletion fails, the DynamoDB deletion still happens
    and search_removed=False tells the caller to retry that part.
  * Supabase removal is best-effort over its REST API and is only
    attempted when SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are
    set (server-only env vars, never the frontend anon key).
    Full Supabase pgvector search arrives in the search step;
    this store only needs to be able to remove a user's rows.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from services.retry import retry_on_transient

MAX_TEXT_CHARS = 20000
MAX_CORRECTION_CHARS = 2000
MAX_SUMMARY_CHARS = 8000
CACHE_TTL_DAYS = 30


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _now_epoch() -> int:
    return int(time.time())


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text[:limit] if len(text) > limit else text


def _decimal(value):
    """DynamoDB (boto3) rejects floats — use Decimal instead.
    None stays None (stored as NULL / removed by callers)."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def google_user_id(google_sub: str) -> str:
    """Stable user key from a Google `sub` claim. Never use email."""
    return f"google#{google_sub}"


def query_hash(normalized_query: str) -> str:
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


class DynamoDBStore:
    """User-scoped DynamoDB access. Create via get_store(), not directly
    in request handlers, so the enabled/disabled logic stays in one place."""

    def __init__(self, region: str | None = None, prefix: str = "shibu",
                 endpoint_url: str | None = None):
        # boto3 is imported here, not at module level, so importing
        # this file (and running the app without DynamoDB) never
        # requires boto3 to be installed.
        import boto3

        self.region = region or os.environ.get("AWS_REGION", "ap-south-1")
        self.prefix = prefix or "shibu"
        self._resource = boto3.resource(
            "dynamodb", region_name=self.region,
            endpoint_url=endpoint_url or os.environ.get("DYNAMODB_ENDPOINT") or None,
        )
        self.users = self._resource.Table(f"{self.prefix}-users")
        self.conversations = self._resource.Table(f"{self.prefix}-conversations")
        self.messages = self._resource.Table(f"{self.prefix}-messages")
        self.cached_answers = self._resource.Table(f"{self.prefix}-cached-answers")

    # -- users ---------------------------------------------------------
    def create_user(self, user_id: str, display_name: str = "",
                    email: str = "", google_sub: str = "") -> dict:
        now = _now_iso()
        item = {
            "user_id": user_id,
            "google_sub": google_sub,
            "display_name": display_name,
            "email": email,
            # Opt-in learning: consent False, do-not-use True by default.
            "consent_for_learning": False,
            "do_not_use_for_learning": True,
            "consent_updated_at": now,
            "created_at": now,
            "updated_at": now,
            "last_login_at": now,
        }
        # First login only: a second simultaneous login must not
        # overwrite the first user's consent choices.
        self.users.put_item(
            Item=item, ConditionExpression="attribute_not_exists(user_id)")
        return item

    def get_or_create_user(self, user_id: str, display_name: str = "",
                           email: str = "", google_sub: str = "") -> dict:
        existing = self.get_user(user_id)
        if existing:
            self.users.update_item(
                Key={"user_id": user_id},
                UpdateExpression="SET #u = :u, #l = :l",
                ExpressionAttributeNames={"#u": "updated_at", "#l": "last_login_at"},
                ExpressionAttributeValues={":u": _now_iso(), ":l": _now_iso()},
            )
            return self.get_user(user_id)
        try:
            return self.create_user(user_id, display_name, email, google_sub)
        except Exception as exc:
            # Race: another request created it between get and put.
            if "ConditionalCheckFailed" in str(exc):
                return self.get_user(user_id)
            raise

    def get_user(self, user_id: str) -> dict | None:
        resp = self.users.get_item(Key={"user_id": user_id})
        return resp.get("Item")

    def set_preferences(self, user_id: str, tone: str, level: str) -> dict | None:
        self.users.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET preferences = :p, updated_at = :u",
            ExpressionAttributeValues={
                ":p": {"tone": tone, "level": level}, ":u": _now_iso()},
        )
        return self.get_user(user_id)

    def set_consent(self, user_id: str, consent_for_learning: bool) -> dict | None:
        """Set both flags consistently. Opting out also removes the
        user's messages from learning (DynamoDB side; search rows are
        removed by _remove_from_search_index)."""
        now = _now_iso()
        self.users.update_item(
            Key={"user_id": user_id},
            UpdateExpression="SET consent_for_learning = :c, "
                             "do_not_use_for_learning = :d, "
                             "consent_updated_at = :u, updated_at = :u",
            ExpressionAttributeValues={
                ":c": bool(consent_for_learning),
                ":d": not bool(consent_for_learning),
                ":u": now,
            },
        )
        if not consent_for_learning:
            self.remove_from_learning(user_id)
        return self.get_user(user_id)

    def _learning_allowed(self, user_id: str) -> bool:
        user = self.get_user(user_id)
        if not user:
            return False
        return bool(user.get("consent_for_learning")) and \
            not bool(user.get("do_not_use_for_learning"))

    # -- conversations --------------------------------------------------
    def create_conversation(self, user_id: str, title: str = "") -> dict:
        now = _now_iso()
        item = {
            "user_id": user_id,
            "conversation_id": f"conv_{uuid.uuid4().hex}",
            "title": _truncate(title, 100),
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
            "message_count": 0,
            "summary": "",
            "active_task": "",
            "last_numeric_answer": None,
            "last_topic": "",
            "last_intent": "",
        }
        self.conversations.put_item(Item=item)
        return item

    def get_conversation(self, user_id: str, conversation_id: str) -> dict | None:
        """Owner-checked: returns None for another user's conversation."""
        resp = self.conversations.get_item(
            Key={"user_id": user_id, "conversation_id": conversation_id})
        return resp.get("Item")

    def list_conversations(self, user_id: str, limit: int = 20,
                           last_key: dict | None = None) -> dict:
        """Newest first, via the by-updated GSI. Paginated."""
        kwargs: dict = {
            "IndexName": "by-updated",
            "KeyConditionExpression": "user_id = :u",
            "ExpressionAttributeValues": {":u": user_id},
            "ScanIndexForward": False,
            "Limit": max(1, min(int(limit), 100)),
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = self.conversations.query(**kwargs)
        return {"items": resp.get("Items", []),
                "last_key": resp.get("LastEvaluatedKey")}

    def update_conversation_metadata(self, user_id: str, conversation_id: str,
                                     summary: str = "", active_task: str = "",
                                     last_numeric_answer=None,
                                     last_topic: str = "", last_intent: str = "") -> dict | None:
        if not self.get_conversation(user_id, conversation_id):
            return None
        self.conversations.update_item(
            Key={"user_id": user_id, "conversation_id": conversation_id},
            UpdateExpression="SET #u = :u, #s = :s, #a = :a, #n = :n, "
                             "#t = :t, #i = :i",
            ExpressionAttributeNames={
                "#u": "updated_at", "#s": "summary", "#a": "active_task",
                "#n": "last_numeric_answer", "#t": "last_topic", "#i": "last_intent",
            },
            ExpressionAttributeValues={
                ":u": _now_iso(), ":s": _truncate(summary, 500),
                ":a": active_task,
                ":n": _decimal(last_numeric_answer), ":t": last_topic, ":i": last_intent,
            },
        )
        return self.get_conversation(user_id, conversation_id)

    # -- messages -------------------------------------------------------
    def save_message(self, user_id: str, conversation_id: str, role: str,
                     text: str, intent: str = "", rule_id: str = "",
                     confidence: float = 0.0, tool: str = "", action: str = "") -> dict:
        if role not in ("user", "bot"):
            raise ValueError("role must be 'user' or 'bot'")
        conv = self.get_conversation(user_id, conversation_id)
        if not conv:
            raise PermissionError("conversation not found for this user")
        now = _now_iso()
        message_id = f"msg_{uuid.uuid4().hex}"
        item = {
            "conversation_id": conversation_id,
            "created_at_id": f"{now}#{message_id}",
            "message_id": message_id,
            "user_id": user_id,
            "role": role,
            "text": _truncate(text, MAX_TEXT_CHARS),
            "intent": intent,
            "rule_id": rule_id,
            "confidence": _decimal(confidence),
            "tool": tool,
            "action": action,
            "created_at": now,
            "rating": None,
            "correction_text": None,
            "approved_for_learning": False,
            "rated_at": None,
            "rated_at_id": None,
            "learning_index_key": None,
        }
        # Remove None values: DynamoDB (and moto) reject None attributes
        # in some paths, and a missing attribute is cleaner than null.
        item = {k: v for k, v in item.items() if v is not None}
        self.messages.put_item(Item=item)
        self.conversations.update_item(
            Key={"user_id": user_id, "conversation_id": conversation_id},
            UpdateExpression="SET #u = :u, #l = :u, "
                             "message_count = if_not_exists(message_count, :z) + :one",
            ExpressionAttributeNames={"#u": "updated_at", "#l": "last_message_at"},
            ExpressionAttributeValues={":u": now, ":z": 0, ":one": 1},
        )
        return item

    @retry_on_transient()
    def save_exchange(self, user_id: str, conversation_id: str,
                      user_text: str, bot_text: str, intent: str = "",
                      rule_id: str = "", confidence: float = 0.0,
                      tool: str = "") -> dict:
        """Save a user+bot pair and return both items. Callers update
        conversation metadata (summary / task / last result) after this."""
        user_item = self.save_message(user_id, conversation_id, "user", user_text)
        bot_item = self.save_message(
            user_id, conversation_id, "bot", bot_text,
            intent=intent, rule_id=rule_id, confidence=confidence, tool=tool)
        return {"user": user_item, "bot": bot_item}

    def load_messages(self, user_id: str, conversation_id: str,
                      limit: int = 50, last_key: dict | None = None,
                      newest_first: bool = False) -> dict:
        if not self.get_conversation(user_id, conversation_id):
            return {"items": [], "last_key": None}
        kwargs: dict = {
            "KeyConditionExpression": "conversation_id = :c",
            "ExpressionAttributeValues": {":c": conversation_id},
            "ScanIndexForward": not newest_first,
            "Limit": max(1, min(int(limit), 100)),
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = self.messages.query(**kwargs)
        return {"items": resp.get("Items", []),
                "last_key": resp.get("LastEvaluatedKey")}

    def _find_message(self, user_id: str, conversation_id: str,
                      message_id: str) -> dict | None:
        """Find a message by its message_id (the UI only knows this ID).
        Walks the conversation's messages; conversations are small."""
        last_key = None
        while True:
            page = self.load_messages(user_id, conversation_id,
                                      limit=100, last_key=last_key)
            for item in page["items"]:
                if item.get("message_id") == message_id:
                    return item
            last_key = page["last_key"]
            if not last_key:
                return None

    def rate_message(self, user_id: str, conversation_id: str, message_id: str,
                     rating: str, correction_text: str = "") -> dict | None:
        if rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'")
        item = self._find_message(user_id, conversation_id, message_id)
        if not item:
            return None
        now = _now_iso()
        self.messages.update_item(
            Key={"conversation_id": conversation_id,
                 "created_at_id": item["created_at_id"]},
            UpdateExpression="SET #r = :r, correction_text = :c, rated_at = :t, "
                             "rated_at_id = :ri",
            ExpressionAttributeNames={"#r": "rating"},
            ExpressionAttributeValues={
                ":r": rating, ":c": _truncate(correction_text, MAX_CORRECTION_CHARS),
                ":t": now, ":ri": f"{now}#{message_id}",
            },
        )
        return self._find_message(user_id, conversation_id, message_id)

    def approve_for_learning(self, user_id: str, conversation_id: str,
                             message_id: str) -> dict:
        """Manual approval (Step 4 default). An up-vote alone never
        puts a message into learning; consent is re-checked here."""
        if not self._learning_allowed(user_id):
            return {"approved": False, "reason": "user has not consented to learning"}
        item = self._find_message(user_id, conversation_id, message_id)
        if not item:
            return {"approved": False, "reason": "message not found"}
        now = _now_iso()
        self.messages.update_item(
            Key={"conversation_id": conversation_id,
                 "created_at_id": item["created_at_id"]},
            UpdateExpression="SET approved_for_learning = :t, "
                             "learning_index_key = :k, "
                             "rated_at = if_not_exists(rated_at, :r), "
                             "rated_at_id = if_not_exists(rated_at_id, :ri)",
            ExpressionAttributeValues={
                ":t": True, ":k": "LEARN_PENDING",
                ":r": now, ":ri": f"{now}#{message_id}",
            },
        )
        return {"approved": True, "message_id": message_id}

    def list_learning_pending(self, limit: int = 50,
                              last_key: dict | None = None) -> dict:
        """Admin / nightly-job view: only LEARN_PENDING messages.
        Returns full messages (GSI is KEYS_ONLY, so each is re-read)."""
        kwargs: dict = {
            "IndexName": "learning-index",
            "KeyConditionExpression": "learning_index_key = :k",
            "ExpressionAttributeValues": {":k": "LEARN_PENDING"},
            "Limit": max(1, min(int(limit), 100)),
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = self.messages.query(**kwargs)
        items = []
        for key_item in resp.get("Items", []):
            full = self.messages.get_item(Key={
                "conversation_id": key_item["conversation_id"],
                "created_at_id": key_item["created_at_id"]}).get("Item")
            if full:
                items.append(full)
        return {"items": items, "last_key": resp.get("LastEvaluatedKey")}

    def mark_learning_synced(self, conversation_id: str, created_at_id: str) -> None:
        self.messages.update_item(
            Key={"conversation_id": conversation_id, "created_at_id": created_at_id},
            UpdateExpression="SET learning_index_key = :k",
            ExpressionAttributeValues={":k": "LEARN_SYNCED"},
        )

    def remove_from_learning(self, user_id: str) -> int:
        """Remove all of one user's messages from learning (opt-out).
        Returns how many were changed. Search rows are removed by the
        caller of set_consent / delete via _remove_from_search_index."""
        changed = 0
        convs = self.list_conversations(user_id, limit=100)
        for conv in convs["items"]:
            last_key = None
            while True:
                page = self.load_messages(user_id, conv["conversation_id"],
                                          limit=100, last_key=last_key)
                for item in page["items"]:
                    if item.get("learning_index_key"):
                        self.messages.update_item(
                            Key={"conversation_id": conv["conversation_id"],
                                 "created_at_id": item["created_at_id"]},
                            UpdateExpression="SET approved_for_learning = :f "
                                             "REMOVE learning_index_key",
                            ExpressionAttributeValues={":f": False},
                        )
                        changed += 1
                last_key = page["last_key"]
                if not last_key:
                    break
        return changed

    # -- delete / export -------------------------------------------------
    def delete_conversation(self, user_id: str, conversation_id: str) -> dict:
        if not self.get_conversation(user_id, conversation_id):
            return {"deleted_messages": 0, "deleted": False}
        self._remove_from_search_index(user_id=user_id,
                                       conversation_id=conversation_id)
        deleted = 0
        last_key = None
        while True:
            page = self.load_messages(user_id, conversation_id,
                                      limit=100, last_key=last_key)
            with self.messages.batch_writer() as batch:
                for item in page["items"]:
                    batch.delete_item(Key={
                        "conversation_id": conversation_id,
                        "created_at_id": item["created_at_id"]})
                    deleted += 1
            last_key = page["last_key"]
            if not last_key:
                break
        self.conversations.delete_item(
            Key={"user_id": user_id, "conversation_id": conversation_id})
        return {"deleted_messages": deleted, "deleted": True}

    def delete_all_user_data(self, user_id: str) -> dict:
        """Export-safe order: search index first, then messages,
        conversations, then the user item. Idempotent."""
        search_removed = self._remove_from_search_index(user_id=user_id)
        total_messages = 0
        total_conversations = 0
        convs = self.list_conversations(user_id, limit=100)
        for conv in convs["items"]:
            result = self.delete_conversation(user_id, conv["conversation_id"])
            total_messages += result["deleted_messages"]
            total_conversations += 1
        self.users.delete_item(Key={"user_id": user_id})
        return {
            "deleted_conversations": total_conversations,
            "deleted_messages": total_messages,
            "deleted_user": True,
            "search_removed": search_removed,
        }

    def export_user_data(self, user_id: str) -> dict:
        user = self.get_user(user_id)
        if not user:
            return {"user": None, "conversations": []}
        conversations = []
        convs = self.list_conversations(user_id, limit=100)
        for conv in convs["items"]:
            messages = []
            last_key = None
            while True:
                page = self.load_messages(user_id, conv["conversation_id"],
                                          limit=100, last_key=last_key)
                messages.extend(page["items"])
                last_key = page["last_key"]
                if not last_key:
                    break
            conversations.append({"conversation": conv, "messages": messages})
        return {"user": user, "conversations": conversations}

    def _remove_from_search_index(self, user_id: str | None = None,
                                  conversation_id: str | None = None):
        """Best-effort Supabase row removal. Returns True if removed
        (or if search is not configured, so there is nothing to remove),
        False if removal was attempted and failed — the caller reports
        that, but DynamoDB deletion is not blocked by it."""
        url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not url or not key:
            return True
        try:
            import requests

            params: dict = {}
            if user_id:
                params["user_id"] = f"eq.{user_id}"
            if conversation_id:
                params["conversation_id"] = f"eq.{conversation_id}"
            resp = requests.delete(
                f"{url}/rest/v1/good_answers", params=params,
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
                timeout=10)
            return resp.status_code in (200, 204)
        except Exception:
            return False

    # -- cached web answers (shared, no user data) -----------------------
    def get_cached_answer(self, normalized_query: str) -> dict | None:
        resp = self.cached_answers.get_item(
            Key={"query_hash": query_hash(normalized_query)})
        item = resp.get("Item")
        if not item:
            return None
        if int(item.get("expires_at", 0)) <= _now_epoch():
            return None  # TTL may not have deleted it yet; treat as miss.
        return item

    def put_cached_answer(self, normalized_query: str, original_query: str,
                          summary: str, sources: list, tool: str = "web",
                          ttl_days: int = CACHE_TTL_DAYS) -> dict:
        sources = [
            {"title": _truncate(str(s.get("title", "")), 200),
             "url": _truncate(str(s.get("url", "")), 500)}
            for s in (sources or [])[:5]
        ]
        item = {
            "query_hash": query_hash(normalized_query),
            "query_normalized": _truncate(normalized_query, 500),
            "query_original": _truncate(original_query, 500),
            "summary": _truncate(summary, MAX_SUMMARY_CHARS),
            "sources": sources,
            "tool": tool,
            "created_at": _now_iso(),
            "expires_at": _now_epoch() + int(ttl_days) * 86400,
            "hit_count": 0,
            "last_hit_at": None,
        }
        item = {k: v for k, v in item.items() if v is not None}
        self.cached_answers.put_item(Item=item)
        return item

    def record_cache_hit(self, normalized_query: str) -> None:
        try:
            self.cached_answers.update_item(
                Key={"query_hash": query_hash(normalized_query)},
                UpdateExpression="SET hit_count = if_not_exists(hit_count, :z) + :one, "
                                 "last_hit_at = :t",
                ExpressionAttributeValues={":z": 0, ":one": 1, ":t": _now_iso()},
            )
        except Exception:
            pass  # A hit counter must never break a chat answer.


_STORE = None
_STORE_Tried = False


def get_store() -> DynamoDBStore | None:
    """Singleton, or None when DynamoDB is disabled / unavailable.
    Enabled only with DYNAMODB_ENABLED=1/true/yes. Any failure here
    returns None instead of raising, so the app keeps working."""
    global _STORE, _STORE_Tried
    if _STORE_Tried:
        return _STORE
    _STORE_Tried = True
    if os.environ.get("DYNAMODB_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return None
    try:
        _STORE = DynamoDBStore(
            prefix=os.environ.get("DYNAMODB_TABLE_PREFIX", "shibu"))
    except Exception:
        _STORE = None
    return _STORE


def reset_store_for_tests() -> None:
    global _STORE, _STORE_Tried
    _STORE = None
    _STORE_Tried = False
