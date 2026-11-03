"""Model manager for Shibu AI: Google Gemini only.

Gemini is the ONLY AI provider. There is no second provider, no
local-model mode, and no fallback leg: generate() makes at most
ONE Gemini call per request. When Gemini is unconfigured or the
call fails, generate() returns (None, error) and callers use the
deterministic fallback (rules / maths / retrieved documents /
search snippets, with a labelled limitation) — never another
provider, never a fabricated answer.

Modes, decided at construction from the environment:
- "gemini": GEMINI_API_KEY is set.
- "off":    no key. The rule engine is the whole brain, and every
            status call says so plainly instead of pretending a
            model is running.

Security: the key comes only from GEMINI_API_KEY, never written
to a file, never logged, never returned by any status endpoint.
Diagnostics are structure-only (HTTP status, response key names,
candidate structure, usage key names — never message text, never
the key), matching services/gemini_service.py, so /api/llm/status
can diagnose a failure without exposing anything Gemini returned.
"""

import os
import re
import time

# Last failed provider attempt in this process, for /api/llm/status
# diagnostics: provider + HTTP status + response top-level keys
# ONLY. No values, no payload, no key. Updated by generate() on a
# provider failure; read (never written) by status(). ModelManager
# instances are created per request, so process-level state is
# what makes the diagnostic survive to the next status call.
_LAST_PROVIDER_DIAGNOSTIC: dict = {}


_ERROR_TYPE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,40}$")


def _record_provider_diagnostic(provider: str, diagnostic) -> None:
    if isinstance(diagnostic, dict):
        error_type = diagnostic.get("error_type")
        if not (isinstance(error_type, str) and _ERROR_TYPE_RE.match(error_type)):
            error_type = None
        _LAST_PROVIDER_DIAGNOSTIC.clear()
        _LAST_PROVIDER_DIAGNOSTIC.update({
            "provider": provider,
            "status": diagnostic.get("status"),
            "response_keys": list(diagnostic.get("response_keys") or []),
            "choice_structure": diagnostic.get("choice_structure"),
            "usage_keys": diagnostic.get("usage_keys"),
            "error_type": error_type,
        })
        try:
            from services.diagnostic_store import write_diagnostic
            write_diagnostic("provider", dict(_LAST_PROVIDER_DIAGNOSTIC))
        except Exception:
            pass


def _provider_diagnostic() -> dict | None:
    """Last provider failure: process memory first, then the
    durable sidecar (a fresh process after a restart)."""
    if _LAST_PROVIDER_DIAGNOSTIC:
        return dict(_LAST_PROVIDER_DIAGNOSTIC)
    try:
        from services.diagnostic_store import read_diagnostic
        return read_diagnostic("provider")
    except Exception:
        return None


def _clear_provider_diagnostic() -> None:
    """A later success supersedes a recorded failure: without
    this, /api/llm/status would report a stale failure forever
    (the sidecar survives restarts), even after Gemini started
    answering again. Best-effort, never raises."""
    _LAST_PROVIDER_DIAGNOSTIC.clear()
    try:
        from services.diagnostic_store import delete_diagnostic
        delete_diagnostic("provider")
    except Exception:
        pass


# Last router decision in this process, for /api/llm/status:
# task kind, chosen provider ("gemini" always), stable reason code.
# No prompt text, no payload values. Process-level for the same
# reason as _LAST_PROVIDER_DIAGNOSTIC.
_LAST_ROUTER_DECISION: dict = {}


def _record_router_decision(decision: dict, fallback_used: bool = False) -> None:
    _LAST_ROUTER_DECISION.clear()
    _LAST_ROUTER_DECISION.update({
        "task_kind": decision.get("task_kind"),
        "provider": decision.get("provider"),
        "reason": decision.get("reason"),
        "fallback_used": False,  # Gemini-only: there is no fallback provider.
    })
    try:
        from services.diagnostic_store import write_diagnostic
        write_diagnostic("router", dict(_LAST_ROUTER_DECISION,
                                        recorded_at=int(time.time())))
    except Exception:
        pass


def _router_decision() -> dict | None:
    """Last router decision: process memory first, then the
    durable sidecar (a fresh process after a restart)."""
    if _LAST_ROUTER_DECISION:
        return dict(_LAST_ROUTER_DECISION)
    try:
        from services.diagnostic_store import read_diagnostic
        stored = read_diagnostic("router")
    except Exception:
        return None
    if not isinstance(stored, dict):
        return None
    decision = {k: stored.get(k) for k in
                ("task_kind", "provider", "reason", "fallback_used")}
    if decision.get("provider") is None:
        return None
    return decision


def _router_preference() -> str:
    try:
        from services.provider_router import router_preference
        return router_preference()
    except Exception:
        return "auto"


def _router_policy_version() -> int:
    try:
        from services.provider_router import POLICY_VERSION
        return POLICY_VERSION
    except Exception:
        return 2


class ModelManager:
    """Gemini-only manager: select -> generate (one call) -> status.

    `mode` is "gemini" when GEMINI_API_KEY is set, else "off".
    `model_id` is always the Gemini model id. `effective_model`
    is the model that actually answered the last successful call.
    """

    def __init__(self, token=None, model=None, use_local=None, gemini_key=None):
        # Legacy parameters (token/use_local) are accepted and
        # IGNORED: no other provider or local mode exists. They stay
        # in the signature only so old call sites do not crash.
        self.gemini_key = (gemini_key if gemini_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.gemini_model = (os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.5-flash")
        self.cloud_only = os.environ.get("CLOUD_ONLY", "").strip().lower() in ("1", "true", "yes")
        self.model_id = self.gemini_model
        self.effective_model = self.gemini_model
        self.mode = "gemini" if self.gemini_key else "off"
        self._load_error = None

    # -- status / selection ------------------------------------------------
    def status(self):
        """Safe to expose: never includes the key or any secret."""
        decided = _router_decision()
        mode = self.mode
        info = {
            "mode": mode,
            "model": self.model_id if self.mode == "gemini" else self.gemini_model,
            "loaded": bool(self.gemini_key),
            "device": "api" if self.mode == "gemini" else None,
            "ram_gb": None,
            "gpu": False,
            "cloud_only": self.cloud_only,
            "cached": None,
            "recommendation": "",
            "error": self._load_error,
            "provider_error": _provider_diagnostic(),
            "router": {
                "preference": _router_preference(),
                "last_decision": decided,
                "policy_version": _router_policy_version(),
            },
        }
        if self.mode == "off":
            info["how_to_enable"] = (
                "Set GEMINI_API_KEY in .env (locally) or in the cloud "
                "host's Environment Variables (Google AI Studio key). "
                "Until then, Shibu AI answers from its rule engine.")
        return info

    def select(self):
        """Report the model that would be used, without any network call."""
        if self.mode == "gemini":
            return {"mode": "gemini", "model": self.model_id,
                    "note": "Google Gemini API — the only AI provider."}
        return {"mode": "off", "model": self.model_id,
                "note": "No LLM is enabled. Set GEMINI_API_KEY to enable Gemini."}

    # -- loading -----------------------------------------------------------
    def load(self):
        """No local model exists to load. Gemini needs no download."""
        if self.mode == "gemini":
            return True, f"Gemini API mode ready: {self.model_id} (Google Gemini API)."
        return False, "No language model is enabled. Set GEMINI_API_KEY — see README section 13."

    # -- generation --------------------------------------------------------
    def generate(self, messages, max_new_tokens=256, temperature=0.7, task_kind=None):
        """Generate a reply for a chat history — ONE Gemini call max.

        `messages` is a list of {"role": "user"|"assistant"|"system",
        "content": str}. Returns (text, error). text is None on
        failure, so callers can fall back to the deterministic
        pipeline instead of showing a traceback or an invented
        answer. The router decision is recorded (diagnostics only)
        before the call; on failure the structure-only diagnostic
        is recorded for /api/llm/status.
        """
        if self.mode == "off":
            return None, "No language model is enabled."
        from services.provider_router import classify
        decision = classify(task_kind, messages)
        text, err = self._attempt_gemini(messages, max_new_tokens, temperature)
        if text:
            _clear_provider_diagnostic()
            _record_router_decision(decision)
            return text, None
        _record_router_decision(decision)
        return None, err or "The language model could not draft this."

    def _attempt_gemini(self, messages, max_new_tokens, temperature):
        """One Gemini call. Records the safe diagnostic on failure."""
        from services.gemini_service import GeminiService
        svc = GeminiService(api_key=self.gemini_key, model=self.model_id)
        text, err = svc.generate(messages, max_tokens=max_new_tokens, temperature=temperature)
        if text:
            self.effective_model = self.model_id
            return text, err
        _record_provider_diagnostic("gemini", getattr(svc, "last_diagnostic", None) or
                                    {"status": None, "response_keys": [],
                                     "error_type": "no_http_response"})
        return text, err

    def test_prompt(self):
        """The documented smoke test: one short prompt, one short answer."""
        text, err = self.generate(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            max_new_tokens=40, temperature=0.0)
        if err:
            return {"ok": False, "error": err}
        return {"ok": True, "model": self.model_id, "reply": text}


def _plain_error(exc):
    """Turn an exception into a helpful message without leaking secrets."""
    msg = re.sub(r"(Bearer|token|key)[=:\s]+\S+", r"\1 [hidden]", str(exc), flags=re.I)
    return f"Model generation failed: {msg[:300]}"
