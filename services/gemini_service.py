"""Google Gemini chat service for Shibu AI — the drafting LLM option.

Why this exists: chat generation used to go through OpenAI's ChatGPT
API; that integration has been removed. Gemini (Google AI Studio)
is the replacement drafting provider, alongside Hugging Face
Inference Providers for short/structured work. Which provider
answers a request is decided per request by services/provider_router.py.

API: Google AI Studio Gemini REST API, no SDK dependency (plain
`requests`, like the other providers):
  POST {GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent
  header: x-goog-api-key: <GEMINI_API_KEY>  (the key is NEVER put
  in the URL query string, and never logged or returned anywhere)
  body: {"contents": [...], "systemInstruction": {...},
         "generationConfig": {"temperature": t, "maxOutputTokens": n}}

Model: gemini-2.5-flash by default (GEMINI_MODEL to change it). The
exact model id must match the owner's Google AI Studio model list.

Security: the key comes only from the GEMINI_API_KEY environment
variable. It is never written to a file by this code, never logged,
never returned by any status endpoint, and never pasted in chat.
This is a DIFFERENT Google credential from Google OAuth sign-in
(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET): the OAuth credentials
identify users, the Gemini key pays for / authorises model calls.
On a cloud host, set it in the host's Environment Variables
dashboard, not in the code or in GitHub. On your PC, put it in
.env (which is in .gitignore, so it is never pushed to GitHub).

Diagnostics are structure-only (HTTP status, response top-level
key names, candidate structure, usage key names — never message
text, never the key, never payload values), matching the other
providers, so /api/llm/status can diagnose a failure without
exposing anything the provider returned.
"""

import os


def _safe_body(resp):
    """Parsed JSON body of a response, or None when unparsable."""
    try:
        return resp.json()
    except Exception:
        return None


def _response_keys(data) -> list:
    if isinstance(data, dict):
        return sorted(str(k) for k in data.keys())
    return []


def _candidate_structure(data) -> dict | None:
    """Structure ONLY of candidates[0]: key names, the content
    part key names, whether text is present, the finishReason enum
    VALUE (a small closed set: STOP/SAFETY/RECITATION/MAX_TOKENS/
    OTHER), and whether any safety rating blocked the response.
    No other values, no payload text, no rating details — so a
    live failure can be diagnosed from /api/llm/status without
    exposing anything the provider returned."""
    if not isinstance(data, dict):
        return None
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return {"candidate_keys": [], "content_keys": None,
                "part_types": None, "content_type": type(first).__name__,
                "finish_reason": None, "blocked_present": False,
                "text_present": False}
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if isinstance(parts, list):
        part_types = sorted({str(k) for p in parts
                             if isinstance(p, dict) for k in p.keys()})
        text_present = any(isinstance(p, dict) and isinstance(p.get("text"), str)
                           and p["text"].strip() for p in parts)
        content_type = "list"
    elif parts is None:
        part_types = None
        text_present = False
        content_type = "NoneType"
    else:
        part_types = None
        text_present = False
        content_type = type(parts).__name__
    finish_reason = first.get("finishReason")
    blocked = finish_reason == "SAFETY"
    ratings = first.get("safetyRatings")
    if isinstance(ratings, list):
        blocked = blocked or any(
            isinstance(r, dict) and r.get("blocked") is True for r in ratings)
    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        blocked = True
    return {
        "candidate_keys": sorted(str(k) for k in first.keys()),
        "content_keys": sorted(str(k) for k in content.keys())
            if isinstance(content, dict) else None,
        "part_types": part_types,
        "content_type": content_type,
        # The finishReason enum value is the decisive fact for an
        # empty 200: "SAFETY" means the response was blocked,
        # "STOP" with no text means an empty generation.
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "blocked_present": bool(blocked),
        "text_present": bool(text_present),
    }


def _usage_keys(data) -> dict | None:
    """Key NAMES of the response "usageMetadata" object only —
    never values. Returns None when there is no usable object."""
    if not isinstance(data, dict):
        return None
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    nested = {}
    for key, value in usage.items():
        if isinstance(value, dict) and value:
            nested[str(key)] = sorted(str(k) for k in value.keys())
    return {"keys": sorted(str(k) for k in usage.keys()), "nested": nested}


def _diagnostic_for(status, body):
    """Structure-only diagnostic for one response (never values)."""
    return {"status": status,
            "response_keys": _response_keys(body),
            "choice_structure": _candidate_structure(body),
            "usage_keys": _usage_keys(body)}


def _message_text(message) -> str:
    """Plain text of one chat message, whatever its shape."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts)
    return ""


def build_request_body(messages, max_tokens=256, temperature=0.7):
    """Map OpenAI-style chat messages to a Gemini generateContent
    body. System messages become systemInstruction; user/assistant
    turns become contents with roles "user"/"model"; consecutive
    same-role turns are merged and empty turns dropped (Gemini
    rejects empty parts and alternating roles are required)."""
    system_texts = []
    contents = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        text = _message_text(message).strip()
        if role == "system":
            if text:
                system_texts.append(text)
            continue
        gemini_role = "model" if role in ("assistant", "model", "bot") else "user"
        if not text:
            continue
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"][0]["text"] += "\n" + text
        else:
            contents.append({"role": gemini_role,
                             "parts": [{"text": text}]})
    body = {"contents": contents,
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens}}
    if system_texts:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
    return body


def _extract_text(data) -> str | None:
    """Joined text of candidates[0].content.parts[].text, or None
    when the shape carries no usable text."""
    if not isinstance(data, dict):
        return None
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        return None
    texts = [p["text"].strip() for p in parts
             if isinstance(p, dict) and isinstance(p.get("text"), str)
             and p["text"].strip()]
    joined = "\n".join(texts).strip()
    return joined or None


class GeminiService:
    def __init__(self, api_key=None, model=None, api_base=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("GEMINI_MODEL", "")).strip() or "gemini-2.5-flash"
        self.api_base = (api_base if api_base is not None
                         else os.environ.get("GEMINI_API_BASE", "")).strip() \
            or "https://generativelanguage.googleapis.com/v1beta"
        self.enabled = bool(self.api_key)
        # Structure-only diagnostic of the last HTTP response
        # (status + response key names + candidate structure + usage
        # key names; never values), for /api/llm/status. None when
        # no HTTP response was received.
        self.last_diagnostic = None

    def status(self):
        # Safe: never includes the key.
        return {"enabled": self.enabled, "model": self.model if self.enabled else None}

    def generate(self, messages, max_tokens=256, temperature=0.7):
        """Return (text, error). text is None on failure."""
        if not self.enabled:
            return None, "No Gemini API key is set (GEMINI_API_KEY)."
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = build_request_body(messages, max_tokens=max_tokens, temperature=temperature)
        if not body["contents"]:
            return None, "Gemini returned a response in an unexpected format."
        url = f"{self.api_base.rstrip('/')}/models/{self.model}:generateContent"
        try:
            import requests
            resp = requests.post(
                url,
                headers={"x-goog-api-key": self.api_key,
                         "Content-Type": "application/json"},
                json=body,
                timeout=60)
            data = _safe_body(resp)
            self.last_diagnostic = _diagnostic_for(resp.status_code, data)
            if resp.status_code == 200:
                if data is None:
                    return None, "Gemini returned a response in an unexpected format."
                struct = self.last_diagnostic["choice_structure"]
                if struct is not None and struct["blocked_present"]:
                    return None, "Gemini blocked the response."
                text = _extract_text(data)
                if text:
                    return text, None
                return None, "Gemini returned a response in an unexpected format."
            msg = ""
            if isinstance(data, dict):
                err = data.get("error") or {}
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
            msg = str(msg)[:300]
            code = resp.status_code
            if code in (401, 403):
                return None, (f"Gemini returned HTTP {code}: {msg} Your GEMINI_API_KEY is invalid, "
                              "expired, or not enabled for the Gemini API. Create a key at "
                              "Google AI Studio (aistudio.google.com) and set it in .env / the "
                              "cloud dashboard.")
            if code == 404:
                return None, (f"Gemini returned HTTP 404: {msg} That model name is not available. "
                              "Use GEMINI_MODEL with an id from your Google AI Studio model list.")
            if code == 429:
                return None, (f"Gemini returned HTTP 429: {msg} Rate limit or quota exhausted. "
                              "Wait and try again, or check your Google AI Studio quota.")
            return None, f"Gemini returned HTTP {code}: {msg}"
        except Exception as exc:
            return None, f"Could not reach Gemini: {exc}"[:300]
