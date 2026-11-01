"""Opt-in Meta-compatible chat API service for Shibu AI.

UNVERIFIED LIVE: no Meta API key was available when this was written,
so no live call has ever been made from this code. It is inactive
unless BOTH META_API_KEY and META_MODEL are set in the environment;
there is deliberately no default model id, so nothing is guessed.

The endpoint is OpenAI-compatible chat completions
({base}/chat/completions, Bearer key), mirroring OpenAIService: one
HTTP call, short timeout, errors returned as text for the caller to
fall through to the existing chain — never a crash, never a fake
answer.

Security: the key comes only from the META_API_KEY environment
variable. It is never written to a file by this code, never logged,
never returned by any status endpoint, and never pasted in chat.
Set it in .env locally or in the host's Environment Variables
dashboard, never in git.
"""

import os


class MetaService:
    def __init__(self, api_key=None, model=None, base=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("META_API_KEY", "")).strip()
        # No default model: the owner must name the exact model id.
        self.model = (model if model is not None else os.environ.get("META_MODEL", "")).strip()
        self.base = (base if base is not None else os.environ.get("META_API_BASE", "")).strip() or "https://api.meta.ai/v1"
        self.enabled = bool(self.api_key and self.model)

    def status(self):
        # Safe: never includes the key.
        return {"enabled": self.enabled, "model": self.model if self.enabled else None}

    def generate(self, messages, max_tokens=256, temperature=0.7):
        """Return (text, error). text is None on failure."""
        if not self.enabled:
            return None, "Meta API is not configured (set both META_API_KEY and META_MODEL)."
        url = self.base.rstrip("/") + "/chat/completions"
        try:
            import requests
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices and choices[0].get("message", {}).get("content"):
                    return choices[0]["message"]["content"].strip(), None
                return None, "The Meta-compatible API returned a response in an unexpected format."
            try:
                err = resp.json().get("error", {})
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
            except Exception:
                msg = resp.text[:300]
            code = resp.status_code
            if code == 401:
                return None, f"Meta-compatible API returned HTTP 401: {msg} The META_API_KEY was rejected. Check the key and restart Shibu."
            if code == 429:
                return None, f"Meta-compatible API returned HTTP 429: {msg} Rate limit or no credits left."
            if code == 404:
                return None, f"Meta-compatible API returned HTTP 404: {msg} That META_MODEL name is not available on the account. Check the exact model id."
            return None, f"Meta-compatible API returned HTTP {code}: {msg}"
        except Exception as exc:
            return None, f"Could not reach the Meta-compatible API: {exc}"[:300]
