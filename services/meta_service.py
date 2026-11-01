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

    @staticmethod
    def _extract_text(data):
        """Pull assistant text from a Meta-compatible response body.

        The exact response shape is not documented here and may vary,
        so several plausible shapes are tried in order. No shape is
        assumed to be *the* real one, and no raw payload is ever
        surfaced: a body that matches none of them is simply
        unparseable. Returns the text, or None.
        """
        if not isinstance(data, dict):
            return None

        def _text_from_content(content):
            """A content value that is a string, or a list of parts."""
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        piece = part.get("text")
                        if isinstance(piece, str) and piece.strip():
                            parts.append(piece.strip())
                    elif isinstance(part, str) and part.strip():
                        parts.append(part.strip())
                if parts:
                    return "\n".join(parts)
            return None

        # 1. OpenAI chat-completions shape: choices[0].message.content
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = _text_from_content((first.get("message") or {}).get("content"))
                if text:
                    return text
                text = _text_from_content(first.get("text"))
                if text:
                    return text
                text = _text_from_content(first.get("content"))
                if text:
                    return text
        # 2. Responses-style shape: output[].content[].text (+ output_text)
        output = data.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if isinstance(item, dict):
                    text = _text_from_content(item.get("content"))
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(parts)
        text = _text_from_content(data.get("output_text"))
        if text:
            return text
        # 3. Messages-style shape: content[].text
        text = _text_from_content(data.get("content"))
        if text:
            return text
        # 4. Top-level plain-text fields.
        for key in ("text", "answer", "response"):
            text = _text_from_content(data.get(key))
            if text:
                return text
        return None

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
                text = self._extract_text(resp.json())
                if text:
                    return text, None
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
