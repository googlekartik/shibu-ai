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
        # Safe diagnostic from the last failed call: HTTP status +
        # top-level response keys ONLY. Never values, never payload,
        # never headers, never the key. Read by ModelManager for
        # /api/llm/status so a live failure can be diagnosed
        # without exposing anything sensitive.
        self.last_diagnostic = None

    def status(self):
        # Safe: never includes the key.
        return {"enabled": self.enabled, "model": self.model if self.enabled else None}

    @staticmethod
    def _extract_text(data, _depth=0):
        """Pull assistant text from a Meta-compatible response body.

        The exact response shape is not documented here and may vary,
        so several plausible shapes are tried in order. No shape is
        assumed to be *the* real one, and no raw payload is ever
        surfaced: a body that matches none of them is simply
        unparseable. Returns the text, or None.
        """
        if isinstance(data, str):
            # A body that is itself a JSON string: parse once more.
            text = data.strip()
            if text.startswith("{") and _depth < 2:
                try:
                    import json
                    return MetaService._extract_text(json.loads(text), _depth + 1)
                except Exception:
                    return None
            return None
        if not isinstance(data, dict) or _depth > 2:
            return None

        def _text_from_content(content):
            """A content value that is a string, a list of parts, or
            a part that nests another content list."""
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        piece = part.get("text")
                        if isinstance(piece, str) and piece.strip():
                            parts.append(piece.strip())
                            continue
                        nested = _text_from_content(part.get("content"))
                        if nested:
                            parts.append(nested)
                            continue
                        piece = part.get("output_text")
                        if isinstance(piece, str) and piece.strip():
                            parts.append(piece.strip())
                    elif isinstance(part, str) and part.strip():
                        parts.append(part.strip())
                if parts:
                    return "\n".join(parts)
            if isinstance(content, dict):
                return _text_from_content(content.get("content"))
            return None

        # 1. OpenAI chat-completions shape: choices[0].message.content
        #    (content may be a string or a list of parts), plus
        #    streaming-style delta fragments joined in order.
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            delta_fragments = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                text = _text_from_content((choice.get("message") or {}).get("content"))
                if text:
                    return text
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    text = _text_from_content(delta.get("content"))
                    if text:
                        delta_fragments.append(text)
                text = _text_from_content(choice.get("text"))
                if text:
                    return text
                text = _text_from_content(choice.get("content"))
                if text:
                    return text
            # Streaming-style delta fragments: joined in order,
            # only after every choice has been examined.
            if delta_fragments:
                return "".join(delta_fragments)
        # 2. Responses-style shape: output[].content[].text (+ output_text)
        output = data.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if isinstance(item, dict):
                    text = _text_from_content(item.get("content"))
                    if text:
                        parts.append(text)
                elif isinstance(item, str) and item.strip():
                    parts.append(item.strip())
            if parts:
                return "\n".join(parts)
        text = _text_from_content(data.get("output_text"))
        if text:
            return text
        # 3. Messages-style shape: content[].text
        text = _text_from_content(data.get("content"))
        if text:
            return text
        # 4. Envelope wrappers: data / result / results / response.
        for key in ("data", "result", "results", "response"):
            value = data.get(key)
            if isinstance(value, dict):
                text = MetaService._extract_text(value, _depth + 1)
                if text:
                    return text
            elif isinstance(value, list):
                parts = [MetaService._extract_text(v, _depth + 1)
                         for v in value if isinstance(v, dict)]
                parts = [p for p in parts if p]
                if parts:
                    return "\n".join(parts)
            elif isinstance(value, str) and value.strip() and key == "response":
                return value.strip()
        # 5. Top-level plain-text fields.
        for key in ("text", "answer"):
            text = _text_from_content(data.get(key))
            if text:
                return text
        return None

    @staticmethod
    def _response_keys(data) -> list:
        """Top-level keys of a response body, for diagnostics only.
        A JSON-string body is parsed once; anything else yields []."""
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except Exception:
                return []
        if isinstance(data, dict):
            return sorted(str(k) for k in data.keys())
        return []

    def generate(self, messages, max_tokens=256, temperature=0.7):
        """Return (text, error). text is None on failure.

        Every failure records self.last_diagnostic = {status,
        response_keys}: status + top-level keys only, never values.
        Error strings carry the same two facts plus plain guidance
        written here — never server message text, which could
        contain payload details.
        """
        if not self.enabled:
            self.last_diagnostic = {"status": None, "response_keys": []}
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
            try:
                body = resp.json()
            except Exception:
                body = None
            keys = self._response_keys(body)
            self.last_diagnostic = {"status": resp.status_code, "response_keys": keys}
            keys_text = ", ".join(keys) if keys else "none"
            if resp.status_code == 200:
                text = self._extract_text(body)
                if text:
                    return text, None
                return None, ("The Meta-compatible API returned a response in an unexpected format "
                              f"(HTTP 200, response keys: {keys_text}).")
            code = resp.status_code
            if code == 401:
                guidance = "The META_API_KEY was rejected. Check the key and restart Shibu."
            elif code == 429:
                guidance = "Rate limit or no credits left."
            elif code == 404:
                guidance = "That META_MODEL name is not available on the account. Check the exact model id."
            else:
                guidance = "Check the model id and the API base URL, then try again."
            return None, (f"Meta-compatible API returned HTTP {code} "
                          f"(response keys: {keys_text}). {guidance}")
        except Exception as exc:
            self.last_diagnostic = {"status": None, "response_keys": []}
            return None, f"Could not reach the Meta-compatible API: {exc}"[:300]
