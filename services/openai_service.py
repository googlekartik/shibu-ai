"""ChatGPT API service for Shibu AI — the fast-response LLM option.

Why this exists: Hugging Face Inference Providers routing (see
model_manager.py) can fail with provider errors, and local models need
a GPU/RAM and a big download. OpenAI's chat API is a direct, fast
alternative: one HTTP call, no download, no provider suffix.

Model: gpt-4o-mini by default (OPENAI_MODEL to change it). It is
chosen for speed and low cost, not for maximum reasoning — for hard
reasoning, set OPENAI_MODEL to a larger model in .env.

Security: the key comes only from the OPENAI_API_KEY environment
variable. It is never written to a file by this code, never logged,
never returned by any status endpoint, and never pasted in chat.
On a cloud host, set it in the host's Environment Variables dashboard,
not in the code or in GitHub. On your PC, put it in .env (which is
in .gitignore, so it is never pushed to GitHub).
"""

import os


class OpenAIService:
    def __init__(self, api_key=None, model=None):
        self.api_key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        self.model = (model or os.environ.get("OPENAI_MODEL", "")).strip() or "gpt-4o-mini"
        self.enabled = bool(self.api_key)

    def status(self):
        # Safe: never includes the key.
        return {"enabled": self.enabled, "model": self.model if self.enabled else None}

    def generate(self, messages, max_tokens=256, temperature=0.7):
        """Return (text, error). text is None on failure."""
        if not self.enabled:
            return None, "No OpenAI API key is set (OPENAI_API_KEY)."
        try:
            import requests
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
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
                return None, "OpenAI returned a response in an unexpected format."
            try:
                err = resp.json().get("error", {})
                msg = err.get("message", "") if isinstance(err, dict) else str(err)
            except Exception:
                msg = resp.text[:300]
            code = resp.status_code
            if code == 401:
                return None, f"OpenAI returned HTTP 401: {msg} Your OPENAI_API_KEY is invalid or expired. Create a new key at platform.openai.com and set it in .env / the cloud dashboard."
            if code == 429:
                return None, f"OpenAI returned HTTP 429: {msg} Rate limit or no credits left. Wait and try again, or check your OpenAI billing."
            if code == 404:
                return None, f"OpenAI returned HTTP 404: {msg} That model name is not available on your OpenAI account. Use OPENAI_MODEL=gpt-4o-mini."
            return None, f"OpenAI returned HTTP {code}: {msg}"
        except Exception as exc:
            return None, f"Could not reach OpenAI: {exc}"[:300]
