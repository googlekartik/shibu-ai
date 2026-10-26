"""Optional Hugging Face service (kept for compatibility).

It never overrides the rule engine: it is only used to enhance UNKNOWN
answers, and results are labelled as AI-assisted.

Note: the old api-inference.huggingface.co endpoint is decommissioned,
so this service now uses the Inference Providers router, exactly like
services/model_manager.py:
  POST https://router.huggingface.co/v1/chat/completions
The token needs the "Make calls to Inference Providers" permission.
"""

import os

import requests


class HuggingFaceService:
    def __init__(self):
        self.token = os.environ.get("HF_TOKEN", "").strip()
        self.model = os.environ.get("HF_MODEL", "").strip() or "Qwen/Qwen2.5-3B-Instruct"
        _prov = os.environ.get("HF_PROVIDER", "").strip()
        if _prov and ":" not in self.model:
            self.model = f"{self.model}:{_prov}"
        self.enabled = bool(self.token)

    def status(self):
        return {"enabled": self.enabled, "model": self.model if self.enabled else None}

    def enhance(self, prompt: str):
        """Return enhanced text, or None if disabled / on any failure."""
        if not self.enabled:
            return None
        try:
            resp = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "application/json"},
                json={"model": self.model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 256, "temperature": 0.7},
                timeout=60)
            if resp.status_code != 200:
                return None
            data = resp.json()
            choices = data.get("choices") or []
            if choices and choices[0].get("message", {}).get("content"):
                return choices[0]["message"]["content"].strip()
            return None
        except Exception:
            return None
