"""Model manager for Shibu AI: Hugging Face language-model integration.

What already existed before this file
--------------------------------------
`services/huggingface_service.py` offered one optional path: call the
Hugging Face Inference API with a user-supplied HF_TOKEN, only to enhance
UNKNOWN answers. It was disabled when no token was set, and there was no
local model, no model selection, and no way to run the conversational
engine without a token.

What this manager adds
----------------------
1. Reuse: if HF_TOKEN is set, the existing API path is reused (it is the
   cheapest way to get a real LLM answer without downloading gigabytes).
2. Automatic selection: if no token is set and local mode is enabled
   (HF_USE_LOCAL=1), a model is picked from the machine's RAM/GPU, not
   from a hardcoded large model. On an ~8 GB machine that is the 0.5B
   model, NOT the 3B/7B models — those need far more memory.
3. Lazy, cached loading: nothing is downloaded at app startup. A local
   model is downloaded once, into HF_HOME / the Hugging Face cache, and
   reused on every later startup. Loading is only attempted when a
   conversational answer is actually requested.
4. Clear failure reporting: every failure returns a status dict with a
   plain-language reason and a fallback, instead of crashing the app.
   The rule engine, maths, search and document tools keep working even
   when no LLM is available.

Candidate models, verified against the Hugging Face Hub API
(2026-10-24; all public, not gated, no login required to download):
- Qwen/Qwen2.5-0.5B-Instruct  — Apache-2.0, ~1 GB, best for <=8 GB RAM.
- Qwen/Qwen2.5-1.5B-Instruct  — Apache-2.0, ~3 GB, for 8-16 GB RAM.
- Qwen/Qwen2.5-3B-Instruct    — Qwen licence ("other" on the Hub), ~6 GB,
  only for 16 GB+ RAM, or 4-bit quantized on 8-16 GB.
- Qwen/Qwen2.5-7B-Instruct    — Apache-2.0, ~15 GB, only for 16 GB+ RAM.
- microsoft/Phi-4-mini-instruct — MIT, ~3.8B params (~8 GB), English-first,
  a good alternative on 16 GB+ RAM, not a first choice for Hindi.
Check the model's Hub page for its current licence before redistributing
anything built on it. Qwen models support many languages including Hindi
to a useful (not perfect) degree; Phi-4-mini is mainly English.

Security notes
--------------
- HF_TOKEN is read from the environment only, never written to a file,
  never logged, and never returned by any status endpoint.
- Local model files come only from the official Hugging Face Hub
  (huggingface_hub / transformers). No model is loaded from an arbitrary
  user-supplied path.
- The LLM never runs user code and never executes commands. Code that a
  user asks to run still goes through services/code_runner.py, with its
  own allow-list and timeout.
"""

import os
import re
import shutil

# RAM thresholds used for automatic selection (see module docstring).
SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MEDIUM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
XL_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_API_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _total_ram_gb():
    """Total system RAM in GB, or None if it cannot be read."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1_048_576
    except Exception:
        pass
    try:
        import psutil  # optional, not a hard dependency
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


def _has_gpu():
    return bool(shutil.which("nvidia-smi"))


def recommend_model(ram_gb=None, gpu=False):
    """Pick the largest model the hardware can plausibly run.

    Rules (conservative, because an OOM kill is worse than a smaller
    model): <=8 GB -> 0.5B; 8-16 GB -> 1.5B (3B only with 4-bit
    quantization, see HF_QUANTIZE); >16 GB -> 3B, or 7B with a GPU.
    Unknown RAM -> the smallest model, never the largest.
    """
    if ram_gb is None:
        ram_gb = _total_ram_gb()
    if ram_gb is None:
        return SMALL_MODEL, "RAM could not be detected, so the smallest model was chosen."
    if ram_gb <= 8:
        return SMALL_MODEL, f"Detected {ram_gb:.1f} GB RAM: the 0.5B model fits; 3B/7B would risk running out of memory."
    if ram_gb <= 16:
        return MEDIUM_MODEL, f"Detected {ram_gb:.1f} GB RAM: the 1.5B model fits. Set HF_MODEL=Qwen/Qwen2.5-3B-Instruct and HF_QUANTIZE=4bit to try the 3B model quantized."
    if gpu:
        return XL_MODEL, f"Detected {ram_gb:.1f} GB RAM with a GPU: the 7B model is the largest practical choice."
    return LARGE_MODEL, f"Detected {ram_gb:.1f} GB RAM, no GPU: the 3B model is the largest practical CPU choice."


def is_cached(model_id):
    """True if this model is already in the local Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id and repo.size_on_disk > 0:
                return True
    except Exception:
        pass
    # Fallback: look for the model folder under HF_HOME / ~/.cache.
    base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(base, "hub")
    folder = "models--" + model_id.replace("/", "--")
    return os.path.isdir(os.path.join(hub, folder))


class ModelManager:
    """One place for: select -> (download once) -> load -> generate.

    Modes, decided at construction from the environment:
    - "openai": OPENAI_API_KEY is set. ChatGPT API (gpt-4o-mini by
              default) — the fastest, most reliable hosted option, and
              the one used first when both OpenAI and HF keys are set.
    - "api":   HF_TOKEN is set. Reuses the existing API service. No
              download, no local RAM cost. This is the recommended mode
              for a small PC.
    - "local": HF_USE_LOCAL=1 and no token (or HF_FORCE_LOCAL=1). Loads
              a transformers model on demand, from the cache if present.
    - "off":   neither. The rule engine is the whole brain, and every
              status call says so plainly instead of pretending an LLM
              is running.
    """

    def __init__(self, token=None, model=None, use_local=None, openai_key=None):
        self.openai_key = (openai_key if openai_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        self.openai_model = (os.environ.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini")
        self.token = (token if token is not None else os.environ.get("HF_TOKEN", "")).strip()
        self.cloud_only = os.environ.get("CLOUD_ONLY", "").strip().lower() in ("1", "true", "yes")
        requested_local = (use_local if use_local is not None
                           else os.environ.get("HF_USE_LOCAL", "").strip() in ("1", "true", "yes"))
        requested_force_local = os.environ.get("HF_FORCE_LOCAL", "").strip() in ("1", "true", "yes")
        # Render production uses hosted APIs only.  Local mode remains available
        # for existing development installs when CLOUD_ONLY is not set.
        self.use_local = bool(requested_local) and not self.cloud_only
        self.force_local = bool(requested_force_local) and not self.cloud_only
        # Optional explicit provider, e.g. featherless-ai, novita, auto.
        # The router needs a provider for many models: a plain model name
        # can fail with "not supported by any provider you have enabled"
        # even when the model itself exists. See _generate_api.
        self.provider = os.environ.get("HF_PROVIDER", "").strip()
        self.effective_model = None
        env_model = (model or os.environ.get("HF_MODEL", "")).strip()
        if self.openai_key and not self.force_local:
            self.mode = "openai"
            self.model_id = self.openai_model
            self.effective_model = self.openai_model
        elif self.token and not self.force_local:
            self.mode = "api"
            self.model_id = env_model or DEFAULT_API_MODEL
        elif self.use_local or self.force_local:
            self.mode = "local"
            if env_model:
                self.model_id = env_model
            else:
                self.model_id, self._recommendation = recommend_model()
        else:
            self.mode = "off"
            self.model_id = env_model or None
        if not hasattr(self, "_recommendation"):
            self._recommendation = ""
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._load_error = None
        self._api_service = None

    # -- status / selection ------------------------------------------------
    def status(self):
        """Safe to expose: never includes the token or any secret."""
        ram = _total_ram_gb()
        info = {
            "mode": self.mode,
            "model": self.model_id,
            "loaded": self._model is not None or (self.mode == "api" and bool(self.token)) or (self.mode == "openai" and bool(self.openai_key)),
            "device": self._device if self.mode == "local" else ("api" if self.mode in ("api", "openai") else None),
            "ram_gb": round(ram, 1) if ram else None,
            "gpu": _has_gpu(),
            "cloud_only": self.cloud_only,
            "cached": is_cached(self.model_id) if self.mode == "local" and self.model_id else None,
            "recommendation": self._recommendation,
            "error": self._load_error,
        }
        if self.mode == "off":
            if self.cloud_only:
                info["how_to_enable"] = (
                    "This deployment is cloud-only. Set OPENAI_API_KEY for gpt-4o-mini "
                    "or HF_TOKEN for Hugging Face hosted inference in the service environment.")
            else:
                info["how_to_enable"] = (
                    "For the fastest answer, set OPENAI_API_KEY in .env (ChatGPT API, gpt-4o-mini). "
                    "Or set HF_TOKEN in a .env file to use the Hugging Face API, "
                    "or set HF_USE_LOCAL=1 and install requirements-llm.txt to run a small "
                    "model on this PC. Until then, Shibu AI answers from its rule engine.")
        return info

    def select(self):
        """Report the model that would be used, without downloading it."""
        if self.mode == "openai":
            return {"mode": "openai", "model": self.model_id, "note": "ChatGPT API (OpenAI) — fast, no download, no provider routing."}
        if self.mode == "api":
            return {"mode": "api", "model": self.model_id, "note": "Reusing the existing Hugging Face API configuration; no download needed."}
        if self.mode == "local":
            model, note = (self.model_id, self._recommendation) if self.model_id else recommend_model()
            return {"mode": "local", "model": model, "cached": is_cached(model), "note": note}
        model, note = recommend_model()
        return {"mode": "off", "model": model, "note": "No LLM is enabled. " + note}

    # -- loading -----------------------------------------------------------
    def load(self):
        """Load (or reuse) the backend. Returns (ok, message).

        Local loading is lazy and happens at most once per process.
        Download happens only here, only once (the Hub cache), and only
        when local mode was explicitly enabled.
        """
        if self.mode == "openai":
            return True, f"OpenAI API mode ready: {self.model_id} (ChatGPT API, fast response)."
        if self.mode == "api":
            return True, f"API mode ready: {self.model_id} via the Hugging Face Inference Providers router."
        if self.mode == "off":
            return False, "No language model is enabled. Set HF_TOKEN (API) or HF_USE_LOCAL=1 (local model) — see README section 13."
        if self._model is not None:
            return True, f"Local model already loaded: {self.model_id}."
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self._load_error = ("transformers/torch are not installed. Install the optional "
                                "LLM dependencies with: pip install -r requirements-llm.txt")
            return False, self._load_error
        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            quantize = os.environ.get("HF_QUANTIZE", "").strip()
            kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
            if self._device == "cuda":
                kwargs["device_map"] = "auto"
            if quantize == "4bit":
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                except Exception:
                    pass  # bitsandbytes missing: fall back to unquantized weights
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if self._device == "cpu":
                self._model.to("cpu")
            self._model.eval()
            return True, f"Local model loaded: {self.model_id} on {self._device}."
        except Exception as exc:  # OOM, network, bad model id, disk full...
            self._load_error = _plain_error(exc)
            self._model = None
            return False, self._load_error

    # -- generation --------------------------------------------------------
    def generate(self, messages, max_new_tokens=256, temperature=0.7):
        """Generate a reply for a chat history.

        `messages` is a list of {"role": "user"|"assistant"|"system",
        "content": str}. Returns (text, error). text is None on failure,
        so callers can fall back to the rule engine instead of showing a
        traceback or an invented answer.
        """
        if self.mode == "openai":
            from services.openai_service import OpenAIService
            svc = OpenAIService(api_key=self.openai_key, model=self.model_id)
            text, err = svc.generate(messages, max_tokens=max_new_tokens, temperature=temperature)
            if text:
                self.effective_model = self.model_id
            return text, err
        if self.mode == "api":
            return self._generate_api(messages, max_new_tokens, temperature)
        if self.mode == "local":
            return self._generate_local(messages, max_new_tokens, temperature)
        return None, "No language model is enabled."

    def _generate_api(self, messages, max_new_tokens, temperature):
        """Call Hugging Face Inference Providers via its router.

        POST https://router.huggingface.co/v1/chat/completions
        (OpenAI-compatible chat completions).

        Provider routing — the fix for HTTP 400 "not supported by any
        provider you have enabled":
        The router does not always pick a provider from a plain model
        name. It needs a provider, either from the user's preferred
        order in huggingface.co/settings/inference-providers, or from
        a ":provider" suffix on the model name. So Shibu tries, in
        order, on a 400 only (a 401/429/503 will not be fixed by
        switching provider, so those are returned at once):
          1. HF_PROVIDER if set, e.g. "Qwen/Qwen2.5-3B-Instruct:featherless-ai"
          2. the model name exactly as configured
          3. model + ":auto"  (router picks a supporting provider)
          4. model + ":featherless-ai" (the only live provider for
             Qwen2.5-3B/7B-Instruct, verified against the Hub API)
        The model that actually answered is stored in
        self.effective_model, so the answer label never claims a
        different model was used. If every candidate fails with 400,
        the last error is returned, with settings/local-mode guidance.
        """
        try:
            import requests
        except Exception as exc:
            return None, _plain_error(exc)
        candidates = []
        base = self.model_id
        if ":" in base:
            candidates.append(base)  # user already gave a provider suffix
        else:
            if self.provider:
                candidates.append(f"{base}:{self.provider}")
            candidates.append(base)
            candidates.append(f"{base}:auto")
            if "Qwen/" in base:
                candidates.append(f"{base}:featherless-ai")
        last_error = None
        for candidate in candidates:
            try:
                resp = requests.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.token}",
                             "Content-Type": "application/json"},
                    json={"model": candidate, "messages": messages,
                          "max_tokens": max_new_tokens, "temperature": temperature},
                    timeout=60)
            except Exception as exc:
                return None, _plain_error(exc)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices and choices[0].get("message", {}).get("content"):
                    self.effective_model = candidate
                    return choices[0]["message"]["content"].strip(), None
                return None, "The router returned a response in an unexpected format."
            last_error = _error_text(resp)
            if resp.status_code != 400:
                return None, last_error
            # 400: try the next provider candidate.
        if last_error:
            return None, last_error
        return None, "Hugging Face API call failed."

    def _generate_api_legacy(self, messages, max_new_tokens, temperature):
        """Deprecated fallback: the old api-inference endpoint.

        Kept only because it costs one extra request on a router 404
        and may still answer for a model the router does not list.
        Its failure is never reported on its own — the router error is
        the useful one.
        """
        try:
            import requests
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"inputs": prompt,
                      "parameters": {"max_new_tokens": max_new_tokens,
                                     "temperature": temperature,
                                     "return_full_text": False}},
                timeout=60)
            if resp.status_code != 200:
                return None, _error_text(resp)
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip(), None
            if isinstance(data, dict) and data.get("generated_text"):
                return data["generated_text"].strip(), None
            return None, "The legacy API returned a response in an unexpected format."
        except Exception as exc:
            return None, _plain_error(exc)

    def _generate_local(self, messages, max_new_tokens, temperature):
        ok, msg = self.load()
        if not ok:
            return None, msg
        try:
            import torch
            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = _plain_chat(messages)
            else:
                text = _plain_chat(messages)
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0, pad_token_id=self._tokenizer.eos_token_id)
            new_tokens = out[0][inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), None
        except Exception as exc:
            return None, _plain_error(exc)

    def test_prompt(self):
        """The documented smoke test: one short prompt, one short answer."""
        text, err = self.generate(
            [{"role": "user", "content": "Say hello in one short sentence."}],
            max_new_tokens=40, temperature=0.0)
        if err:
            return {"ok": False, "error": err}
        return {"ok": True, "model": self.model_id, "reply": text}


def _error_text(resp):
    """Turn an HF HTTP error into a plain, actionable message (no token)."""
    try:
        data = resp.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            err = err.get("message") or str(err)
        elif not err:
            err = resp.text[:300]
    except Exception:
        err = resp.text[:300] if hasattr(resp, "text") else ""
    err = str(err).strip()
    code = resp.status_code
    if code == 401:
        return (f"Hugging Face API returned HTTP 401: {err} "
                "Your token was rejected, or it does not have the 'Make calls to "
                "Inference Providers' permission. In Hugging Face Settings > Access "
                "Tokens, edit the token, turn that permission on, save, and restart Shibu.")
    if code == 400:
        return (f"Hugging Face API returned HTTP 400: {err} "
                "The requested model is not supported by any provider you have enabled. "
                "Fix in order: (1) set HF_PROVIDER=featherless-ai in .env and restart Shibu "
                "(for Qwen models this is the only live provider); (2) go to "
                "huggingface.co/settings/inference-providers and enable featherless-ai; "
                "(3) if providers need payment on your account, use local mode instead: "
                "install requirements-llm.txt and set HF_USE_LOCAL=1.")
    if code == 404:
        return (f"Hugging Face API returned HTTP 404: {err} "
                "This model is not available through the Inference Providers router. "
                "Check the model's page on huggingface.co for its Inference Providers, "
                "or set HF_MODEL to a model that lists one.")
    if code == 429:
        return (f"Hugging Face API returned HTTP 429: {err} "
                "Rate limit or no Inference Providers credits left. Wait a little and try again.")
    if code in (502, 503):
        return (f"Hugging Face API returned HTTP {code}: {err} "
                "The model is loading or temporarily unavailable. Wait 30 seconds and try again.")
    return f"Hugging Face API returned HTTP {code}: {err}"


def _plain_chat(messages):
    parts = []
    for m in messages:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], m["role"])
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts) + "\nAssistant:"


def _plain_error(exc):
    """Turn an exception into a helpful message without leaking secrets."""
    msg = str(exc)
    if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
        return ("The model ran out of memory. Use a smaller model (for example "
                "Qwen/Qwen2.5-0.5B-Instruct), set HF_QUANTIZE=4bit, or use the "
                "Hugging Face API instead (set HF_TOKEN).")
    if "Connection" in type(exc).__name__ or "connect" in msg.lower():
        return "Could not reach Hugging Face (network problem?). Check your internet connection and try again."
    if "401" in msg or "Unauthorized" in msg:
        return "Hugging Face rejected the credentials. Check HF_TOKEN; no token is needed to download the public models listed in this file."
    # Never echo a token-like value.
    msg = re.sub(r"(Bearer|token)[=:\s]+\S+", r"\1 [hidden]", msg, flags=re.I)
    return f"Model loading/generation failed: {msg[:300]}"
