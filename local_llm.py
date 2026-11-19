"""The local model main.py and studio/graph.py run on: Qwen 3.6 35B-A3B in LM Studio.

Change MODEL here to switch both. The LM Studio address, the model and its
context can also be set per run with TRADINGAGENTS_BACKEND_URL, LMSTUDIO_MODEL
and LMSTUDIO_CONTEXT_LENGTH.

LM Studio loads a model on first use with a small context (8192 tokens), which
the agents' prompts outgrow. ensure_model_loaded(), called before a run, loads
MODEL with CONTEXT_LENGTH (65536) tokens instead, replacing a copy loaded with
less. LMSTUDIO_CONTEXT_LENGTH=0 leaves loading to LM Studio.
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# An empty variable (e.g. a blank line in .env) keeps the default.
MODEL = os.environ.get("LMSTUDIO_MODEL") or "qwen/qwen3.6-35b-a3b"
BACKEND_URL = os.environ.get("TRADINGAGENTS_BACKEND_URL") or "http://192.168.1.246:1234/v1"
CONTEXT_LENGTH = int(os.environ.get("LMSTUDIO_CONTEXT_LENGTH") or 65536)

LOAD_TIMEOUT_SECONDS = 600   # a 35B model takes a minute or two to load


def use_local_model(config: dict) -> dict:
    """Point both model tiers at MODEL on LM Studio, in place; returns ``config``."""
    # The LLM client reads OPENAI_COMPATIBLE_API_KEY; LM Studio's own name for the
    # token is LM_API_TOKEN, so either one in .env is enough.
    if os.environ.get("LM_API_TOKEN"):
        os.environ.setdefault("OPENAI_COMPATIBLE_API_KEY", os.environ["LM_API_TOKEN"])

    config["llm_provider"] = "openai_compatible"
    config["backend_url"] = BACKEND_URL
    config["deep_think_llm"] = MODEL
    config["quick_think_llm"] = MODEL
    # Qwen "thinks" before answering; without a higher cap the server's default
    # 2048-token output limit can be used up by reasoning, leaving the answer empty.
    config["max_tokens"] = 16384
    # Turn Qwen's thinking off: LM Studio maps reasoning_effort "none" to no
    # reasoning, which cuts each answer from hundreds of tokens to a handful.
    config["openai_reasoning_effort"] = "none"
    # Without a forced tool call Qwen answers the managers' prompts in prose, so the
    # structured decision is lost; LM Studio's grammar-constrained JSON fixes that.
    config["structured_output_method"] = "json_schema"
    return config


def _api(method: str, path: str, body: dict | None = None, timeout: float = 30) -> dict:
    """A call to LM Studio's management API (``/api/v1``) on the BACKEND_URL server."""
    root = BACKEND_URL.rstrip("/").removesuffix("/v1")
    token = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or os.environ.get("LM_API_TOKEN")
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(root + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        raise RuntimeError(f"LM Studio {method} {path} failed ({exc.code}): {detail or exc.reason}") from None


def ensure_model_loaded(model: str | None = None, context_length: int | None = None) -> str:
    """Have LM Studio hold ``model`` with at least ``context_length`` tokens of context.

    A copy already loaded with enough context is used as it is; one loaded with less
    is unloaded first (two copies would not fit in memory), so a request it is
    serving right now fails. Never raises: when LM Studio cannot do it, this warns
    and the run goes on with whatever LM Studio gives it. Returns what was done:
    "skipped", "loaded", "reloaded", "already loaded" or "failed".
    """
    model = model or MODEL
    context_length = CONTEXT_LENGTH if context_length is None else context_length
    if context_length <= 0:
        return "skipped"
    try:
        entry = next((m for m in _api("GET", "/api/v1/models").get("models", []) if m.get("key") == model), None)
        if entry is None:
            logger.warning("LM Studio has no model %r downloaded; cannot load it.", model)
            return "failed"
        instances = entry.get("loaded_instances") or []
        if any((i.get("config") or {}).get("context_length", 0) >= context_length for i in instances):
            return "already loaded"
        for instance in instances:
            logger.warning("Unloading %s (context %s) to load it with %d tokens.",
                           instance["id"], (instance.get("config") or {}).get("context_length"), context_length)
            _api("POST", "/api/v1/models/unload", {"instance_id": instance["id"]})
        print(f"Loading {model} in LM Studio with a {context_length}-token context...", flush=True)
        _api("POST", "/api/v1/models/load", {"model": model, "context_length": context_length},
             timeout=LOAD_TIMEOUT_SECONDS)
        return "reloaded" if instances else "loaded"
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Could not load %s with a %d-token context in LM Studio (%s); "
                       "LM Studio will load it with its own settings.", model, context_length, exc)
        return "failed"
