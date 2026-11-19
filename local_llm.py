"""The local model main.py and studio/graph.py run on: Qwen 3.6 35B-A3B in LM Studio.

Change MODEL here to switch both. The LM Studio address and the model can also
be set per run with TRADINGAGENTS_BACKEND_URL and LMSTUDIO_MODEL.

LM Studio loads a model on first use with a small context (8192 tokens), which
the agents' prompts outgrow, so load it yourself first with a larger one:

    lms load qwen/qwen3.6-35b-a3b --context-length 65536 -y
"""

import os

# An empty variable (e.g. a blank line in .env) keeps the default.
MODEL = os.environ.get("LMSTUDIO_MODEL") or "qwen/qwen3.6-35b-a3b"
BACKEND_URL = os.environ.get("TRADINGAGENTS_BACKEND_URL") or "http://192.168.1.246:1234/v1"


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
