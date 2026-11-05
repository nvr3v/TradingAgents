"""OpenAI ``reasoning_effort`` is gated to reasoning models.

Non-reasoning OpenAI models (gpt-4.1, gpt-4o, ...) 400 with "Unsupported
parameter: 'reasoning.effort'". The client must drop the kwarg for those rather
than forward it and crash the run. The GPT-5 family and the o-series accept it.
"""

import pytest

from tradingagents.llm_clients.openai_client import (
    OpenAIClient,
    _supports_reasoning_effort,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.5", True), ("gpt-5.4", True), ("gpt-5.4-mini", True),
        ("gpt-5.5-pro", True), ("gpt-6-astra", True), ("o1", True), ("o3-mini", True),
        ("gpt-4.1", False), ("gpt-4o", False), ("gpt-4o-mini", False),
        ("gpt-3.5-turbo", False), ("gpt-10", True),
        ("gpt-5foo", False), ("gpt-60x", False), ("o3rd-party", False),
    ],
)
def test_supports_reasoning_effort(model, expected):
    assert _supports_reasoning_effort(model) is expected


def _effort_on(model, monkeypatch):
    # A fake key lets get_llm() construct the client without a network call.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAIClient(model, provider="openai", reasoning_effort="low").get_llm()
    return getattr(llm, "reasoning_effort", None)


def test_reasoning_model_receives_effort(monkeypatch):
    assert _effort_on("gpt-5.4-mini", monkeypatch) == "low"


def test_non_reasoning_model_drops_effort(monkeypatch):
    # gpt-4.1 would 400 with reasoning_effort — it must be dropped.
    assert _effort_on("gpt-4.1", monkeypatch) is None


def test_openai_compatible_forwards_effort_for_any_model(monkeypatch):
    # A local server (LM Studio) decides what the effort means: "none" turns
    # thinking off for Qwen3.5, whose name the native-OpenAI gate would reject.
    from tradingagents.llm_clients.factory import build_llm_kwargs

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    kwargs = build_llm_kwargs({"llm_provider": "openai_compatible", "openai_reasoning_effort": "none"})
    assert kwargs["reasoning_effort"] == "none"
    llm = OpenAIClient("qwen/qwen3.5-9b", base_url="http://localhost:1234/v1",
                       provider="openai_compatible", **kwargs).get_llm()
    assert llm.reasoning_effort == "none"
