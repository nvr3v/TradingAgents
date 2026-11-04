"""``structured_output_method`` config override for OpenAI-compatible providers.

A local thinking model (e.g. Qwen 3.5 on LM Studio) given an unforced tool may
answer in prose, losing the structured decision; ``json_schema`` constrains the
reply to the schema instead.
"""

import pytest
from pydantic import BaseModel

from tradingagents.llm_clients.factory import build_llm_kwargs, create_tier_client


class _Schema(BaseModel):
    rating: str


def _config(**overrides):
    return {"llm_provider": "openai_compatible", "backend_url": "http://localhost:1234/v1",
            "quick_think_llm": "qwen/qwen3.5-9b", "deep_think_llm": "qwen/qwen3.5-9b", **overrides}


def _bound_kwargs(runnable):
    # with_structured_output returns llm.bind(...) | parser; the first step holds the request kwargs.
    return runnable.first.kwargs


@pytest.mark.unit
def test_unset_keeps_default_function_calling():
    assert "structured_output_method" not in build_llm_kwargs(_config())
    llm = create_tier_client(_config(), "deep").get_llm()
    kwargs = _bound_kwargs(llm.with_structured_output(_Schema))
    assert "tools" in kwargs and kwargs.get("tool_choice") is None


@pytest.mark.unit
def test_json_schema_override_uses_response_format():
    llm = create_tier_client(_config(structured_output_method="json_schema"), "deep").get_llm()
    assert llm.structured_output_method == "json_schema"
    kwargs = _bound_kwargs(llm.with_structured_output(_Schema))
    assert "response_format" in kwargs
    assert "tools" not in kwargs


@pytest.mark.unit
def test_explicit_method_argument_wins():
    llm = create_tier_client(_config(structured_output_method="json_schema"), "deep").get_llm()
    assert "tools" in _bound_kwargs(llm.with_structured_output(_Schema, method="function_calling"))


@pytest.mark.unit
def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="structured_output_method"):
        build_llm_kwargs(_config(structured_output_method="xml"))


@pytest.mark.unit
def test_not_forwarded_to_non_openai_providers():
    assert "structured_output_method" not in build_llm_kwargs(
        {"llm_provider": "anthropic", "structured_output_method": "json_schema"})


@pytest.mark.unit
def test_env_override(monkeypatch):
    from tradingagents.default_config import build_default_config
    monkeypatch.setenv("TRADINGAGENTS_STRUCTURED_OUTPUT_METHOD", "json_schema")
    assert build_default_config()["structured_output_method"] == "json_schema"
