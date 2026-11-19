"""The local LM Studio model that main.py and studio/graph.py run on (local_llm.py)."""

import importlib

import pytest

import local_llm
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import create_tier_client
from tradingagents.llm_clients.openai_client import LocalCompatibleChatOpenAI


@pytest.fixture
def lmstudio_env(monkeypatch):
    for name in ("LMSTUDIO_MODEL", "TRADINGAGENTS_BACKEND_URL", "OPENAI_COMPATIBLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LM_API_TOKEN", "sk-lm-test-token")
    importlib.reload(local_llm)
    yield
    monkeypatch.undo()
    importlib.reload(local_llm)


@pytest.mark.unit
def test_both_tiers_use_qwen_3_6_35b_a3b(lmstudio_env):
    config = local_llm.use_local_model(DEFAULT_CONFIG.copy())
    assert config["llm_provider"] == "openai_compatible"
    assert config["deep_think_llm"] == config["quick_think_llm"] == "qwen/qwen3.6-35b-a3b"
    assert config["backend_url"] == "http://192.168.1.246:1234/v1"
    assert config["openai_reasoning_effort"] == "none"
    assert config["structured_output_method"] == "json_schema"


@pytest.mark.unit
@pytest.mark.parametrize("tier", ["quick", "deep"])
def test_builds_an_lmstudio_client_with_the_token(lmstudio_env, tier):
    llm = create_tier_client(local_llm.use_local_model(DEFAULT_CONFIG.copy()), tier).get_llm()
    assert isinstance(llm, LocalCompatibleChatOpenAI)
    assert llm.model_name == "qwen/qwen3.6-35b-a3b"
    assert llm.openai_api_base == "http://192.168.1.246:1234/v1"
    # LM_API_TOKEN alone is enough: it is sent as the client's key.
    assert llm.openai_api_key.get_secret_value() == "sk-lm-test-token"
    assert llm.reasoning_effort == "none"
    assert llm.max_tokens == 16384
    assert llm.structured_output_method == "json_schema"


@pytest.mark.unit
def test_explicit_key_wins_over_lm_api_token(lmstudio_env, monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-lm-explicit")
    llm = create_tier_client(local_llm.use_local_model(DEFAULT_CONFIG.copy()), "quick").get_llm()
    assert llm.openai_api_key.get_secret_value() == "sk-lm-explicit"


@pytest.mark.unit
def test_model_and_address_from_the_environment(lmstudio_env, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen/qwen3.5-9b")
    monkeypatch.setenv("TRADINGAGENTS_BACKEND_URL", "http://localhost:1234/v1")
    importlib.reload(local_llm)
    config = local_llm.use_local_model(DEFAULT_CONFIG.copy())
    assert config["quick_think_llm"] == "qwen/qwen3.5-9b"
    assert config["backend_url"] == "http://localhost:1234/v1"


@pytest.mark.unit
def test_blank_environment_values_keep_the_defaults(lmstudio_env, monkeypatch):
    monkeypatch.setenv("LMSTUDIO_MODEL", "")
    monkeypatch.setenv("TRADINGAGENTS_BACKEND_URL", "")
    importlib.reload(local_llm)
    config = local_llm.use_local_model(DEFAULT_CONFIG.copy())
    assert config["quick_think_llm"] == "qwen/qwen3.6-35b-a3b"
    assert config["backend_url"] == "http://192.168.1.246:1234/v1"
