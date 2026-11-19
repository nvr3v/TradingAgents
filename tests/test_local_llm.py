"""The local LM Studio model that main.py and studio/graph.py run on (local_llm.py)."""

import importlib

import pytest

import local_llm
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.factory import create_tier_client
from tradingagents.llm_clients.openai_client import LocalCompatibleChatOpenAI


@pytest.fixture
def lmstudio_env(monkeypatch):
    for name in ("LMSTUDIO_MODEL", "LMSTUDIO_CONTEXT_LENGTH", "TRADINGAGENTS_BACKEND_URL", "OPENAI_COMPATIBLE_API_KEY"):
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


@pytest.fixture
def lmstudio_api(lmstudio_env, monkeypatch):
    """A fake LM Studio management API: set ``state["instances"]``; records each call."""
    state = {"instances": [], "downloaded": True, "load_error": None}
    calls = []

    def api(method, path, body=None, timeout=30):
        calls.append((method, path, body))
        if path == "/api/v1/models":
            models = [{"key": "qwen/qwen3.6-35b-a3b", "loaded_instances": state["instances"]}]
            return {"models": models if state["downloaded"] else []}
        if path == "/api/v1/models/unload":
            state["instances"] = [i for i in state["instances"] if i["id"] != body["instance_id"]]
            return {"instance_id": body["instance_id"]}
        if path == "/api/v1/models/load":
            if state["load_error"]:
                raise RuntimeError(state["load_error"])
            state["instances"] = [{"id": body["model"], "config": {"context_length": body["context_length"]}}]
            return {"status": "loaded"}
        raise AssertionError(path)

    monkeypatch.setattr(local_llm, "_api", api)
    return state, calls


def _instance(context_length):
    return {"id": "qwen/qwen3.6-35b-a3b", "config": {"context_length": context_length}}


@pytest.mark.unit
def test_loads_the_model_with_65k_context_by_default(lmstudio_api):
    state, calls = lmstudio_api
    assert local_llm.CONTEXT_LENGTH == 65536
    assert local_llm.ensure_model_loaded() == "loaded"
    assert ("POST", "/api/v1/models/load", {"model": "qwen/qwen3.6-35b-a3b", "context_length": 65536}) in calls
    assert state["instances"] == [_instance(65536)]


@pytest.mark.unit
def test_replaces_a_copy_loaded_with_a_smaller_context(lmstudio_api):
    """LM Studio's own first-use load gives 8192 tokens; that copy is swapped out."""
    state, calls = lmstudio_api
    state["instances"] = [_instance(8192)]
    assert local_llm.ensure_model_loaded() == "reloaded"
    assert [p for _, p, _ in calls] == ["/api/v1/models", "/api/v1/models/unload", "/api/v1/models/load"]
    assert state["instances"] == [_instance(65536)]


@pytest.mark.unit
@pytest.mark.parametrize("loaded", [65536, 131072])
def test_keeps_a_copy_with_enough_context(lmstudio_api, loaded):
    state, calls = lmstudio_api
    state["instances"] = [_instance(loaded)]
    assert local_llm.ensure_model_loaded() == "already loaded"
    assert [p for _, p, _ in calls] == ["/api/v1/models"]


@pytest.mark.unit
@pytest.mark.parametrize("value,expected", [("32768", 32768), ("", 65536), ("0", 0)])
def test_context_length_from_the_environment(lmstudio_env, monkeypatch, value, expected):
    monkeypatch.setenv("LMSTUDIO_CONTEXT_LENGTH", value)
    importlib.reload(local_llm)
    assert local_llm.CONTEXT_LENGTH == expected


@pytest.mark.unit
def test_an_explicit_context_length_is_loaded(lmstudio_api):
    state, _ = lmstudio_api
    assert local_llm.ensure_model_loaded(context_length=32768) == "loaded"
    assert state["instances"] == [_instance(32768)]


@pytest.mark.unit
def test_zero_context_length_leaves_loading_to_lm_studio(lmstudio_api):
    _, calls = lmstudio_api
    assert local_llm.ensure_model_loaded(context_length=0) == "skipped"
    assert not calls


@pytest.mark.unit
@pytest.mark.parametrize("problem", ["not downloaded", "load fails", "server down"])
def test_never_raises(lmstudio_api, monkeypatch, problem, caplog):
    state, _ = lmstudio_api
    if problem == "not downloaded":
        state["downloaded"] = False
    elif problem == "load fails":
        state["load_error"] = "LM Studio POST /api/v1/models/load failed (500): insufficient system resources"
    else:
        def down(*_a, **_k):
            raise OSError("connection refused")
        monkeypatch.setattr(local_llm, "_api", down)
    assert local_llm.ensure_model_loaded() == "failed"
    assert caplog.records and caplog.records[-1].levelname == "WARNING"


@pytest.mark.unit
def test_api_calls_the_server_root_with_the_token(lmstudio_env, monkeypatch):
    seen = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return b'{"models": []}'

    def urlopen(request, timeout):
        seen.update(url=request.full_url, auth=request.get_header("Authorization"), timeout=timeout)
        return Reply()

    monkeypatch.setattr(local_llm.urllib.request, "urlopen", urlopen)
    assert local_llm._api("GET", "/api/v1/models") == {"models": []}
    assert seen == {"url": "http://192.168.1.246:1234/api/v1/models", "auth": "Bearer sk-lm-test-token", "timeout": 30}
