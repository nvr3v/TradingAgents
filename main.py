from datetime import date

from tradingagents.default_config import DEFAULT_CONFIG, use_tradingview
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Local Qwen 3.5 9B served by LM Studio (OpenAI-compatible API, no key needed).
config["llm_provider"] = "openai_compatible"
config["backend_url"] = "http://192.168.1.246:1234/v1"
config["deep_think_llm"] = "qwen/qwen3.5-9b"
config["quick_think_llm"] = "qwen/qwen3.5-9b"
# Qwen 3.5 "thinks" before answering; without a higher cap the server's default
# 2048-token output limit is used up by reasoning and the answer comes back empty.
config["max_tokens"] = 16384
# Turn Qwen's thinking off: LM Studio maps reasoning_effort "none" to no
# reasoning, which cuts each answer from hundreds of tokens to a handful.
config["openai_reasoning_effort"] = "none"
# Without a forced tool call Qwen answers the managers' prompts in prose, so the
# structured decision is lost; LM Studio's grammar-constrained JSON fixes that.
config["structured_output_method"] = "json_schema"

# Market data from your TradingView account via its hosted MCP server: realtime
# prices (per your plan), news, fundamentals and macro series, with the default
# vendors behind it as fallbacks. Sign in once: `tradingagents login tradingview`.
use_tradingview(config)

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", date.today().isoformat())
print(decision)
