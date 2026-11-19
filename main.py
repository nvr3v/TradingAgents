from datetime import date

from local_llm import use_local_model
from tradingagents.default_config import DEFAULT_CONFIG, use_tradingview
from tradingagents.graph.trading_graph import TradingAgentsGraph

# DEFAULT_CONFIG already applies TRADINGAGENTS_* env-var overrides
# (llm_provider, deep_think_llm, quick_think_llm, backend_url, etc.),
# so users can switch models or endpoints purely via .env without
# editing this script. Override individual keys here only when you
# want a hard-coded value that should ignore the environment.
config = DEFAULT_CONFIG.copy()

# Local Qwen 3.6 35B-A3B served by LM Studio; the model and its settings are in local_llm.py.
use_local_model(config)

# Market data from your TradingView account via its hosted MCP server: realtime
# prices (per your plan), news and macro series (fundamentals come from SEC EDGAR
# and yfinance: TradingView's MCP server no longer serves them), with the default
# vendors behind it as fallbacks. Sign in once: `tradingagents login tradingview`.
use_tradingview(config)

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", date.today().isoformat())
print(decision)
