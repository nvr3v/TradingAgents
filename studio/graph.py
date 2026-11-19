"""TradingAgents for LangGraph Studio (`langgraph dev`, see langgraph.json).

Start the server with `langgraph dev --no-reload`, and restart it after a code
change: on Windows its auto-reload can report a reload yet keep the old code.

Studio runs a graph and checkpoints it itself, so this exposes the agents'
workflow without the project's own SQLite checkpointer, behind one step that
builds a run's initial state the way ``propagate()`` does. Start a run in Studio
with just a ticker and a date:

    {"company_of_interest": "NVDA", "trade_date": "2026-11-05"}

The agents' workflow shows as a nested graph ("TradingAgents") you can open to
follow each analyst, the debates and the managers, and to inspect the state.

Unlike ``propagate()``, a Studio run does not write the state log or add its
decision to the memory log.

Model and data settings match main.py: the local model in local_llm.py (Qwen
3.6 35B-A3B on LM Studio), and TradingView's hosted MCP server first for
market data.
"""

from langgraph.graph import END, START, StateGraph
# Not typing's: Studio builds its input form with pydantic, which rejects typing.TypedDict on Python < 3.12.
from typing_extensions import NotRequired, TypedDict

from local_llm import use_local_model
from tradingagents.agents.state import AgentState
from tradingagents.default_config import DEFAULT_CONFIG, use_tradingview
from tradingagents.graph.trading_graph import TradingAgentsGraph

config = DEFAULT_CONFIG.copy()
use_local_model(config)
use_tradingview(config)

ta = TradingAgentsGraph(config=config)


class RunInput(TypedDict):
    """What a Studio run starts from."""

    company_of_interest: str  # ticker, e.g. "NVDA"
    trade_date: str  # YYYY-MM-DD
    asset_type: NotRequired[str]  # "stock" (default) or "crypto"


def prepare_run(state: RunInput) -> dict:
    """Build the run's initial state, as propagate() does."""
    return ta.create_run_state(
        state["company_of_interest"], state["trade_date"], state.get("asset_type") or "stock"
    )


builder = StateGraph(AgentState, input_schema=RunInput)
builder.add_node("Prepare Run", prepare_run)
builder.add_node("TradingAgents", ta.workflow.compile())
builder.add_edge(START, "Prepare Run")
builder.add_edge("Prepare Run", "TradingAgents")
builder.add_edge("TradingAgents", END)

graph = builder.compile().with_config(recursion_limit=ta.propagator.max_recur_limit)
