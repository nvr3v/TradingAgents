"""TradingView hosted-MCP data vendor: symbol mapping, formats, error mapping, config.

The MCP call is replaced by canned payloads shaped like the server's responses,
so nothing here signs in or touches the network.
"""

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.vendors.tradingview_mcp import client, vendor as tv


def _ts(day: str, hour: int, minute: int = 0) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(hour=hour, minute=minute, tzinfo=UTC).timestamp())


SEARCH = {"data": {"symbols": [
    {"symbol": "BCBA:NVDA", "exchange": "BYMA", "type": "dr"},
    {"symbol": "NASDAQ:NVDA", "exchange": "NASDAQ", "type": "stock"},
]}, "success": True}


@pytest.fixture
def server(monkeypatch):
    """Route call_tool to per-tool handlers; records every call."""
    handlers, calls = {}, []

    def fake_call(name, arguments):
        calls.append((name, arguments))
        reply = handlers[name]
        return json.dumps(reply(arguments) if callable(reply) else reply)

    monkeypatch.setattr(client, "call_tool", fake_call)
    tv.resolve_symbol.cache_clear()
    handlers["mcp-tv-search-symbols"] = SEARCH
    yield handlers, calls
    tv.resolve_symbol.cache_clear()


@pytest.mark.unit
def test_resolve_prefers_primary_us_listing(server):
    assert tv.resolve_symbol("nvda") == "NASDAQ:NVDA"


@pytest.mark.unit
@pytest.mark.parametrize("ticker,expected", [
    ("0700.HK", "HKEX:700"), ("7203.T", "TSE:7203"), ("SHOP.TO", "TSX:SHOP"),
    ("2330.TW", "TWSE:2330"), ("6488.TWO", "TPEX:6488"), ("AMEX:SPY", "AMEX:SPY"),
])
def test_resolve_suffix_and_explicit(server, ticker, expected):
    handlers, calls = server
    assert tv.resolve_symbol(ticker) == expected
    assert not calls  # built directly, no search


@pytest.mark.unit
def test_resolve_no_exact_match(server):
    handlers, _ = server
    handlers["mcp-tv-search-symbols"] = {"data": {"symbols": [{"symbol": "NASDAQ:NVDL"}]}, "success": True}
    with pytest.raises(NoMarketDataError):
        tv.resolve_symbol("NVDX")


@pytest.mark.unit
def test_session_open_dates():
    assert tv._utc_date(_ts("2026-11-04", 14, 30)) == "2026-11-04"   # NASDAQ
    assert tv._utc_date(_ts("2026-11-04", 1, 30)) == "2026-11-04"    # HKEX
    assert tv._utc_date(_ts("2026-11-03", 22)) == "2026-11-04"       # FX opens the evening before
    assert tv._utc_date(_ts("2026-10-29", 23)) == "2026-10-30"       # TVC yields


@pytest.mark.unit
def test_stock_data_csv_and_window(server):
    handlers, calls = server
    handlers["mcp-tv-get-ohlcv"] = {"bars": {
        "t": [_ts("2026-11-02", 14, 30), _ts("2026-11-03", 14, 30), _ts("2026-11-04", 14, 30)],
        "o": [1, 2, 3], "h": [2, 3, 4], "l": [0.5, 1.5, 2.5], "c": [1.5, 2.5, 3.5], "v": [10, 20, 30]},
        "success": True}
    out = tv.get_stock_data("NVDA", "2026-11-03", "2026-11-04")
    assert "NASDAQ:NVDA" in out and "Total records: 2" in out
    assert "2026-11-03,2,3,1.5,2.5,20" in out and "2026-11-02" not in out
    ohlcv_args = [a for n, a in calls if n == "mcp-tv-get-ohlcv"][0]
    assert ohlcv_args["date_from"] == "2026-11-02"  # a day early, filtered locally


@pytest.mark.unit
def test_stock_data_empty_is_no_data(server):
    handlers, _ = server
    handlers["mcp-tv-get-ohlcv"] = {"bars": {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}, "success": True}
    with pytest.raises(NoMarketDataError):
        tv.get_stock_data("NVDA", "2026-11-03", "2026-11-04")


@pytest.mark.unit
def test_indicator_from_bars(server):
    handlers, _ = server
    start = datetime(2026, 1, 1, tzinfo=UTC)
    days = [start + timedelta(days=i) for i in range(300)]
    closes = [100 + i for i in range(300)]
    handlers["mcp-tv-get-ohlcv"] = {"bars": {
        "t": [int(d.replace(hour=14, minute=30).timestamp()) for d in days],
        "o": closes, "h": closes, "l": closes, "c": closes, "v": [1] * 300}, "success": True}
    last = days[-1].strftime("%Y-%m-%d")
    out = tv.get_indicators("NVDA", "close_10_sma", last, 1)
    assert f"{last}: {sum(closes[-10:]) / 10:g}" in out


@pytest.mark.unit
def test_fundamentals_snapshot_refuses_old_dates(server):
    with pytest.raises(VendorUnavailableError, match="leak"):
        tv.get_fundamentals("NVDA", "2020-01-02")


@pytest.mark.unit
def test_fundamentals_current(server):
    handlers, _ = server
    handlers["mcp-tv-get-financials"] = {"data": {"price_earnings_ttm": 25.5, "market_cap_basic": 4.8e12}, "success": True}
    handlers["mcp-tv-get-forecasts"] = {"data": {"analyst_rating": {"recommendation": "Buy", "total_analysts": 3},
                                                 "price_targets": {"average": 300}, "estimates": {}}, "success": True}
    out = tv.get_fundamentals("NVDA", date.today().isoformat())
    assert "price_earnings_ttm: 25.5" in out and "recommendation: Buy" in out


@pytest.mark.unit
def test_news_window_filter(server):
    handlers, _ = server
    handlers["mcp-tv-get-news"] = {"data": {"has_more": False, "headlines": [
        {"title": "In window", "published": _ts("2026-11-03", 12), "provider": {"name": "P"}, "link": "L"},
        {"title": "Too new", "published": _ts("2026-11-05", 12), "provider": {"name": "P"}},
        {"title": "Too old", "published": _ts("2026-10-01", 12), "provider": {"name": "P"}},
    ]}, "success": True}
    out = tv.get_news("NVDA", "2026-11-02", "2026-11-04")
    assert "In window" in out and "Too new" not in out and "Too old" not in out


@pytest.mark.unit
def test_macro_alias_and_unknown(server):
    handlers, calls = server
    handlers["mcp-tv-get-economic-data"] = {"description": "US Inflation", "unit": "%",
                                           "series": [{"date": "2026-08-01", "value": 3.4},
                                                      {"date": "2026-09-01", "value": 3.7}], "success": True}
    out = tv.get_macro_indicators("cpi", "2026-11-04", 120)
    assert "Latest: 3.7%" in out
    name, args = calls[-1]
    assert name == "mcp-tv-get-economic-data" and args["symbol"] == "ECONOMICS:USIRYY"
    assert args["date_from"] <= "2025-06-01" and args["date_to"] == "2026-11-04"
    assert "unknown macro indicator" in tv.get_macro_indicators("bogus", "2026-11-04")


@pytest.mark.unit
def test_macro_short_window_keeps_latest_releases(server):
    # A 30-day window on a monthly series holds no release; the latest ones are shown.
    handlers, _ = server
    handlers["mcp-tv-get-economic-data"] = {"description": "US Interest Rate", "unit": "%", "series": [
        {"date": "2026-08-01", "value": 4.25}, {"date": "2026-09-01", "value": 4.25},
        {"date": "2026-10-01", "value": 4.0}], "success": True}
    out = tv.get_macro_indicators("fed_funds_rate", "2026-11-04", 30)
    assert "Latest: 4%" in out and "| 2026-08-01 | 4.25 |" in out


@pytest.mark.unit
def test_error_mapping(monkeypatch):
    tv.resolve_symbol.cache_clear()

    def not_signed_in(*_):
        raise client.TradingViewNotSignedIn(client.LOGIN_HINT)
    monkeypatch.setattr(client, "call_tool", not_signed_in)
    with pytest.raises(VendorNotConfiguredError):
        tv.get_stock_data("NVDA", "2026-11-03", "2026-11-04")

    monkeypatch.setattr(client, "call_tool", lambda *_: json.dumps({"success": False, "error": "429"}))
    with pytest.raises(VendorUnavailableError):
        tv.get_stock_data("AMEX:SPY", "2026-11-03", "2026-11-04")


@pytest.mark.unit
def test_use_tradingview_chains(monkeypatch):
    from tradingagents.dataflows.router import VENDOR_METHODS
    from tradingagents.default_config import (
        TRADINGVIEW_VENDORS,
        build_default_config,
        use_tradingview,
    )

    config = use_tradingview(build_default_config())
    for category, chain in TRADINGVIEW_VENDORS.items():
        assert config["data_vendors"][category] == chain
    assert config["data_vendors"]["prediction_markets"] == "polymarket"  # untouched
    for method in ("get_stock_data", "get_indicators", "get_fundamentals",
                   "get_income_statement", "get_news", "get_macro_indicators"):
        assert "tradingview" in VENDOR_METHODS[method]
    monkeypatch.setenv("TRADINGAGENTS_USE_TRADINGVIEW", "true")
    assert build_default_config()["data_vendors"]["core_stock_apis"] == "tradingview,yfinance"


@pytest.mark.unit
def test_router_falls_back_when_not_signed_in(monkeypatch):
    from tradingagents.dataflows import router
    from tradingagents.dataflows.config import get_config, set_config

    tv.resolve_symbol.cache_clear()
    monkeypatch.setattr(client, "is_signed_in", lambda: False)
    monkeypatch.setitem(router.VENDOR_METHODS["get_stock_data"], "yfinance", lambda *a: "from yahoo")
    saved = get_config()
    set_config({"data_vendors": {**saved["data_vendors"], "core_stock_apis": "tradingview,yfinance"}})
    try:
        assert router.route_to_vendor("get_stock_data", "NVDA", "2026-11-03", "2026-11-04") == "from yahoo"
    finally:
        set_config(saved)


@pytest.mark.unit
def test_auth_file_private_and_stale_registration(tmp_path, monkeypatch):
    import asyncio
    import stat

    from mcp.shared.auth import OAuthClientInformationFull

    monkeypatch.setattr(client, "AUTH_FILE", tmp_path / "tv.json")
    storage = client._FileTokenStorage()
    info = OAuthClientInformationFull(client_id="abc", redirect_uris=["http://127.0.0.1:8765/callback"])
    asyncio.run(storage.set_client_info(info))
    assert stat.S_IMODE((tmp_path / "tv.json").stat().st_mode) == 0o600
    # Registered for another redirect address: discarded so the SDK registers afresh.
    assert asyncio.run(storage.get_client_info()) is None
    current = OAuthClientInformationFull(client_id="xyz", redirect_uris=[client._redirect_uri(client.CALLBACK_PORT)])
    asyncio.run(storage.set_client_info(current))
    assert asyncio.run(storage.get_client_info()).client_id == "xyz"


@pytest.mark.unit
def test_redirect_avoids_cloudfront_blocked_hosts():
    uri = client._redirect_uri(8765)
    assert "localhost" not in uri and "127.0.0.1" not in uri and uri.startswith("http://[::1]:8765/")


@pytest.mark.unit
def test_token_expiry_recorded_and_inferred(tmp_path, monkeypatch):
    """A later process must know the 15-minute access token lapsed, or it never refreshes."""
    import asyncio
    import os
    import time

    from mcp.shared.auth import OAuthToken

    monkeypatch.setattr(client, "AUTH_FILE", tmp_path / "tv.json")
    before = time.time()
    asyncio.run(client._FileTokenStorage().set_tokens(
        OAuthToken(access_token="a", token_type="Bearer", expires_in=900, refresh_token="r")))
    assert before + 900 <= client._token_expiry() <= time.time() + 900

    # A sign-in saved before expires_at existed is dated by the file's mtime.
    data = json.loads((tmp_path / "tv.json").read_text())
    del data["expires_at"]
    (tmp_path / "tv.json").write_text(json.dumps(data))
    os.utime(tmp_path / "tv.json", (1000, 1000))
    assert client._token_expiry() == 1900


@pytest.mark.unit
def test_token_storage_off_the_event_loop(tmp_path, monkeypatch):
    """Under an event loop that forbids blocking calls (``langgraph dev``) a refreshed
    token must still be saved: the refresh rotates the refresh token, so a failed
    write leaves the sign-in on disk spent."""
    import asyncio

    blockbuster = pytest.importorskip("blockbuster")
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    monkeypatch.setattr(client, "AUTH_FILE", tmp_path / "sub" / "tv.json")
    storage = client._FileTokenStorage()
    token = OAuthToken(access_token="a", token_type="Bearer", expires_in=900, refresh_token="r2")
    info = OAuthClientInformationFull(client_id="xyz", redirect_uris=[client._redirect_uri(client.CALLBACK_PORT)])

    async def roundtrip():
        await storage.set_client_info(info)
        await storage.set_tokens(token)
        return await storage.get_tokens(), await storage.get_client_info()

    with blockbuster.blockbuster_ctx():
        tokens, saved_info = asyncio.run(roundtrip())
    assert tokens.refresh_token == "r2" and saved_info.client_id == "xyz"
