"""TradingView (hosted MCP server) as a TradingAgents data vendor.

Serves the router's ``get_stock_data``, ``get_indicators``, ``get_fundamentals``,
``get_income_statement``, ``get_news`` and ``get_macro_indicators`` from the
person's own TradingView account, so prices are realtime where their plan has
the exchange's data (delayed otherwise). Only read-only tools are called; the
server's alert and watchlist tools, which change the account, are never used.

Errors map onto the router's taxonomy so a configured fallback vendor takes
over: no sign-in -> ``VendorNotConfiguredError``; a failed or throttled request
-> ``VendorUnavailableError``; an empty window -> ``NoMarketDataError``.

Point-in-time caveat: bars, news and statement history honour the analysis
date, but ``get_fundamentals`` is TradingView's *current* snapshot, so it serves
only analyses dated within ``SNAPSHOT_MAX_AGE_DAYS`` of today; older dates fall
through to the next vendor rather than leak today's figures into a backtest.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import pandas as pd

from tradingagents.dataflows.errors import (
    NoMarketDataError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)

from . import client

logger = logging.getLogger(__name__)

SNAPSHOT_MAX_AGE_DAYS = 3
MAX_BARS = 5000
INDICATOR_WARMUP_DAYS = 400       # history before the window so 200-day averages settle
NEWS_PAGE = 100
NEWS_MAX = 200                    # the server's cap
MACRO_MAX_ROWS = 40
MACRO_LOOKBACK_DAYS = 365
MACRO_MIN_HISTORY_DAYS = 550      # enough for the last few quarterly releases
MACRO_MIN_POINTS = 3              # latest releases shown when the window holds fewer

# Primary US listing venues, preferred when a bare ticker matches several exchanges.
_US_EXCHANGES = ("NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "ARCA", "CBOE", "BATS")

# Yahoo-style suffix -> TradingView exchange prefix.
_SUFFIX_EXCHANGE = {
    ".HK": "HKEX", ".T": "TSE", ".L": "LSE", ".TO": "TSX", ".V": "TSXV", ".NS": "NSE",
    ".BO": "BSE", ".DE": "XETR", ".F": "FWB", ".PA": "EURONEXT", ".AS": "EURONEXT",
    ".BR": "EURONEXT", ".MI": "MIL", ".MC": "BME", ".SW": "SIX", ".AX": "ASX",
    ".SI": "SGX", ".KS": "KRX", ".KQ": "KRX", ".TW": "TWSE", ".TWO": "TPEX",
    ".SS": "SSE", ".SZ": "SZSE", ".SA": "BMFBOVESPA", ".ST": "OMXSTO", ".OL": "OSL",
    ".CO": "OMXCOP", ".HE": "OMXHEX",
}

# Friendly macro aliases (the FRED vendor's names) -> TradingView series. ECONOMICS:
# series come from get_economic_data; the others are market symbols read as daily closes.
MACRO_SERIES = {
    "fed_funds_rate": "ECONOMICS:USINTR", "federal_funds_rate": "ECONOMICS:USINTR",
    "fed_funds": "ECONOMICS:USINTR", "interest_rate": "ECONOMICS:USINTR",
    "2y_treasury": "TVC:US02Y", "10y_treasury": "TVC:US10Y", "30y_treasury": "TVC:US30Y",
    "cpi": "ECONOMICS:USIRYY", "inflation": "ECONOMICS:USIRYY", "core_cpi": "ECONOMICS:USCIR",
    "pce": "ECONOMICS:USPCEPIAC", "core_pce": "ECONOMICS:USCPCEPIAC",
    "gdp": "ECONOMICS:USGDPQQ", "real_gdp": "ECONOMICS:USGDPQQ",
    "industrial_production": "ECONOMICS:USIPYY",
    "unemployment_rate": "ECONOMICS:USUR", "unemployment": "ECONOMICS:USUR",
    "nonfarm_payrolls": "ECONOMICS:USNFP", "payrolls": "ECONOMICS:USNFP",
    "initial_claims": "ECONOMICS:USIJC",
    "m2": "ECONOMICS:USM2", "money_supply": "ECONOMICS:USM2",
    "vix": "TVC:VIX", "dollar_index": "TVC:DXY",
    "consumer_confidence": "ECONOMICS:USCCI", "consumer_sentiment": "ECONOMICS:USCCI",
    "housing_starts": "ECONOMICS:USHST", "retail_sales": "ECONOMICS:USRSMM",
    "eur_usd": "FX:EURUSD", "germany_10y": "TVC:DE10Y", "france_10y": "TVC:FR10Y",
}


class TradingViewNotConfiguredError(VendorNotConfiguredError):
    """No TradingView sign-in on this machine."""


# --- calling the server -----------------------------------------------------------------


def _call(tool: str, arguments: dict) -> dict:
    """Call a read-only tool and return its JSON payload, mapped to vendor errors."""
    try:
        text = client.call_tool(tool, arguments)
    except client.TradingViewNotSignedIn as exc:
        raise TradingViewNotConfiguredError(str(exc)) from exc
    except client.TradingViewMCPError as exc:
        raise VendorUnavailableError(str(exc)) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VendorUnavailableError(f"TradingView {tool} returned non-JSON: {text[:200]}") from exc
    if isinstance(payload, dict) and payload.get("success") is False:
        raise VendorUnavailableError(f"TradingView {tool} failed: {payload.get('error') or payload}")
    return payload


@lru_cache(maxsize=256)
def resolve_symbol(ticker: str) -> str:
    """A ticker as TradingView routes it (``NVDA`` -> ``NASDAQ:NVDA``).

    ``EXCHANGE:TICKER`` passes through. A Yahoo suffix (``0700.HK``) picks that
    exchange; a bare ticker prefers its primary US listing, else the first exact match.
    """
    ticker = ticker.strip().upper()
    if ":" in ticker:
        return ticker
    # The longest matching suffix wins (".TWO" before ".TW"). With the exchange
    # known, the symbol is built directly: search ranks other venues first.
    for suffix in sorted(_SUFFIX_EXCHANGE, key=len, reverse=True):
        if ticker.endswith(suffix):
            base = ticker[: -len(suffix)]
            if suffix == ".HK":
                base = base.lstrip("0") or "0"  # TradingView lists 0700.HK as HKEX:700
            return f"{_SUFFIX_EXCHANGE[suffix]}:{base}"
    base = ticker
    found = _call("mcp-tv-search-symbols", {"query": base}).get("data", {}).get("symbols", [])
    exact = [s for s in found if s.get("symbol", "").split(":")[-1].upper() == base]
    us = [s for s in exact if s.get("exchange", "").upper() in _US_EXCHANGES]
    exact = us or exact
    if not exact:
        raise NoMarketDataError(ticker, ticker, "TradingView symbol search found no exact match")
    return exact[0]["symbol"]


def _utc_date(ts: int) -> str:
    """The trading date of a daily bar stamped with its session open, in UTC.

    Sessions in the Americas, Europe and most of Asia open the same UTC day
    (NASDAQ 13:30/14:30, HKEX 01:30). Forex, bond and futures sessions, and
    Sydney or Wellington, open the evening before in UTC (21:00-23:00): a bar
    stamped from 18:00 UTC on belongs to the next calendar day.
    """
    opened = datetime.fromtimestamp(ts, UTC)
    return (opened + timedelta(days=1) if opened.hour >= 18 else opened).strftime("%Y-%m-%d")


def daily_bars(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Daily OHLCV between two dates (inclusive), as Date/Open/High/Low/Close/Volume."""
    symbol = resolve_symbol(ticker)
    # A day early: an evening-UTC session open for date_from is stamped the day before.
    query_from = (pd.to_datetime(date_from) - timedelta(days=1)).strftime("%Y-%m-%d")
    payload = _call("mcp-tv-get-ohlcv", {
        "symbol": symbol, "interval": "1D", "date_from": query_from, "date_to": date_to,
        "count": MAX_BARS, "format": "columns",
    })
    bars = payload.get("bars") or {}
    if isinstance(bars, list):  # rows layout, should the server ignore format
        bars = {k: [b[k] for b in bars] for k in ("t", "o", "h", "l", "c", "v")} if bars else {}
    if not bars or not bars.get("t"):
        raise NoMarketDataError(ticker, symbol, f"no daily bars between {date_from} and {date_to}")
    df = pd.DataFrame({
        "Date": pd.to_datetime([_utc_date(t) for t in bars["t"]]),
        "Open": bars["o"], "High": bars["h"], "Low": bars["l"], "Close": bars["c"],
        "Volume": bars.get("v") or [None] * len(bars["t"]),
    })
    df = df[(df["Date"] >= pd.to_datetime(date_from)) & (df["Date"] <= pd.to_datetime(date_to))]
    if df.empty:
        raise NoMarketDataError(ticker, symbol, f"no daily bars between {date_from} and {date_to}")
    return df.sort_values("Date").reset_index(drop=True)


def load_ohlcv(ticker: str, as_of_date: str, years: int = 2) -> pd.DataFrame:
    """Daily bars for ``years`` up to ``as_of_date``: the verified snapshot's input."""
    start = (pd.to_datetime(as_of_date) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    return daily_bars(ticker, start, as_of_date)


def _fmt_num(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
    if isinstance(value, int) and abs(value) >= 1e6:
        return f"{value:,}"
    return str(value)


# --- router methods ---------------------------------------------------------------------


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    df = daily_bars(symbol, start_date, end_date)
    tv = resolve_symbol(symbol)
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    header = (f"# Stock data for {tv} (from {symbol}) from {start_date} to {end_date}, via TradingView\n"
              f"# Total records: {len(out)}. Realtime or delayed per the user's TradingView plan; "
              "the latest bar may still be forming. Split-adjusted, regular session.\n\n")
    return header + out.to_csv(index=False)


def get_indicators(symbol: str, indicator: str, as_of_date: str, look_back_days: int) -> str:
    """A stockstats indicator over the window, computed from TradingView daily bars."""
    from stockstats import wrap

    end = pd.to_datetime(as_of_date)
    window_start = end - timedelta(days=int(look_back_days))
    history_start = (window_start - timedelta(days=INDICATOR_WARMUP_DAYS)).strftime("%Y-%m-%d")
    df = daily_bars(symbol, history_start, as_of_date)
    stock = wrap(df.copy())
    try:
        values = stock[indicator]
    except Exception as exc:  # noqa: BLE001 — an unknown indicator name from the LLM
        raise NoMarketDataError(symbol, resolve_symbol(symbol), f"indicator {indicator!r} unavailable: {exc}") from exc
    by_date = {d.strftime("%Y-%m-%d"): v for d, v in zip(df["Date"], values, strict=False)}
    lines, day = [], end
    while day >= window_start:
        key = day.strftime("%Y-%m-%d")
        value = by_date.get(key)
        lines.append(f"{key}: {'N/A: Not a trading day (weekend or holiday)' if value is None else _fmt_num(float(value)) if pd.notna(value) else 'N/A'}")
        day -= timedelta(days=1)
    return (f"## {indicator} values from {window_start:%Y-%m-%d} to {as_of_date} "
            f"(computed from TradingView daily bars for {resolve_symbol(symbol)}):\n\n" + "\n".join(lines))


def _require_recent(ticker: str, as_of_date: str, what: str) -> None:
    age = (datetime.now().date() - datetime.strptime(as_of_date, "%Y-%m-%d").date()).days
    if age > SNAPSHOT_MAX_AGE_DAYS:
        raise VendorUnavailableError(
            f"TradingView serves only the current {what}; {as_of_date} is {age} days old "
            "and would leak today's figures into a historical analysis")


def get_fundamentals(ticker: str, as_of_date: str) -> str:
    _require_recent(ticker, as_of_date, "fundamentals snapshot")
    symbol = resolve_symbol(ticker)
    fin = _call("mcp-tv-get-financials", {"symbol": symbol}).get("data", {})
    lines = [f"## Fundamentals for {symbol} (TradingView, current as of {datetime.now():%Y-%m-%d})", ""]
    lines += [f"- {k}: {_fmt_num(v)}" for k, v in fin.items()]
    try:
        fc = _call("mcp-tv-get-forecasts", {"symbol": symbol}).get("data", {})
    except VendorUnavailableError as exc:
        logger.warning("TradingView forecasts unavailable for %s: %s", symbol, exc)
        fc = {}
    if fc:
        rating, targets, est = fc.get("analyst_rating", {}), fc.get("price_targets", {}), fc.get("estimates", {})
        lines += ["", "### Analyst consensus",
                  f"- recommendation: {rating.get('recommendation')} (score {rating.get('score')}, "
                  f"{rating.get('total_analysts')} analysts: buy {rating.get('buy')}, overweight "
                  f"{rating.get('overweight')}, hold {rating.get('hold')}, underweight "
                  f"{rating.get('underweight')}, sell {rating.get('sell')})",
                  f"- price targets: average {targets.get('average')}, high {targets.get('high')}, "
                  f"low {targets.get('low')}, upside {targets.get('upside_pct')}%"]
        lines += [f"- {k}: {_fmt_num(v)}" for k, v in est.items()]
    return "\n".join(lines)


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Reported fundamentals history (revenue through EPS, FCF and debt) up to as_of_date."""
    symbol = resolve_symbol(ticker)
    period = "fy" if str(freq).lower().startswith("a") else "fq"
    args = {"symbol": symbol, "period": period}
    if as_of_date:
        args["date_to"] = as_of_date
    payload = _call("mcp-tv-get-financial-history", args)
    labels, series = payload.get("labels") or [], payload.get("series") or {}
    if not labels or not series:
        raise NoMarketDataError(ticker, symbol, "no reported financial history")
    table = pd.DataFrame(
        {name: [_fmt_num(p.get("value")) + (f" ({p['yoy_pct']:+}% YoY)" if p.get("yoy_pct") is not None else "")
                for p in points] for name, points in series.items()},
        index=labels[-len(next(iter(series.values()))):],
    ).T
    return (f"# {'Annual' if period == 'fy' else 'Quarterly'} reported financials for {symbol} "
            f"(TradingView){f', reports up to {as_of_date}' if as_of_date else ''}\n\n" + table.to_csv())


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    symbol = resolve_symbol(ticker)
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    kept, offset, oldest = [], 0, None
    while offset < NEWS_MAX:
        data = _call("mcp-tv-get-news", {"symbol": symbol, "limit": NEWS_PAGE, "offset": offset}).get("data", {})
        headlines = data.get("headlines") or []
        for h in headlines:
            published = datetime.fromtimestamp(h.get("published", 0), UTC)
            oldest = published if oldest is None else min(oldest, published)
            if start <= published < end:
                kept.append((published, h))
        if not data.get("has_more") or not headlines or (oldest and oldest < start):
            break
        offset += len(headlines)
    if not kept:
        if oldest is None or oldest > end:
            return (f"<TradingView news unavailable for {start_date}..{end_date}: its feed for {symbol} "
                    "reaches back only to its latest ~200 headlines, so this is not an absence of news>")
        return f"No TradingView news for {symbol} between {start_date} and {end_date}"
    lines = [f"## {symbol} news from {start_date} to {end_date} (TradingView, {len(kept)} headlines)", ""]
    for published, h in sorted(kept, key=lambda x: x[0], reverse=True):
        provider = (h.get("provider") or {}).get("name", "")
        lines.append(f"### {h.get('title', '').strip()}")
        lines.append(f"{provider}, {published:%Y-%m-%d %H:%M} UTC{' (paywalled)' if h.get('paywall') else ''}"
                     f" — {h.get('link', '')}")
        lines.append("")
    return "\n".join(lines)


def get_macro_indicators(indicator: str, as_of_date: str, look_back_days: int | None = None) -> str:
    key = indicator.strip()
    series = key.upper() if ":" in key else MACRO_SERIES.get(key.lower())
    if series is None:
        return (f"TradingView: unknown macro indicator {indicator!r}. Use one of "
                f"{', '.join(sorted(MACRO_SERIES))}, or a TradingView ECONOMICS: ticker "
                "such as ECONOMICS:USIRYY.")
    start = (datetime.strptime(as_of_date, "%Y-%m-%d")
             - timedelta(days=look_back_days or MACRO_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    if series.startswith("ECONOMICS:"):
        # Monthly and quarterly series may have no release inside a short window:
        # read a longer history and keep the window, or failing that the latest releases.
        history_from = min(start, (datetime.strptime(as_of_date, "%Y-%m-%d")
                                   - timedelta(days=MACRO_MIN_HISTORY_DAYS)).strftime("%Y-%m-%d"))
        payload = _call("mcp-tv-get-economic-data", {"symbol": series, "date_from": history_from, "date_to": as_of_date})
        points = [(p["date"], p["value"]) for p in payload.get("series") or []
                  if p.get("value") is not None and p["date"] <= as_of_date]
        in_window = [p for p in points if p[0] >= start]
        points = in_window if len(in_window) >= MACRO_MIN_POINTS else points[-MACRO_MIN_POINTS:]
        title, unit = payload.get("description") or series, payload.get("unit") or ""
        note = payload.get("notice", "")
    else:
        df = daily_bars(series, start, as_of_date)
        points = [(d.strftime("%Y-%m-%d"), c) for d, c in zip(df["Date"], df["Close"], strict=False)]
        title, unit, note = f"{series} daily close", "", ""
    if not points:
        raise NoMarketDataError(indicator, series, f"no observations between {start} and {as_of_date}")
    first, last = points[0][1], points[-1][1]
    lines = [f"## {title} ({series}, TradingView)", "",
             f"- Latest: {_fmt_num(last)}{unit} on {points[-1][0]}",
             f"- Change since {points[0][0]}: {_fmt_num(last - first)}{unit}",
             *([f"- Note: {note}"] if note else []), "", "| Date | Value |", "|---|---|"]
    lines += [f"| {d} | {_fmt_num(v)} |" for d, v in points[-MACRO_MAX_ROWS:]]
    return "\n".join(lines)
