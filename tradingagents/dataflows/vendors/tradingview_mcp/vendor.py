"""TradingView (hosted MCP server) as a TradingAgents data vendor.

Serves the router's ``get_stock_data``, ``get_indicators``, ``get_news`` and
``get_macro_indicators`` from the person's own TradingView account, so prices
are realtime where their plan has the exchange's data (delayed otherwise). Only
read-only tools are called; the server's alert and watchlist tools, which
change the account, are never used.

Fundamentals are not served: the server withdrew its financials, financial
history and forecasts tools, and its symbol-data tools drop fundamentals,
estimates, earnings and dividends columns. Those come from the next vendors in
the chain (SEC EDGAR, yfinance).

Errors map onto the router's taxonomy so a configured fallback vendor takes
over: no sign-in -> ``VendorNotConfiguredError``; a failed or throttled request
-> ``VendorUnavailableError``; an empty window -> ``NoMarketDataError``.

The account has a call quota (per minute and per day). Daily bars are fetched
once per symbol and reused for a few minutes, since every indicator reads the
same history; and once the server reports the quota spent, TradingView is
skipped until its cooldown ends, rather than spending the run on refused calls.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
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

MAX_BARS = 5000
INDICATOR_WARMUP_DAYS = 400       # history before the window so 200-day averages settle
BARS_TTL_SECONDS = 600            # how long fetched daily bars are reused
NEWS_PAGE = 100
NEWS_MIN_PAGE = 25                # smallest page tried when a large one comes back broken
NEWS_MAX = 200                    # the server's cap
QUOTA_COOLDOWN_SECONDS = 600      # when the quota reply names no wait (it has said 10 minutes)
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


class TradingViewBadReplyError(VendorUnavailableError):
    """A reply that does not hold a JSON payload (e.g. one cut short)."""


# --- calling the server -----------------------------------------------------------------

_quota_lock = threading.Lock()
_quota_blocked_until = 0.0   # time.time() before which TradingView is not called


def _note_quota(message: str) -> None:
    """Skip TradingView until the cooldown a quota refusal names (or a default) ends."""
    global _quota_blocked_until
    wait = re.search(r"in (\d+)\s*min", message)
    seconds = int(wait.group(1)) * 60 if wait else QUOTA_COOLDOWN_SECONDS
    with _quota_lock:
        _quota_blocked_until = max(_quota_blocked_until, time.time() + seconds)
    logger.warning("TradingView call quota spent; skipping TradingView until %s",
                   datetime.fromtimestamp(_quota_blocked_until).strftime("%H:%M"))


def _decode(tool: str, text: str):
    """The JSON payload of a reply, ignoring any text the server put after it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    stripped = text.lstrip()
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise TradingViewBadReplyError(
            f"TradingView {tool} returned a reply that is not JSON ({len(text)} chars, {exc}; "
            f"starts {text[:120]!r}, ends {text[-120:]!r})") from exc
    if stripped[end:].strip():
        logger.debug("TradingView %s: text after its JSON reply: %.200s", tool, stripped[end:].strip())
    return payload


def _call(tool: str, arguments: dict) -> dict:
    """Call a read-only tool and return its JSON payload, mapped to vendor errors."""
    if time.time() < _quota_blocked_until:
        raise VendorUnavailableError(
            "TradingView call quota spent; skipping TradingView until "
            f"{datetime.fromtimestamp(_quota_blocked_until):%H:%M}")
    try:
        text = client.call_tool(tool, arguments)
    except client.TradingViewNotSignedIn as exc:
        raise TradingViewNotConfiguredError(str(exc)) from exc
    except client.TradingViewMCPError as exc:
        if "TOOL_QUOTA_EXCEEDED" in str(exc):
            _note_quota(str(exc))
        raise VendorUnavailableError(str(exc)) from exc
    payload = _decode(tool, text)
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


# (symbol, date_to) -> (fetched at, date_from fetched, bars): a later request for the
# same end date and a shorter history is served from these, without a call.
_bars_cache: dict[tuple[str, str], tuple[float, str, pd.DataFrame]] = {}
_bars_lock = threading.Lock()


def _fetch_bars(ticker: str, symbol: str, date_from: str, date_to: str) -> pd.DataFrame:
    with _bars_lock:
        cached = _bars_cache.get((symbol, date_to))
    if cached and cached[1] <= date_from and time.monotonic() - cached[0] < BARS_TTL_SECONDS:
        return cached[2]
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
    with _bars_lock:
        _bars_cache[(symbol, date_to)] = (time.monotonic(), date_from, df)
    return df


def daily_bars(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Daily OHLCV between two dates (inclusive), as Date/Open/High/Low/Close/Volume."""
    symbol = resolve_symbol(ticker)
    # One spelling of each date, so cached windows compare as text.
    date_from, date_to = (pd.to_datetime(d).strftime("%Y-%m-%d") for d in (date_from, date_to))
    df = _fetch_bars(ticker, symbol, date_from, date_to)
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


def _news_page(symbol: str, offset: int, page: int) -> tuple[dict, int]:
    """One page of headlines and the page size that worked: a broken reply to a large
    page (a long one can come back cut short) is asked again in smaller pages."""
    while True:
        try:
            return _call("mcp-tv-get-news", {"symbol": symbol, "limit": page, "offset": offset}).get("data", {}), page
        except TradingViewBadReplyError:
            if page <= NEWS_MIN_PAGE:
                raise
            page = max(NEWS_MIN_PAGE, page // 2)
            logger.warning("TradingView news reply for %s unreadable; retrying with %d headlines", symbol, page)


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    symbol = resolve_symbol(ticker)
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
    kept, offset, oldest, page = [], 0, None, NEWS_PAGE
    while offset < NEWS_MAX:
        data, page = _news_page(symbol, offset, page)
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
