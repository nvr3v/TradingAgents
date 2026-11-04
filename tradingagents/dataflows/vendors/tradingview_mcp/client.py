"""Client for TradingView's hosted MCP server (https://mcp.tradingview.com/mcp).

The server uses the MCP authorization flow: OAuth 2.1 with dynamic client
registration and PKCE, authorized by the person in their own browser at
tradingview.com. ``sign_in`` runs that flow once with a loopback redirect on
127.0.0.1; the registered client and its tokens are kept in one file under
~/.tradingagents, readable only by its owner, and the SDK refreshes the access
token from it. A run never opens a browser: without a sign-in, calls raise
``TradingViewNotSignedIn`` so the router can fall back to the next vendor.

Requires the optional ``mcp`` package: pip install "tradingagents[tradingview]".
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
import urllib.parse
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from tradingagents.default_config import _TRADINGAGENTS_HOME

SERVER_URL = os.environ.get("TRADINGVIEW_MCP_URL") or "https://mcp.tradingview.com/mcp"
AUTH_FILE = Path(os.environ.get("TRADINGVIEW_MCP_AUTH_FILE")
                 or Path(_TRADINGAGENTS_HOME) / "tradingview-mcp-auth.json")
CALLBACK_PORT = int(os.environ.get("TRADINGVIEW_MCP_CALLBACK_PORT") or 8765)
CALLBACK_PATH = "/callback"
REDIRECT_HOST = os.environ.get("TRADINGVIEW_MCP_REDIRECT_HOST") or "[::1]"
SCOPE = "mcp:read mcp:tools"
SIGN_IN_TIMEOUT = 300
REFRESH_MARGIN = 60               # refresh this long before the access token lapses
CALL_TIMEOUT = 60
LOGIN_HINT = "Not signed in to TradingView; run `tradingagents login tradingview`."


class TradingViewMCPError(RuntimeError):
    """The TradingView MCP server could not serve a request."""


class TradingViewNotSignedIn(TradingViewMCPError):
    """No usable TradingView sign-in on this machine."""


def _require_mcp():
    try:
        import mcp  # noqa: F401
    except ImportError as exc:
        raise TradingViewMCPError(
            'The TradingView vendor needs the mcp package: pip install "tradingagents[tradingview]"'
        ) from exc


# --- the auth file ---------------------------------------------------------------------


def _read_auth() -> dict:
    try:
        return json.loads(AUTH_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_auth(data: dict) -> None:
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = AUTH_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh)
    os.replace(tmp, AUTH_FILE)


def _token_expiry() -> float | None:
    """When the stored access token lapses (epoch seconds), or None if unknown.

    A sign-in saved before ``expires_at`` was recorded is dated by the file.
    """
    data = _read_auth()
    if data.get("expires_at"):
        return float(data["expires_at"])
    lifetime = (data.get("tokens") or {}).get("expires_in")
    return AUTH_FILE.stat().st_mtime + lifetime if lifetime and AUTH_FILE.exists() else None


def is_signed_in() -> bool:
    return bool(_read_auth().get("tokens"))


class _FileTokenStorage:
    """``mcp.client.auth.TokenStorage`` backed by AUTH_FILE."""

    _lock = threading.Lock()

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        tokens = _read_auth().get("tokens")
        return OAuthToken.model_validate(tokens) if tokens else None

    async def set_tokens(self, tokens) -> None:
        with self._lock:
            data = _read_auth()
            data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            # The token carries only its lifetime; keep when it lapses, so a later
            # process knows to refresh instead of sending an expired token.
            data["expires_at"] = time.time() + tokens.expires_in if tokens.expires_in else None
            _write_auth(data)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        info = _read_auth().get("client_info")
        # A registration made for another redirect address cannot complete a
        # sign-in here; returning None makes the SDK register afresh.
        if not info or _redirect_uri(CALLBACK_PORT) not in info.get("redirect_uris", []):
            return None
        return OAuthClientInformationFull.model_validate(info)

    async def set_client_info(self, client_info) -> None:
        with self._lock:
            data = _read_auth()
            data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            _write_auth(data)


# --- interactive sign-in ---------------------------------------------------------------


class _CallbackServer:
    """One-shot loopback HTTP server that receives the authorization redirect."""

    def __init__(self, port: int):
        self.result: dict[str, str] = {}
        self.done = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (http.server API)
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path != CALLBACK_PATH:
                    self.send_error(404)
                    return
                params = dict(urllib.parse.parse_qsl(parsed.query))
                outer.result = params
                ok = "code" in params
                body = ("Signed in to TradingView. You can close this tab and return to TradingAgents."
                        if ok else f"TradingView sign-in failed: {params.get('error', 'no code returned')}")
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode())
                outer.done.set()

            def log_message(self, *args):
                pass

        # Listen on both loopback families; the redirect uses one of them.
        class Server6(HTTPServer):
            address_family = socket.AF_INET6

        self.servers = [HTTPServer(("127.0.0.1", port), Handler)]
        with contextlib.suppress(OSError):  # no IPv6 loopback on this machine
            self.servers.append(Server6(("::1", port), Handler))
        self.redirect_uri = _redirect_uri(port)
        for httpd in self.servers:
            threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def close(self):
        for httpd in self.servers:
            httpd.shutdown()
            httpd.server_close()


def _redirect_uri(port: int) -> str:
    """The loopback redirect, on the IPv6 literal [::1] (RFC 8252 section 7.3).

    TradingView's CloudFront rejects its sign-in page with 403 when the pending
    authorize URL redirects to ``localhost`` or ``127.0.0.1`` (a rule against
    local addresses), so a person not yet signed in could never get past it.
    ``[::1]`` passes that rule and every browser routes it to this machine.
    """
    return f"http://{REDIRECT_HOST}:{port}{CALLBACK_PATH}"


_metadata_cache: dict = {}


async def _load_metadata(context) -> None:
    """Fill the SDK context with the server's resource and authorization-server metadata."""
    import httpx2
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

    if not _metadata_cache:
        origin = urllib.parse.urlsplit(SERVER_URL)
        prm_url = f"{origin.scheme}://{origin.netloc}/.well-known/oauth-protected-resource{origin.path}"
        async with httpx2.AsyncClient(timeout=CALL_TIMEOUT, headers={"User-Agent": _user_agent()}) as http:
            prm = ProtectedResourceMetadata.model_validate_json((await http.get(prm_url)).raise_for_status().content)
            issuer = str(prm.authorization_servers[0]).rstrip("/")
            asm = OAuthMetadata.model_validate_json(
                (await http.get(f"{issuer}/.well-known/oauth-authorization-server")).raise_for_status().content)
        _metadata_cache.update(prm=prm, issuer=issuer, asm=asm)
    context.protected_resource_metadata = _metadata_cache["prm"]
    context.auth_server_url = _metadata_cache["issuer"]
    context.oauth_metadata = _metadata_cache["asm"]


def _client_metadata(redirect_uri: str):
    from mcp.shared.auth import OAuthClientMetadata
    return OAuthClientMetadata(
        client_name="TradingAgents",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        scope=SCOPE,
    )


def _provider(interactive: bool, say: Callable[[str], None] = print, open_browser: bool = True):
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import AuthorizationCodeResult

    callback: _CallbackServer | None = _CallbackServer(CALLBACK_PORT) if interactive else None
    redirect_uri = callback.redirect_uri if callback else _redirect_uri(CALLBACK_PORT)

    async def redirect_handler(url: str) -> None:
        if callback is None:
            raise TradingViewNotSignedIn(LOGIN_HINT)
        say("Open this address to sign in to TradingView and approve TradingAgents:")
        say(url)
        if open_browser:
            webbrowser.open(url)

    async def callback_handler() -> AuthorizationCodeResult:
        if callback is None:
            raise TradingViewNotSignedIn(LOGIN_HINT)
        got = await asyncio.to_thread(callback.done.wait, SIGN_IN_TIMEOUT)
        if not got:
            raise TradingViewMCPError("Timed out waiting for the TradingView sign-in.")
        if "code" not in callback.result:
            raise TradingViewMCPError(f"TradingView sign-in failed: {callback.result.get('error', 'no code')}")
        return AuthorizationCodeResult(code=callback.result["code"], state=callback.result.get("state"))

    class Provider(OAuthClientProvider):
        # The SDK loads stored tokens without their expiry and treats them as valid
        # forever; TradingView's last 15 minutes, and a 401 then sends the SDK into
        # a full browser sign-in instead of a refresh. Restore the expiry (with a
        # minute's margin) so an expired token is refreshed with the refresh token.
        async def _initialize(self) -> None:
            await super()._initialize()
            expiry = _token_expiry()
            if self.context.current_tokens and expiry:
                self.context.token_expiry_time = expiry - REFRESH_MARGIN
                # A refresh before any 401 needs the metadata the SDK only discovers
                # on a 401; without it the SDK guesses <server>/token, which 404s.
                if expiry - REFRESH_MARGIN <= time.time():
                    await _load_metadata(self.context)

    provider = Provider(
        server_url=SERVER_URL,
        client_metadata=_client_metadata(redirect_uri),
        storage=_FileTokenStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
    return provider, callback


def _user_agent() -> str:
    from tradingagents import __version__
    return f"TradingAgents/{__version__}"


def _http_client(auth):
    """An httpx client for the MCP session whose every request carries a User-Agent.

    The SDK builds its OAuth discovery and registration requests bare, without the
    client's default headers, and www.tradingview.com answers a request with no
    User-Agent with 403. Discovery then falls back to a wrong registration URL and
    sign-in fails, so the header is set at the transport, which every request passes.
    """
    import httpx2
    from mcp.client.streamable_http import create_mcp_http_client

    agent = _user_agent()

    class _UATransport(httpx2.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            request.headers.setdefault("User-Agent", agent)
            return await super().handle_async_request(request)

    template = create_mcp_http_client()  # the SDK's default timeouts
    return httpx2.AsyncClient(auth=auth, timeout=template.timeout, transport=_UATransport(),
                              headers={"User-Agent": agent})


async def _with_session(fn, *, interactive=False, say=print, open_browser=True):
    _require_mcp()
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    provider, callback = _provider(interactive, say, open_browser)
    try:
        async with (
            _http_client(provider) as http,
            streamable_http_client(SERVER_URL, http_client=http) as (read, write, *_),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return await fn(session)
    finally:
        if callback:
            callback.close()


def _unwrap(exc: BaseException) -> BaseException:
    """Return the first meaningful exception inside anyio/asyncio exception groups."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


_call_lock = threading.Lock()


def _run(coro_fn, **kwargs):
    # One request at a time per process: each refresh rotates the refresh token,
    # and two refreshing at once would leave one holding a token already spent.
    with _call_lock:
        return _run_unlocked(coro_fn, **kwargs)


def _run_unlocked(coro_fn, **kwargs):
    try:
        return asyncio.run(asyncio.wait_for(_with_session(coro_fn, **kwargs),
                                            SIGN_IN_TIMEOUT if kwargs.get("interactive") else CALL_TIMEOUT))
    except BaseException as exc:  # noqa: BLE001 — re-raised as a typed error below
        inner = _unwrap(exc)
        if isinstance(inner, TradingViewMCPError):
            raise inner from None
        if isinstance(inner, (KeyboardInterrupt, SystemExit)):
            raise inner from None
        raise TradingViewMCPError(f"TradingView MCP request failed: {type(inner).__name__}: {inner}") from inner


def _field(obj, *names):
    """The first present attribute: SDK 2.x names fields snake_case, 1.x camelCase."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


# --- public API ------------------------------------------------------------------------


def sign_in(say: Callable[[str], None] = print, open_browser: bool = True) -> list[str]:
    """Run the browser sign-in and return the names of the server's tools."""
    async def names(session):
        return [t.name for t in (await session.list_tools()).tools]
    return _run(names, interactive=True, say=say, open_browser=open_browser)


def sign_out() -> bool:
    """Forget this machine's TradingView sign-in. True when there was one."""
    if AUTH_FILE.exists():
        AUTH_FILE.unlink()
        return True
    return False


def list_tools() -> list[dict[str, Any]]:
    """The server's tools with their descriptions and input schemas."""
    if not is_signed_in():
        raise TradingViewNotSignedIn(LOGIN_HINT)

    async def tools(session):
        return [{"name": t.name, "description": t.description, "input_schema": _field(t, "input_schema", "inputSchema")}
                for t in (await session.list_tools()).tools]
    return _run(tools)


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Call one tool and return its text content (joined) or raise on a tool error."""
    if not is_signed_in():
        raise TradingViewNotSignedIn(LOGIN_HINT)

    async def call(session):
        return await session.call_tool(name, arguments)
    result = _run(call)
    text = "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "text", None))
    structured = _field(result, "structured_content", "structuredContent")
    if not text and structured:
        text = json.dumps(structured)
    if _field(result, "is_error", "isError"):
        raise TradingViewMCPError(f"TradingView tool {name} failed: {text or 'no detail'}")
    return text
