"""Minimal pearld JSON-RPC client. Same API on regtest, testnet, and mainnet —
caller picks the URL.

Auth is sent as an HTTP Basic header derived from the user:pass in the URL.
"""

from __future__ import annotations

import json
import urllib.request
from base64 import b64encode
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def public_rpc_url(url: str) -> str:
    """Return an RPC URL safe to print, without credentials, query, or fragment."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parsed.port is not None:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid RPC URL>"


class PearldRpc:
    def __init__(self, url: str, timeout: float = 30.0):
        parsed = urlsplit(url)
        userinfo = (parsed.username or "") + ":" + (parsed.password or "")
        self.auth_header = "Basic " + b64encode(userinfo.encode()).decode()
        self.url = public_rpc_url(url)
        if not parsed.path:
            self.url += "/"
        self.timeout = timeout

    def call(self, method: str, params: list[Any]) -> Any:
        # jsonrpc "2.0" required by pearld v1.0.6+ — it accepts the TCP
        # connection on "1.0" requests but returns no body (curl sees
        # "Empty reply from server"; client logs Connection: close).
        body = json.dumps({"jsonrpc": "2.0", "id": "1", "method": method, "params": params}).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": self.auth_header},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        if data.get("error"):
            raise RuntimeError(f"RPC error: {data['error']}")
        return data["result"]
