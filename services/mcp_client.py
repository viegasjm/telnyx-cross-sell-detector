#!/usr/bin/env python3
"""Lightweight MCP streamable-http client — bypasses mcporter subprocess overhead.

Connects once, initializes, then makes multiple tool calls over the same session.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class MCPClient:
    """Minimal MCP streamable-http client."""

    def __init__(
        self,
        endpoint: str,
        auth_token: str,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._request_id = 0
        self._client = httpx.Client(timeout=timeout)

    # -- Session lifecycle ----------------------------------------------------

    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP session. Must be called before any tool calls."""
        resp = self._send_request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "cross-sell-detector", "version": "0.1.0"},
            },
        )
        result = resp.get("result", {})
        logger.info("MCP session initialized: %s", result.get("serverInfo", {}))

        # Send initialized notification (required by MCP spec)
        self._send_notification("notifications/initialized", {})
        return result

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    # -- Tool calls -----------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools on the MCP server."""
        resp = self._send_request("tools/list", {})
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Any:
        """Call a tool on the MCP server and return the parsed result."""
        args = arguments or {}
        resp = self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": args,
            },
        )
        result = resp.get("result", {})
        content = result.get("content", [])

        # MCP returns content as a list of items. Extract the first text payload
        # and normalize it. The private-users server has returned all of these
        # shapes in practice:
        #   API Response (Status: 200):\n{"data": [...], "meta": {...}}
        #   API Response (Status: 200):\n"{\"data\": [...]}"   # double encoded
        #   ```json\n{"data": [...]}\n```
        if isinstance(content, list) and len(content) >= 1:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return self._parse_tool_text(item.get("text", ""))

        return content

    @staticmethod
    def _parse_tool_text(text: str) -> Any:
        """Parse a private-users MCP text payload into JSON when possible."""
        if not isinstance(text, str):
            return text

        # Strip the "API Response (Status: NNN):\n" prefix that the
        # private-users MCP server prepends to responses.
        prefix = "API Response (Status: "
        if text.startswith(prefix):
            nl = text.find("\n")
            if nl >= 0:
                text = text[nl + 1:]

        text = text.strip()

        # Maybe the JSON is wrapped in a markdown code block.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Parse repeatedly to handle double-encoded JSON strings.
        parsed: Any = text
        for _ in range(3):
            if not isinstance(parsed, str):
                return parsed
            candidate = parsed.strip()
            if not candidate:
                return candidate
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # Last resort: extract the first object/array embedded in text.
                starts = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0]
                if starts:
                    start = min(starts)
                    end_obj = candidate.rfind("}")
                    end_arr = candidate.rfind("]")
                    end = max(end_obj, end_arr)
                    if end > start:
                        try:
                            parsed = json.loads(candidate[start : end + 1])
                            continue
                        except (json.JSONDecodeError, ValueError):
                            pass
                return parsed
        return parsed

    # -- Transport layer ------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        request_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.auth_token}",
        }

        # If we have a session ID, include it
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        resp = self._client.post(
            self.endpoint,
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()

        # Capture session ID from response
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

        # Parse response — could be JSON or SSE
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse_response(resp.text)
        else:
            return resp.json()

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        resp = self._client.post(
            self.endpoint,
            json=payload,
            headers=headers,
        )
        # Notifications don't expect a response, but we update session ID
        sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
        if sid:
            self.session_id = sid

    @staticmethod
    def _parse_sse_response(text: str) -> Dict[str, Any]:
        """Parse Server-Sent Events response from MCP server."""
        for line in text.split("\n"):
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    return json.loads(data_str)
                except json.JSONDecodeError:
                    continue
        return {}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import subprocess

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Get auth token from keychain
    try:
        token = subprocess.check_output(
            ["security", "find-generic-password", "-s", "TELNYX_PROD_API_KEY", "-w"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print("ERROR: Could not get TELNYX_PROD_API_KEY from keychain")
        raise SystemExit(1)

    client = MCPClient(
        endpoint="https://private-users-mcp.query.prod.telnyx.io:3000/mcp",
        auth_token=token,
    )

    try:
        client.initialize()

        # Test: list 2 users
        result = client.call_tool("listPrivateUsers", {"page_size": 2, "page_number": 1})
        users = result.get("data", [])
        print(f"Got {len(users)} users")

        if users:
            user_id = users[0].get("id")
            # Test: list connections for first user
            conns = client.call_tool(
                "listPrivateConnections",
                {"filter_user_id": user_id, "page_size": 10},
            )
            conn_data = conns.get("data", []) if isinstance(conns, dict) else conns
            print(f"User {users[0].get('email')}: {len(conn_data) if isinstance(conn_data, list) else '?'} connections")

    finally:
        client.close()
