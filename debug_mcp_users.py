#!/usr/bin/env python3
"""Safe diagnostics for private-users MCP listPrivateUsers contract.

Run with:
  TELNYX_PROD_API_KEY='...' .venv/bin/python debug_mcp_users.py

Prints schemas and redacted response shapes only; does not dump emails/user rows.
"""
from __future__ import annotations

import json
import os
from typing import Any

from services.mcp_client import MCPClient
from services.account_scanner import MCP_ENDPOINT, AccountScanner


def summarize(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k.lower() in {"email", "name", "business_name", "first_name", "last_name"}:
                out[k] = "<redacted>"
            elif k.lower() in {"id", "user_id"} and isinstance(v, str):
                out[k] = f"<id:{len(v)}>"
            else:
                out[k] = summarize(v, depth + 1)
        return out
    if isinstance(value, list):
        return {
            "__type__": "list",
            "len": len(value),
            "first": summarize(value[0], depth + 1) if value else None,
        }
    if isinstance(value, str):
        if len(value) > 300:
            return value[:300] + "...<truncated>"
        return value
    return value


def main() -> None:
    token = os.environ.get("TELNYX_PROD_API_KEY") or os.environ.get("CORE_API_KEY")
    if not token:
        raise SystemExit("Set TELNYX_PROD_API_KEY or CORE_API_KEY in env")

    client = MCPClient(endpoint=MCP_ENDPOINT, auth_token=token, timeout=30.0)
    try:
        client.initialize()
        tools = client.list_tools()
        print(f"TOOLS count={len(tools)}")
        for tool in tools:
            name = tool.get("name")
            if name and ("user" in name.lower() or name == "listPrivateUsers"):
                print("\nTOOL", name)
                print("description:", (tool.get("description") or "")[:700])
                schema = tool.get("inputSchema") or tool.get("schema") or tool.get("parameters")
                print("schema:", json.dumps(schema, indent=2)[:2000])

        variants = [
            ("current", {"page_size": 100, "page_number": 1}),
            ("camel", {"pageSize": 100, "pageNumber": 1}),
            ("page/per_page", {"page": 1, "per_page": 100}),
            ("page/page_size", {"page": 1, "page_size": 100}),
            ("limit/offset", {"limit": 100, "offset": 0}),
            ("limit/page", {"limit": 100, "page": 1}),
            ("empty args", {}),
        ]
        for label, args in variants:
            print(f"\nCALL {label} args={args}")
            try:
                result = client.call_tool("listPrivateUsers", args)
                print(json.dumps(summarize(result), indent=2)[:4000])
            except Exception as exc:
                print("ERROR", type(exc).__name__, str(exc)[:1000])
    finally:
        client.close()


if __name__ == "__main__":
    main()
