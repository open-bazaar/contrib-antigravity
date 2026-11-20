"""Read available models and MCP tools without opening a conversation."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from pathlib import Path

from bazaar_contrib_harness import (
    HarnessInspection,
    HarnessMcpServer,
    HarnessMcpTool,
    HarnessModel,
    McpConnection,
)


async def cli(workspace: Path, *args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        "agy",
        *args,
        cwd=workspace,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
    if process.returncode:
        raise OSError(
            stderr.decode(errors="replace").strip()
            or f"agy exited {process.returncode}"
        )
    return stdout.decode().strip()


def mcp_servers(listing: str) -> tuple[HarnessMcpServer, ...]:
    configured: dict[str, str] = {}
    for line in listing.splitlines():
        columns = re.split(r"\s{2,}", line.strip(), maxsplit=3)
        if len(columns) == 4 and columns[2] in {"enabled", "disabled"}:
            configured[columns[0]] = columns[2]

    tools: dict[str, list[HarnessMcpTool]] = {
        name: [] for name, status in configured.items() if status == "enabled"
    }
    cache = Path.home() / ".gemini" / "antigravity-cli" / "mcp"
    for path in sorted(cache.glob("*/*.json")):
        if (server := tools.get(path.parent.name)) is not None:
            schema = json.loads(path.read_bytes())
            server.append(
                HarnessMcpTool(
                    name=schema["name"],
                    description=schema.get("description"),
                    input_schema=schema.get("parameters"),
                )
            )
    return tuple(
        HarnessMcpServer(
            name=name,
            connection=McpConnection.DISABLED if status == "disabled" else None,
            tools=tuple(tools.get(name, ())),
        )
        for name, status in configured.items()
    )


async def inspect(workspace: Path, *, timeout: float) -> HarnessInspection:  # noqa: ASYNC109
    version = None
    models: dict[str, str] = {}
    efforts: dict[str, list[str]] = {}
    try:
        async with asyncio.timeout(timeout):
            version = await cli(workspace, "--version")
            listing = await cli(workspace, "models")
            configured = await cli(workspace, "mcp", "list")
            servers = await asyncio.to_thread(mcp_servers, configured)
        for line in listing.splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                continue
            identifier, name = fields
            base, _, effort = identifier.rpartition("-")
            if effort in ("low", "medium", "high", "xhigh", "max"):
                identifier = base
                name = name.partition(" (")[0]
                efforts.setdefault(identifier, []).append(effort)
            models[identifier] = name
    except (TimeoutError, OSError, json.JSONDecodeError) as error:
        return HarnessInspection(
            available=version is not None,
            version=version,
            error=str(error) or type(error).__name__,
        )
    return HarnessInspection(
        available=True,
        version=version,
        models=tuple(
            HarnessModel(
                id=identifier, name=name, efforts=tuple(efforts.get(identifier, ()))
            )
            for identifier, name in models.items()
        ),
        mcp_servers=servers,
    )
