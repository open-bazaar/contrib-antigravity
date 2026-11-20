"""Antigravity specifics for the shared Harness acceptance suite."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import psutil
from bazaar_contrib_harness import CommandDetail, ToolCallCompleted, ToolCallFailed
from bazaar_core import ContribEntry
from watchfiles import awatch

from bazaar_contrib_antigravity import harness
from bazaar_contrib_antigravity.events import current_option

KIND = "antigravity"
PACKAGE = "bazaar-contrib-antigravity"
BUILDER = "antigravity-builder"
HOME = "HOME"
HOME_DIRECTORY = "antigravity-home"
API_KEY = "GEMINI_API_KEY"
PRIVATE = (API_KEY, "GOOGLE_GEMINI_BASE_URL", "AGY_CLI_DISABLE_AUTO_UPDATE")
SCENARIOS = ("mcp-once", "bcc", "full-access", "recovery", "refresh", "background")
FULL_ACCESS = frozenset({"full-access", "background"})
INSTRUCTION_FILES = ("AGENTS.md", "GEMINI.md")
MCP_TOOL = "call_mcp_tool"
MCP_APPROVAL = "mcp__test__get_value"
COMMAND_APPROVAL = "command_execution"
BACKGROUND = (
    "with the Bash tool's `run_in_background` option, without a trailing `&`. "
    "Keep that background task running after this turn."
)
HARNESS_ENTRY = ContribEntry(name="harness", definition=harness)
EXIT_CLOSES_SOURCE = True
STEERING_INPUT = "Reply with one short sentence."


async def discover_fixture_tools(home: Path, environment: dict[str, str]) -> None:
    data = home / ".gemini" / "antigravity-cli"
    schema = data / "mcp" / "test" / "get_value.json"

    async def discover() -> None:
        if schema.is_file():
            return
        async for _ in awatch(data):
            if schema.is_file():
                return

    # ACP returns session/new before connecting to the configured MCP server.
    async with asyncio.TaskGroup() as group:
        discovery = group.create_task(discover())
        process = await asyncio.create_subprocess_exec(
            "agy",
            "acp",
            "--process-per-session",
            cwd=home,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None
        writer, reader = process.stdin, process.stdout

        async def request(
            identifier: int, method: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            packet = {
                "jsonrpc": "2.0",
                "id": identifier,
                "method": method,
                "params": params,
            }
            writer.write(json.dumps(packet).encode() + b"\n")
            await writer.drain()
            async for line in reader:
                response = json.loads(line)
                if response.get("id") == identifier:
                    assert "result" in response, response
                    return response["result"]
            raise RuntimeError(f"agy closed before replying to {method}")

        try:
            await request(1, "initialize", {"protocolVersion": 1})
            session = await request(
                2, "session/new", {"cwd": str(home), "mcpServers": []}
            )
            await discovery
            await request(3, "session/close", {"sessionId": session["sessionId"]})
        finally:
            writer.close()
            await process.wait()


def configure(
    environment: dict[str, str], home: Path, python: Path, model: str
) -> None:
    environment.update(
        GEMINI_API_KEY=environment["AI_API_KEY"],
        GOOGLE_GEMINI_BASE_URL=environment["AI_GATEWAY_BASE_URL"],
        AGY_CLI_DISABLE_AUTO_UPDATE="true",
    )
    gemini = home / ".gemini"
    settings = gemini / "antigravity-cli" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        json.dumps(
            {
                "modelProvider": "gemini",
                "setupComplete": True,
                "onboardingComplete": True,
                "agentMode": "",
                "toolPermission": "request-review",
                "model": model + "-low",
            }
        )
    )
    config = gemini / "config" / "mcp_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "test": {
                        "command": str(python),
                        "args": ["-m", "bcn_test_channel.mcp"],
                    }
                }
            }
        )
    )
    asyncio.run(discover_fixture_tools(home, environment))


def bcc_command(call: dict[str, Any]) -> str | None:
    detail = call["detail"]
    return detail["command"] if detail["kind"] == "command" else None


def session_model(direction: str, packet: dict[str, Any]) -> str | None:
    if direction == "received":
        if result := packet.get("result"):
            return current_option(result, "model")
        if packet.get("method") == "session/update":
            return current_option(packet["params"]["update"], "model")
    return None


def verify_call(payload: ToolCallCompleted | ToolCallFailed) -> None:
    detail = payload.call.detail
    if isinstance(detail, CommandDetail):
        assert detail.command, payload
        if isinstance(payload, ToolCallCompleted):
            assert detail.output is not None, payload


def approval_call_id(approval: dict[str, Any]) -> str:
    return approval["details"]["toolCall"]["toolCallId"]


def steering_input(direction: str, packet: dict[str, Any]) -> str | None:
    if direction == "sent" and packet.get("method") == "session/prompt":
        return str(packet["id"])
    return None


def steering_settled(direction: str, packet: dict[str, Any]) -> str | None:
    if direction == "received" and "method" not in packet and "id" in packet:
        return str(packet["id"])
    return None


def steering_result(direction: str, packet: dict[str, Any]) -> dict[str, Any] | None:
    if direction == "received" and "method" not in packet:
        result = packet.get("result")
        if result and "stopReason" in result:
            return {"id": packet["id"], "stopReason": result["stopReason"]}
    return None


def session_processes(
    parent: psutil.Process, environment: dict[str, str]
) -> list[psutil.Process]:
    processes = []
    for child in parent.children():
        try:
            if (
                "--process-per-session" in child.cmdline()
                and child.environ().get(HOME) == environment[HOME]
            ):
                processes.append(child)
        except psutil.NoSuchProcess:
            continue
    return processes
