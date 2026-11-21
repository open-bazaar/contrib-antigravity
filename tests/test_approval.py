"""Unit tests for Antigravity ACP permission flags, routing, and detail normalization."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from bazaar_contrib_audit import AUDIT, AuditContext, audit
from bazaar_contrib_harness import (
    ApprovalRequest,
    ApprovalResult,
    ApprovalScope,
    HarnessFailure,
    HarnessFailureReason,
    HarnessInput,
    HarnessSession,
    HarnessTurn,
    SandboxPolicy,
    SessionInput,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
)
from bazaar_contrib_timer import TIMER
from bazaar_core import (
    Composition,
    Context,
    Contrib,
    ContribEntry,
    ContribHub,
    ContribSpec,
    load,
)

from bazaar_contrib_antigravity.events import Usage
from bazaar_contrib_antigravity.runtime import AntigravityHarness, Connection, Turn
from bazaar_contrib_antigravity.source import Source
from bazaar_contrib_antigravity.transport import AcpProcess, Packet


class RecordingApproval:
    def __init__(
        self,
        decide: Callable[[ApprovalRequest], ApprovalResult] | None = None,
        *,
        on_call: Callable[[ApprovalRequest], None] | None = None,
    ) -> None:
        self.requests: list[ApprovalRequest] = []
        self._decide = decide
        self._on_call = on_call

    async def request_approval(
        self,
        request: ApprovalRequest,
        *,
        timeout: float,  # noqa: ASYNC109
    ) -> ApprovalResult:
        del timeout
        self.requests.append(request)
        if self._on_call is not None:
            self._on_call(request)
        if self._decide is not None:
            return self._decide(request)
        return ApprovalResult(
            request_id=request.request_id,
            decision="approved",
            scope=ApprovalScope.ONCE,
        )


def _permission_packet(
    *,
    packet_id: str = "jetski-acp-front:1",
    session_id: str = "provider-session-1",
    tool_call: dict[str, Any] | None = None,
    options: list[dict[str, str]] | None = None,
) -> Packet:
    if tool_call is None:
        tool_call = {
            "sessionUpdate": "tool_call",
            "toolCallId": "call_123",
            "name": "run_command",
            "kind": "execute",
            "title": "echo hi",
            "rawInput": {
                "CommandLine": "echo hi",
                "Cwd": "/tmp/work",
                "toolSummary": "Print greeting",
            },
        }
    if options is None:
        options = [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "always", "name": "Always", "kind": "allow_always"},
            {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
        ]
    return Packet(
        jsonrpc="2.0",
        id=packet_id,
        method="session/request_permission",
        params={
            "sessionId": session_id,
            "toolCall": tool_call,
            "options": options,
        },
    )


def _make_connection(
    tmp_path: Path,
    *,
    sandbox_policy: SandboxPolicy = SandboxPolicy.WORKSPACE_WRITE,
    provider_thread_id: str = "provider-session-1",
) -> tuple[Connection, Any]:
    request = SessionInput(
        id="session-1",
        workspace=tmp_path,
        sandbox_policy=sandbox_policy,
    )
    session = HarnessSession(id="session-1", provider_thread_id=provider_thread_id)
    connection = Connection(request, session, Usage())

    class _DummySource:
        context = AuditContext(agent_id="agent-1", harness="antigravity")

    class _DummyProcess:
        def __init__(self) -> None:
            self.running = True
            self.incoming: dict[int | str, asyncio.Task[None]] = {}
            self.source = _DummySource()

        async def has_background_job(self) -> bool:
            return False

    dummy = _DummyProcess()
    connection.process = cast(AcpProcess, dummy)
    return connection, dummy


def _attach_turn(
    connection: Connection,
    approval: RecordingApproval,
    *,
    turn_id: str = "turn-1",
) -> Turn:
    turn = Turn(
        connection,
        HarnessTurn(id=turn_id, session_id=connection.session.id, input_text="hi"),
        approval,
        timeout=30.0,
    )
    connection.turn = turn
    return turn


@pytest.mark.parametrize(
    ("policy", "expected_flag", "unexpected_flag"),
    [
        (
            SandboxPolicy.WORKSPACE_WRITE,
            "--sandbox",
            "--dangerously-skip-permissions",
        ),
        (
            SandboxPolicy.DANGER_FULL_ACCESS,
            "--dangerously-skip-permissions",
            "--sandbox",
        ),
    ],
)
async def test_open_passes_sandbox_and_permission_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: SandboxPolicy,
    expected_flag: str,
    unexpected_flag: str,
) -> None:
    captured_args: list[str] = []

    class _FakeTransport:
        def __init__(self) -> None:
            self.watcher = asyncio.create_task(asyncio.sleep(0))
            self.running = True
            self.incoming: dict[int | str, asyncio.Task[None]] = {}

        async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            del params
            if method == "initialize":
                return {"protocolVersion": 1, "agentCapabilities": {}}
            if method == "session/new":
                return {
                    "sessionId": "provider-1",
                    "configOptions": [{"id": "model", "currentValue": "gemini-test"}],
                }
            raise AssertionError(method)

        async def aclose(self) -> None:
            await self.watcher

    async def fake_start(
        cls: type[AcpProcess],
        command: list[str],
        source: Source,
        receive: Any,
        answer: Any,
        disconnected: Any,
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> _FakeTransport:
        del cls, source, receive, answer, disconnected, cwd, env
        captured_args.extend(command)
        return _FakeTransport()

    monkeypatch.setattr(AcpProcess, "start", classmethod(fake_start))

    timer_spec = await asyncio.to_thread(load, (ContribSpec(contrib="timer-wheel"),))

    @asynccontextmanager
    async def exercise(ctx: Context):
        harness = AntigravityHarness(
            HarnessInput(
                agent_id="agent-1",
                agent_name="Agent",
                workspace=tmp_path,
                startup_timeout=5,
                shutdown_timeout=5,
            ),
            ctx.accept(TIMER),
            ctx.accept(AUDIT),
            model=None,
            effort=None,
        )
        connection = await harness.open(
            SessionInput(id="session-1", workspace=tmp_path, sandbox_policy=policy),
            HarnessSession(id="session-1"),
            Usage(),
            timeout=5,
        )
        await connection.process.aclose()
        yield

    async with (
        ContribHub() as hub,
        hub.enable(
            Composition(
                entries=(
                    ContribEntry(name="audit", definition=audit),
                    *timer_spec.entries,
                    ContribEntry(
                        name="exercise",
                        definition=Contrib(accepts=[AUDIT, TIMER], lifespan=exercise),
                    ),
                ),
                diagnostics=timer_spec.diagnostics,
            )
        ),
    ):
        pass

    assert expected_flag in captured_args
    assert unexpected_flag not in captured_args


async def test_answer_allows_provider_wakeup_after_completed_turn(
    tmp_path: Path,
) -> None:
    connection, _ = _make_connection(tmp_path)
    approval = RecordingApproval(
        lambda req: ApprovalResult(
            request_id=req.request_id,
            decision="approved",
            scope=ApprovalScope.ALWAYS,
        )
    )
    turn = _attach_turn(connection, approval)
    turn.finish(TurnCompleted(event_name="session/prompt"))
    assert turn.finished and turn.completed

    response = await connection.answer(_permission_packet())
    assert response == {"outcome": {"outcome": "selected", "optionId": "always"}}
    assert len(approval.requests) == 1
    req = approval.requests[0]
    assert req.action == "command_execution"
    assert req.details["command"] == "echo hi"
    assert req.details["cwd"] == "/tmp/work"
    assert req.details["reason"] == "Print greeting"
    assert cast(dict[str, Any], req.details["toolCall"])["toolCallId"] == "call_123"


async def test_answer_cancels_when_turn_failed_cancelled_or_absent(
    tmp_path: Path,
) -> None:
    connection, _ = _make_connection(tmp_path)
    assert await connection.answer(_permission_packet()) == {
        "outcome": {"outcome": "cancelled"}
    }

    approval = RecordingApproval()
    turn = _attach_turn(connection, approval)
    turn.finish(TurnCancelled(event_name="session/prompt"))
    assert await connection.answer(_permission_packet()) == {
        "outcome": {"outcome": "cancelled"}
    }
    assert not approval.requests

    turn2 = _attach_turn(connection, approval, turn_id="turn-2")
    turn2.finish(
        TurnFailed(
            event_name="session/prompt",
            failure=HarnessFailure(HarnessFailureReason.EXECUTION, "boom"),
        )
    )
    assert await connection.answer(_permission_packet()) == {
        "outcome": {"outcome": "cancelled"}
    }
    assert not approval.requests

    turn3 = _attach_turn(connection, approval, turn_id="turn-3")
    turn3.cancel_requested = True
    assert await connection.answer(_permission_packet()) == {
        "outcome": {"outcome": "cancelled"}
    }
    assert not approval.requests

    turn4 = _attach_turn(connection, approval, turn_id="turn-4")
    assert await connection.answer(_permission_packet(session_id="other")) == {
        "outcome": {"outcome": "cancelled"}
    }
    assert not approval.requests
    assert not turn4.finished


async def test_answer_cancels_if_active_turn_finishes_during_approval(
    tmp_path: Path,
) -> None:
    connection, _ = _make_connection(tmp_path)
    turn_holder: list[Turn] = []

    def finish_mid_approval(_req: ApprovalRequest) -> None:
        turn_holder[0].finish(TurnCancelled(event_name="session/prompt"))

    approval = RecordingApproval(on_call=finish_mid_approval)
    turn_holder.append(_attach_turn(connection, approval))

    response = await connection.answer(_permission_packet())
    assert response == {"outcome": {"outcome": "cancelled"}}


async def test_answer_normalizes_mcp_and_edit_details_and_handles_missing_keys(
    tmp_path: Path,
) -> None:
    connection, _ = _make_connection(tmp_path)
    approval = RecordingApproval()
    _attach_turn(connection, approval)

    mcp_packet = _permission_packet(
        packet_id="req-mcp",
        tool_call={
            "toolCallId": "call_mcp_1",
            "name": "call_mcp_tool",
            "kind": "other",
            "title": "Query test MCP",
            "rawInput": {
                "ServerName": "test",
                "ToolName": "get_value",
                "Arguments": {"key": "once"},
            },
        },
    )
    assert await connection.answer(mcp_packet) == {
        "outcome": {"outcome": "selected", "optionId": "allow"}
    }
    mcp_req = approval.requests[-1]
    assert mcp_req.action == "mcp__test__get_value"
    assert mcp_req.details["tool_name"] == "mcp__test__get_value"
    assert mcp_req.details["tool_input"] == '{"key":"once"}'
    assert mcp_req.details["reason"] == "Query test MCP"
    assert (
        cast(dict[str, Any], mcp_req.details["toolCall"])["toolCallId"] == "call_mcp_1"
    )

    malformed_mcp = _permission_packet(
        packet_id="req-mcp-malformed",
        tool_call={
            "toolCallId": "call_mcp_2",
            "name": "call_mcp_tool",
            "kind": "other",
            "rawInput": {},
        },
    )
    assert await connection.answer(malformed_mcp) == {
        "outcome": {"outcome": "selected", "optionId": "allow"}
    }
    assert approval.requests[-1].action == "other"

    long_content = "x" * 5000
    edit_packet = _permission_packet(
        packet_id="req-edit",
        tool_call={
            "toolCallId": "call_edit_1",
            "name": "write_file",
            "kind": "edit",
            "title": "Write notes.txt",
            "args": {"Path": "/tmp/notes.txt", "Content": long_content},
        },
    )
    assert await connection.answer(edit_packet) == {
        "outcome": {"outcome": "selected", "optionId": "allow"}
    }
    edit_req = approval.requests[-1]
    assert edit_req.action == "file_change"
    assert edit_req.details["tool_name"] == "write_file"
    assert edit_req.details["reason"] == "Write notes.txt"
    tool_input = cast(str, edit_req.details["tool_input"])
    assert len(tool_input) == 4_000 and tool_input.endswith("…")


async def test_answer_catches_unexpected_approval_handler_exception(
    tmp_path: Path,
) -> None:
    connection, _ = _make_connection(tmp_path)

    def boom(_req: ApprovalRequest) -> ApprovalResult:
        raise LookupError("channel exploded")

    approval = RecordingApproval(decide=boom)
    _attach_turn(connection, approval)

    response = await connection.answer(_permission_packet())
    assert response == {"outcome": {"outcome": "cancelled"}}


async def test_has_background_job_true_while_incoming_permission_pending(
    tmp_path: Path,
) -> None:
    connection, dummy = _make_connection(tmp_path)
    harness = AntigravityHarness(
        HarnessInput(
            agent_id="agent-1",
            agent_name="Agent",
            workspace=tmp_path,
            startup_timeout=5,
            shutdown_timeout=5,
        ),
        cast(Any, None),
        cast(Any, None),
        model=None,
        effort=None,
    )
    harness.sessions[connection.session.id] = connection
    assert not await harness.has_background_job(connection.session, timeout=5)

    task = asyncio.create_task(asyncio.sleep(0))
    dummy.incoming["req-1"] = task
    try:
        assert await harness.has_background_job(connection.session, timeout=5)
    finally:
        dummy.incoming.clear()
        await task


async def test_transport_respond_replies_internal_error_on_unexpected_exception(
    tmp_path: Path,
) -> None:
    received: asyncio.Future[dict[str, Any]] = (
        asyncio.get_running_loop().create_future()
    )

    script = """
import json, sys
print(json.dumps({"jsonrpc": "2.0", "id": "srv-1", "method": "session/request_permission", "params": {}}), flush=True)
line = sys.stdin.readline()
if line:
    print(json.dumps({"jsonrpc": "2.0", "method": "echo", "params": {"reply": json.loads(line)}}), flush=True)
"""

    @asynccontextmanager
    async def exercise(ctx: Context):
        def receive(packet: Packet) -> None:
            if packet.method == "echo" and not received.done():
                received.set_result(packet.params["reply"])

        async def answer(_packet: Packet) -> dict[str, Any]:
            raise ZeroDivisionError("unexpected")

        transport = await AcpProcess.start(
            (sys.executable, "-c", script),
            Source(
                ctx.accept(AUDIT),
                AuditContext(agent_id="approval-test", harness="antigravity"),
            ),
            receive,
            answer,
            lambda _error: None,
            cwd=tmp_path,
            env=dict(os.environ),
        )
        try:
            reply = await asyncio.wait_for(received, timeout=5)
            assert reply["id"] == "srv-1"
            assert reply["error"]["code"] == -32603
            assert "unexpected" in reply["error"]["message"]
            yield
        finally:
            await transport.aclose()
            await transport.watcher

    async with (
        ContribHub() as hub,
        hub.enable(
            Composition(
                entries=(
                    ContribEntry(name="audit", definition=audit),
                    ContribEntry(
                        name="run",
                        definition=Contrib(accepts=[AUDIT], lifespan=exercise),
                    ),
                )
            )
        ),
    ):
        pass
