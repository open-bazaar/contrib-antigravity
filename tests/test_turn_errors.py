"""Tests for Antigravity ACP executor errors reported with stopReason=end_turn."""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from bazaar_contrib_audit import AUDIT, AuditContext, audit
from bazaar_contrib_harness import (
    ApprovalRequest,
    ApprovalResult,
    ContentDelta,
    HarnessFailureReason,
    HarnessInput,
    HarnessOutput,
    HarnessSession,
    HarnessTurn,
    SessionInput,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    UsageUpdated,
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
from bazaar_contrib_antigravity.transport import AcpProcess, Packet

QUOTA_MESSAGE = (
    "Individual quota reached. Please upgrade your subscription to increase "
    "your limits. Resets in 2h49m56s."
)


class _DummyApproval:
    async def request_approval(
        self,
        request: ApprovalRequest,
        *,
        timeout: float,  # noqa: ASYNC109
    ) -> ApprovalResult:
        return ApprovalResult(request_id=request.request_id, decision="rejected")


def _make_turn(
    tmp_path: Path,
    *,
    session_id: str = "provider-session-1",
    pending: set[int] | None = None,
) -> tuple[Connection, Turn]:
    request = SessionInput(id="session-1", workspace=tmp_path)
    session = HarnessSession(id="session-1", provider_thread_id=session_id)
    connection = Connection(request, session, Usage())

    class _DummySource:
        context = AuditContext(
            agent_id="agent-1",
            harness="antigravity",
            session_id="session-1",
            provider_session_id=session_id,
            turn_id="turn-1",
        )

    class _DummyProcess:
        def __init__(self) -> None:
            self.running = True
            self.incoming: dict[int | str, asyncio.Task[None]] = {}
            self.source = _DummySource()

        def observe_tool(self, _update: dict[str, Any]) -> None:
            pass

    connection.process = cast(AcpProcess, _DummyProcess())
    turn = Turn(
        connection,
        HarnessTurn(id="turn-1", session_id="session-1", input_text="hi"),
        _DummyApproval(),
        timeout=30.0,
    )
    turn.pending.update(pending if pending is not None else {1})
    connection.turn = turn
    return connection, turn


def _update_packet(
    update: dict[str, Any], *, session_id: str = "provider-session-1"
) -> Packet:
    return Packet(
        jsonrpc="2.0",
        method="session/update",
        params={"sessionId": session_id, "update": update},
    )


def _prompt_reply(
    request_id: int,
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, int] | None = None,
) -> Packet:
    result: dict[str, Any] = {"stopReason": stop_reason}
    if usage is not None:
        result["usage"] = usage
    return Packet(jsonrpc="2.0", id=request_id, result=result)


async def _collect(turn: Turn) -> list[HarnessOutput]:
    return [item async for item in turn]


async def test_quota_warning_with_end_turn_fails_turn(tmp_path: Path) -> None:
    connection, turn = _make_turn(tmp_path)
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection.receive(_prompt_reply(1))

    events = await _collect(turn)
    assert not turn.completed
    assert isinstance(events[-1].payload, TurnFailed)
    failure = events[-1].payload.failure
    assert failure.reason is HarnessFailureReason.EXECUTION
    assert failure.message == QUOTA_MESSAGE
    assert failure.needs_reconciliation is False


async def test_mid_turn_quota_warning_preserves_usage_and_fails_turn(
    tmp_path: Path,
) -> None:
    connection, turn = _make_turn(tmp_path)
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_1",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "echo hi"},
                "status": "in_progress",
            }
        )
    )
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_1",
                "status": "completed",
                "rawOutput": {"exit_code": 0, "output": "hi\n"},
            }
        )
    )
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection.receive(
        _prompt_reply(
            1,
            usage={
                "inputTokens": 1200,
                "outputTokens": 300,
                "totalTokens": 1500,
            },
        )
    )

    events = await _collect(turn)
    assert any(isinstance(item.payload, UsageUpdated) for item in events)
    assert isinstance(events[-1].payload, TurnFailed)
    assert events[-1].payload.failure.reason is HarnessFailureReason.EXECUTION
    assert events[-1].payload.failure.message == QUOTA_MESSAGE


async def test_config_option_update_does_not_mask_trailing_executor_error(
    tmp_path: Path,
) -> None:
    connection, turn = _make_turn(tmp_path)
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": "\n⚠ Eligibility check failed: unavailable in your location.\n",
                },
            }
        )
    )
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "config_option_update",
                "configOptions": [{"id": "model", "currentValue": "gemini-4-neon"}],
            }
        )
    )
    connection.receive(_prompt_reply(1))

    events = await _collect(turn)
    assert isinstance(events[-1].payload, TurnFailed)
    assert (
        events[-1].payload.failure.message
        == "Eligibility check failed: unavailable in your location."
    )


async def test_normal_assistant_warning_followed_by_usage_update_completes(
    tmp_path: Path,
) -> None:
    connection, turn = _make_turn(tmp_path)
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "usage_update",
                "used": 1000,
                "size": 1048576,
            }
        )
    )
    connection.receive(
        _prompt_reply(1, usage={"inputTokens": 800, "outputTokens": 200})
    )

    events = await _collect(turn)
    assert turn.completed
    assert isinstance(events[-1].payload, TurnCompleted)
    assert any(
        isinstance(item.payload, ContentDelta) and QUOTA_MESSAGE in item.payload.text
        for item in events
    )


async def test_steered_turn_recovers_or_fails_according_to_final_outcome(
    tmp_path: Path,
) -> None:
    # Steered prompt fails with 429 after initial prompt settles with end_turn.
    connection, turn = _make_turn(tmp_path, pending={1, 2})
    connection.receive(_prompt_reply(1))
    assert not turn.finished
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection.receive(_prompt_reply(2))
    events = await _collect(turn)
    assert isinstance(events[-1].payload, TurnFailed)
    assert events[-1].payload.failure.message == QUOTA_MESSAGE

    # Steered prompt succeeds after an earlier warning chunk.
    connection2, turn2 = _make_turn(tmp_path, pending={1, 2})
    connection2.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection2.receive(_prompt_reply(1))
    connection2.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Recovered cleanly."},
            }
        )
    )
    connection2.receive(
        _update_packet({"sessionUpdate": "usage_update", "used": 200, "size": 1048576})
    )
    connection2.receive(_prompt_reply(2))
    events2 = await _collect(turn2)
    assert turn2.completed
    assert isinstance(events2[-1].payload, TurnCompleted)


async def test_cancelled_stop_reason_remains_cancelled_after_warning(
    tmp_path: Path,
) -> None:
    connection, turn = _make_turn(tmp_path)
    connection.receive(
        _update_packet(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"\n⚠ {QUOTA_MESSAGE}\n"},
            }
        )
    )
    connection.receive(_prompt_reply(1, stop_reason="cancelled"))

    events = await _collect(turn)
    assert isinstance(events[-1].payload, TurnCancelled)


async def test_acp_session_reports_quota_failure_and_recovers_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = f"""
import json, sys

turns = 0
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    req_id = req.get("id")
    if method == "initialize":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"protocolVersion": 1, "agentCapabilities": {{}}}}}}), flush=True)
    elif method == "session/new":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"sessionId": "acp-sess-1", "configOptions": [{{"id": "model", "currentValue": "gemini-4-neon"}}]}}}}), flush=True)
    elif method == "session/prompt":
        turns += 1
        if turns == 1:
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "acp-sess-1", "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "\\n⚠ {QUOTA_MESSAGE}\\n"}}}}}}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn"}}}}), flush=True)
        else:
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "acp-sess-1", "update": {{"sessionUpdate": "agent_message_chunk", "content": {{"type": "text", "text": "Back online"}}}}}}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "method": "session/update", "params": {{"sessionId": "acp-sess-1", "update": {{"sessionUpdate": "usage_update", "used": 100, "size": 1048576}}}}}}), flush=True)
            print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{"stopReason": "end_turn", "usage": {{"inputTokens": 80, "outputTokens": 20, "totalTokens": 100}}}}}}), flush=True)
    elif method == "session/close":
        print(json.dumps({{"jsonrpc": "2.0", "id": req_id, "result": {{}}}}), flush=True)
        break
"""

    real_start = AcpProcess.start

    async def start_fake_acp(
        cls: type[AcpProcess],
        command: list[str],
        source: Any,
        receive: Any,
        answer: Any,
        disconnected: Any,
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> AcpProcess:
        del cls, command
        return await real_start(
            (sys.executable, "-c", script),
            source,
            receive,
            answer,
            disconnected,
            cwd=cwd,
            env=dict(os.environ) | env,
        )

    monkeypatch.setattr(AcpProcess, "start", classmethod(start_fake_acp))
    timer_spec = await asyncio.to_thread(load, (ContribSpec(contrib="timer-wheel"),))

    @asynccontextmanager
    async def exercise(ctx: Context):
        harness = AntigravityHarness(
            HarnessInput(
                agent_id="agent-1",
                agent_name="Agent",
                workspace=tmp_path,
                startup_timeout=10,
                shutdown_timeout=5,
            ),
            ctx.accept(TIMER),
            ctx.accept(AUDIT),
            model=None,
            effort=None,
        )
        started = await harness.start_session(
            SessionInput(id="session-1", workspace=tmp_path), timeout=10
        )
        assert started.value is not None
        session = started.value
        try:
            stream1 = await harness.start_turn(
                session,
                HarnessTurn(id="turn-1", session_id=session.id, input_text="hello"),
                _DummyApproval(),
                timeout=10,
            )
            events1 = [item async for item in stream1]
            await stream1.aclose()
            assert isinstance(events1[-1].payload, TurnFailed)
            assert events1[-1].payload.failure.reason is HarnessFailureReason.EXECUTION
            assert events1[-1].payload.failure.message == QUOTA_MESSAGE

            stream2 = await harness.start_turn(
                session,
                HarnessTurn(
                    id="turn-2", session_id=session.id, input_text="hello again"
                ),
                _DummyApproval(),
                timeout=10,
            )
            events2 = [item async for item in stream2]
            await stream2.aclose()
            assert isinstance(events2[-1].payload, TurnCompleted)
            yield
        finally:
            await harness.stop_session(session, timeout=5)
            await harness.aclose()

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
