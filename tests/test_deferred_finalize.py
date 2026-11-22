"""Tests for deferred turn finalization and native local-state inspection."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
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
    HarnessSessionUnavailable,
    HarnessTurn,
    ProviderCallStatus,
    SessionInput,
    ToolCallCompleted,
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

from bazaar_contrib_antigravity import runtime as runtime_mod
from bazaar_contrib_antigravity.events import Usage
from bazaar_contrib_antigravity.runtime import AntigravityHarness, Connection, Turn
from bazaar_contrib_antigravity.state import (
    active_subagent_calls,
    classify_step_error,
    has_pending_reply,
    read_last_step_index,
    read_step_error,
)
from bazaar_contrib_antigravity.transport import AcpProcess, Disconnected, Packet


class _VirtualHandle:
    def __init__(self, timer: _VirtualTimer, deadline_ms: int) -> None:
        self._timer = timer
        self.deadline_ms = deadline_ms
        self._event = asyncio.Event()
        self.cancelled = False

    def fire(self) -> None:
        if not self.cancelled:
            self._event.set()

    def cancel(self) -> None:
        self.cancelled = True
        self._timer.handles.discard(self)

    async def wait(self) -> None:
        await self._event.wait()


class _VirtualTimer:
    tick_ms = 1
    maximum_delay_ms = 3_600_000

    def __init__(self, start_ms: int = 10_000) -> None:
        self.now_ms = start_ms
        self.handles: set[_VirtualHandle] = set()

    def create(self, delay_ms: int) -> _VirtualHandle:
        handle = _VirtualHandle(self, self.now_ms + max(0, delay_ms))
        self.handles.add(handle)
        if delay_ms <= 0:
            handle.fire()
        return handle

    async def advance(self, delta_ms: int) -> None:
        self.now_ms += delta_ms
        due = [h for h in tuple(self.handles) if h.deadline_ms <= self.now_ms]
        for handle in due:
            self.handles.discard(handle)
            handle.fire()
        await _yield_loop()


class _DummyApproval:
    async def request_approval(
        self,
        request: ApprovalRequest,
        *,
        timeout: float,  # noqa: ASYNC109
    ) -> ApprovalResult:
        return ApprovalResult(request_id=request.request_id, decision="approved")


class _DummySource:
    def __init__(self, session_id: str) -> None:
        self.context = AuditContext(
            agent_id="agent-1",
            harness="antigravity",
            session_id="session-1",
            provider_session_id=session_id,
            turn_id="turn-1",
        )


class _DummyProcess:
    def __init__(self, session_id: str) -> None:
        self.running = True
        self.incoming: dict[int | str, asyncio.Task[None]] = {}
        self.source = _DummySource(session_id)
        self.sent: list[dict[str, Any]] = []
        self._seq = 1

    def request_id(self) -> int:
        self._seq += 1
        return self._seq

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    def observe_tool(self, _update: dict[str, Any]) -> None:
        pass

    async def has_background_job(self) -> bool:
        return False


def _make_turn(
    tmp_path: Path,
    *,
    session_id: str = "provider-session-1",
    timer: _VirtualTimer | None = None,
    profile_dir: Path | None = None,
    start_step_index: int | None = None,
    pending: set[int] | None = None,
) -> tuple[Connection, Turn, _DummyProcess]:
    request = SessionInput(id="session-1", workspace=tmp_path)
    session = HarnessSession(id="session-1", provider_thread_id=session_id)
    connection = Connection(
        request,
        session,
        Usage(),
        timer=cast(Any, timer),
        profile_dir=profile_dir,
    )
    dummy = _DummyProcess(session_id)
    connection.process = cast(AcpProcess, dummy)
    turn = Turn(
        connection,
        HarnessTurn(id="turn-1", session_id="session-1", input_text="hi"),
        _DummyApproval(),
        timeout=30.0,
    )
    turn.start_step_index = start_step_index
    turn.pending.update(pending if pending is not None else {1})
    connection.turn = turn
    return connection, turn, dummy


def _update(
    update: dict[str, Any], *, session_id: str = "provider-session-1"
) -> Packet:
    return Packet(
        jsonrpc="2.0",
        method="session/update",
        params={"sessionId": session_id, "update": update},
    )


def _prompt_reply(request_id: int, *, stop_reason: str = "end_turn") -> Packet:
    return Packet(
        jsonrpc="2.0",
        id=request_id,
        result={"stopReason": stop_reason},
    )


async def _yield_loop(steps: int = 20) -> None:
    for _ in range(steps):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


async def _wait_finished(turn: Turn, *, timeout: float = 2.0) -> None:  # noqa: ASYNC109
    if not turn.finished and turn.settle_task is not None:
        await asyncio.wait_for(asyncio.shield(turn.settle_task), timeout=timeout)
    assert turn.finished


def _drain_sync(turn: Turn) -> list[HarnessOutput]:
    items: list[HarnessOutput] = []
    while not turn.output.empty():
        item = turn.output.get_nowait()
        if item is not None:
            items.append(item)
    return items


def _create_steps_db(db_path: Path, rows: list[tuple[int, int, bytes]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS steps "
            "(idx INTEGER PRIMARY KEY, step_type INTEGER, step_payload BLOB)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO steps (idx, step_type, step_payload) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


async def test_deferred_finalize_holds_open_for_background_execute_and_autonomous_reply(
    tmp_path: Path,
) -> None:
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(tmp_path, timer=timer)

    # Background run_command starts and prompt prematurely returns end_turn.
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "sleep 20"},
                "status": "in_progress",
            }
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn.finished

    # Still running after several timer ticks.
    await timer.advance(5_000)
    assert not turn.finished

    # Background command completes -> enters wake grace window.
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_bg",
                "status": "completed",
                "rawOutput": {"exit_code": 0, "output": "done\n"},
            }
        )
    )
    await _yield_loop()
    assert not turn.finished

    # Autonomous reply streams text and closes with usage_update.
    connection.receive(
        _update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Background command finished."},
            }
        )
    )
    await timer.advance(10_000)
    assert not turn.finished

    connection.receive(
        _update({"sessionUpdate": "usage_update", "used": 500, "size": 1_048_576})
    )
    await _yield_loop()
    assert not turn.finished

    # After quiet settle debounce, turn completes cleanly with all late events preserved.
    await timer.advance(runtime_mod._DEFERRED_QUIET_MS)
    await _wait_finished(turn)
    assert turn.completed
    events = _drain_sync(turn)
    assert any(isinstance(e.payload, ToolCallCompleted) for e in events)
    assert any(
        isinstance(e.payload, ContentDelta) and "Background command" in e.payload.text
        for e in events
    )
    assert any(isinstance(e.payload, UsageUpdated) for e in events)
    assert isinstance(events[-1].payload, TurnCompleted)


async def test_deferred_wakeup_quota_warning_fails_turn_while_prose_completes(
    tmp_path: Path,
) -> None:
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(tmp_path, timer=timer)
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "pytest"},
                "status": "in_progress",
            }
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn.finished

    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_bg",
                "status": "completed",
                "rawOutput": {"exit_code": 0, "output": "ok"},
            }
        )
    )
    connection.receive(
        _update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": "\n⚠ Individual quota reached. Resets in 3h.\n",
                },
            }
        )
    )
    await _yield_loop()
    assert not turn.finished

    # No usage_update arrives -> quiet window elapses and emits TurnFailed!
    await timer.advance(runtime_mod._DEFERRED_QUIET_MS)
    await _wait_finished(turn)
    assert not turn.completed
    events = _drain_sync(turn)
    assert isinstance(events[-1].payload, TurnFailed)
    assert events[-1].payload.failure.reason is HarnessFailureReason.EXECUTION
    assert "Individual quota reached" in events[-1].payload.failure.message

    # Contrast: if usage_update arrives right after a "⚠" chunk, it is normal prose.
    timer2 = _VirtualTimer()
    conn2, turn2, _ = _make_turn(tmp_path, timer=timer2)
    conn2.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg2",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "echo hi"},
                "status": "in_progress",
            }
        )
    )
    conn2.receive(_prompt_reply(1))
    await _yield_loop()
    conn2.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_bg2",
                "status": "completed",
            }
        )
    )
    conn2.receive(
        _update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "\n⚠ Just quoting a warning.\n"},
            }
        )
    )
    conn2.receive(
        _update({"sessionUpdate": "usage_update", "used": 100, "size": 1_048_576})
    )
    await timer2.advance(runtime_mod._DEFERRED_QUIET_MS)
    assert turn2.finished and turn2.completed


async def test_daemon_execute_and_daemon_schedule_do_not_defer_turn(
    tmp_path: Path,
) -> None:
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(tmp_path, timer=timer)
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_daemon",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "npm run dev", "IsDaemon": True},
                "status": "in_progress",
            }
        )
    )
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_cron",
                "name": "schedule",
                "kind": "other",
                "title": "Run schedule",
                "rawInput": {"CronExpression": "*/5 * * * *", "IsDaemon": True},
                "status": "in_progress",
            }
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    assert turn.finished and turn.completed

    # Non-daemon schedule tool DOES hold the turn open until it finishes.
    timer2 = _VirtualTimer()
    conn2, turn2, _ = _make_turn(tmp_path, timer=timer2)
    conn2.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_timer",
                "name": "schedule",
                "kind": "other",
                "title": "Running schedule",
                "rawInput": {"DurationSeconds": 10},
                "status": "in_progress",
            }
        )
    )
    conn2.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn2.finished
    conn2.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_timer",
                "status": "completed",
            }
        )
    )
    await timer2.advance(runtime_mod._POST_WORK_WAKE_GRACE_MS)
    assert turn2.finished and turn2.completed


async def test_steer_turn_during_deferred_wait_takes_over_settlement(
    tmp_path: Path,
) -> None:
    timer = _VirtualTimer()
    connection, turn, dummy = _make_turn(tmp_path, timer=timer)
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "sleep 30"},
                "status": "in_progress",
            }
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn.finished and turn.prompt_settled

    # User sends a follow-up while background command is still running -> steer_turn!
    await connection.submit(turn, [{"type": "text", "text": "actually stop and reply"}])
    await _yield_loop()
    assert not turn.prompt_settled
    assert 2 in turn.pending
    assert any(m.get("method") == "session/prompt" for m in dummy.sent)

    # Background command completes, then steered prompt returns end_turn.
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_bg",
                "status": "completed",
            }
        )
    )
    connection.receive(_prompt_reply(2))
    await _yield_loop()
    assert turn.finished and turn.completed


async def test_subagent_and_pending_reply_on_disk_hold_turn_and_background_job(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    root_sid = "parent-session-1"
    child_sid = "child-session-1"
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(
        tmp_path, session_id=root_sid, timer=timer, profile_dir=profile
    )

    # Create detached subagent descriptor + child launch inbox message.
    subagents_dir = profile / "brain" / root_sid / ".system_generated" / "subagents"
    parent_inbox = profile / "brain" / root_sid / ".system_generated" / "messages"
    child_inbox = profile / "brain" / child_sid / ".system_generated" / "messages"
    await asyncio.to_thread(subagents_dir.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(parent_inbox.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(child_inbox.mkdir, parents=True, exist_ok=True)

    await asyncio.to_thread(
        (subagents_dir / f"{child_sid}.json").write_text,
        json.dumps(
            {
                "conversationId": child_sid,
                "state": "SUBAGENT_STATE_ALIVE",
                "subagentDescriptor": {"role": "Researcher"},
            }
        ),
        encoding="utf-8",
    )
    await asyncio.to_thread(
        (child_inbox / "launch.json").write_text,
        json.dumps(
            {
                "sender": root_sid,
                "sourceMetadata": {
                    "tool": {
                        "conversationId": root_sid,
                        "toolCall": {
                            "id": "call_sub_1",
                            "name": "invoke_subagent",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # ACP reports invoke_subagent completed immediately, then prompt returns end_turn.
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_sub_1",
                "name": "invoke_subagent",
                "kind": "other",
                "title": "Run invoke_subagent",
                "status": "in_progress",
            },
            session_id=root_sid,
        )
    )
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_sub_1",
                "status": "completed",
            },
            session_id=root_sid,
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn.finished

    # Still waiting after poll tick while child has not reported back.
    await timer.advance(runtime_mod._DEFERRED_POLL_MS)
    assert not turn.finished

    # Child finishes and writes an unread high-priority message to parent inbox!
    await asyncio.to_thread(
        (parent_inbox / "msg-1.json").write_text,
        json.dumps(
            {
                "id": "msg-1",
                "sender": child_sid,
                "priority": "MESSAGE_PRIORITY_HIGH",
                "content": "Findings ready",
            }
        ),
        encoding="utf-8",
    )
    await timer.advance(runtime_mod._DEFERRED_POLL_MS)
    # Subagent reported, but high-priority wake message is still unread -> STILL held!
    assert not turn.finished

    # Parent reads the message, streams autonomous reply, and emits usage_update.
    await asyncio.to_thread(
        (parent_inbox / "read.json").write_text,
        json.dumps({"msg-1": True}),
        encoding="utf-8",
    )
    connection.receive(
        _update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Subagent finished."},
            },
            session_id=root_sid,
        )
    )
    connection.receive(
        _update(
            {"sessionUpdate": "usage_update", "used": 300, "size": 1_048_576},
            session_id=root_sid,
        )
    )
    await timer.advance(runtime_mod._DEFERRED_QUIET_MS)
    await _wait_finished(turn)
    assert turn.completed

    # Verify harness.has_background_job checks disk state too!
    harness = AntigravityHarness(
        HarnessInput(
            agent_id="agent-1",
            agent_name="Agent",
            workspace=tmp_path,
            startup_timeout=5,
            shutdown_timeout=5,
        ),
        cast(Any, timer),
        cast(Any, None),
        model=None,
        effort=None,
    )
    harness.sessions[connection.session.id] = connection
    assert not await harness.has_background_job(connection.session, timeout=5)

    # Trailing SYSTEM_MESSAGE in transcript.jsonl counts as live background work.
    transcript = (
        profile / "brain" / root_sid / ".system_generated" / "logs" / "transcript.jsonl"
    )
    await asyncio.to_thread(transcript.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(
        transcript.write_text,
        json.dumps(
            {
                "step_index": 5,
                "type": "SYSTEM_MESSAGE",
                "content": "priority=MESSAGE_PRIORITY_HIGH wake",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert await harness.has_background_job(connection.session, timeout=5)


async def test_sqlite_hidden_step_error_classification_and_session_broken(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    sid = "sess-sql-1"
    db_path = profile / "conversations" / f"{sid}.db"

    # Historical step_type=17 at idx=2 should be ignored when start_step_index=2.
    await asyncio.to_thread(
        _create_steps_db,
        db_path,
        [(2, 17, b"RESOURCE_EXHAUSTED (code 429): old error")],
    )
    start_idx = await asyncio.to_thread(read_last_step_index, profile, sid)
    assert start_idx == 2
    assert await asyncio.to_thread(read_step_error, profile, sid, start_idx) is None

    # New real-format 429 error step at idx=3 without any "⚠" text chunk fails turn!
    await asyncio.to_thread(
        _create_steps_db,
        db_path,
        [
            (2, 17, b"RESOURCE_EXHAUSTED (code 429): old error"),
            (
                3,
                17,
                (
                    b"\x0a\x10RESOURCE_EXHAUSTED (code 429): Individual quota reached. "
                    b"Resets in 3h4m49s.\x00trace"
                ),
            ),
        ],
    )
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(
        tmp_path,
        session_id=sid,
        timer=timer,
        profile_dir=profile,
        start_step_index=start_idx,
    )
    connection.receive(_prompt_reply(1))
    await _wait_finished(turn)
    assert not turn.completed
    events = _drain_sync(turn)
    assert isinstance(events[-1].payload, TurnFailed)
    assert events[-1].payload.failure.code == "RESOURCE_EXHAUSTED"
    assert "Individual quota reached" in events[-1].payload.failure.message
    assert not connection.broken

    # INVALID_THINKING_SIGNATURE marks connection.broken = True!
    await asyncio.to_thread(
        _create_steps_db,
        db_path,
        [
            (
                4,
                17,
                (
                    b"Error 400, Message: messages.12.content.0.thinking.signature: "
                    b"Field required, Status: INVALID_ARGUMENT"
                ),
            )
        ],
    )
    conn2, turn2, _ = _make_turn(
        tmp_path,
        session_id=sid,
        timer=_VirtualTimer(),
        profile_dir=profile,
        start_step_index=3,
    )
    conn2.receive(_prompt_reply(1))
    await _wait_finished(turn2)
    assert not turn2.completed
    assert conn2.broken is True
    events2 = _drain_sync(turn2)
    assert isinstance(events2[-1].payload, TurnFailed)
    assert events2[-1].payload.failure.code == "INVALID_THINKING_SIGNATURE"

    # A subsequent PLANNER_RESPONSE (step_type=15) after step 4 clears the error!
    await asyncio.to_thread(
        _create_steps_db,
        db_path,
        [(5, 15, b"ok")],
    )
    assert await asyncio.to_thread(read_step_error, profile, sid, 3) is None


async def test_state_readers_fail_open_on_symlinks_and_classify_variants(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    conversations = profile / "conversations"
    await asyncio.to_thread(conversations.mkdir, parents=True, exist_ok=True)
    real_db = tmp_path / "outside.db"
    await asyncio.to_thread(
        _create_steps_db,
        real_db,
        [(1, 17, b"Error 500, Message: boom, Status: INTERNAL")],
    )
    link_db = conversations / "linked-sess.db"
    symlink_created = False
    try:
        await asyncio.to_thread(link_db.symlink_to, real_db)
        symlink_created = True
    except OSError:
        # Unprivileged Windows CI runners may disallow creating symlinks.
        pass
    if symlink_created:
        assert (
            await asyncio.to_thread(read_last_step_index, profile, "linked-sess")
            is None
        )
        assert (
            await asyncio.to_thread(read_step_error, profile, "linked-sess", -1) is None
        )
    assert await asyncio.to_thread(read_step_error, profile, "../escape", -1) is None
    assert (
        await asyncio.to_thread(
            active_subagent_calls, profile, "missing-sess", ("call_1",)
        )
        == ()
    )
    assert not await asyncio.to_thread(has_pending_reply, profile, "missing-sess")

    # Unit checks for classify_step_error variants
    o1 = classify_step_error(
        "Error 503, Message: Service unavailable, Status: UNAVAILABLE"
    )
    assert o1.message == "Service unavailable" and o1.code == "UNAVAILABLE"
    o2 = classify_step_error("reason:QUOTA_EXHAUSTED quotaResetDelay:2h5m")
    assert o2.code == "QUOTA_EXHAUSTED" and "2h5m" in o2.message
    o3 = classify_step_error("unstructured crash")
    assert o3.code is None and not o3.session_broken


async def test_inactivity_timeout_disconnect_and_aclose_during_deferred_wait(
    tmp_path: Path,
) -> None:
    # 1. Autonomous reply hangs without usage_update -> inactivity timeout completes cleanly.
    timer = _VirtualTimer()
    connection, turn, _ = _make_turn(tmp_path, timer=timer)
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "echo hi"},
                "status": "in_progress",
            }
        )
    )
    connection.receive(_prompt_reply(1))
    await _yield_loop()
    connection.receive(
        _update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_bg",
                "status": "completed",
            }
        )
    )
    connection.receive(
        _update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Partial reply without usage"},
            }
        )
    )
    await _yield_loop()
    assert not turn.finished
    await timer.advance(runtime_mod._DEFERRED_REPLY_TIMEOUT_MS + 1_000)
    assert turn.finished and turn.completed

    # 2. Disconnect during deferred wait fails turn with CONNECTION_LOST.
    timer2 = _VirtualTimer()
    conn2, turn2, _ = _make_turn(tmp_path, timer=timer2)
    conn2.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "sleep 60"},
                "status": "in_progress",
            }
        )
    )
    conn2.receive(_prompt_reply(1))
    await _yield_loop()
    assert not turn2.finished
    conn2.disconnected(Disconnected("pipe broken"))
    await _yield_loop()
    assert turn2.finished and not turn2.completed
    events2 = _drain_sync(turn2)
    assert isinstance(events2[-1].payload, TurnFailed)
    assert events2[-1].payload.failure.reason is HarnessFailureReason.CONNECTION_LOST
    assert events2[-1].payload.failure.needs_reconciliation is True

    # 3. aclose() during deferred wait cancels settle_task cleanly.
    timer3 = _VirtualTimer()
    conn3, turn3, dummy3 = _make_turn(tmp_path, timer=timer3)
    conn3.receive(
        _update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "call_bg",
                "name": "run_command",
                "kind": "execute",
                "rawInput": {"CommandLine": "sleep 60"},
                "status": "in_progress",
            }
        )
    )
    conn3.receive(_prompt_reply(1))
    await _yield_loop()
    assert turn3.settle_task is not None
    await turn3.aclose()
    assert turn3.settle_task is None
    assert any(m.get("method") == "session/cancel" for m in dummy3.sent)


async def test_acp_e2e_deferred_background_quota_error_and_broken_session_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_mod, "_DEFERRED_POLL_MS", 20)
    monkeypatch.setattr(runtime_mod, "_DEFERRED_QUIET_MS", 40)
    monkeypatch.setattr(runtime_mod, "_POST_WORK_WAKE_GRACE_MS", 200)

    home_dir = tmp_path / "home"
    profile_dir = home_dir / ".gemini" / "antigravity-cli"
    await asyncio.to_thread(
        (profile_dir / "conversations").mkdir, parents=True, exist_ok=True
    )

    session_counter = {"n": 0}
    script = """
import json, os, sqlite3, sys, time

session_id = os.environ["FAKE_SESSION_ID"]
db_path = os.environ["FAKE_DB_PATH"]
turns = 0
for line in sys.stdin:
    req = json.loads(line)
    method = req.get("method")
    req_id = req.get("id")
    if method == "initialize":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"protocolVersion": 1, "agentCapabilities": {}}}), flush=True)
    elif method == "session/new":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": session_id, "configOptions": [{"id": "model", "currentValue": "gemini-4-neon"}]}}), flush=True)
    elif method == "session/prompt":
        turns += 1
        if session_id == "sess-1" and turns == 1:
            # Start background command and return end_turn prematurely.
            print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": {"sessionUpdate": "tool_call", "toolCallId": "bg-1", "name": "run_command", "kind": "execute", "rawInput": {"CommandLine": "echo work"}, "status": "in_progress"}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}), flush=True)
            time.sleep(0.05)
            # Command finishes after end_turn, autonomous wakeup hits quota error!
            print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": {"sessionUpdate": "tool_call_update", "toolCallId": "bg-1", "status": "completed", "rawOutput": {"exit_code": 0, "output": "work"}}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "\\n⚠ Individual quota reached. Resets in 1h.\\n"}}}}), flush=True)
        elif session_id == "sess-1" and turns == 2:
            # Write INVALID_THINKING_SIGNATURE to SQLite and return silent end_turn.
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS steps (idx INTEGER PRIMARY KEY, step_type INTEGER, step_payload BLOB)")
            conn.execute("INSERT INTO steps VALUES (1, 17, ?)", (b"Error 400, Message: messages.4.content.0.thinking.signature: Field required, Status: INVALID_ARGUMENT",))
            conn.commit()
            conn.close()
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}), flush=True)
        else:
            print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Fresh session OK"}}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": {"sessionUpdate": "usage_update", "used": 50, "size": 1048576}}}), flush=True)
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}}), flush=True)
    elif method == "session/close":
        print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}}), flush=True)
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
        session_counter["n"] += 1
        sid = f"sess-{session_counter['n']}"
        fake_env = (
            dict(os.environ)
            | env
            | {
                "HOME": str(home_dir),
                "USERPROFILE": str(home_dir),
                "FAKE_SESSION_ID": sid,
                "FAKE_DB_PATH": str(profile_dir / "conversations" / f"{sid}.db"),
            }
        )
        return await real_start(
            (sys.executable, "-c", script),
            source,
            receive,
            answer,
            disconnected,
            cwd=cwd,
            env=fake_env,
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
            SessionInput(
                id="session-1",
                workspace=tmp_path,
                environment={"HOME": str(home_dir), "USERPROFILE": str(home_dir)},
            ),
            timeout=10,
        )
        assert started.value is not None
        session = started.value
        assert session.provider_thread_id == "sess-1"
        try:
            # Turn 1: background tool finishes after end_turn and autonomous wakeup hits 429.
            s1 = await harness.start_turn(
                session,
                HarnessTurn(id="turn-1", session_id=session.id, input_text="run"),
                _DummyApproval(),
                timeout=10,
            )
            e1 = [item async for item in s1]
            await s1.aclose()
            assert any(isinstance(i.payload, ToolCallCompleted) for i in e1)
            assert isinstance(e1[-1].payload, TurnFailed)
            assert "Individual quota reached" in e1[-1].payload.failure.message

            # Turn 2: silent INVALID_THINKING_SIGNATURE in SQLite -> fails & marks broken.
            s2 = await harness.start_turn(
                session,
                HarnessTurn(id="turn-2", session_id=session.id, input_text="again"),
                _DummyApproval(),
                timeout=10,
            )
            e2 = [item async for item in s2]
            await s2.aclose()
            assert isinstance(e2[-1].payload, TurnFailed)
            assert e2[-1].payload.failure.code == "INVALID_THINKING_SIGNATURE"

            # Next start_turn refuses broken session; reconcile_session opens fresh sess-2!
            with pytest.raises(HarnessSessionUnavailable):
                await harness.start_turn(
                    session,
                    HarnessTurn(id="turn-3", session_id=session.id, input_text="hi"),
                    _DummyApproval(),
                    timeout=10,
                )
            reconciled = await harness.reconcile_session(
                session, None, None, timeout=10
            )
            assert reconciled.status is ProviderCallStatus.CONFIRMED
            assert reconciled.value is not None
            fresh = reconciled.value.session
            assert fresh.provider_thread_id == "sess-2"

            # Turn 3 on fresh session succeeds cleanly!
            s3 = await harness.start_turn(
                fresh,
                HarnessTurn(id="turn-3", session_id=fresh.id, input_text="hi"),
                _DummyApproval(),
                timeout=10,
            )
            e3 = [item async for item in s3]
            await s3.aclose()
            assert isinstance(e3[-1].payload, TurnCompleted)
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
