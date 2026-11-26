"""BCN sessions backed by the installed Antigravity ACP CLI."""

# The lifecycle and approval budgets come from the Harness contract.
# ruff: noqa: ASYNC109

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import Mapping
from contextlib import suppress
from importlib.metadata import version
from math import ceil
from mimetypes import guess_file_type
from pathlib import Path
from time import time_ns
from typing import Any

from bazaar_contrib_audit import Audit, AuditContext
from bazaar_contrib_harness import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalScope,
    HarnessAttachment,
    HarnessEventPayload,
    HarnessExpire,
    HarnessFailure,
    HarnessFailureReason,
    HarnessInput,
    HarnessOutput,
    HarnessSession,
    HarnessSessionReconciliation,
    HarnessSessionUnavailable,
    HarnessTurn,
    HarnessTurnStream,
    ModelUpdated,
    ProviderCallResult,
    ProviderCallStatus,
    SandboxPolicy,
    SessionInput,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from bazaar_contrib_timer import Timer
from bazaar_contrib_timer import timeout as timer_timeout
from pydantic import JsonValue, ValidationError
from watchfiles import awatch

from .events import TurnEvents, Usage, current_option
from .source import Source
from .state import (
    StepErrorOutcome,
    active_subagent_calls,
    has_live_disk_work,
    has_pending_reply,
    profile_root,
    read_last_step_index,
    read_step_error,
)
from .transport import (
    AcpProcess,
    Disconnected,
    NotSent,
    Packet,
    ProtocolError,
    RpcError,
)

logger = logging.getLogger(__name__)
_MAX_APPROVAL_DETAIL = 4_000
_DEFERRED_POLL_MS = 1_000
_DEFERRED_QUIET_MS = 500
_POST_WORK_WAKE_GRACE_MS = 3_000
_DEFERRED_REPLY_TIMEOUT_MS = 900_000


def _truncate_detail(value: str) -> str:
    if len(value) > _MAX_APPROVAL_DETAIL:
        return value[: _MAX_APPROVAL_DETAIL - 1] + "…"
    return value


def _approval_action(
    tool: Mapping[str, Any], arguments: Mapping[str, Any] | None
) -> str:
    if tool.get("name") == "call_mcp_tool" and arguments is not None:
        server = arguments.get("ServerName")
        mcp_tool = arguments.get("ToolName")
        if (
            isinstance(server, str)
            and server
            and isinstance(mcp_tool, str)
            and mcp_tool
        ):
            return f"mcp__{server}__{mcp_tool}"
    kind = tool.get("kind")
    if isinstance(kind, str) and kind:
        return {"execute": "command_execution", "edit": "file_change"}.get(kind, kind)
    return "tool_permission"


def _approval_details(
    params: Mapping[str, Any],
    tool: Mapping[str, Any],
    arguments: Mapping[str, Any] | None,
    action: str,
) -> dict[str, JsonValue]:
    details: dict[str, JsonValue] = dict(params)
    kind = tool.get("kind")
    name = tool.get("name")
    title = tool.get("title")
    if kind == "execute":
        command = arguments.get("CommandLine") if arguments is not None else None
        if not isinstance(command, str) or not command.strip():
            command = title if isinstance(title, str) else None
        if isinstance(command, str) and command.strip():
            details.setdefault("command", _truncate_detail(command.strip()))
        cwd = arguments.get("Cwd") if arguments is not None else None
        if isinstance(cwd, str) and cwd.strip():
            details.setdefault("cwd", cwd.strip())
        summary = arguments.get("toolSummary") if arguments is not None else None
        if (
            isinstance(summary, str)
            and summary.strip()
            and summary.strip() != details.get("command")
        ):
            details.setdefault("reason", _truncate_detail(summary.strip()))
        return details

    if isinstance(title, str) and title.strip():
        details.setdefault("reason", _truncate_detail(title.strip()))
    if name == "call_mcp_tool":
        details.setdefault("tool_name", action)
        raw_input: Any = (
            arguments.get("Arguments", arguments) if arguments is not None else None
        )
    else:
        if isinstance(name, str) and name.strip():
            details.setdefault("tool_name", name.strip())
        raw_input = arguments
    if raw_input is not None:
        try:
            encoded = json.dumps(
                raw_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except TypeError, ValueError:
            encoded = ""
        if encoded:
            details.setdefault("tool_input", _truncate_detail(encoded))
    return details


class Turn:
    def __init__(
        self,
        connection: Connection,
        turn: HarnessTurn,
        approval: ApprovalHandler,
        timeout: float,
    ) -> None:
        self.connection = connection
        self.id = turn.id
        self.approval = approval
        self.timeout = timeout
        self.timer = connection.timer
        self.profile_dir = connection.profile_dir
        self.start_step_index: int | None = None
        self.events = TurnEvents(connection.usage)
        self.output: asyncio.Queue[HarnessOutput | None] = asyncio.Queue()
        self.pending: set[int] = set()
        self.replies: dict[int, Packet] = {}
        self.finished = False
        self.completed = False
        self.cancel_requested = False
        self.prompt_settled = False
        self.settled_reason: str | None = None
        self.settled_packet: Packet | None = None
        self.settle_task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        self._post_settle_reply_open = False
        self._post_work_grace_deadline_ms: int | None = None
        self._quiet_deadline_ms: int | None = None
        self._reply_inactivity_deadline_ms: int | None = None

    def __aiter__(self) -> Turn:
        return self

    async def __anext__(self) -> HarnessOutput:
        item = await self.output.get()
        if item is None:
            self.output.put_nowait(None)
            raise StopAsyncIteration
        return item

    def _now_ms(self) -> int:
        if self.timer is not None:
            return self.timer.now_ms
        return time_ns() // 1_000_000

    def _cancel_settle_task(self) -> asyncio.Task[None] | None:
        task = self.settle_task
        self.settle_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
        return task

    async def aclose(self) -> None:
        task = self._cancel_settle_task()
        if not self.finished and not self.cancel_requested:
            self.cancel_requested = True
            if self.pending or self.prompt_settled:
                with suppress(Disconnected):
                    await self.connection.process.send(
                        {
                            "method": "session/cancel",
                            "params": {
                                "sessionId": self.connection.session.provider_thread_id
                            },
                        }
                    )
        if task is not None and task is not asyncio.current_task():
            with suppress(asyncio.CancelledError):
                await task

    def emit(
        self, payload: HarnessEventPayload, event: str, packet: Packet | None = None
    ) -> None:
        session = self.connection.session
        self.output.put_nowait(
            HarnessOutput(
                session_id=session.id,
                provider_session_id=session.provider_thread_id,
                turn_id=self.id,
                provider="antigravity",
                provider_event=event,
                occurred_at_ms=time_ns() // 1_000_000,
                payload=payload,
                raw=packet.model_dump(mode="json", exclude_unset=True)
                if packet
                else None,
            )
        )

    def finish(
        self,
        payload: TurnCompleted | TurnCancelled | TurnFailed,
        packet: Packet | None = None,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        self.completed = isinstance(payload, TurnCompleted)
        self._cancel_settle_task()
        self.emit(payload, payload.event_name, packet)
        self.output.put_nowait(None)
        if self.connection.turn is self:
            source = self.connection.process.source
            source.context = source.context.model_copy(update={"turn_id": None})

    def on_prompt_submitted(self, identifier: int) -> None:
        self.pending.add(identifier)
        self.prompt_settled = False
        self.settled_reason = None
        self.settled_packet = None
        self._post_settle_reply_open = False
        self._post_work_grace_deadline_ms = None
        self._quiet_deadline_ms = None
        self._reply_inactivity_deadline_ms = None
        self._wake.set()

    def handle_update(self, update: dict[str, Any], event: str, packet: Packet) -> None:
        had_outliving = bool(self.events.outliving_tool_ids())
        for payload in self.events.update(update):
            self.emit(payload, event, packet)
        if self.events.subagent_calls:
            self.connection.subagent_calls.update(self.events.subagent_calls)
        if not self.prompt_settled:
            return
        now = self._now_ms()
        if event in {"agent_message_chunk", "agent_thought_chunk"}:
            content = update.get("content")
            if isinstance(content, Mapping) and content.get("text"):
                if self.events.trailing_error is not None:
                    # Wait a short quiet window so normal assistant text quoting
                    # "\n⚠ ..." followed immediately by usage_update is not
                    # misclassified as an executor error.
                    self._post_settle_reply_open = False
                    self._post_work_grace_deadline_ms = None
                    self._quiet_deadline_ms = now + _DEFERRED_QUIET_MS
                else:
                    self._post_settle_reply_open = True
                    self._quiet_deadline_ms = None
                    self._post_work_grace_deadline_ms = None
                    self._reply_inactivity_deadline_ms = (
                        now + _DEFERRED_REPLY_TIMEOUT_MS
                    )
        elif event == "usage_update":
            self._post_settle_reply_open = False
            self._reply_inactivity_deadline_ms = None
            self._post_work_grace_deadline_ms = None
            self._quiet_deadline_ms = now + _DEFERRED_QUIET_MS
        elif event in {"tool_call", "tool_call_update"}:
            now_outliving = bool(self.events.outliving_tool_ids())
            if now_outliving:
                self._quiet_deadline_ms = None
                self._post_work_grace_deadline_ms = None
                self._reply_inactivity_deadline_ms = None
            elif had_outliving and not now_outliving:
                # Background execute/schedule tool just completed; give agy a
                # grace window to start streaming its autonomous reply.
                self._post_work_grace_deadline_ms = now + _POST_WORK_WAKE_GRACE_MS
                self._quiet_deadline_ms = None
            elif event == "tool_call":
                self._post_settle_reply_open = True
                self._quiet_deadline_ms = None
                self._post_work_grace_deadline_ms = None
                self._reply_inactivity_deadline_ms = now + _DEFERRED_REPLY_TIMEOUT_MS
        self._wake.set()

    def note_approval_resolved(self) -> None:
        if self.prompt_settled and not self.finished:
            now = self._now_ms()
            self._post_settle_reply_open = True
            self._quiet_deadline_ms = None
            self._post_work_grace_deadline_ms = None
            self._reply_inactivity_deadline_ms = now + _DEFERRED_REPLY_TIMEOUT_MS
            self._wake.set()

    def _finish_from_step_error(
        self, outcome: StepErrorOutcome, packet: Packet | None
    ) -> None:
        if outcome.session_broken:
            self.connection.broken = True
        self.finish(
            TurnFailed(
                event_name="session/prompt",
                failure=HarnessFailure(
                    HarnessFailureReason.EXECUTION,
                    outcome.message,
                    code=outcome.code,
                ),
            ),
            packet,
        )

    async def _wait_or_wake(self, delay_ms: int) -> None:
        if self._wake.is_set():
            return
        wake_task = asyncio.create_task(self._wake.wait())
        handle = None
        if self.timer is not None:
            clamped = max(1, min(delay_ms, self.timer.maximum_delay_ms))
            handle = self.timer.create(clamped)
            timer_task = asyncio.create_task(handle.wait())
        else:
            timer_task = asyncio.create_task(asyncio.sleep(max(delay_ms, 1) / 1_000))
        try:
            await asyncio.wait(
                (wake_task, timer_task), return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            if handle is not None:
                handle.cancel()
            wake_task.cancel()
            timer_task.cancel()
            await asyncio.gather(wake_task, timer_task, return_exceptions=True)

    def _still_settling(self) -> bool:
        return (
            not self.finished
            and not self.cancel_requested
            and self.prompt_settled
            and not self.pending
        )

    async def _read_step_error(self) -> StepErrorOutcome | None:
        if self.profile_dir is None or self.start_step_index is None:
            return None
        return await asyncio.to_thread(
            read_step_error,
            self.profile_dir,
            self.connection.session.provider_thread_id,
            self.start_step_index,
        )

    async def _active_subagents(self) -> tuple[str, ...]:
        if self.profile_dir is None or not self.events.subagent_calls:
            return ()
        return await asyncio.to_thread(
            active_subagent_calls,
            self.profile_dir,
            self.connection.session.provider_thread_id,
            tuple(self.events.subagent_calls),
        )

    async def _has_pending_reply(self) -> bool:
        if self.profile_dir is None:
            return False
        return await asyncio.to_thread(
            has_pending_reply,
            self.profile_dir,
            self.connection.session.provider_thread_id,
        )

    async def _settle_loop(self) -> None:
        deferred = False
        saw_subagents = False
        try:
            while self._still_settling():
                self._wake.clear()
                packet = self.settled_packet
                reason = self.settled_reason

                has_open_tools = bool(self.events.outliving_tool_ids())
                has_incoming = bool(self.connection.process.incoming)
                subagents = await self._active_subagents()
                if not self._still_settling():
                    return
                has_subagents = bool(subagents)
                if has_subagents:
                    saw_subagents = True
                elif saw_subagents:
                    saw_subagents = False
                    if self._post_work_grace_deadline_ms is None:
                        self._post_work_grace_deadline_ms = (
                            self._now_ms() + _POST_WORK_WAKE_GRACE_MS
                        )

                if has_open_tools or has_incoming or has_subagents:
                    deferred = True
                    self._reply_inactivity_deadline_ms = None
                    await self._wait_or_wake(_DEFERRED_POLL_MS)
                    continue

                # No background tools, approvals, or subagents are active.
                # Check for a hidden error step (SQLite step_type=17) or a
                # refusal stop reason before letting pending_reply delay failure.
                step_error = await self._read_step_error()
                if not self._still_settling():
                    return
                if step_error is not None:
                    self._finish_from_step_error(step_error, packet)
                    return

                if reason == "refusal":
                    message = (
                        self.events.trailing_error
                        or "Antigravity stopped the turn with refusal"
                    )
                    self.finish(
                        TurnFailed(
                            event_name="session/prompt",
                            failure=HarnessFailure(
                                HarnessFailureReason.EXECUTION,
                                message,
                                code="refusal",
                            ),
                        ),
                        packet,
                    )
                    return

                # If a trailing "\n⚠ ...\n" arrived on the foreground prompt
                # (not deferred), fail immediately. If it arrived during a
                # deferred autonomous wakeup, wait out the short quiet window
                # so a normal usage_update immediately after prose can clear it.
                if self.events.trailing_error is not None:
                    now = self._now_ms()
                    if (
                        deferred
                        and self._quiet_deadline_ms is not None
                        and now < self._quiet_deadline_ms
                    ):
                        await self._wait_or_wake(self._quiet_deadline_ms - now)
                        continue
                    self.finish(
                        TurnFailed(
                            event_name="session/prompt",
                            failure=HarnessFailure(
                                HarnessFailureReason.EXECUTION,
                                self.events.trailing_error,
                            ),
                        ),
                        packet,
                    )
                    return

                now = self._now_ms()

                # Keep waiting while an autonomous reply is streaming or queued
                # on disk, bounded by the inactivity timeout.
                pending_reply = await self._has_pending_reply()
                if not self._still_settling():
                    return
                if self._post_settle_reply_open or pending_reply:
                    deferred = True
                    if self._reply_inactivity_deadline_ms is None:
                        self._reply_inactivity_deadline_ms = (
                            now + _DEFERRED_REPLY_TIMEOUT_MS
                        )
                    remaining = self._reply_inactivity_deadline_ms - now
                    if remaining > 0:
                        await self._wait_or_wake(min(_DEFERRED_POLL_MS, remaining))
                        continue
                    # Inactivity timeout expired while waiting for a reply to
                    # close; do one final error-step check before completing.
                    final_error = await self._read_step_error()
                    if not self._still_settling():
                        return
                    if final_error is not None:
                        self._finish_from_step_error(final_error, packet)
                        return
                    self.finish(TurnCompleted(event_name="session/prompt"), packet)
                    return

                # Wait out the post-tool wake grace window if a background tool
                # or subagent just finished and agy hasn't started streaming yet.
                if self._post_work_grace_deadline_ms is not None:
                    remaining = self._post_work_grace_deadline_ms - now
                    if remaining > 0:
                        deferred = True
                        await self._wait_or_wake(min(_DEFERRED_POLL_MS, remaining))
                        continue
                    self._post_work_grace_deadline_ms = None

                # Wait out the short quiet settle debounce after usage_update.
                if self._quiet_deadline_ms is not None:
                    remaining = self._quiet_deadline_ms - now
                    if remaining > 0:
                        deferred = True
                        await self._wait_or_wake(remaining)
                        continue
                    self._quiet_deadline_ms = None

                # Final synchronous check against any race that woke us.
                if self._wake.is_set():
                    continue
                self.finish(TurnCompleted(event_name="session/prompt"), packet)
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Antigravity deferred turn settlement failed")
            if self._still_settling():
                self.finish(
                    TurnCompleted(event_name="session/prompt"), self.settled_packet
                )
        finally:
            if self.settle_task is asyncio.current_task():
                self.settle_task = None

    def _start_settle_task(self) -> None:
        if self.settle_task is not None and not self.settle_task.done():
            self._wake.set()
            return
        self.settle_task = asyncio.create_task(
            self._settle_loop(), name=f"antigravity-settle:{self.id}"
        )

    def reply(self, identifier: int, packet: Packet) -> None:
        self.pending.remove(identifier)
        self.replies[identifier] = packet
        if (
            packet.result is not None
            and (usage := packet.result.get("usage")) is not None
        ):
            self.emit(self.connection.usage.account(usage), "session/prompt", packet)
        if self.pending:
            return

        # A second session/prompt interrupts the first RPC's ownership of the
        # outcome. The old RPC may return end_turn while the model is still busy.
        first = min(self.replies)
        final = next(
            reply
            for key, reply in sorted(self.replies.items(), reverse=True)
            if key == first or reply.error is None or reply.error.code != -32602
        )
        if final.error is not None:
            terminal = TurnFailed(
                event_name="session/prompt",
                failure=HarnessFailure(
                    HarnessFailureReason.EXECUTION,
                    final.error.message,
                    code=final.error.code,
                ),
            )
            self.finish(terminal, final)
            return

        reason = final.value().get("stopReason")
        match reason:
            case "end_turn":
                # Bare unit tests without a Timer or local profile_dir settle
                # synchronously unless background tools are visibly open on ACP.
                if (
                    self.timer is None
                    and self.profile_dir is None
                    and not self.events.outliving_tool_ids()
                    and not self.connection.process.incoming
                ):
                    if self.events.trailing_error is not None:
                        self.finish(
                            TurnFailed(
                                event_name="session/prompt",
                                failure=HarnessFailure(
                                    HarnessFailureReason.EXECUTION,
                                    self.events.trailing_error,
                                ),
                            ),
                            final,
                        )
                    else:
                        self.finish(TurnCompleted(event_name="session/prompt"), final)
                    return
                self.prompt_settled = True
                self.settled_reason = "end_turn"
                self.settled_packet = final
                self._start_settle_task()
            case "cancelled":
                self.finish(TurnCancelled(event_name="session/prompt"), final)
            case "refusal":
                if self.profile_dir is not None and self.start_step_index is not None:
                    self.prompt_settled = True
                    self.settled_reason = "refusal"
                    self.settled_packet = final
                    self._start_settle_task()
                    return
                self.finish(
                    TurnFailed(
                        event_name="session/prompt",
                        failure=HarnessFailure(
                            HarnessFailureReason.EXECUTION,
                            self.events.trailing_error
                            or "Antigravity stopped the turn with refusal",
                            code="refusal",
                        ),
                    ),
                    final,
                )
            case "max_tokens" | "max_turn_requests":
                self.finish(
                    TurnFailed(
                        event_name="session/prompt",
                        failure=HarnessFailure(
                            HarnessFailureReason.EXECUTION,
                            f"Antigravity stopped the turn with {reason}",
                            code=reason,
                        ),
                    ),
                    final,
                )
            case _:
                self.finish(
                    TurnFailed(
                        event_name="session/prompt",
                        failure=HarnessFailure(
                            HarnessFailureReason.PROTOCOL,
                            f"Unrecognized Antigravity stop reason: {reason!r}",
                        ),
                    ),
                    final,
                )


class Connection:
    def __init__(
        self,
        request: SessionInput,
        session: HarnessSession,
        usage: Usage,
        *,
        timer: Timer | None = None,
        profile_dir: Path | None = None,
    ) -> None:
        self.request = request
        self.session = session
        self.usage = usage
        self.timer = timer
        self.profile_dir = profile_dir
        self.process: AcpProcess
        self.watch_task: asyncio.Task[None] | None = None
        self.turn: Turn | None = None
        self.model: str | None = None
        self.broken = False
        self.subagent_calls: set[str] = set()

    def receive(self, packet: Packet) -> None:
        turn = self.turn
        if packet.method == "session/update":
            if packet.params["sessionId"] != self.session.provider_thread_id:
                return
            update = packet.params["update"]
            event = update["sessionUpdate"]
            if event in {"tool_call", "tool_call_update"}:
                self.process.observe_tool(update)
            if event == "config_option_update":
                self.model = current_option(update, "model") or self.model
            if turn is not None and not turn.finished:
                turn.handle_update(update, event, packet)
        elif (
            turn is not None
            and not turn.finished
            and packet.method is None
            and isinstance(packet.id, int)
            and packet.id in turn.pending
        ):
            turn.reply(packet.id, packet)

    async def answer(self, packet: Packet) -> dict[str, Any]:
        if packet.method != "session/request_permission":
            raise RpcError(f"ACP client method is unavailable: {packet.method}", -32601)
        turn = self.turn
        cancelled = {"outcome": {"outcome": "cancelled"}}
        if (
            turn is None
            or turn.cancel_requested
            or (turn.finished and not turn.completed)
            or packet.params.get("sessionId") != self.session.provider_thread_id
        ):
            return cancelled

        raw_options = packet.params.get("options")
        if not isinstance(raw_options, list):
            return cancelled
        options = {
            option["kind"]: option["optionId"]
            for option in raw_options
            if isinstance(option, dict)
            and isinstance(option.get("kind"), str)
            and isinstance(option.get("optionId"), str)
        }
        allow = {
            scope: options[kind]
            for scope, kind in (
                (ApprovalScope.ONCE, "allow_once"),
                (ApprovalScope.ALWAYS, "allow_always"),
            )
            if kind in options
        }
        reject = options.get("reject_once") or options.get("reject_always")
        choice = reject
        if allow:
            raw_tool = packet.params.get("toolCall")
            tool = raw_tool if isinstance(raw_tool, dict) else {}
            raw_input = tool.get("rawInput")
            raw_args = tool.get("args")
            arguments = (
                raw_input
                if isinstance(raw_input, dict)
                else raw_args
                if isinstance(raw_args, dict)
                else None
            )
            action = _approval_action(tool, arguments)
            details = _approval_details(packet.params, tool, arguments, action)
            active_before = not turn.finished
            try:
                result = await turn.approval.request_approval(
                    ApprovalRequest(
                        request_id=str(packet.id),
                        session_id=self.session.id,
                        turn_id=turn.id,
                        action=action,
                        details=details,
                        allowed_scopes=tuple(allow),
                    ),
                    timeout=turn.timeout,
                )
            except Exception:
                logger.exception(
                    "Could not handle Antigravity permission request %s", packet.id
                )
                return cancelled
            if turn.cancel_requested or (active_before and turn.finished):
                return cancelled
            if result.decision == "approved":
                choice = allow.get(result.scope, reject)
        if choice is None:
            return cancelled
        turn.note_approval_resolved()
        return {"outcome": {"outcome": "selected", "optionId": choice}}

    def disconnected(self, error: Disconnected | ProtocolError) -> None:
        if self.turn is not None and not self.turn.finished:
            self.turn.finish(
                TurnFailed(
                    event_name="acp.disconnected",
                    failure=HarnessFailure(
                        HarnessFailureReason.PROTOCOL
                        if isinstance(error, ProtocolError)
                        else HarnessFailureReason.CONNECTION_LOST,
                        str(error),
                        needs_reconciliation=True,
                    ),
                )
            )

    async def submit(self, turn: Turn, prompt: list[dict[str, Any]]) -> None:
        identifier = self.process.request_id()
        turn.on_prompt_submitted(identifier)
        try:
            await self.process.send(
                {
                    "id": identifier,
                    "method": "session/prompt",
                    "params": {
                        "sessionId": self.session.provider_thread_id,
                        "prompt": prompt,
                    },
                }
            )
        except NotSent:
            turn.pending.discard(identifier)
            raise
        except Disconnected as error:
            self.disconnected(error)

    async def close(self, timer: Timer, timeout: float) -> None:
        if self.turn is not None:
            task = self.turn._cancel_settle_task()
            if task is not None and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError):
                    await task
        if self.watch_task is not None:
            self.watch_task.cancel()
        try:
            if timeout > 0 and self.process.running:
                async with timer_timeout(timer, ceil(timeout * 1000)):
                    await self.process.capture_children()
                    if self.session.provider_thread_id is not None:
                        await self.process.request(
                            "session/close",
                            {"sessionId": self.session.provider_thread_id},
                        )
                    await self.process.wait()
        except TimeoutError, Disconnected, ProtocolError, RpcError:
            pass
        finally:
            try:
                await self.process.aclose()
            finally:
                if self.watch_task is not None:
                    await asyncio.gather(self.watch_task, return_exceptions=True)


def attachment_block(attachment: HarnessAttachment) -> dict[str, Any]:
    path = attachment.path
    mime = (
        attachment.media_type or guess_file_type(path)[0] or "application/octet-stream"
    )
    data = path.read_bytes()
    if mime.startswith("image/"):
        return {
            "type": "image",
            "mimeType": mime,
            "data": base64.b64encode(data).decode(),
        }
    resource = {"uri": path.absolute().as_uri(), "mimeType": mime}
    if mime.startswith("text/"):
        resource["text"] = data.decode()
    else:
        resource["blob"] = base64.b64encode(data).decode()
    return {"type": "resource", "resource": resource}


def instruction_files(workspace: Path, env: dict[str, str]) -> tuple[Path, ...]:
    home = Path(env.get("HOME") or env.get("USERPROFILE") or Path.home()) / ".gemini"
    return tuple(
        path
        for root in (workspace, workspace / ".agents", home, home / "config")
        for name in ("AGENTS.md", "GEMINI.md")
        if (path := root / name).is_file()
    )


class AntigravityHarness:
    def __init__(
        self,
        input: HarnessInput,
        timer: Timer,
        audit: Audit,
        model: str | None,
        effort: str | None,
        api_key: str | None = None,
    ) -> None:
        self.input = input
        self.timer = timer
        self.audit = audit
        self.model = model
        self.effort = effort
        self.api_key = api_key
        self.sessions: dict[str, Connection] = {}
        self.lifecycle: asyncio.Queue[HarnessExpire] = asyncio.Queue()
        self.process_tasks: set[asyncio.Task[None]] = set()

    async def open(
        self,
        request: SessionInput,
        session: HarnessSession,
        usage: Usage,
        *,
        timeout: float,
    ) -> Connection:
        args = ["agy", "acp", "--process-per-session"]
        if request.sandbox_policy is SandboxPolicy.WORKSPACE_WRITE:
            args.append("--sandbox")
        elif request.sandbox_policy is SandboxPolicy.DANGER_FULL_ACCESS:
            args.append("--dangerously-skip-permissions")
        environment = os.environ | request.environment
        if self.api_key is not None:
            environment["GEMINI_API_KEY"] = self.api_key
        connection = Connection(
            request,
            session,
            usage,
            timer=self.timer,
            profile_dir=profile_root(environment),
        )
        try:
            async with timer_timeout(
                self.timer, ceil(min(timeout, self.input.startup_timeout) * 1000)
            ):
                source = Source(
                    self.audit,
                    AuditContext(
                        agent_id=self.input.agent_id,
                        harness="antigravity",
                        session_id=session.id,
                        provider_session_id=session.provider_thread_id,
                    ),
                )
                transport = connection.process = await AcpProcess.start(
                    args,
                    source,
                    connection.receive,
                    connection.answer,
                    connection.disconnected,
                    cwd=request.workspace,
                    env=environment,
                )
                self.process_tasks.add(transport.watcher)
                transport.watcher.add_done_callback(self.process_tasks.discard)
                initialized = await transport.request(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {},
                        "clientInfo": {
                            "name": "bazaar-contrib-antigravity",
                            "version": await asyncio.to_thread(
                                version, "bazaar-contrib-antigravity"
                            ),
                        },
                    },
                )
                if initialized["protocolVersion"] != 1:
                    raise ProtocolError("Antigravity did not negotiate ACP v1")
                params: dict[str, Any] = {
                    "cwd": str(request.workspace),
                    "mcpServers": [],
                }
                if request.instructions:
                    native = (
                        initialized["agentCapabilities"]
                        .get("_meta", {})
                        .get("antigravity", {})
                    )
                    if native.get("systemPrompt", {}).get("append") is not True:
                        raise ProtocolError(
                            "Antigravity cannot append BCN session instructions"
                        )
                    params["_meta"] = {
                        "antigravity": {
                            "systemPrompt": {
                                "append": "\n\n".join(request.instructions)
                            }
                        }
                    }
                if session.provider_thread_id is None:
                    configured = await transport.request("session/new", params)
                    session = HarnessSession(
                        id=session.id, provider_thread_id=configured["sessionId"]
                    )
                    connection.session = session
                    source.context = source.context.model_copy(
                        update={"provider_session_id": session.provider_thread_id}
                    )
                    # agy discovers globally configured MCP tools after session/new.
                    await self.timer.create(2_000).wait()
                else:
                    params["sessionId"] = session.provider_thread_id
                    configured = await transport.request("session/resume", params)
                for key, value in (
                    ("model", self.model),
                    ("thought_level", self.effort),
                ):
                    if value is None or current_option(configured, key) == value:
                        continue
                    configured = await transport.request(
                        "session/set_config_option",
                        {
                            "sessionId": session.provider_thread_id,
                            "configId": key,
                            "value": value,
                        },
                    )
                    if current_option(configured, key) != value:
                        raise ProtocolError(f"Antigravity did not select {key}={value}")
                connection.model = current_option(configured, "model")
            paths = await asyncio.to_thread(
                instruction_files, request.workspace, environment
            )
            if paths:
                connection.watch_task = asyncio.create_task(
                    self.watch_files(session.id, paths)
                )
        except BaseException:
            if hasattr(connection, "process"):
                await connection.process.aclose()
            raise
        return connection

    async def receive_event(self) -> HarnessExpire:
        return await self.lifecycle.get()

    async def watch_files(self, session_id: str, paths: tuple[Path, ...]) -> None:
        async for _ in awatch(*paths):
            self.lifecycle.put_nowait(HarnessExpire(session_id=session_id))

    async def start_session(
        self, request: SessionInput, *, timeout: float
    ) -> ProviderCallResult[HarnessSession]:
        previous = self.sessions.pop(request.id, None)
        if previous is not None:
            await previous.close(self.timer, timeout)
        try:
            connection = await self.open(
                request, HarnessSession(id=request.id), Usage(), timeout=timeout
            )
        except (
            Disconnected,
            ProtocolError,
            RpcError,
            TimeoutError,
            OSError,
            ValidationError,
            KeyError,
            TypeError,
        ) as exc:
            return ProviderCallResult(
                status=ProviderCallStatus.FAILED, failure=failure(exc)
            )
        self.sessions[request.id] = connection
        return ProviderCallResult(
            status=ProviderCallStatus.CONFIRMED, value=connection.session
        )

    async def reconcile_session(
        self,
        session: HarnessSession,
        turn: HarnessTurn | None,
        approval_handler: ApprovalHandler | None,
        *,
        timeout: float,
    ) -> ProviderCallResult[HarnessSessionReconciliation]:
        connection = self.sessions.get(session.id)
        if connection is None:
            return ProviderCallResult(
                status=ProviderCallStatus.FAILED,
                failure=HarnessFailure(
                    HarnessFailureReason.UNAVAILABLE,
                    "No Antigravity session owns this request",
                ),
            )
        active = connection.turn
        if turn is not None or (active is not None and not active.finished):
            if (
                turn is not None
                and active is not None
                and active.id == turn.id
                and not active.finished
                and connection.process.running
            ):
                return ProviderCallResult(
                    status=ProviderCallStatus.CONFIRMED,
                    value=HarnessSessionReconciliation(connection.session, active),
                )
            return ProviderCallResult(
                status=ProviderCallStatus.FAILED,
                failure=HarnessFailure(
                    HarnessFailureReason.STATE_MISMATCH,
                    "Antigravity cannot recover an unconfirmed turn after disconnecting",
                ),
            )
        if not connection.process.running or connection.broken:
            broken = connection.broken
            await connection.close(self.timer, timeout)
            try:
                connection = await self.open(
                    connection.request,
                    HarnessSession(id=connection.session.id)
                    if broken
                    else connection.session,
                    Usage() if broken else connection.usage,
                    timeout=timeout,
                )
            except (
                Disconnected,
                ProtocolError,
                RpcError,
                TimeoutError,
                OSError,
                ValidationError,
                KeyError,
                TypeError,
            ) as exc:
                return ProviderCallResult(
                    status=ProviderCallStatus.FAILED, failure=failure(exc)
                )
            self.sessions[session.id] = connection
        return ProviderCallResult(
            status=ProviderCallStatus.CONFIRMED,
            value=HarnessSessionReconciliation(connection.session),
        )

    async def start_turn(
        self,
        session: HarnessSession,
        turn: HarnessTurn,
        approval_handler: ApprovalHandler,
        *,
        timeout: float,
    ) -> HarnessTurnStream:
        connection = self.sessions.get(session.id)
        if connection is None or not connection.process.running:
            raise HarnessSessionUnavailable(
                "The Antigravity ACP session is unavailable"
            )
        if connection.broken:
            raise HarnessSessionUnavailable(
                "The Antigravity ACP session history is corrupted and must be reopened"
            )

        prompt: list[dict[str, Any]] = [{"type": "text", "text": turn.input_text}]
        stream = Turn(connection, turn, approval_handler, timeout)
        try:
            for attachment in turn.attachments:
                prompt.append(await asyncio.to_thread(attachment_block, attachment))
        except (OSError, UnicodeError) as error:
            stream.finish(
                TurnFailed(
                    event_name="attachment.read",
                    failure=HarnessFailure(HarnessFailureReason.EXECUTION, str(error)),
                )
            )
            return stream

        if connection.turn is not None and not connection.turn.finished:
            raise HarnessSessionUnavailable(
                "The Antigravity session already owns an active turn"
            )
        if connection.profile_dir is not None:
            stream.start_step_index = await asyncio.to_thread(
                read_last_step_index,
                connection.profile_dir,
                connection.session.provider_thread_id,
            )
        connection.turn = stream
        source = connection.process.source
        source.context = source.context.model_copy(update={"turn_id": turn.id})
        stream.emit(TurnStarted(event_name="session/prompt"), "session/prompt")
        if connection.model is not None:
            stream.emit(ModelUpdated(model=connection.model), "session/prompt")
        try:
            await connection.submit(stream, prompt)
        except NotSent as error:
            stream.finished = True
            source.context = source.context.model_copy(update={"turn_id": None})
            raise HarnessSessionUnavailable(str(error)) from error
        return stream

    async def steer_turn(
        self,
        session: HarnessSession,
        turn: HarnessTurn,
        input_text: str,
        *,
        timeout: float,
    ) -> bool:
        connection = self.sessions.get(session.id)
        if connection is None or not connection.process.running:
            return False
        stream = connection.turn
        if (
            stream is None
            or stream.id != turn.id
            or stream.finished
            or stream.cancel_requested
            or connection.process.incoming
        ):
            return False
        try:
            await connection.submit(stream, [{"type": "text", "text": input_text}])
        except NotSent:
            return False
        return True

    async def has_background_job(
        self, session: HarnessSession, *, timeout: float
    ) -> bool:
        connection = self.sessions.get(session.id)
        if connection is None:
            return False
        if (
            bool(connection.process.incoming)
            or (connection.turn is not None and not connection.turn.finished)
            or await connection.process.has_background_job()
        ):
            return True
        if connection.profile_dir is not None:
            return await asyncio.to_thread(
                has_live_disk_work,
                connection.profile_dir,
                connection.session.provider_thread_id,
                tuple(connection.subagent_calls),
            )
        return False

    async def stop_session(
        self, session: HarnessSession, *, timeout: float
    ) -> ProviderCallResult[HarnessSession]:
        connection = self.sessions.pop(session.id, None)
        if connection is not None:
            await connection.close(self.timer, timeout)
            session = connection.session
        return ProviderCallResult(status=ProviderCallStatus.CONFIRMED, value=session)

    async def aclose(self) -> None:
        connections = tuple(self.sessions.values())
        self.sessions.clear()
        await asyncio.gather(
            *(
                connection.close(self.timer, self.input.shutdown_timeout)
                for connection in connections
            )
        )
        await asyncio.gather(*self.process_tasks)


def failure(error: Exception) -> HarnessFailure:
    match error:
        case TimeoutError():
            reason = HarnessFailureReason.TIMEOUT
        case Disconnected() | OSError():
            reason = HarnessFailureReason.UNAVAILABLE
        case RpcError():
            reason = HarnessFailureReason.EXECUTION
        case _:
            reason = HarnessFailureReason.PROTOCOL
    return HarnessFailure(
        reason,
        str(error) or type(error).__name__,
        code=error.code if isinstance(error, RpcError) else None,
    )
