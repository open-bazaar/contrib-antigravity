"""Read-only inspection of Antigravity's native local session state.

Antigravity ACP hides several lifecycle facts from the JSON-RPC stream:
- Upstream API errors (429/404/5xx, quota exhaustion, invalid thinking
  signatures) can finish with ``stopReason="end_turn"`` while recording a
  ``CORTEX_STEP_TYPE_ERROR_MESSAGE`` (``step_type = 17``) row in the session's
  SQLite conversation database.
- Detached subagents launched by ``invoke_subagent`` report ``completed`` on
  the ACP tool call immediately at launch, while tracking their liveness under
  ``.system_generated/subagents/`` and ``.system_generated/messages/``.
- Queued high-priority wakeups sit in ``.system_generated/messages/`` or as a
  trailing ``SYSTEM_MESSAGE`` in ``.system_generated/logs/transcript.jsonl``
  before the model begins streaming its autonomous reply.

Every reader in this module is strictly read-only, rejects symlinks, enforces
size/count bounds, and fails open (returning ``None`` / ``False`` / ``()``) so
missing files or upstream schema changes never break the ACP turn loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections.abc import Collection, Mapping
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")

# Native trajectory step_type values from Antigravity's SQLite steps table.
_ERROR_STEP_TYPE = 17

# Bounds matching T3 Code's defensive limits, with a larger inbox cap for
# long-lived BCN sessions (idle_timeout=0).
_MAX_ERROR_PAYLOAD_BYTES = 16_384
_MAX_JSON_FILE_BYTES = 1024 * 1024
_MAX_DETACHED_CHILDREN = 64
_MAX_MESSAGE_FILES = 2_048
_TRANSCRIPT_TAIL_BYTES = 64 * 1024
_TRANSCRIPT_MAX_TAIL_BYTES = 256 * 1024

_HIGH_PRIORITY_MESSAGE = "MESSAGE_PRIORITY_HIGH"
_LOW_PRIORITY_MESSAGE = "MESSAGE_PRIORITY_LOW"
_KILLED_SUBAGENT_STATE = "SUBAGENT_STATE_KILLED"

_API_ERROR_RE = re.compile(
    r"Error (\d{3}), Message: (.+?), Status: ([A-Z_]+)", re.DOTALL
)
_STATUS_ERROR_RE = re.compile(
    r"(?<![A-Z_])([A-Z_]+) \(code (\d{3})\): ([^\x00-\x1f\x7f]+)"
)
_QUOTA_EXHAUSTED_RE = re.compile(r"\breason:QUOTA_EXHAUSTED\b")
_INVALID_THINKING_SIGNATURE_RE = re.compile(
    r"\bmessages\.\d+\.content\.\d+"
    r"(?:\.thinking\.signature: Field required|: Invalid `signature` in `thinking` block)"
)
_RESET_TIMESTAMP_RE = re.compile(r"\bquotaResetTimeStamp:([^\s\]]+)")
_RESET_DELAY_RE = re.compile(
    r"\bquotaResetDelay:((?:\d+(?:\.\d+)?(?:ns|us|µs|μs|ms|h|m|s))+)(?=[\s\]]|$)"
)
_RETRY_DELAY_RE = re.compile(
    r"\bretryDelay:((?:\d+(?:\.\d+)?(?:ns|us|µs|μs|ms|h|m|s))+)(?=[\s\]]|$)"
)


@dataclass(frozen=True, slots=True)
class StepErrorOutcome:
    """Classified error read from an Antigravity SQLite ``steps`` row."""

    message: str
    code: str | None = None
    session_broken: bool = False


def profile_root(env: Mapping[str, str] | None = None) -> Path:
    """Locate the native ``~/.gemini/antigravity-cli`` storage root."""
    source = env if env is not None else os.environ
    home = Path(source.get("HOME") or source.get("USERPROFILE") or Path.home())
    return home / ".gemini" / "antigravity-cli"


def _valid_session_id(session_id: str | None) -> str | None:
    if isinstance(session_id, str) and _SESSION_ID_RE.fullmatch(session_id):
        return session_id
    return None


def _managed_path(root: Path, *parts: str, want_dir: bool = False) -> Path | None:
    """Resolve ``root / parts`` while rejecting symlinks on managed segments."""
    try:
        if not root.exists() or root.is_symlink() or not root.is_dir():
            return None
        current = root
        for part in parts:
            if not part or part in {".", ".."} or "/" in part or "\\" in part:
                return None
            current = current / part
            if current.is_symlink():
                return None
        if want_dir:
            return current if current.is_dir() else None
        return current if current.is_file() else None
    except OSError:
        return None


def _read_json_file(path: Path) -> Any | None:
    try:
        if path.is_symlink():
            return None
        stat = path.stat()
        if not path.is_file() or stat.st_size > _MAX_JSON_FILE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        return None


def _open_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=0.1,
    )
    conn.execute("PRAGMA busy_timeout = 100")
    return conn


def _has_steps_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'steps'"
    ).fetchone()
    return row is not None


def read_last_step_index(root: Path, session_id: str | None) -> int | None:
    """Return current ``max(idx)`` in ``conversations/<session_id>.db``.

    Returns ``-1`` if the conversation DB does not exist yet (so any step
    created during the prompt is observed), or ``None`` if the storage root is
    unavailable or unreadable (fail-open).
    """
    valid_id = _valid_session_id(session_id)
    if valid_id is None:
        return None
    conversations = _managed_path(root, "conversations", want_dir=True)
    if conversations is None:
        return None
    candidate = conversations / f"{valid_id}.db"
    try:
        if candidate.is_symlink():
            return None
        if not candidate.exists():
            return -1
        if not candidate.is_file():
            return None
    except OSError:
        return None
    try:
        with closing(_open_readonly_sqlite(candidate)) as conn:
            if not _has_steps_table(conn):
                return None
            row = conn.execute("SELECT max(idx) FROM steps").fetchone()
            if row is None or not isinstance(row[0], int):
                return -1
            return row[0]
    except sqlite3.Error, OSError:
        return None


def _parse_api_error(text: str) -> tuple[str, str, str] | None:
    if (match := _API_ERROR_RE.search(text)) is not None:
        return match.group(1), match.group(2), match.group(3)
    if (match := _STATUS_ERROR_RE.search(text)) is not None:
        return match.group(2), match.group(3), match.group(1)
    return None


def _reset_hint(text: str) -> str | None:
    if (ts := _RESET_TIMESTAMP_RE.search(text)) is not None:
        return f"Resets at {ts.group(1)}."
    if (delay := _RESET_DELAY_RE.search(text)) is not None:
        return f"Resets in {delay.group(1)}."
    if (retry := _RETRY_DELAY_RE.search(text)) is not None:
        return f"Retry after {retry.group(1)}."
    return None


def classify_step_error(text: str) -> StepErrorOutcome:
    """Classify decoded text from a ``step_type = 17`` error payload."""
    if _INVALID_THINKING_SIGNATURE_RE.search(text) is not None:
        return StepErrorOutcome(
            message=(
                "Antigravity's saved conversation has an invalid thinking signature "
                "and can no longer be sent to the model. The next turn will start "
                "a fresh Antigravity session."
            ),
            code="INVALID_THINKING_SIGNATURE",
            session_broken=True,
        )

    api_error = _parse_api_error(text)
    quota_exhausted = _QUOTA_EXHAUSTED_RE.search(text) is not None

    if api_error is not None:
        http_status, raw_message, status = api_error
        message = raw_message.strip() or f"Antigravity API error ({http_status})."
        if (
            quota_exhausted
            and "reset" not in message.lower()
            and (hint := _reset_hint(text)) is not None
        ):
            message = f"{message.rstrip('.')} ({hint})"
        code = status or ("QUOTA_EXHAUSTED" if quota_exhausted else http_status)
        return StepErrorOutcome(message=message, code=code)

    if quota_exhausted:
        message = "You have reached your Antigravity quota for this period."
        if (hint := _reset_hint(text)) is not None:
            message = f"{message} {hint}"
        return StepErrorOutcome(message=message, code="QUOTA_EXHAUSTED")

    return StepErrorOutcome(
        message="Antigravity stopped this turn because of an error.",
        code=None,
    )


def read_step_error(
    root: Path,
    session_id: str | None,
    after_index: int | None,
) -> StepErrorOutcome | None:
    """Inspect the latest step added after ``after_index`` for a hidden error."""
    if after_index is None:
        return None
    valid_id = _valid_session_id(session_id)
    if valid_id is None:
        return None
    db_path = _managed_path(root, "conversations", f"{valid_id}.db", want_dir=False)
    if db_path is None:
        return None
    try:
        with closing(_open_readonly_sqlite(db_path)) as conn:
            if not _has_steps_table(conn):
                return None
            row = conn.execute(
                "SELECT step_type, substr(step_payload, 1, ?) "
                "FROM steps WHERE idx > ? ORDER BY idx DESC LIMIT 1",
                (_MAX_ERROR_PAYLOAD_BYTES, after_index),
            ).fetchone()
    except sqlite3.Error, OSError:
        return None
    if row is None or row[0] != _ERROR_STEP_TYPE:
        return None
    payload = row[1]
    if isinstance(payload, bytes | bytearray | memoryview):
        text = bytes(payload).decode("utf-8", errors="replace")
    elif isinstance(payload, str):
        text = payload
    else:
        text = ""
    return classify_step_error(text)


def _inbox_entries(root: Path, conversation_id: str) -> tuple[Path, list[str]] | None:
    inbox = _managed_path(
        root,
        "brain",
        conversation_id,
        ".system_generated",
        "messages",
        want_dir=True,
    )
    if inbox is None:
        return None
    try:
        names = [
            name
            for name in os.listdir(inbox)
            if name.endswith(".json") and name != "read.json"
        ]
    except OSError:
        return None
    if len(names) > _MAX_MESSAGE_FILES:
        return None
    return inbox, names


def _read_inbox_messages(root: Path, conversation_id: str) -> list[dict[str, Any]]:
    listing = _inbox_entries(root, conversation_id)
    if listing is None:
        return []
    inbox, names = listing
    messages: list[dict[str, Any]] = []
    for name in names:
        parsed = _read_json_file(inbox / name)
        if isinstance(parsed, dict):
            messages.append(parsed)
    return messages


def _detached_launch_call_id(
    message: Mapping[str, Any], root_session_id: str
) -> str | None:
    if message.get("sender") != root_session_id:
        return None
    metadata = message.get("sourceMetadata")
    if not isinstance(metadata, Mapping):
        return None
    tool = metadata.get("tool")
    if not isinstance(tool, Mapping):
        return None
    conv_id = tool.get("conversationId")
    if conv_id is not None and conv_id != root_session_id:
        return None
    call = tool.get("toolCall")
    if not isinstance(call, Mapping) or call.get("name") != "invoke_subagent":
        return None
    call_id = call.get("id")
    return call_id if isinstance(call_id, str) and call_id else None


def active_subagent_calls(
    root: Path,
    session_id: str | None,
    tool_call_ids: Collection[str],
) -> tuple[str, ...]:
    """Return launching ``invoke_subagent`` tool call IDs whose child is still alive.

    A detached child is considered done once it posts any message back to the
    parent session's inbox or its descriptor records ``SUBAGENT_STATE_KILLED``.
    """
    if not tool_call_ids:
        return ()
    valid_id = _valid_session_id(session_id)
    if valid_id is None:
        return ()
    subagents_dir = _managed_path(
        root, "brain", valid_id, ".system_generated", "subagents", want_dir=True
    )
    if subagents_dir is None:
        return ()
    try:
        child_ids = [
            name[:-5]
            for name in os.listdir(subagents_dir)
            if name.endswith(".json") and _SESSION_ID_RE.fullmatch(name[:-5])
        ]
    except OSError:
        return ()
    if not child_ids or len(child_ids) > _MAX_DETACHED_CHILDREN:
        return ()

    wanted = set(tool_call_ids)
    reported = {
        sender
        for msg in _read_inbox_messages(root, valid_id)
        if isinstance(sender := msg.get("sender"), str)
    }

    active: list[str] = []
    for child_id in child_ids:
        if child_id in reported:
            continue
        descriptor = _read_json_file(subagents_dir / f"{child_id}.json")
        if not isinstance(descriptor, dict):
            continue
        recorded_id = descriptor.get("conversationId")
        if recorded_id is not None and recorded_id != child_id:
            continue
        if descriptor.get("state") == _KILLED_SUBAGENT_STATE:
            continue
        for msg in _read_inbox_messages(root, child_id):
            launch_id = _detached_launch_call_id(msg, valid_id)
            if launch_id is not None and launch_id in wanted:
                active.append(launch_id)
                break
    return tuple(active)


def _has_unread_high_priority_message(root: Path, session_id: str) -> bool:
    listing = _inbox_entries(root, session_id)
    if listing is None:
        return False
    inbox, names = listing
    if not names:
        return False

    receipts: dict[str, bool] = {}
    read_path = inbox / "read.json"
    if read_path.exists():
        raw_receipts = _read_json_file(read_path)
        if not isinstance(raw_receipts, dict):
            return False
        receipts = {k: v for k, v in raw_receipts.items() if isinstance(v, bool)}

    for name in names:
        file_id = name[:-5]
        if receipts.get(file_id) is True:
            continue
        parsed = _read_json_file(inbox / name)
        if not isinstance(parsed, dict):
            continue
        msg_id = parsed.get("id")
        if isinstance(msg_id, str) and receipts.get(msg_id) is True:
            continue
        if parsed.get("priority") == _HIGH_PRIORITY_MESSAGE:
            return True
    return False


def _read_last_transcript_step(root: Path, session_id: str) -> dict[str, Any] | None:
    transcript = _managed_path(
        root,
        "brain",
        session_id,
        ".system_generated",
        "logs",
        "transcript.jsonl",
        want_dir=False,
    )
    if transcript is None:
        return None
    try:
        size = transcript.stat().st_size
        if size <= 0:
            return None
        with transcript.open("rb") as handle:
            for cap in (_TRANSCRIPT_TAIL_BYTES, _TRANSCRIPT_MAX_TAIL_BYTES):
                start = max(0, size - cap)
                handle.seek(start)
                data = handle.read(size - start).rstrip(b"\r\n")
                if not data:
                    if start == 0:
                        return None
                    continue
                newline = data.rfind(b"\n")
                if newline == -1 and start > 0:
                    continue
                line = data[newline + 1 :].decode("utf-8", errors="strict")
                parsed = json.loads(line)
                return parsed if isinstance(parsed, dict) else None
    except OSError, UnicodeDecodeError, json.JSONDecodeError:
        return None
    return None


def has_pending_reply(root: Path, session_id: str | None) -> bool:
    """Return True if Antigravity has a queued high-priority wake or pending reply."""
    valid_id = _valid_session_id(session_id)
    if valid_id is None:
        return False
    if _has_unread_high_priority_message(root, valid_id):
        return True
    last_step = _read_last_transcript_step(root, valid_id)
    if last_step is None or last_step.get("type") != "SYSTEM_MESSAGE":
        return False
    content = last_step.get("content")
    if not isinstance(content, str):
        return True
    return (
        f"priority={_HIGH_PRIORITY_MESSAGE}" in content
        or f"priority={_LOW_PRIORITY_MESSAGE}" not in content
    )


def has_live_disk_work(
    root: Path,
    session_id: str | None,
    subagent_call_ids: Collection[str],
) -> bool:
    """Combined check for active detached subagents or queued wakes on disk."""
    with suppress(Exception):
        if active_subagent_calls(root, session_id, subagent_call_ids):
            return True
        if has_pending_reply(root, session_id):
            return True
    return False
