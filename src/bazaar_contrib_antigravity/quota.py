"""Read Antigravity account quota via non-interactive `agy -p /usage`."""

# The timeout budget comes from HarnessInput.startup_timeout.
# ruff: noqa: ASYNC109

from __future__ import annotations

import asyncio
import math
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from math import ceil
from pathlib import Path
from time import time_ns
from typing import Any

import psutil
from bazaar_contrib_harness import QuotaSnapshot, QuotaWindow
from bazaar_contrib_timer import Timer
from bazaar_contrib_timer import timeout as timer_timeout
from pydantic import AwareDatetime, BaseModel, ValidationError

if sys.platform == "win32":
    from .windows_job import CREATE_SUSPENDED, WindowsJob
else:
    from .transport import kill_group

_UNTOUCHED_RESET_TOLERANCE_MS = 60_000
_WINDOW_UNITS: dict[str, int] = {
    "m": 1,
    "h": 60,
    "d": 24 * 60,
    "w": 7 * 24 * 60,
}
_WINDOW_RE = re.compile(r"^(\d+)([mhdw])$")


class UsageBucket(BaseModel):
    id: str | None = None
    name: str | None = None
    description: str | None = None
    window: str | None = None
    remaining_fraction: float | None = None
    reset_time: AwareDatetime | None = None
    disabled: bool = False


class UsageGroup(BaseModel):
    name: str | None = None
    description: str | None = None
    buckets: tuple[UsageBucket, ...] = ()


class UsageData(BaseModel):
    groups: tuple[UsageGroup, ...] = ()
    buckets: tuple[UsageBucket, ...] = ()


class UsageCommand(BaseModel):
    name: str
    data: UsageData | None = None


class UsageReport(BaseModel):
    status: str
    command: UsageCommand | None = None


def _parse_window_minutes(value: str | None) -> int | None:
    if not value:
        return None
    token = value.strip().lower()
    if token == "weekly":
        return 7 * 24 * 60
    if token == "daily":
        return 24 * 60
    if match := _WINDOW_RE.fullmatch(token):
        amount = int(match.group(1))
        return amount * _WINDOW_UNITS[match.group(2)] if amount > 0 else None
    return None


def _bucket_duration_minutes(bucket: UsageBucket) -> int | None:
    duration = _parse_window_minutes(bucket.window)
    if duration is not None or not bucket.id:
        return duration
    return _parse_window_minutes(bucket.id.rsplit("-", 1)[-1])


def _bucket_resets_at_ms(
    bucket: UsageBucket,
    duration_minutes: int | None,
    observed_at_ms: int,
) -> int | None:
    if bucket.reset_time is None:
        return None
    resets_at_ms = int(bucket.reset_time.timestamp() * 1000)
    # Untouched rolling windows report remaining_fraction=1, no usage
    # description, and reset_time ~= observed_at + duration.
    if (
        bucket.remaining_fraction is not None
        and bucket.remaining_fraction >= 1.0
        and not bucket.description
        and duration_minutes is not None
        and abs(resets_at_ms - observed_at_ms - duration_minutes * 60_000)
        <= _UNTOUCHED_RESET_TOLERANCE_MS
    ):
        return None
    return resets_at_ms


def _clean_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def parse_usage_report(
    raw: str | bytes,
    *,
    observed_at_ms: int,
) -> QuotaSnapshot | None:
    """Parse `agy -p /usage --output-format json` stdout into a QuotaSnapshot."""
    try:
        report = UsageReport.model_validate_json(raw)
    except ValidationError:
        return None
    if (
        report.status != "SUCCESS"
        or report.command is None
        or report.command.name not in {"usage", "quota"}
        or report.command.data is None
    ):
        return None

    data = report.command.data
    groups = data.groups
    if not groups and data.buckets:
        groups = (UsageGroup(buckets=data.buckets),)

    windows: list[QuotaWindow] = []
    for group in groups:
        group_name = _clean_label(group.name)
        group_windows: list[QuotaWindow] = []
        for bucket in group.buckets:
            if bucket.disabled:
                continue
            fraction = bucket.remaining_fraction
            if fraction is None or not math.isfinite(fraction):
                continue
            duration_minutes = _bucket_duration_minutes(bucket)
            remaining_pct = max(0.0, min(1.0, fraction)) * 100.0
            name = group_name or _clean_label(bucket.name) or _clean_label(bucket.id)
            group_windows.append(
                QuotaWindow(
                    name=name,
                    duration_minutes=duration_minutes,
                    remaining_pct=remaining_pct,
                    resets_at_ms=_bucket_resets_at_ms(
                        bucket, duration_minutes, observed_at_ms
                    ),
                )
            )
        group_windows.sort(
            key=lambda window: (
                window.duration_minutes is None,
                window.duration_minutes or 0,
            )
        )
        windows.extend(group_windows)

    return (
        QuotaSnapshot(observed_at_ms=observed_at_ms, windows=tuple(windows))
        if windows
        else None
    )


async def read_account_quota(
    workspace: Path,
    *,
    timer: Timer,
    timeout: float,
    command: Sequence[str] = ("agy", "-p", "/usage", "--output-format", "json"),
    env: dict[str, str] | None = None,
) -> QuotaSnapshot | None:
    """Query Antigravity account quota without opening a conversation."""
    job = None
    options: dict[str, Any]
    if sys.platform == "win32":
        job = WindowsJob()
        options = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
        }
    else:
        options = {"start_new_session": True}

    process: asyncio.subprocess.Process | None = None
    try:
        async with timer_timeout(timer, ceil(timeout * 1000)):
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workspace,
                env=dict(os.environ) if env is None else env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **options,
            )
            if sys.platform == "win32" and job is not None:
                job.assign(process.pid)
                await asyncio.to_thread(
                    lambda pid=process.pid: psutil.Process(pid).resume()
                )
            stdout, _ = await process.communicate()
            observed_at_ms = time_ns() // 1_000_000
    except OSError, TimeoutError, psutil.Error:
        return None
    finally:
        if process is not None and process.returncode is None:
            if sys.platform == "win32":
                if job is not None:
                    with suppress(OSError):
                        job.close()
                    job = None
                with suppress(ProcessLookupError):
                    process.kill()
            else:
                with suppress(OSError):
                    await asyncio.to_thread(kill_group, process.pid)
                with suppress(ProcessLookupError):
                    process.kill()
            with suppress(OSError):
                await process.wait()
        if job is not None:
            with suppress(OSError):
                job.close()

    if process is None or process.returncode != 0 or not stdout.strip():
        return None
    return parse_usage_report(stdout, observed_at_ms=observed_at_ms)
