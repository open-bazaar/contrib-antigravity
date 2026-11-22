"""Tests for Antigravity account quota reading (`QuotaReader`)."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import pytest
from bazaar_contrib_audit import AUDIT, audit
from bazaar_contrib_harness import (
    HarnessInput,
    QuotaReader,
    QuotaSnapshot,
    QuotaWindow,
)
from bazaar_contrib_timer import TIMER, Timer
from bazaar_core import (
    Composition,
    Context,
    Contrib,
    ContribEntry,
    ContribHub,
    ContribSpec,
    load,
)

from bazaar_contrib_antigravity.quota import parse_usage_report, read_account_quota
from bazaar_contrib_antigravity.runtime import AntigravityHarness


def _epoch_ms(iso_utc: str) -> int:
    return int(datetime.fromisoformat(iso_utc).astimezone(UTC).timestamp() * 1000)


SAMPLE_USAGE_PAYLOAD: dict[str, Any] = {
    "conversation_id": "",
    "status": "SUCCESS",
    "response": (
        "Gemini Models\tWeekly Limit Remaining\t80%\t2026-11-29T04:24:56Z\n"
        "Gemini Models\tFive Hour Limit Remaining\t100%\t2026-11-22T19:24:56Z\n"
        "Claude and GPT models\tWeekly Limit Remaining\t100%\t2026-11-29T14:25:23Z\n"
        "Claude and GPT models\tFive Hour Limit Remaining\t100%\t2026-11-22T19:25:23Z\n"
    ),
    "duration_seconds": 0,
    "num_turns": 0,
    "usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "cache_read_tokens": 0,
        "total_tokens": 0,
    },
    "command": {
        "name": "usage",
        "data": {
            "description": "Within each group, models share a weekly limit and a 5-hour limit.",
            "groups": [
                {
                    "name": "Gemini Models",
                    "description": "Models within this group: Gemini 4 Neon, ...",
                    "buckets": [
                        {
                            "id": "gemini-weekly",
                            "name": "Weekly Limit Remaining",
                            "description": "You have used some of your weekly limit.",
                            "window": "weekly",
                            "remaining_fraction": 0.8043148517608643,
                            "reset_time": "2026-11-29T04:24:56Z",
                        },
                        {
                            "id": "gemini-5h",
                            "name": "Five Hour Limit Remaining",
                            "description": "You have used some of your 5-hour limit.",
                            "window": "5h",
                            "remaining_fraction": 0.9995733499526978,
                            "reset_time": "2026-11-22T19:24:56Z",
                        },
                    ],
                },
                {
                    "name": "Claude and GPT models",
                    "description": "Models within this group: Claude Opus 5.5, ...",
                    "buckets": [
                        {
                            "id": "3p-weekly",
                            "name": "Weekly Limit Remaining",
                            "window": "weekly",
                            "remaining_fraction": 1,
                            "reset_time": "2026-11-29T14:25:23Z",
                        },
                        {
                            "id": "3p-5h",
                            "name": "Five Hour Limit Remaining",
                            "window": "5h",
                            "remaining_fraction": 1,
                            "reset_time": "2026-11-22T19:25:23Z",
                        },
                    ],
                },
            ],
        },
    },
}


def test_parse_usage_report_maps_groups_orders_windows_and_drops_untouched_resets() -> (
    None
):
    observed = _epoch_ms("2026-11-22T14:25:23Z")
    snapshot = parse_usage_report(
        json.dumps(SAMPLE_USAGE_PAYLOAD), observed_at_ms=observed
    )
    assert snapshot is not None
    assert snapshot.observed_at_ms == observed
    assert snapshot.windows == (
        QuotaWindow(
            name="Gemini Models",
            duration_minutes=300,
            remaining_pct=0.9995733499526978 * 100.0,
            resets_at_ms=_epoch_ms("2026-11-22T19:24:56Z"),
        ),
        QuotaWindow(
            name="Gemini Models",
            duration_minutes=10080,
            remaining_pct=0.8043148517608643 * 100.0,
            resets_at_ms=_epoch_ms("2026-11-29T04:24:56Z"),
        ),
        QuotaWindow(
            name="Claude and GPT models",
            duration_minutes=300,
            remaining_pct=100.0,
            resets_at_ms=None,
        ),
        QuotaWindow(
            name="Claude and GPT models",
            duration_minutes=10080,
            remaining_pct=100.0,
            resets_at_ms=None,
        ),
    )


def test_parse_usage_report_supports_flat_buckets_clamping_and_filtering() -> None:
    observed = _epoch_ms("2026-11-22T14:00:00Z")
    payload = {
        "status": "SUCCESS",
        "command": {
            "name": "quota",
            "data": {
                "buckets": [
                    {
                        "id": "disabled-5h",
                        "window": "5h",
                        "remaining_fraction": 0.5,
                        "disabled": True,
                    },
                    {
                        "id": "missing-fraction",
                        "window": "5h",
                    },
                    {
                        "id": "gemini-daily",
                        "name": "Daily Limit",
                        "remaining_fraction": -0.25,
                        "reset_time": "2026-11-23T14:00:00Z",
                    },
                    {
                        "id": "burst-30m",
                        "window": "30m",
                        "description": "Active burst window",
                        "remaining_fraction": 1.2,
                        "reset_time": "2026-11-22T14:30:00Z",
                    },
                    {
                        "id": "custom",
                        "name": "Custom Pool",
                        "window": "unknown",
                        "remaining_fraction": 0.42,
                        "reset_time": "2026-11-22T18:00:00Z",
                    },
                ]
            },
        },
    }
    snapshot = parse_usage_report(json.dumps(payload), observed_at_ms=observed)
    assert snapshot is not None
    assert snapshot.windows == (
        QuotaWindow(
            name="burst-30m",
            duration_minutes=30,
            remaining_pct=100.0,
            resets_at_ms=_epoch_ms("2026-11-22T14:30:00Z"),
        ),
        QuotaWindow(
            name="Daily Limit",
            duration_minutes=1440,
            remaining_pct=0.0,
            resets_at_ms=_epoch_ms("2026-11-23T14:00:00Z"),
        ),
        QuotaWindow(
            name="Custom Pool",
            duration_minutes=None,
            remaining_pct=0.42 * 100.0,
            resets_at_ms=_epoch_ms("2026-11-22T18:00:00Z"),
        ),
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        json.dumps({"status": "ERROR", "error": "Eligibility check failed"}),
        json.dumps({"status": "SUCCESS", "command": {"name": "model", "data": {}}}),
        json.dumps({"status": "SUCCESS", "command": {"name": "usage", "data": {}}}),
        json.dumps(
            {
                "status": "SUCCESS",
                "command": {
                    "name": "usage",
                    "data": {
                        "groups": [
                            {
                                "name": "Gemini Models",
                                "buckets": [{"id": "a", "disabled": True}],
                            }
                        ]
                    },
                },
            }
        ),
    ],
)
def test_parse_usage_report_returns_none_for_unusable_payloads(raw: str) -> None:
    assert parse_usage_report(raw, observed_at_ms=1_000) is None


async def _with_timer_and_audit(
    action: Any,
) -> None:
    timer_spec = await asyncio.to_thread(load, (ContribSpec(contrib="timer-wheel"),))

    @asynccontextmanager
    async def runner(ctx: Context):
        await action(ctx.accept(TIMER), ctx.accept(AUDIT))
        yield

    async with (
        ContribHub() as hub,
        hub.enable(
            Composition(
                entries=(
                    ContribEntry(name="audit", definition=audit),
                    *timer_spec.entries,
                    ContribEntry(
                        name="runner",
                        definition=Contrib(accepts=[TIMER, AUDIT], lifespan=runner),
                    ),
                ),
                diagnostics=timer_spec.diagnostics,
            )
        ),
    ):
        pass


async def test_harness_implements_quota_reader_and_skips_when_api_key_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    async def fake_read_account_quota(*args: Any, **kwargs: Any) -> QuotaSnapshot:
        nonlocal called
        called = True
        return QuotaSnapshot(
            observed_at_ms=123,
            windows=(QuotaWindow(name="Gemini Models", remaining_pct=88.0),),
        )

    monkeypatch.setattr(
        "bazaar_contrib_antigravity.runtime.read_account_quota",
        fake_read_account_quota,
    )

    async def exercise(timer: Timer, audit_svc: Any) -> None:
        harness_input = HarnessInput(
            agent_id="agent-1",
            agent_name="test-agent",
            workspace=tmp_path,
            startup_timeout=10.0,
            shutdown_timeout=5.0,
        )
        keyed = AntigravityHarness(
            harness_input, timer, audit_svc, None, None, api_key="secret"
        )
        assert isinstance(keyed, QuotaReader)
        assert await keyed.read_quota() is None
        assert not called

        oauth = AntigravityHarness(harness_input, timer, audit_svc, None, None)
        assert isinstance(oauth, QuotaReader)
        snapshot = await oauth.read_quota()
        assert called
        assert snapshot is not None
        assert snapshot.windows[0].remaining_pct == 88.0

    await _with_timer_and_audit(exercise)


async def test_read_account_quota_subprocess_success_and_failures(
    tmp_path: Path,
) -> None:
    async def exercise(timer: Timer, _audit: Any) -> None:
        ok_script = f"import json; print(json.dumps({SAMPLE_USAGE_PAYLOAD!r}))"
        snapshot = await read_account_quota(
            tmp_path,
            timer=timer,
            timeout=10.0,
            command=(sys.executable, "-c", ok_script),
        )
        assert snapshot is not None
        assert len(snapshot.windows) == 4
        assert snapshot.windows[0].name == "Gemini Models"
        assert snapshot.windows[0].duration_minutes == 300

        assert (
            await read_account_quota(
                tmp_path,
                timer=timer,
                timeout=10.0,
                command=(sys.executable, "-c", "import sys; sys.exit(2)"),
            )
            is None
        )
        assert (
            await read_account_quota(
                tmp_path,
                timer=timer,
                timeout=10.0,
                command=("nonexistent-agy-binary-for-quota-test",),
            )
            is None
        )

    await _with_timer_and_audit(exercise)


async def test_read_account_quota_timeout_cleans_up_child_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = f"""
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
time.sleep(60)
"""

    async def exercise(timer: Timer, _audit: Any) -> None:
        result = await read_account_quota(
            tmp_path,
            timer=timer,
            timeout=0.4,
            command=(sys.executable, "-c", script),
        )
        assert result is None
        if await asyncio.to_thread(pid_file.exists):
            child_pid = int((await asyncio.to_thread(pid_file.read_text)).strip())
            try:
                proc = await asyncio.to_thread(psutil.Process, child_pid)
                await asyncio.to_thread(proc.wait, 2.0)
                assert not await asyncio.to_thread(proc.is_running)
            except psutil.NoSuchProcess:
                pass

    await _with_timer_and_audit(exercise)


async def test_read_account_quota_cancellation_reaps_process(tmp_path: Path) -> None:
    started = tmp_path / "started"
    script = f"""
import pathlib, time
pathlib.Path({str(started)!r}).write_text("1")
time.sleep(60)
"""

    async def exercise(timer: Timer, _audit: Any) -> None:
        task = asyncio.create_task(
            read_account_quota(
                tmp_path,
                timer=timer,
                timeout=30.0,
                command=(sys.executable, "-c", script),
            )
        )
        for _ in range(50):
            if await asyncio.to_thread(started.exists):
                break
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await _with_timer_and_audit(exercise)
