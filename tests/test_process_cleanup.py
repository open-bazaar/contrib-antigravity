"""Exercise ACP process ownership through real child process lifecycles."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, cast

import psutil
import pytest
from bazaar_contrib_audit import AUDIT, AUDIT_EVENT, AuditContext, AuditEvent, audit
from bazaar_core import Composition, Context, Contrib, ContribEntry, ContribHub

from bazaar_contrib_antigravity.source import Source
from bazaar_contrib_antigravity.transport import AcpProcess


def command(descendant: str) -> tuple[str, ...]:
    leaf = "import threading; threading.Event().wait()"
    group = (
        "{'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}"
        " if os.name == 'nt' else {'start_new_session': True}"
    )
    launch = f"""
import json, os, subprocess, sys

leaf = subprocess.Popen(
    [sys.executable, "-c", {leaf!r}],
    stdin=subprocess.DEVNULL,
    **({group if descendant == "detached" else "{}"}),
)
print(json.dumps({{"jsonrpc": "2.0", "method": "leaf", "params": {{"pid": leaf.pid}}}}), flush=True)
"""
    if descendant == "grandchild":
        launch = f"""
import json, subprocess, sys
subprocess.run([sys.executable, "-c", {launch!r}], check=True)
"""
    return (
        sys.executable,
        "-c",
        launch
        + """
print(json.dumps({"jsonrpc": "2.0", "method": "ready"}), flush=True)
if line := sys.stdin.readline():
    request = json.loads(line)
    if request["method"] == "session/close":
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {}}), flush=True)
    else:
        sys.exit(7)
""",
    )


@pytest.mark.parametrize("descendant", ("child", "grandchild", "detached"))
@pytest.mark.parametrize("ending", ("exit", "stop", "zero", "cancel"))
async def test_process_tree(tmp_path: Path, descendant: str, ending: str) -> None:
    if descendant == "detached" and ending == "exit" and os.name != "nt":
        pytest.skip("The exiting root has already lost its separately grouped child")
    closed: list[dict[str, Any]] = []
    ready = asyncio.Event()
    leaf: psutil.Process

    async def record(ctx: Context, event: AuditEvent) -> None:
        nonlocal leaf
        if event.event_name == "harness.source.received":
            payload = cast(dict[str, Any], event.payload)
            packet = json.loads(payload["data"])
            if packet.get("method") == "leaf":
                leaf = await asyncio.to_thread(psutil.Process, packet["params"]["pid"])
            elif packet.get("method") == "ready":
                ready.set()
        elif event.event_name == "harness.source.closed":
            closed.append(cast(dict[str, Any], event.payload))

    @asynccontextmanager
    async def exercise(ctx: Context):
        async def answer(packet):
            raise AssertionError(packet)

        transport = await AcpProcess.start(
            command(descendant),
            Source(
                ctx.accept(AUDIT),
                AuditContext(agent_id="cleanup", harness="antigravity"),
            ),
            lambda packet: None,
            answer,
            lambda error: None,
            cwd=tmp_path,
            env=dict(os.environ),
        )
        try:
            await ready.wait()
            if ending == "exit":
                await transport.send({"method": "exit"})
                await transport.watcher
            elif ending == "stop":
                await transport.capture_children()
                await transport.request("session/close", {})
                await transport.wait()
            elif ending == "cancel":
                task = asyncio.create_task(transport.aclose())
                asyncio.get_running_loop().call_soon(task.cancel)
                with pytest.raises(asyncio.CancelledError):
                    await task
            await transport.aclose()
            await transport.watcher
            await asyncio.to_thread(leaf.wait)
            assert not await asyncio.to_thread(leaf.is_running)
            assert len(closed) == 1
            assert closed[0]["sequence"] == {"exit": 3, "stop": 4}.get(ending, 2)
            if ending in {"exit", "stop"}:
                assert closed[0]["returncode"] == (7 if ending == "exit" else 0)
            yield
        finally:
            await transport.aclose()
            if ready.is_set():
                with suppress(psutil.NoSuchProcess):
                    await asyncio.to_thread(leaf.kill)

    async with (
        ContribHub() as hub,
        hub.enable(
            Composition(
                entries=(
                    ContribEntry(name="audit", definition=audit),
                    ContribEntry(
                        name="record",
                        definition=Contrib(listen={AUDIT_EVENT: record}),
                    ),
                    ContribEntry(
                        name="run",
                        definition=Contrib(accepts=[AUDIT], lifespan=exercise),
                    ),
                )
            )
        ),
    ):
        pass
