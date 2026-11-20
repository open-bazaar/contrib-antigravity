"""Carry ACP requests, responses, and notifications over the CLI's stdio."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import psutil
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .source import Source

if sys.platform == "win32":
    from .windows_job import CREATE_SUSPENDED, WindowsJob

    WINDOWS_COMMAND_PREFIX = (
        "try { $OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "[Console]::OutputEncoding = $OutputEncoding } catch {}; "
    )


def children(pid: int, *, recursive: bool = True) -> list[psutil.Process]:
    with suppress(psutil.NoSuchProcess):
        return psutil.Process(pid).children(recursive=recursive)
    return []


def worker_children(pid: int) -> Iterator[psutil.Process]:
    for worker in children(pid, recursive=False):
        with suppress(psutil.NoSuchProcess):
            if worker.cmdline()[1:3] == ["acp", "--worker"]:
                yield from children(worker.pid, recursive=False)


if sys.platform != "win32":

    def background_jobs(pid: int) -> bool:
        # agy's ACP worker starts terminal commands in their own process groups.
        # The worker and its stdio MCP servers inherit the ACP process group.
        for child in worker_children(pid):
            with suppress(psutil.NoSuchProcess, ProcessLookupError):
                if (
                    os.getpgid(child.pid) != pid
                    and child.status() != psutil.STATUS_ZOMBIE
                ):
                    return True
        return False

    def kill_group(pid: int) -> None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Darwin reports EPERM for a group with only zombie members.
            if sys.platform != "darwin":
                raise
            for process in psutil.process_iter(("pid", "status")):
                with suppress(ProcessLookupError):
                    if (
                        process.info["status"] != psutil.STATUS_ZOMBIE
                        and os.getpgid(process.pid) == pid
                    ):
                        raise
else:

    def live_windows_commands(pid: int, commands: set[str]) -> set[str]:
        # Match the command agy reported against its PowerShell child. A stdio
        # MCP server can also use PowerShell and the same UTF-8 initialization.
        launched = {WINDOWS_COMMAND_PREFIX + command: command for command in commands}
        live: set[str] = set()
        for child in worker_children(pid):
            with suppress(psutil.NoSuchProcess):
                arguments = child.cmdline()
                if (
                    len(arguments) == 3
                    and arguments[1] == "-Command"
                    and (command := launched.get(arguments[2])) is not None
                ):
                    live.add(command)
        return live


class RpcError(RuntimeError):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


class Disconnected(RuntimeError):
    pass


class NotSent(Disconnected):
    pass


class ProtocolError(RuntimeError):
    pass


class Error(BaseModel):
    message: str
    code: int


class Packet(BaseModel):
    model_config = ConfigDict(extra="allow")
    jsonrpc: Literal["2.0"]
    method: str | None = None
    id: int | str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: Error | None = None

    def value(self) -> dict[str, Any]:
        if self.error is not None:
            raise RpcError(self.error.message, self.error.code)
        return self.result or {}


async def lines(stream: asyncio.StreamReader) -> AsyncIterator[str]:
    fragments: list[bytes] = []
    while True:
        try:
            fragment = await stream.readuntil(b"\n")
        except asyncio.LimitOverrunError as error:
            fragments.append(await stream.readexactly(error.consumed))
            continue
        except asyncio.IncompleteReadError as error:
            if fragments or error.partial:
                yield b"".join([*fragments, error.partial]).decode(errors="replace")
            return
        yield (
            b"".join([*fragments, fragment]).decode(errors="replace").removesuffix("\n")
        )
        fragments.clear()


class AcpProcess:
    @classmethod
    async def start(
        cls,
        command: Sequence[str],
        source: Source,
        receive: Callable[[Packet], None],
        answer: Callable[[Packet], Awaitable[dict[str, Any]]],
        disconnected: Callable[[Disconnected | ProtocolError], None],
        *,
        cwd: Path,
        env: dict[str, str],
    ) -> AcpProcess:
        job = None
        options: dict[str, Any]
        if sys.platform == "win32":
            job = WindowsJob()
            options = {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED
            }
        else:
            options = {"start_new_session": True}
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **options,
            )
            if job is not None:
                try:
                    job.assign(process.pid)
                    psutil.Process(process.pid).resume()
                except OSError, psutil.Error:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()
                    raise
        except BaseException:
            if job is not None:
                job.close()
            raise
        return cls(
            process,
            source,
            receive,
            answer,
            disconnected,
            job.close if job is not None else None,
        )

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        source: Source,
        receive: Callable[[Packet], None],
        answer: Callable[[Packet], Awaitable[dict[str, Any]]],
        disconnected: Callable[[Disconnected | ProtocolError], None],
        close_job: Callable[[], None] | None,
    ) -> None:
        self.process = process
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        self.stdin = process.stdin
        self.source = source
        self.receive = receive
        self.answer = answer
        self.disconnected = disconnected
        self.failure: Disconnected | ProtocolError | None = None
        self.sequence = 0
        self.waiting: dict[int, asyncio.Future[Packet]] = {}
        self.incoming: dict[int | str, asyncio.Task[None]] = {}
        self.close_job = close_job
        self.descendants: set[psutil.Process] = set()
        if sys.platform == "win32":
            self.windows_commands: dict[str, str] = {}
        self.cleanup: asyncio.Task[None] | None = None
        self.reader = asyncio.create_task(self.read(process.stdout))
        self.stderr = asyncio.create_task(self.read_stderr(process.stderr))
        self.watcher = asyncio.create_task(self.watch())

    @property
    def running(self) -> bool:
        return (
            self.failure is None
            and self.process.returncode is None
            and not self.stdin.is_closing()
        )

    def request_id(self) -> int:
        self.sequence += 1
        return self.sequence

    async def send(self, message: dict[str, Any]) -> None:
        if not self.running:
            raise NotSent(
                "The Antigravity ACP process is unavailable"
            ) from self.failure
        data = json.dumps({"jsonrpc": "2.0", **message})
        try:
            self.stdin.write(data.encode() + b"\n")
            await self.source.packet(data, "sent")
            await self.stdin.drain()
        except OSError as error:
            raise Disconnected("The Antigravity ACP input pipe closed") from error

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        identifier = self.request_id()
        response = asyncio.get_running_loop().create_future()
        self.waiting[identifier] = response
        try:
            await self.send({"id": identifier, "method": method, "params": params})
            return (await response).value()
        finally:
            del self.waiting[identifier]
            if response.done() and not response.cancelled():
                response.exception()
            else:
                response.cancel()

    async def read(self, stdout: asyncio.StreamReader) -> None:
        failure: Disconnected | ProtocolError = Disconnected(
            "The Antigravity ACP output pipe closed"
        )
        try:
            async for line in lines(stdout):
                await self.source.packet(line, "received")
                packet = Packet.model_validate_json(line)
                if packet.method is not None and packet.id is not None:
                    self.incoming[packet.id] = asyncio.create_task(self.respond(packet))
                elif packet.method == "$/cancel_request":
                    identifier = packet.params["requestId"]
                    task = self.incoming.get(identifier)
                    if task is not None:
                        task.cancel()
                elif (
                    packet.method is None
                    and isinstance(packet.id, int)
                    and packet.id in self.waiting
                ):
                    response = self.waiting[packet.id]
                    if not response.done():
                        response.set_result(packet)
                else:
                    self.receive(packet)
        except (ValidationError, KeyError, TypeError, ValueError) as error:
            failure = ProtocolError(f"Invalid Antigravity ACP message: {error}")
        except OSError as error:
            failure = Disconnected(str(error))
        finally:
            self.failure = failure
            for response in self.waiting.values():
                if not response.done():
                    response.set_exception(failure)
            for task in self.incoming.values():
                task.cancel()
            self.disconnected(failure)

    async def respond(self, packet: Packet) -> None:
        assert packet.id is not None
        try:
            try:
                result = {"result": await self.answer(packet)}
            except RpcError as error:
                result = {"error": {"code": error.code, "message": str(error)}}
            except (ValidationError, KeyError, TypeError, ValueError) as error:
                result = {"error": {"code": -32602, "message": str(error)}}
            except (RuntimeError, OSError) as error:
                result = {"error": {"code": -32603, "message": str(error)}}
            if self.running:
                await self.send({"id": packet.id, **result})
        except Disconnected:
            pass
        finally:
            self.incoming.pop(packet.id, None)

    async def read_stderr(self, stderr: asyncio.StreamReader) -> None:
        async for line in lines(stderr):
            await self.source.stderr(line)

    async def wait(self) -> None:
        self.stdin.close()
        await self.process.wait()

    async def watch(self) -> None:
        await self.process.wait()
        await self.aclose()
        await self.source.close(self.process.returncode)

    async def capture_children(self) -> None:
        if sys.platform != "win32" and self.process.returncode is None:
            self.descendants.update(await asyncio.to_thread(children, self.process.pid))

    def observe_tool(self, update: dict[str, Any]) -> None:
        if sys.platform == "win32" and update.get("name") == "run_command":
            call = update["toolCallId"]
            match update.get("status"):
                case "completed" | "failed":
                    self.windows_commands.pop(call, None)
                case "in_progress":
                    if command := (update.get("rawInput") or {}).get("CommandLine"):
                        self.windows_commands[call] = command

    async def has_background_job(self) -> bool:
        if not self.running:
            return False
        if sys.platform == "win32":
            commands = self.windows_commands.copy()
            if not commands:
                return False
            live = await asyncio.to_thread(
                live_windows_commands, self.process.pid, set(commands.values())
            )
            for call, command in commands.items():
                if command not in live:
                    self.windows_commands.pop(call, None)
            return bool(live)
        return await asyncio.to_thread(background_jobs, self.process.pid)

    async def aclose(self) -> None:
        if self.cleanup is None:
            self.cleanup = asyncio.create_task(self.close_process())
        await asyncio.shield(self.cleanup)

    async def close_process(self) -> None:
        await self.capture_children()
        self.stdin.close()
        requests = tuple(self.incoming.values())
        for task in requests:
            task.cancel()
        if sys.platform == "win32":
            assert self.close_job is not None
            self.close_job()
        else:
            await asyncio.to_thread(kill_group, self.process.pid)

            def kill_children() -> None:
                for child in self.descendants:
                    with suppress(psutil.NoSuchProcess):
                        child.kill()

            await asyncio.to_thread(kill_children)
        await self.process.wait()
        await asyncio.gather(
            self.reader, self.stderr, *requests, return_exceptions=True
        )
