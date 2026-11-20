"""Project native Antigravity session updates into Harness events."""

from __future__ import annotations

from typing import Any

from bazaar_contrib_harness import (
    CommandDetail,
    ContentDelta,
    ContentDeltaKind,
    HarnessEventPayload,
    McpDetail,
    ModelUpdated,
    OtherDetail,
    TokenUsage,
    ToolCall,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    UsageUpdated,
)

_USAGE_FIELDS = {
    "inputTokens": "input_tokens",
    "cachedReadTokens": "cached_input_tokens",
    "cachedWriteTokens": "cache_write_input_tokens",
    "outputTokens": "output_tokens",
    "thoughtTokens": "reasoning_output_tokens",
    "totalTokens": "total_tokens",
}


class Usage:
    def __init__(self) -> None:
        self.tokens: dict[str, int] = {}
        self.context_window: int | None = None

    def account(self, usage: dict[str, int]) -> UsageUpdated:
        counts = {
            key: usage[name] for name, key in _USAGE_FIELDS.items() if name in usage
        }
        for key, count in counts.items():
            self.tokens[key] = self.tokens.get(key, 0) + count
        return self.event(last=TokenUsage(**counts))

    def event(self, *, last: TokenUsage | None = None) -> UsageUpdated:
        return UsageUpdated(
            total=TokenUsage(**self.tokens),
            last=last,
            model_context_window=self.context_window,
        )


class TurnEvents:
    def __init__(self, usage: Usage) -> None:
        self.usage = usage
        self.tools: dict[str, dict[str, Any]] = {}

    def update(self, data: dict[str, Any]) -> tuple[HarnessEventPayload, ...]:
        match data["sessionUpdate"]:
            case "agent_message_chunk" | "agent_thought_chunk":
                content = data["content"]
                if content["type"] == "text":
                    kind = (
                        ContentDeltaKind.AGENT_MESSAGE
                        if data["sessionUpdate"] == "agent_message_chunk"
                        else ContentDeltaKind.REASONING_TEXT
                    )
                    return (ContentDelta(kind=kind, text=content["text"]),)
            case "plan":
                return (
                    ContentDelta(
                        kind=ContentDeltaKind.PLAN,
                        text="\n".join(entry["content"] for entry in data["entries"]),
                    ),
                )
            case "tool_call" | "tool_call_update":
                return self.tool(data)
            case "usage_update":
                self.usage.context_window = data["size"]
                return (self.usage.event(),)
            case "config_option_update":
                model = current_option(data, "model")
                if model is not None:
                    return (ModelUpdated(model=model),)
        return ()

    def tool(self, update: dict[str, Any]) -> tuple[HarnessEventPayload, ...]:
        call_id = update["toolCallId"]
        first = call_id not in self.tools
        record = self.tools.setdefault(call_id, {})
        record.update(update)
        name = record.get("name") or record.get("title") or record.get("kind") or "tool"
        arguments = record.get("rawInput")
        output = record.get("rawOutput")
        if name == "call_mcp_tool" and isinstance(arguments, dict):
            detail = McpDetail(
                server=arguments["ServerName"],
                tool=arguments["ToolName"],
                arguments=arguments["Arguments"],
                result=output if isinstance(output, str) else None,
            )
        elif (
            record.get("kind") == "execute"
            and isinstance(arguments, dict)
            and isinstance(arguments.get("CommandLine"), str)
        ):
            result = output if isinstance(output, dict) else {}
            detail = CommandDetail(
                command=arguments["CommandLine"],
                cwd=arguments.get("Cwd"),
                description=arguments.get("toolSummary"),
                exit_code=result.get("exit_code"),
                output=result.get("output"),
            )
        else:
            detail = OtherDetail(name=name, input=arguments, output=output)
        call = ToolCall(call_id=call_id, name=name, detail=detail)
        events: list[HarnessEventPayload] = []
        if first:
            events.append(ToolCallStarted(call=call))
        match update.get("status"):
            case "completed":
                events.append(ToolCallCompleted(call=call))
            case "failed":
                error = output.get("error") if isinstance(output, dict) else None
                events.append(
                    ToolCallFailed(
                        call=call,
                        error_message=error if isinstance(error, str) else None,
                    )
                )
        return tuple(events)


def current_option(packet: dict[str, Any], identifier: str) -> str | None:
    return next(
        (
            entry.get("currentValue")
            for entry in packet.get("configOptions", ())
            if entry["id"] == identifier
        ),
        None,
    )
