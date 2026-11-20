"""Record the native ACP stream through the BCN Audit service."""

from __future__ import annotations

from time import monotonic_ns, time_ns
from typing import Literal
from uuid import uuid7

from bazaar_contrib_audit import Audit, AuditContext
from bazaar_contrib_harness import HarnessSourceEvent


class Source:
    def __init__(self, audit: Audit, context: AuditContext) -> None:
        self.audit = audit
        self.context = context
        self.id = str(uuid7())
        self.sequence = 0

    async def packet(self, data: str, direction: Literal["sent", "received"]) -> None:
        timestamp = time_ns() // 1_000_000
        self.sequence += 1
        event = HarnessSourceEvent(
            provider="antigravity",
            source_id=self.id,
            sequence=self.sequence,
            direction=direction,
            data=data,
            occurred_at_ms=timestamp,
            monotonic_ns=monotonic_ns(),
        )
        await self.audit.publish(
            f"harness.source.{direction}",
            context=self.context,
            payload=event.model_dump(mode="json"),
            occurred_at_ms=timestamp,
        )

    async def stderr(self, data: str) -> None:
        await self.audit.publish(
            "harness.source.stderr",
            context=self.context,
            payload={"provider": "antigravity", "source_id": self.id, "data": data},
        )

    async def close(self, returncode: int | None) -> None:
        await self.audit.publish(
            "harness.source.closed",
            context=self.context,
            payload={
                "provider": "antigravity",
                "source_id": self.id,
                "sequence": self.sequence,
                "returncode": returncode,
            },
        )
