"""Model adapters.

Two implementations:

  ContextBoundAgent  a deterministic simulator that decides using ONLY the
                     messages it is handed. It is not a model and proves
                     nothing about models. Its job is to validate the
                     instrument: if a strategy evicts a fact, this agent
                     provably cannot use that fact, so any probe that still
                     scores as recalled indicates a leak in the harness.

  AnthropicAdapter   the real thing, for the actual study.

Every reported run must record which adapter and which pinned model version
produced it. A result from the simulator is a plumbing test, not a finding.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

# Loads ANTHROPIC_API_KEY (and friends) from a .env file in the working
# directory if present. No-op, not an error, if the file or the key is
# absent — ContextBoundAgent and the offline path never touch this.
load_dotenv()

Message = dict[str, Any]


@dataclass
class Action:
    kind: str  # "tool" | "text"
    name: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    def act(self, system: str, messages: list[Message], tools: list[dict]) -> Action: ...


# --------------------------------------------------------------------- offline

_INTAKE = re.compile(r'"code":\s*"(AC-[A-Z0-9]{4}-\d{4})".*?"vendor":\s*"(vendor-\d\d)"')
_INTAKE_ALT = re.compile(r'"vendor":\s*"(vendor-\d\d)".*?"code":\s*"(AC-[A-Z0-9]{4}-\d{4})"')
# Success payloads only. The *call* text also contains tx_id, and counting
# those would credit the agent for failed attempts.
_POSTED = re.compile(r'"new_balance":\s*[\d.]+,\s*"tx_id":\s*"(TX-\d{3})"')
_FROZEN = re.compile(r'"frozen_vendors":\s*\[([^\]]*)\]')
_PENDING = re.compile(r'"id":\s*"(TX-\d{3})",\s*"vendor":\s*"(vendor-\d\d)"')
_AMOUNT = re.compile(r'"amount":\s*([\d.]+),\s*"id":\s*"(TX-\d{3})"')


@dataclass
class ContextBoundAgent:
    """A competent agent with no memory beyond its visible context."""

    seed: int = 0
    hallucination_rate: float = 0.5
    name: str = "context_bound_sim"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- what the agent can currently see -------------------------------------

    def _view(self, messages: list[Message]) -> dict[str, Any]:
        blob = "\n".join(str(m.get("content", "")) for m in messages)
        codes: dict[str, str] = {}
        for code, vendor in _INTAKE.findall(blob):
            codes[vendor] = code
        for vendor, code in _INTAKE_ALT.findall(blob):
            codes[vendor] = code
        pending: dict[str, str] = dict(_PENDING.findall(blob))
        amounts: dict[str, float] = {tx: float(a) for a, tx in _AMOUNT.findall(blob)}
        frozen_m = _FROZEN.search(blob)
        frozen = set(re.findall(r"vendor-\d\d", frozen_m.group(1))) if frozen_m else None
        return {
            "blob": blob,
            "codes": codes,
            "pending": pending,
            "amounts": amounts,
            "posted": set(_POSTED.findall(blob)),
            "frozen": frozen,
            "saw_policy": frozen_m is not None,
            "saw_pending": bool(pending),
            "wire_dead": "decommissioned" in blob,
        }

    # -- probe answering ------------------------------------------------------

    def _answer_probe(self, q: str, v: dict[str, Any]) -> str:
        found: str | None = None
        # Order matters: the provenance question's text contains the substring
        # "account code", so it must be checked before the identifier branch
        # or it gets misrouted and silently graded wrong. Caught by
        # test_no_leak_under_full_history.
        if "tool call produced" in q:
            found = "intake_vendor" if "intake_vendor" in v["blob"] and v["codes"].get("vendor-00") else None
        elif "account code" in q:
            found = v["codes"].get("vendor-00")
        elif "frozen list" in q:
            found = " ".join(sorted(v["frozen"])) if v["frozen"] else None
        elif "settlement method" in q:
            found = "wire" if v["wire_dead"] else None
        elif "currency" in q:
            found = "USD" if "USD" in v["blob"] else None
        elif "successfully posted" in q:
            found = str(len(v["posted"])) if v["saw_pending"] or v["posted"] else None
        if found:
            return found
        # Nothing in context. Either admit it or confidently invent something,
        # which is the failure mode the three-valued outcome exists to catch.
        if self._rng.random() < self.hallucination_rate:
            return {
                "account code": "AC-0000-1000",
                "frozen list": "vendor-01 vendor-07",
                "settlement method": "ach",
                "currency": "EUR",
                "successfully posted": "0",
                "tool call produced": "policy_lookup",
            }.get(
                next(
                    (
                        k
                        for k in (
                            "tool call produced",  # checked first: substring of "account code" question
                            "account code",
                            "frozen list",
                            "settlement method",
                            "currency",
                            "successfully posted",
                        )
                        if k in q
                    ),
                    "",
                ),
                "unclear",
            )
        return "I do not recall that from earlier in this session."

    # -- policy ---------------------------------------------------------------

    def act(self, system: str, messages: list[Message], tools: list[dict]) -> Action:
        v = self._view(messages)
        last = str(messages[-1].get("content", "")) if messages else ""

        if messages and messages[-1].get("kind") == "probe":
            return Action("text", content=self._answer_probe(last, v))

        if not v["saw_policy"]:
            return Action("tool", name="policy_lookup")
        if not v["saw_pending"]:
            return Action("tool", name="list_pending")

        todo = [
            (tx, vendor)
            for tx, vendor in sorted(v["pending"].items())
            if tx not in v["posted"] and not (v["frozen"] and vendor in v["frozen"])
        ]
        if not todo:
            total = round(sum(v["amounts"].get(tx, 0.0) for tx in v["posted"]), 2)
            return Action("tool", name="file_report", args={"total_posted": total, "accounts_touched": len(v["codes"])})

        tx, vendor = todo[0]
        if vendor not in v["codes"]:
            return Action("tool", name="intake_vendor", args={"vendor": vendor})
        method = "ach" if v["wire_dead"] else "wire"
        return Action(
            "tool",
            name="post_transaction",
            args={"code": v["codes"][vendor], "tx_id": tx, "amount": v["amounts"].get(tx, 0.0), "method": method},
        )


# ---------------------------------------------------------------------- online


@dataclass
class AnthropicAdapter:
    """Real runs. Pin the version string in every report; an unpinned model
    name makes a result unreproducible within weeks."""

    model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    cache: bool = True
    name: str = "anthropic"
    max_retries: int = 6
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_client(self):
        import anthropic  # imported lazily so the offline path has no dependency

        # Built once per adapter instance and reused across every turn of an
        # episode (and across seeds, since a strategy/model pair is typically
        # reused), rather than reconnecting on every call. max_retries is
        # explicit rather than left at the SDK default: a full seed sweep is
        # dozens of consecutive calls, and 429s from bursty request rates are
        # routine, not exceptional, at that volume.
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=self.max_retries)
        return self._client

    def act(self, system: str, messages: list[Message], tools: list[dict]) -> Action:
        client = self._get_client()
        sys_block: Any = [{"type": "text", "text": system}]
        if self.cache:
            sys_block[0]["cache_control"] = {"type": "ephemeral"}
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=sys_block,
            messages=[{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages],
            tools=[
                {
                    "name": t["name"],
                    "description": t["doc"],
                    "input_schema": {
                        "type": "object",
                        "properties": {a: {"type": "string"} for a in t["args"]},
                        "required": t["args"],
                    },
                }
                for t in tools
            ],
        )
        usage = {
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_write": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        }
        for block in resp.content:
            if block.type == "tool_use":
                return Action("tool", name=block.name, args=dict(block.input), usage=usage)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Action("text", content=text, usage=usage)
