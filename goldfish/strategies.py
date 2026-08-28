"""Compaction strategies.

The whole reuse surface of this project is one method:

    reduce(messages, budget) -> messages

If someone can implement that, they can measure their own compaction logic with
the same instrument. Keep it that way. Any strategy that needs to change the
agent loop itself (scratchpad, structured notes) declares it via `needs_tools`
rather than reaching into the runner.

Token counts here are approximate (chars // 4). That is fine for eviction
decisions and NOT fine for cost reporting, which reads real usage off the API
response. Do not conflate the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

Message = dict[str, Any]


def approx_tokens(msgs: list[Message]) -> int:
    return sum(len(str(m.get("content", ""))) for m in msgs) // 4


class Strategy(Protocol):
    name: str
    needs_tools: list[str]

    def reduce(self, messages: list[Message], budget: int) -> list[Message]: ...


@dataclass
class FullHistory:
    """Control. Never evicts. The cost baseline that prompt caching makes
    interesting: this is the only strategy with a stable cache prefix."""

    name: str = "full_history"
    needs_tools: list[str] = field(default_factory=list)

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        return messages


@dataclass
class SlidingWindow:
    """Keep the most recent messages that fit. Oldest-first eviction, nothing
    preserved. Cheap, cache friendly at the suffix, and structurally unable to
    hold a fact planted before the window."""

    name: str = "sliding_window"
    needs_tools: list[str] = field(default_factory=list)

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        kept: list[Message] = []
        used = 0
        for m in reversed(messages):
            cost = len(str(m.get("content", ""))) // 4
            if used + cost > budget and kept:
                break
            kept.append(m)
            used += cost
        return list(reversed(kept))


@dataclass
class ToolOutputMasking:
    """Keep every turn, drop old tool payloads. Preserves the shape of the
    trajectory (what was called, in what order) while discarding the bulk.
    Often the strongest cost/fidelity trade in coding scaffolds, and usually
    left out of comparisons."""

    name: str = "tool_masking"
    keep_recent: int = 6
    min_payload: int = 120  # small observations are cheap; masking them buys nothing
    needs_tools: list[str] = field(default_factory=list)

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        # Mask only under pressure. Masking unconditionally made this strategy
        # differ from the control even at an unbounded budget, which confounded
        # every comparison against it. Caught by test_control_invariance.
        if approx_tokens(messages) <= budget:
            return messages
        out: list[Message] = []
        n = len(messages)
        for i, m in enumerate(messages):
            big = len(str(m.get("content", ""))) >= self.min_payload
            if m.get("kind") == "observation" and big and i < n - self.keep_recent:
                stub = f"[masked observation from {m.get('tool', 'tool')}, {len(str(m['content']))} chars]"
                out.append({**m, "content": stub})
            else:
                out.append(m)
        if approx_tokens(out) <= budget:
            return out
        return SlidingWindow().reduce(out, budget)


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9\-]+", s.lower())


@dataclass
class RetrievalOverTurns:
    """Evicted turns go to a store and are retrieved against the current query.

    Deliberately uses lexical retrieval, not embeddings. Embedding choice is a
    second free variable, and one free variable per run is the rule. Swap it in
    as a separate arm if you want to measure it.
    """

    name: str = "retrieval"
    top_k: int = 4
    needs_tools: list[str] = field(default_factory=list)

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        if approx_tokens(messages) <= budget:
            return messages
        query = _tokenize(str(messages[-1].get("content", "")))
        head_budget = int(budget * 0.7)
        kept = SlidingWindow().reduce(messages, head_budget)
        kept_ids = {id(m) for m in kept}
        evicted = [m for m in messages if id(m) not in kept_ids]
        scored = []
        for m in evicted:
            toks = set(_tokenize(str(m.get("content", ""))))
            score = sum(1 for q in query if q in toks)
            if score:
                scored.append((score, m))
        scored.sort(key=lambda x: -x[0])
        recalled = [m for _, m in scored[: self.top_k]]
        if not recalled:
            return kept
        blob = "\n".join(f"- {str(m.get('content'))[:400]}" for m in recalled)
        note = {"role": "user", "kind": "retrieved", "content": f"[retrieved from earlier in this session]\n{blob}"}
        return [note] + kept


@dataclass
class ExternalScratchpad:
    """Agent maintains a durable file via write_note/read_note rather than
    relying on the transcript. Eviction still happens oldest-first like
    SlidingWindow, but any write_note/read_note call and its observation are
    pinned so they survive eviction regardless of age. The bet under test is
    axis B (where evicted content goes), not axis A: whether an agent that is
    told to externalize facts before they age out actually outperforms one
    that has nowhere to put them.

    Declares needs_tools so the runner tells the agent this channel is
    durable; a strategy-blind agent has no reason to reach for it.
    """

    name: str = "scratchpad"
    keep_recent: int = 6
    needs_tools: list[str] = field(default_factory=lambda: ["write_note", "read_note"])

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        if approx_tokens(messages) <= budget:
            return messages
        pinned: set[int] = set()
        for i, m in enumerate(messages):
            if m.get("kind") == "observation" and m.get("tool") in ("write_note", "read_note"):
                pinned.add(i)
                if i > 0:
                    pinned.add(i - 1)  # the call that produced this observation
        pinned_tokens = approx_tokens([m for i, m in enumerate(messages) if i in pinned])
        rest = [m for i, m in enumerate(messages) if i not in pinned]
        kept_rest = SlidingWindow().reduce(rest, max(0, budget - pinned_tokens))
        kept_rest_ids = {id(m) for m in kept_rest}
        return [m for i, m in enumerate(messages) if i in pinned or id(m) in kept_rest_ids]


Summarizer = Callable[[list[Message]], str]


def extractive_summarizer(msgs: list[Message]) -> str:
    """Offline default so the harness runs without API keys.

    This is a stand-in for the plumbing, NOT the thing under test. Real runs
    must pass an LLM summarizer; an extractive baseline does not hallucinate,
    and hallucination during compaction is one of the effects being measured.
    """
    lines = []
    for m in msgs:
        c = str(m.get("content", ""))
        lines.append(c if len(c) <= 160 else c[:160] + "...")
    return "Summary of earlier turns:\n" + "\n".join(f"- {l}" for l in lines[-40:])


@dataclass
class Summarization:
    """The industry default: six of eight surveyed frameworks use this.

    `generation` counts how many times this instance has summarized. Fidelity
    as a function of generation is a headline result, so the counter is part of
    the strategy rather than bookkeeping in the runner.
    """

    summarizer: Summarizer = extractive_summarizer
    keep_recent: int = 6
    name: str = "summarization"
    needs_tools: list[str] = field(default_factory=list)
    generation: int = 0

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        if approx_tokens(messages) <= budget:
            return messages
        head, tail = messages[: -self.keep_recent], messages[-self.keep_recent :]
        if not head:
            return messages
        self.generation += 1
        summary = self.summarizer(head)
        node = {
            "role": "user",
            "kind": "summary",
            "generation": self.generation,
            "content": summary,
        }
        out = [node] + tail
        if approx_tokens(out) > budget:
            out = SlidingWindow().reduce(out, budget)
        return out


REGISTRY: dict[str, Callable[[], Strategy]] = {
    "full_history": FullHistory,
    "sliding_window": SlidingWindow,
    "tool_masking": ToolOutputMasking,
    "retrieval": RetrievalOverTurns,
    "summarization": Summarization,
    "scratchpad": ExternalScratchpad,
}
