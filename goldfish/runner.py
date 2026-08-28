"""Episode runner.

One design decision worth flagging, because it determines whether the
compaction-generation result is real: compaction MUTATES the persistent history
rather than being applied as a view at call time. That is what Claude Code,
Codex CLI, Aider and OpenHands actually do, and it is why loss compounds when
you summarize a summary. A harness that recomputes the reduction from the full
transcript each turn would quietly make generation-compounding unmeasurable.

A separate append-only `transcript` is kept for bookkeeping only. It is never
shown to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .env import LedgerEnv
from .models import Action, ModelAdapter
from .probes import Outcome, Probe, ProbeClass, score_behavioural, default_probe_suite
from .strategies import Strategy, approx_tokens

SYSTEM = (
    "You are an accounts-payable agent operating a ledger through tools. "
    "Work through the pending queue. Call exactly one tool per turn. "
    "When a question is asked directly, answer it from what you know and say so "
    "plainly if you no longer have the information."
)

SCRATCHPAD_HINT = (
    " Your context window is bounded and older turns may be evicted. "
    "write_note/read_note are a durable scratchpad that survives eviction: "
    "write down anything you might need much later, such as minted account "
    "codes or policy facts, as soon as you learn it."
)

OPENING = (
    "Close out this ledger period. Requirements:\n"
    "- Post every pending transaction to the correct vendor account.\n"
    "- The closing report must be settled in USD.\n"
    "- Never post to a vendor on the frozen list. Check the posting policy.\n"
    "- File the closing report when the queue is clear."
)

# What must appear in the raw transcript for a probe's fact to count as planted.
PLANT_MARKERS = {
    "ident-1": lambda blob, env: env.vendor_to_code.get("vendor-00", "\0") in blob,
    "constraint-1": lambda blob, env: "frozen_vendors" in blob,
    "negative-1": lambda blob, env: "decommissioned" in blob,
    "goal-1": lambda blob, env: "USD" in blob,
    "artifact-1": lambda blob, env: '"new_balance"' in blob,
    "prov-1": lambda blob, env: "intake_vendor(" in blob and env.vendor_to_code.get("vendor-00", "\0") in blob,
}


@dataclass
class EpisodeResult:
    strategy: str
    model: str
    seed: int
    turns: int
    probes: list[Probe]
    behavioural: Any
    usage: dict[str, int]
    approx_context_peak: int
    compaction_generations: int
    redundant_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model": self.model,
            "seed": self.seed,
            "turns": self.turns,
            "compaction_generations": self.compaction_generations,
            "redundant_calls": self.redundant_calls,
            "approx_context_peak": self.approx_context_peak,
            "usage": self.usage,
            "behavioural": vars(self.behavioural),
            "probes": [
                {"id": p.id, "class": p.cls.value, "distance": p.distance, "outcome": p.outcome.value, "answer": p.answer}
                for p in self.probes
            ],
        }


def run_episode(
    strategy: Strategy,
    model: ModelAdapter,
    seed: int = 0,
    budget: int = 1200,
    max_turns: int = 60,
    probes: list[Probe] | None = None,
) -> EpisodeResult:
    env = LedgerEnv(seed=seed)
    probes = probes or default_probe_suite()
    system = SYSTEM
    needed = set(getattr(strategy, "needs_tools", ()))
    if {"write_note", "read_note"} & needed:
        system += SCRATCHPAD_HINT
    history: list[dict[str, Any]] = [{"role": "user", "kind": "goal", "content": OPENING}]
    transcript: list[str] = [OPENING]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    peak = 0
    seen_calls: set[str] = set()
    redundant = 0

    for turn in range(max_turns):
        blob = "\n".join(transcript)
        for p in probes:
            if p.planted_turn is None and PLANT_MARKERS[p.id](blob, env):
                p.planted_turn = turn

        due = [p for p in probes if p.test_turn == turn and p.asked_turn is None]
        if due:
            p = due[0]
            history.append({"role": "user", "kind": "probe", "content": p.question})
            transcript.append(p.question)
            history = strategy.reduce(history, budget)
            action = model.act(system, history, env.tool_schema())
            _accumulate(usage, action.usage)
            p.asked_turn = turn
            p.grade(action.content, env)
            history.append({"role": "assistant", "kind": "answer", "content": action.content})
            transcript.append(action.content)
            peak = max(peak, approx_tokens(history))
            continue

        history = strategy.reduce(history, budget)
        peak = max(peak, approx_tokens(history))
        action = model.act(system, history, env.tool_schema())
        _accumulate(usage, action.usage)

        if action.kind == "text":
            history.append({"role": "assistant", "kind": "answer", "content": action.content})
            transcript.append(action.content)
            continue

        sig = f"{action.name}:{json.dumps(action.args, sort_keys=True)}"
        if sig in seen_calls:
            redundant += 1
        seen_calls.add(sig)

        result = env.call(action.name, action.args)
        call_txt = f"{action.name}({json.dumps(action.args, sort_keys=True)})"
        obs_txt = result.render()
        history.append({"role": "assistant", "kind": "action", "content": call_txt})
        history.append({"role": "user", "kind": "observation", "tool": action.name, "content": obs_txt})
        transcript.extend([call_txt, obs_txt])

        if env.report_filed is not None:
            # Keep going only if probes remain, so late probes still get asked.
            if all(p.asked_turn is not None for p in probes):
                break

    return EpisodeResult(
        strategy=getattr(strategy, "name", type(strategy).__name__),
        model=model.name,
        seed=seed,
        turns=turn + 1,
        probes=probes,
        behavioural=score_behavioural(env),
        usage=usage,
        approx_context_peak=peak,
        compaction_generations=getattr(strategy, "generation", 0),
        redundant_calls=redundant,
    )


def _accumulate(total: dict[str, int], delta: dict[str, int]) -> None:
    for k, v in (delta or {}).items():
        total[k] = total.get(k, 0) + v
