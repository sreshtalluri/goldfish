"""Probes: the measurement instrument.

A probe is a piece of information of a known class, introduced into the
trajectory at turn k and tested at turn k+n. Two kinds:

  interrogative  an injected user turn with a programmatically checkable answer
  behavioural    checked against environment state at the end, never asked

Outcome is three-valued, not binary. The distinction between an agent that says
"I no longer have that" and one that confidently invents a replacement is the
most operationally important thing this benchmark measures, and a pass/fail
score erases it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ProbeClass(str, Enum):
    GOAL = "goal"
    CONSTRAINT = "constraint"
    NEGATIVE = "negative_knowledge"
    ARTIFACT = "artifact_state"
    IDENTIFIER = "identifier"
    PROVENANCE = "provenance"


class Outcome(str, Enum):
    RECALLED = "recalled"          # correct
    ADMITTED = "admitted"          # wrong, but the agent said it did not know
    HALLUCINATED = "hallucinated"  # wrong, stated confidently
    NOT_TESTED = "not_tested"


_ADMISSION = re.compile(
    r"\b(do not recall|don't recall|do not remember|don't remember|not sure|"
    r"unsure|no longer have|cannot find|can't find|unknown|i don't know|"
    r"i do not know|not in my context|lost)\b",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _matches(truth_norm: str, answer_norm: str) -> bool:
    """Plain substring containment is unsafe for short numeric truths: "0" is
    a substring of "10", "20", "104"... which would silently grade a wrong
    numeric answer as recalled. Real models answer artifact/count probes in
    free text ("I've posted 10 transactions"), where this fires constantly.

    Word identifiers (account codes, vendor names) still use containment,
    since separators are already stripped by _norm and there is no digit-glue
    risk. Purely numeric truths require the match to sit on digit boundaries.
    """
    if not truth_norm:
        return False
    if truth_norm.isdigit():
        return re.search(rf"(?<!\d){re.escape(truth_norm)}(?!\d)", answer_norm) is not None
    return truth_norm in answer_norm


@dataclass
class Probe:
    id: str
    cls: ProbeClass
    question: str
    # Resolved lazily against the environment, because ground truth for things
    # like a minted account code does not exist until the agent mints it.
    expected: Callable[[Any], str]
    test_turn: int
    planted_turn: int | None = None
    asked_turn: int | None = None
    answer: str | None = None
    outcome: Outcome = Outcome.NOT_TESTED

    @property
    def distance(self) -> int | None:
        if self.planted_turn is None or self.asked_turn is None:
            return None
        if self.asked_turn < self.planted_turn:
            # Asked before the dependency existed (e.g. artifact_state asked
            # before any transaction was posted, so "0" is trivially the
            # current state rather than a fact recalled from earlier). This
            # is not a short distance, it is not a valid test of recall at
            # all, and treating it as a small positive distance would
            # understate real decay. Surfaced by a real distance sweep where
            # an early test_turn landed before a late-planting probe.
            return None
        return self.asked_turn - self.planted_turn

    def grade(self, answer: str, env) -> Outcome:
        self.answer = answer
        truth = self.expected(env)
        if _matches(_norm(truth), _norm(answer)):
            self.outcome = Outcome.RECALLED
        elif _ADMISSION.search(answer):
            self.outcome = Outcome.ADMITTED
        else:
            self.outcome = Outcome.HALLUCINATED
        return self.outcome


def default_probe_suite(test_turns: list[int] | None = None) -> list[Probe]:
    """Interrogative probes, one per class, staggered across the episode.

    Test turns are staggered so a single episode yields several points on the
    forgetting curve rather than one.
    """
    turns = test_turns or [14, 19, 24, 29, 34, 39]
    return [
        Probe(
            id="ident-1",
            cls=ProbeClass.IDENTIFIER,
            question="What is the exact account code that intake_vendor returned for vendor-00? Answer with the code only.",
            expected=lambda env: env.vendor_to_code.get("vendor-00", ""),
            test_turn=turns[0],
        ),
        Probe(
            id="constraint-1",
            cls=ProbeClass.CONSTRAINT,
            question="Which vendors are on the frozen list? Answer with vendor names only.",
            expected=lambda env: " ".join(sorted(env.frozen_vendors)),
            test_turn=turns[1],
        ),
        Probe(
            id="negative-1",
            cls=ProbeClass.NEGATIVE,
            question="Which settlement method has already been shown to fail on this ledger? One word.",
            expected=lambda env: env.FAILING_METHOD,
            test_turn=turns[2],
        ),
        Probe(
            id="goal-1",
            cls=ProbeClass.GOAL,
            question="What currency must the closing report be settled in? One word.",
            expected=lambda env: "USD",
            test_turn=turns[3],
        ),
        Probe(
            id="artifact-1",
            cls=ProbeClass.ARTIFACT,
            question="How many transactions have you successfully posted so far? A number only.",
            expected=lambda env: str(sum(1 for t in env.pending if t.get("posted"))),
            test_turn=turns[4],
        ),
        Probe(
            id="prov-1",
            cls=ProbeClass.PROVENANCE,
            question="Which tool call produced the account code for vendor-00? Name the tool only.",
            expected=lambda env: "intake_vendor",
            test_turn=turns[5],
        ),
    ]


@dataclass
class BehaviouralResult:
    constraint_violations: int
    wire_retries_after_first: int
    report_correct: bool
    completion: float


def score_behavioural(env) -> BehaviouralResult:
    """Behavioural probes. No question asked, so no observer effect."""
    t = env.truth()
    rep = t["report_filed"] or {}
    report_correct = bool(rep) and abs(float(rep.get("total_posted", -1)) - t["total_posted"]) < 0.01
    return BehaviouralResult(
        constraint_violations=t["constraint_violations"],
        wire_retries_after_first=t["wire_retries_after_first"],
        report_correct=report_correct,
        completion=t["n_posted"] / max(1, t["n_pending_total"]),
    )
