# goldfish

How fast does your agent forget, and what does remembering cost?

A benchmark for context compaction strategies. Not a memory system, not a
conversational recall benchmark. It measures **which class of information each
compaction strategy destroys, and how fast**.

## The metric

**Context half life**: turns until recall of a planted fact of class C, under
strategy S, drops below 50 percent. Reported per class, never as one number.

Outcomes are three-valued, because the difference between an agent that says
"I no longer have that" and one that confidently invents a replacement is the
whole operational story:

| outcome | meaning |
| --- | --- |
| `recalled` | correct |
| `admitted` | wrong, and the agent said it did not know |
| `hallucinated` | wrong, stated confidently |

## Probe classes

`goal`, `constraint`, `negative_knowledge`, `artifact_state`, `identifier`,
`provenance`. Artifact state and negative knowledge are the ones expected to
die first and the ones least discussed.

## Status

M0 complete: environment, probes, checkers, strategy interface, runner, and an
offline simulator for validating the instrument.

M1: first real recall curves obtained. All 6 strategies (`full_history`,
`sliding_window`, `tool_masking`, `retrieval`, `summarization`, `scratchpad`)
run against a live model (`claude-sonnet-5`, `AnthropicAdapter`), 3 seeds
each, budget=700. Raw results in `results_full_matrix_real.jsonl`.

Getting a live run working surfaced and fixed several real bugs the offline
simulator could not catch, since `ContextBoundAgent` never exercises message
role structure, tool-call formatting, or free-text answer parsing the way a
real model does: a turn-loop bug that made every real episode fail
immediately (conversation ending in assistant role — an "assistant prefill"
the API rejects), a probe-grading substring bug that could silently score a
wrong numeric answer as recalled, an unenforced negative-knowledge dependency
that a real model could simply route around and never discover, unvalidated
tool arguments that could crash scoring on a malformed call, and a
tool-offered-during-probe issue that miscounted deflection as hallucination.
See commit history for each.

**Preliminary findings (n=3 seeds, single model, single budget — not yet the
seeded/CI-backed numbers M2 will produce, treat as suggestive):**
- `artifact_state` recall collapses under every compacting strategy (0.00-0.33)
  vs. 0.67 under the full-history control — reproduces the Factory.ai finding
  this project is built around (section 0 of the PRD).
- Admission rate is 0.00 across every strategy: the model never once said "I
  don't know" when wrong. Every miss was a confident, silent substitution,
  not a blank — the failure mode the PRD calls out as the operationally
  dangerous one (section 6).
- `scratchpad` is the best-performing non-control strategy (0.83 overall
  recall, matching full_history on 4 of 6 classes) — a real result in favor
  of the "give the agent a durable file" hypothesis, not just a plausible one.
- `summarization` lost `negative_knowledge` completely (0.00) despite
  `tool_masking` and `scratchpad` holding it at 1.00 — summarizing over a
  failed-approach fact seems to be where that fact specifically dies.

## Run

```bash
python run_demo.py          # offline sweep, no API key needed
python -m pytest tests/     # instrument validity tests
```

## Adding a strategy

The entire reuse surface is one method:

```python
class MyStrategy:
    name = "mine"
    needs_tools: list[str] = []
    def reduce(self, messages, budget): ...
```

Register it in `strategies.REGISTRY` and it appears in every report.

## Instrument validity

Two invariants gate every change, and both have already caught real bugs:

- **No leak.** `full_history` must score a perfect ceiling. Less means the
  environment or the probes are losing information the strategy did not.
- **Control invariance.** At an unbounded budget every strategy must be
  byte-identical to the control. `tool_masking` originally failed this because
  it masked unconditionally rather than under pressure, which would have
  confounded every comparison against it.
