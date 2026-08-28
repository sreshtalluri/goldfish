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

M2: context half life, confidence intervals, cache-aware cost accounting.
`goldfish/metrics.py` (Wilson score intervals, linear-interpolated half life
with explicit left/right censoring, actual-vs-no-cache cost). Extended the
real dataset to 3 turn-distance batteries (`early`/`mid`/`late`) x 3 seeds x
6 strategies = 54 episodes, `results_full_matrix_real.jsonl`. Reproduce the
full report with:

```bash
python report_m2.py results_full_matrix_real.jsonl
```

See `m2_report.txt` for the last full run's output.

**Context half life (turns to 50% recall), per strategy x class** — the
project's headline metric, `>N` means recall never dropped below 50% within
the tested range (right-censored, true half life is at least N), `<N` means
it was already below 50% at the shortest distance tested:

| strategy | identifier | constraint | negative_knowledge | goal | artifact_state | provenance |
|---|---|---|---|---|---|---|
| full_history | >23 | >29 | >27 | >41 | **<9** | >48 |
| sliding_window | >23 | >29 | 14.0 | >41 | 17.5 | >48 |
| tool_masking | 20.0 | >29 | 20.0 | >41 | 17.5 | >48 |
| retrieval | 20.0 | >29 | 5.0 | 38.0 | **<9** | >48 |
| summarization | >23 | >29 | **<4** | **<19** | 10.5 | >48 |
| scratchpad | >22 | >30 | >25 | >41 | 6.5 | <21 |

**Findings, n=3 seeds per distance point (small — read as directional, not
final; M3 is where seed count and model coverage grow):**
- `artifact_state` has the shortest half life under every strategy, including
  the control (full_history's own half life for it is under 9 turns — the
  environment makes agents lose track of what they've done even with nothing
  evicted). This is the strongest, most consistent signal in the dataset and
  matches the Factory.ai finding the project is named after.
- `constraint` and `goal` never dropped below 50% recall within ~30-40 turns
  under *any* strategy tested here — the most durable classes by a wide
  margin, which the single-probe-per-class M1 numbers didn't surface.
- `summarization` uniquely drove `negative_knowledge` (<4 turns) and `goal`
  (<19 turns) to fast, left-censored collapse — actively summarizing over
  these facts looks worse than mechanically evicting them (sliding_window
  holds negative_knowledge to 14 turns, almost 4x longer).
- **Cache-aware cost inversion, confirmed with more data:** `full_history`'s
  cache benefit ratio (actual cost vs. a no-cache counterfactual) is 2.2x;
  every compacting strategy sits at 1.3-1.4x, because rewriting history on
  every compaction invalidates the cache prefix. Consequence: `full_history`
  is the *cheapest* strategy per episode ($0.20) of all six — every
  compacting strategy costs 49-59% more, exactly inverting the standard
  "compact to save money" justification. Full table in `m2_report.txt`.
- Admission rate is still 0.00 across all 54 episodes — no model answer was
  ever an honest "I don't know," only correct or confidently wrong.

Overall recall with 95% Wilson CI, pooled across all 3 batteries (n=54
probes/strategy): full_history 0.85 [0.73, 0.92], scratchpad 0.70 [0.57,
0.81], tool_masking 0.69 [0.55, 0.79], sliding_window 0.65 [0.51, 0.76],
retrieval 0.63 [0.50, 0.75], summarization 0.54 [0.41, 0.66].

## Run

```bash
python run_demo.py          # offline sweep, no API key needed
python -m pytest             # instrument + metrics validity tests
python report_m2.py results_full_matrix_real.jsonl   # half life / CI / cost report
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
