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

M1 in progress. All strategies required by the PRD are behind the common
interface, model-free: `full_history`, `sliding_window`, `summarization`,
`retrieval`, `scratchpad`, plus the optional `tool_masking`. What is missing
for M1 to be complete is a real model adapter run (`AnthropicAdapter` exists
but is untested against a live API key in this environment) and the first
real recall curves that come from it.

No real model results yet. Anything produced by `ContextBoundAgent` is a
plumbing test, not a finding — and it is specifically unable to distinguish
`scratchpad` from `sliding_window`, since it never reasons about what to
persist; that differentiation only shows up with a real model that reads the
scratchpad hint in the system prompt.

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
