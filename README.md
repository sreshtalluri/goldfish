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

M3, in progress. Two pieces landed:

**`structured_notes`** (PRD section 5 item 6, the one strategy actually still
missing — `tool_masking` was already built in M0 despite the PRD listing it
as an M3 item too). A single overwriting `annotate` note, pinned across
eviction, with the rest of the budget filled by as-much-recent-context-as-fits
rather than an LLM summary or an unbounded pinned message set. Two real bugs
found and fixed by the same real-model-smoke-test discipline as M1/M2 (see
commit history): the note wasn't actually pinned under pressure, and the
fallback path could leave a dangling assistant-role message as the new
history tail, which the runner's own invariant assertion caught immediately.

**Compaction generation tracking.** Every strategy now records how many real
compactions had happened by the time each probe was asked
(`Probe.generation_at_ask`), not just a final count at episode end. Ran a
21-episode sweep (7 strategies x 3 seeds, budget=200 to force generation 5+
within one episode) — `results_generation_study.jsonl`, `sweep_generation.py`
to reproduce.

**What the data actually shows, honestly:** the generation study surfaced a
real methodological limitation before it surfaced a finding. Only
`summarization` has generation semantics matching the PRD's "summarizing a
summary" framing (a discrete re-summarization count, maxing at 22-25 over 60
turns). Every oldest-first strategy's generation counter fires on nearly
every turn once a tight budget is exceeded (max generation 41-53 out of ~60
turns) — it's a proxy for turns-under-pressure, not a comparable "compaction
generation" in the same sense. Worse: because `default_probe_suite()` tests
each class once per episode at one fixed turn, probe *class* and *generation*
are confounded (identifier always lands around generation 5-6, constraint
7-8, negative_knowledge 9-10, goal 11-12, artifact_state 13-14, provenance
15-16, consistent across all 3 seeds) — so "recall by generation," pooled
across classes, is largely relabeling "recall by which class happens to be
tested at that point in the episode," not a within-fact compounding-decay
curve. A genuine "does loss compound linearly or cliff" answer needs a probe
design that re-tests the *same* fact at multiple generations, which is a
real gap for future work, not something this dataset can answer yet. Full
generation tables (with this caveat) via
`python3 report_m2.py results_generation_study.jsonl`.

**Multi-model coverage (PRD section 7: one Anthropic, one OpenAI, one open
weight).** Built and blocked only on API keys, not on any remaining code.
`OpenAICompatibleAdapter` (`goldfish/models.py`) serves both OpenAI itself and
any OpenAI-compatible open-weight endpoint via `base_url` — one class instead
of two, satisfying two of the three required adapters. Deliberately scoped as
a coverage check rather than a full replication of the Anthropic matrix: full
6-strategy x 3-battery x 3-seed x 3-model replication would cost roughly
3x the M2 run (~$45-50) to answer a question this project doesn't need to
claim ("do the curves match in every detail"). Instead, `sweep_multimodel.py`
runs the 3 strategies with the clearest M2 findings (`full_history` control,
`summarization`, `sliding_window`) x 2 seeds x 1 battery (budget=700) against
`gpt-5-mini` and Groq-hosted `llama-3.3-70b-versatile`, to check the cheaper,
still-real question: does the ranking on the two headline findings (cache
inversion, per-class half life) hold up on a second and third model family.
`report_m2.py` now groups by (model, strategy) so a file mixing models
reports each correctly, including per-model pricing (`goldfish/metrics.py`
`PRICES_BY_MODEL`).

To run: add `OPENAI_API_KEY` and `GROQ_API_KEY` to `.env`, then

```bash
python3 sweep_multimodel.py            # smoke-test the first line, then let it finish (~12 episodes)
python3 report_m2.py results_multimodel.jsonl
```

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
