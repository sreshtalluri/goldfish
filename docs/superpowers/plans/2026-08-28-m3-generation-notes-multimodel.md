# M3: Structured Notes, Compaction Generation Study, Multi-Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the PRD's M3 milestone — the structured-notes strategy, a compaction-generation fidelity study (does loss compound linearly or cliff), and multi-model coverage — with real data behind each, matching the rigor M1/M2 established.

**Architecture:** Three independent subsystems sharing the existing `goldfish/` harness. (1) `StructuredNotes` is a new `Strategy` implementation plus one new environment tool, following the exact pattern `ExternalScratchpad` already established. (2) The generation study adds one field to `Probe` and a few lines to `runner.py` to record which compaction generation was active when each probe was asked, then a new `metrics.py` function to bucket recall by generation — no new strategies needed, this runs against strategies that already exist. (3) Multi-model adds one adapter class, `OpenAICompatibleAdapter`, parametrized by `base_url` so the same code serves OpenAI itself and any OpenAI-compatible open-weight provider (Together, Fireworks, Groq), rather than writing bespoke code twice.

**Tech Stack:** Python 3.13 (pyenv env `goldfish`), pytest, `anthropic` SDK (installed), `openai` SDK (not yet installed — Task 8 installs it).

**Spec:** `goldfish-context-compaction-prd.md` sections 5-8 (strategy list, milestones), section 6 (generation metric), section 7 (model adapters). This plan also resolves one spec/reality discrepancy: PRD line 234 calls masking one of "the two added strategies" for M3, but `tool_masking` has been in `strategies.REGISTRY` since the initial M0 commit and was used throughout M1 and M2. The only strategy actually missing is structured notes; this plan does not re-add masking.

## Global Constraints

- Every new strategy must pass the existing instrument invariants unmodified: `test_no_leak_under_full_history` and `test_control_invariance` in `test_instrument.py` (the latter auto-parametrizes over `strategies.REGISTRY`, so registering a strategy is enough to get coverage — no new test needed for that part).
- Real API calls cost money and this session's background jobs have been killed unpredictably around 20-30 minutes; any run script must write results incrementally (append + flush per episode, per the pattern in `report_m2.py`'s companion scripts) and support resuming only the missing `(strategy, seed)` pairs.
- Pricing/model-ID claims must come from the official docs (`platform.claude.com/docs/en/about-claude/pricing`), not memory — this bit the project once already (a stale `claude-sonnet-4-6` model string shipped in M0 and failed the first live call).
- Run `python3 -m pytest -q` after every task; it must stay green (30 tests currently: `test_instrument.py` + `test_metrics.py`).

---

## Phase 1: Structured Notes Strategy

### Task 1: Add the `annotate` tool to the environment

**Files:**
- Modify: `goldfish/env.py` — `tool_schema()` method and `LedgerEnv` dataclass fields
- Test: `test_instrument.py`

**Interfaces:**
- Produces: `LedgerEnv.annotation: str` (single evolving string, not a dict like `notes`), tool name `"annotate"` with arg `{"note": "string"}`

Structured notes is deliberately not "another scratchpad": `write_note`/`read_note` is a keyed store where every call is a separate message `ExternalScratchpad` pins verbatim forever. `annotate` is a single running buffer — each call *overwrites* it — so the compaction strategy can fold "whatever the agent currently thinks is worth remembering" into one compact node, with no LLM call and no unbounded message growth.

- [ ] **Step 1: Add the field and tool implementation to `LedgerEnv`**

In `goldfish/env.py`, add to the `LedgerEnv` dataclass fields (next to `notes: dict[str, str] = field(default_factory=dict)`):

```python
    annotation: str = ""
```

Add a new tool method next to `_t_write_note`/`_t_read_note`:

```python
    def _t_annotate(self, note: str) -> ToolResult:
        self.annotation = note
        return ToolResult(True, {"bytes": len(note)})
```

- [ ] **Step 2: Register it in `tool_schema()`**

Add to the list returned by `tool_schema()`, after the `read_note` entry:

```python
            {
                "name": "annotate",
                "args": {"note": "string"},
                "doc": "Overwrite your running notes with the current important facts. Replaces any previous note.",
            },
```

- [ ] **Step 3: Write a test that the tool works and overwrites**

Add to `test_instrument.py`:

```python
def test_annotate_overwrites_not_accumulates():
    env = LedgerEnv(seed=0)
    env.call("annotate", {"note": "first"})
    env.call("annotate", {"note": "second"})
    assert env.annotation == "second"
```

- [ ] **Step 4: Run test, confirm it passes**

Run: `python3 -m pytest test_instrument.py -k annotate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add goldfish/env.py test_instrument.py
git commit -m "Add annotate tool: single overwriting note buffer for structured notes"
```

### Task 2: Implement the `StructuredNotes` strategy

**Files:**
- Modify: `goldfish/strategies.py`
- Test: `test_instrument.py`

**Interfaces:**
- Consumes: `LedgerEnv.annotation` is read via the transcript (the tool's observation message), not directly from the env — the strategy only ever sees `messages`, same contract as every other strategy.
- Produces: `strategies.StructuredNotes`, registered in `strategies.REGISTRY["structured_notes"]`.

Design: on eviction, find the *most recent* `annotate` observation in the messages (if any), keep it plus the recent tail (same `keep_recent` pattern as `Summarization`/`ExternalScratchpad`), and discard everything else — no LLM call, no unbounded pinning.

- [ ] **Step 1: Write the strategy**

Add to `goldfish/strategies.py`, after `ExternalScratchpad`:

```python
@dataclass
class StructuredNotes:
    """The agent annotates as it goes via a single overwriting `annotate`
    call; eviction keeps only the latest annotation plus the recent tail,
    discarding the rest. Distinct from both Summarization (no LLM call, no
    hallucination risk from re-deriving old content, and no per-compaction
    cost) and ExternalScratchpad (one compact evolving note, not an
    unbounded set of pinned raw messages) — this is the "Manus style"
    variant PRD section 5 calls out: compaction preserves what the agent
    chose to write down, not the raw transcript.
    """

    name: str = "structured_notes"
    keep_recent: int = 6
    needs_tools: list[str] = field(default_factory=lambda: ["annotate"])

    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        if approx_tokens(messages) <= budget:
            return messages
        latest_note = None
        for m in messages:
            if m.get("kind") == "observation" and m.get("tool") == "annotate":
                latest_note = m
        tail = messages[-self.keep_recent :] if len(messages) > self.keep_recent else messages
        tail_ids = {id(m) for m in tail}
        out = ([latest_note] if latest_note is not None and id(latest_note) not in tail_ids else []) + tail
        if approx_tokens(out) <= budget:
            return out
        return SlidingWindow().reduce(out, budget)
```

- [ ] **Step 2: Register it**

In `goldfish/strategies.py`, add to `REGISTRY`:

```python
    "structured_notes": StructuredNotes,
```

- [ ] **Step 3: Write a strategy-specific test**

The auto-parametrized `test_control_invariance` already covers the "unbounded budget = control" invariant for any registered strategy. Add one behavior-specific test to `test_instrument.py` confirming the note survives eviction that would otherwise drop it:

```python
def test_structured_notes_preserves_latest_annotation_under_pressure():
    messages = [{"role": "user", "kind": "goal", "content": "x" * 500}]
    messages.append({"role": "assistant", "kind": "action", "content": "annotate({'note': 'critical'})"})
    messages.append({"role": "user", "kind": "observation", "tool": "annotate", "content": '{"bytes": 8}'})
    # Pad with enough bulk that a tight budget forces eviction of the old goal turn.
    for i in range(20):
        messages.append({"role": "assistant", "kind": "action", "content": f"noise {i}" * 20})
        messages.append({"role": "user", "kind": "observation", "tool": "list_pending", "content": "y" * 200})
    out = strategies.StructuredNotes().reduce(messages, budget=50)
    assert any(m.get("tool") == "annotate" for m in out)
```

- [ ] **Step 4: Run full suite**

Run: `python3 -m pytest -q`
Expected: all pass, including the new `test_control_invariance[structured_notes]` case that appears automatically.

- [ ] **Step 5: Commit**

```bash
git add goldfish/strategies.py test_instrument.py
git commit -m "Add StructuredNotes strategy (structure notes, M3 PRD section 5 item 6)"
```

### Task 3: Wire the system-prompt hint for `annotate`, matching the scratchpad pattern

**Files:**
- Modify: `goldfish/runner.py`

**Interfaces:**
- Consumes: `strategy.needs_tools` (already read in `run_episode`)
- Produces: extends the existing `needed` check so `"annotate"` also triggers a hint

- [ ] **Step 1: Add the hint constant**

In `goldfish/runner.py`, next to `SCRATCHPAD_HINT`:

```python
NOTES_HINT = (
    " Your context window is bounded and older turns may be evicted. "
    "annotate is a single running note that survives eviction: call it "
    "whenever the important facts change, overwriting the previous note "
    "with everything you would need if you lost the rest of this "
    "conversation."
)
```

- [ ] **Step 2: Extend the hint-selection logic**

Change:

```python
    needed = set(getattr(strategy, "needs_tools", ()))
    if {"write_note", "read_note"} & needed:
        system += SCRATCHPAD_HINT
```

to:

```python
    needed = set(getattr(strategy, "needs_tools", ()))
    if {"write_note", "read_note"} & needed:
        system += SCRATCHPAD_HINT
    if "annotate" in needed:
        system += NOTES_HINT
```

- [ ] **Step 3: Run full suite, then the offline demo**

Run: `python3 -m pytest -q && python3 run_demo.py`
Expected: all green; `structured_notes` row appears in the demo table (offline recall will likely equal `sliding_window`'s, same known limitation as scratchpad — `ContextBoundAgent` never calls `annotate` either, since it doesn't reason about what's worth persisting. Document this in the commit message, not as a bug.)

- [ ] **Step 4: Commit**

```bash
git add goldfish/runner.py
git commit -m "Wire annotate system-prompt hint for structured_notes"
```

### Task 4: One real smoke-test episode before spending on the full matrix

**Files:**
- None (ad-hoc script in `$CLAUDE_JOB_DIR/tmp`, not committed)

- [ ] **Step 1: Run one real episode with `structured_notes`**, same pattern as the M1 smoke test (`AnthropicAdapter`, seed=0, default probe suite). Confirm: no crash, `annotate` gets called at least once (check `r.usage` and print each probe's answer), and probe grading behaves sanely.
- [ ] **Step 2: If it crashes or behaves nonsensically**, treat it exactly like the M1/M2 bugs — fix at the source, add a regression test, re-verify offline suite, re-run this smoke test. Do not proceed to Phase 4 (the real run) until this passes clean.

---

## Phase 2: Compaction Generation Study

### Task 5: Track per-probe compaction generation

**Files:**
- Modify: `goldfish/probes.py` (`Probe` dataclass)
- Modify: `goldfish/strategies.py` (give every strategy a `generation` counter, not just `Summarization`)
- Modify: `goldfish/runner.py` (record generation at ask-time)
- Test: `test_instrument.py`

**Interfaces:**
- Produces: `Probe.generation_at_ask: int | None`, set alongside `asked_turn`/`outcome` in the probe branch of `run_episode`.
- Consumes: `strategy.generation` (already exists on `Summarization`; this task adds it to every other strategy so the study covers all of them, not just summarization).

Right now only `Summarization` increments a `generation` counter, and only when it actually re-summarizes. For the generation study to mean anything across strategies, every strategy needs to report "how many times have I actually evicted something by now" — not "how many times was `reduce()` called" (called every turn regardless of whether eviction happened).

- [ ] **Step 1: Add a `generation` field to every strategy, incremented only on real eviction**

In `goldfish/strategies.py`, add `generation: int = 0` to `SlidingWindow`, `ToolOutputMasking`, `RetrievalOverTurns`, `ExternalScratchpad`, and `StructuredNotes` (mirroring the field already on `Summarization`). Each `reduce()` method already has an early-return guard `if approx_tokens(messages) <= budget: return messages` — increment generation on the line immediately after that guard fails (i.e., the eviction path was taken), for example in `SlidingWindow`:

```python
    def reduce(self, messages: list[Message], budget: int) -> list[Message]:
        if approx_tokens(messages) <= budget:
            return messages
        self.generation += 1
        kept: list[Message] = []
        ...
```

Apply the same one-line insertion (`self.generation += 1` right after the budget guard) to `ToolOutputMasking`, `RetrievalOverTurns`, `ExternalScratchpad`, and `StructuredNotes`. `FullHistory` never evicts, so it correctly has no generation counter — `getattr(strategy, "generation", 0)` in `runner.py` already handles that by defaulting to 0.

- [ ] **Step 2: Add `generation_at_ask` to `Probe`**

In `goldfish/probes.py`, add to the `Probe` dataclass fields (next to `answer: str | None = None`):

```python
    generation_at_ask: int | None = None
```

- [ ] **Step 3: Record it in the runner's probe branch**

In `goldfish/runner.py`, in the `if due:` block, right after `p.asked_turn = turn`:

```python
            p.asked_turn = turn
            p.generation_at_ask = getattr(strategy, "generation", 0)
```

- [ ] **Step 4: Serialize it in `EpisodeResult.to_dict()`**

In `goldfish/runner.py`, add `"generation_at_ask": p.generation_at_ask` to the dict comprehension inside `to_dict()`:

```python
            "probes": [
                {
                    "id": p.id,
                    "class": p.cls.value,
                    "distance": p.distance,
                    "outcome": p.outcome.value,
                    "answer": p.answer,
                    "generation_at_ask": p.generation_at_ask,
                }
                for p in self.probes
            ],
```

- [ ] **Step 5: Test it end-to-end offline**

```python
def test_generation_at_ask_increments_under_pressure():
    """A tight budget with a long episode should force multiple real
    compactions before the last probe is asked, and each probe's recorded
    generation should be non-decreasing over the episode."""
    r = run_episode(strategies.SlidingWindow(), ContextBoundAgent(seed=0), seed=0,
                     budget=150, probes=default_probe_suite())
    gens = [p.generation_at_ask for p in r.probes if p.asked_turn is not None]
    assert any(g and g > 0 for g in gens), "expected at least one probe asked after a real compaction"
    assert gens == sorted(gens)
```

- [ ] **Step 6: Run full suite**

Run: `python3 -m pytest -q`
Expected: all pass (30+ tests now).

- [ ] **Step 7: Commit**

```bash
git add goldfish/probes.py goldfish/strategies.py goldfish/runner.py test_instrument.py
git commit -m "Track compaction generation per probe, across all strategies"
```

### Task 6: Add `recall_by_generation` to metrics, with tests

**Files:**
- Modify: `goldfish/metrics.py`
- Modify: `test_metrics.py`

**Interfaces:**
- Produces: `recall_by_generation(points: list[tuple[int, bool]]) -> dict[int, tuple[int, int]]` — generation bucket -> (recalled count, total count). Reuses the same `(x, bool)` pair shape `half_life()` already takes, so the report script can build both from the same query.

- [ ] **Step 1: Write the failing test**

```python
def test_recall_by_generation_buckets_correctly():
    points = [(0, True), (0, True), (1, True), (1, False), (2, False)]
    buckets = recall_by_generation(points)
    assert buckets == {0: (2, 2), 1: (1, 2), 2: (0, 1)}
```

- [ ] **Step 2: Run it, confirm it fails** (`recall_by_generation` doesn't exist yet)

Run: `python3 -m pytest test_metrics.py -k generation -v`
Expected: FAIL with `ImportError` or `NameError`

- [ ] **Step 3: Implement it**

In `goldfish/metrics.py`:

```python
def recall_by_generation(points: list[tuple[int, bool]]) -> dict[int, tuple[int, int]]:
    """points: (generation, recalled) pairs. Returns generation -> (recalled,
    total), for the section 6 question: does fidelity loss compound linearly
    across compaction generations, or does it cliff after some threshold.
    """
    buckets: dict[int, list[bool]] = {}
    for gen, recalled in points:
        buckets.setdefault(gen, []).append(recalled)
    return {gen: (sum(vals), len(vals)) for gen, vals in sorted(buckets.items())}
```

- [ ] **Step 4: Run test, confirm it passes**

Run: `python3 -m pytest test_metrics.py -q`
Expected: PASS, all metrics tests green.

- [ ] **Step 5: Commit**

```bash
git add goldfish/metrics.py test_metrics.py
git commit -m "Add recall_by_generation to metrics for the compaction generation study"
```

### Task 7: Run the real generation-study sweep and extend the report

**Files:**
- Create: `sweep_generation.py` (repo root, same pattern as the existing `report_m2.py` — a committed, reusable script, not a throwaway)
- Modify: `report_m2.py` (add a "recall by compaction generation" section)

Budget=700 (used throughout M1/M2) barely forces eviction at all in a ~60-turn episode — M0's own finding #3 already flagged this. To see generation 3-5 within one episode, this needs a materially tighter budget. Use budget=200, seeds 0-2, all 6 existing strategies (skip `structured_notes` here unless Phase 1 already landed — sequence Phase 1 before this task if both are in scope for this session), default probe suite, and append to a new file `results_generation_study.jsonl` (do not mix into `results_full_matrix_real.jsonl`, since budget=700 vs 200 are different experiments and conflating them would silently confound any report that reads the file without filtering by budget).

- [ ] **Step 1: Write `sweep_generation.py`** at repo root:

```python
"""Compaction generation study: budget=200 to force multiple real
compactions within one ~60-turn episode (budget=700, used throughout
M1/M2, barely forces eviction at all — M0 finding #3). Writes one JSON
line per episode to results_generation_study.jsonl, flushed immediately,
and skips (strategy, seed) pairs already present so a killed background
job can be resumed by just rerunning this script unchanged.
"""
import json
import sys
import time

sys.path.insert(0, ".")

from goldfish import strategies, runner
from goldfish.models import AnthropicAdapter
from goldfish.probes import default_probe_suite

SEEDS = [0, 1, 2]
BUDGET = 200
OUT = "results_generation_study.jsonl"

done = set()
try:
    for line in open(OUT):
        d = json.loads(line)
        done.add((d["strategy"], d["seed"]))
except FileNotFoundError:
    pass

todo = [(s, seed) for s in strategies.REGISTRY for seed in SEEDS if (s, seed) not in done]
print(f"{len(todo)} episodes remaining: {todo}")

t0 = time.time()
with open(OUT, "a") as f:
    for sname, seed in todo:
        model = AnthropicAdapter()
        r = runner.run_episode(
            strategies.REGISTRY[sname](),
            model,
            seed=seed,
            budget=BUDGET,
            probes=default_probe_suite(),
        )
        f.write(json.dumps(r.to_dict()) + "\n")
        f.flush()
        print(f"[{time.time()-t0:6.0f}s] {sname:<16} seed={seed} turns={r.turns} "
              f"generations={r.compaction_generations} usage={r.usage}")

print(f"\ndone, wrote {OUT}")
```

Run strategies x seeds 0-2 at `budget=200`.
- [ ] **Step 2: Run it in the background** (`run_in_background: true`, since prior M1/M2 background jobs got killed unpredictably around 20-30 min — expect the same here and be ready to resume only the missing `(strategy, seed)` pairs, same recovery pattern as `sweep_resume.py`).
- [ ] **Step 3: Add a "recall by compaction generation" section to `report_m2.py`**, using `recall_by_generation()` from Task 6, printing a table of generation -> recall per strategy, plus a one-line takeaway on whether decay looks linear or cliff-shaped per strategy.
- [ ] **Step 4: Run the extended report against `results_generation_study.jsonl`, inspect the output for sanity** (in particular: does `full_history` show generation=0 for every probe, since it never evicts? If not, that is a bug in Task 5's generation-increment logic, not a finding — stop and fix before trusting anything else in this table).
- [ ] **Step 5: Update `README.md`** with the generation-study findings (add an "M3" status paragraph analogous to the existing M1/M2 ones), and commit everything together:

```bash
git add sweep_generation.py report_m2.py results_generation_study.jsonl README.md
git commit -m "M3: compaction generation study — does fidelity loss compound or cliff"
git push
```

---

## Phase 3: Multi-Model (blocked pending API key)

**This phase cannot start until the user provides at minimum an `OPENAI_API_KEY`.** Unlike Phase 1/2, there is no offline path to validate this — get explicit confirmation before writing code that will sit unused, or before spending the user's OpenAI credits.

### Task 8: Install the OpenAI SDK and add `.env` support for the new key

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1:** `pip install openai` inside the `goldfish` pyenv virtualenv, then `pip freeze | grep -i openai` to get the exact pinned version.
- [ ] **Step 2:** Add the pinned line to `requirements.txt`.
- [ ] **Step 3:** Add `OPENAI_API_KEY=` to `.env.example` (and tell the user to paste their key into the real `.env`, same flow as the Anthropic key).
- [ ] **Step 4: Commit**

```bash
git add requirements.txt .env.example
git commit -m "Add openai SDK dependency for M3 multi-model coverage"
```

### Task 9: Implement `OpenAICompatibleAdapter`

**Files:**
- Modify: `goldfish/models.py`
- Test: needs a real key to test meaningfully; write the code, defer verification to Task 10's smoke test.

**Interfaces:**
- Produces: `OpenAICompatibleAdapter` implementing `ModelAdapter` (same `act(system, messages, tools) -> Action` contract every adapter already follows). Parametrized by `base_url` and `api_key_env` so the same class serves OpenAI itself (`base_url=None`, uses the SDK default) and any OpenAI-compatible open-weight provider (e.g. Together AI at `https://api.together.xyz/v1`) — this is one adapter class covering two of the PRD's three required model adapters ("one Anthropic, one OpenAI, one open weight" — section 7), not two separate implementations.

- [ ] **Step 1: Write the adapter**

Add to `goldfish/models.py`, in the online section next to `AnthropicAdapter`:

```python
@dataclass
class OpenAICompatibleAdapter:
    """Real runs against OpenAI or any OpenAI-compatible endpoint (Together,
    Fireworks, Groq, a local vLLM/Ollama server). One class instead of two:
    the wire format is the same, only base_url/model/api_key differ, and the
    PRD's "one OpenAI, one open weight" requirement (section 7) is satisfied
    by pointing two instances of this class at different base_urls rather
    than duplicating the request-building logic.
    """

    model: str
    base_url: str | None = None  # None = OpenAI's default endpoint
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 1024
    name: str = "openai_compatible"
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_client(self):
        import openai

        if self._client is None:
            self._client = openai.OpenAI(
                api_key=os.environ[self.api_key_env],
                base_url=self.base_url,
                max_retries=6,
            )
        return self._client

    def act(self, system: str, messages: list[Message], tools: list[dict]) -> Action:
        client = self._get_client()
        chat_messages = [{"role": "system", "content": system}]
        chat_messages.extend(
            {"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in messages
        )
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, messages=chat_messages)
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["doc"],
                        "parameters": {
                            "type": "object",
                            "properties": {a: {"type": typ} for a, typ in t["args"].items()},
                            "required": list(t["args"]),
                        },
                    },
                }
                for t in tools
            ]
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        usage = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
            "cache_read": getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            "cache_write": 0,  # OpenAI-compatible caching is automatic and not billed as a separate write
        }
        if choice.tool_calls:
            call = choice.tool_calls[0]
            import json as _json

            return Action("tool", name=call.function.name, args=_json.loads(call.function.arguments), usage=usage)
        return Action("text", content=choice.content or "", usage=usage)
```

**Note on `role: "system"` placement:** unlike the Anthropic adapter's separate `system` parameter, OpenAI's Chat Completions API takes the system prompt as the first message in the same list. This is a real difference between providers' wire formats, not a simplification — do not "fix" it to look like the Anthropic adapter.

- [ ] **Step 2: Confirm it imports and instantiates without a key present** (constructing the dataclass must not touch the network or read the env var — only `_get_client()` does, lazily, matching `AnthropicAdapter`'s pattern)

Before running this, confirm the current OpenAI model ID via WebSearch/their official docs — do not guess it. The same mistake shipped once already in this project (a stale Anthropic model string in M0 that failed the first live call) and pricing/model-ID claims are a Global Constraint above for exactly that reason. Then run, substituting the confirmed ID:

```bash
python3 -c "from goldfish.models import OpenAICompatibleAdapter; a = OpenAICompatibleAdapter(model='<confirmed-model-id>'); print(a.name)"
```
Expected: prints `openai_compatible`, no error, no network call (constructing the dataclass must not touch `os.environ` or the network until `_get_client()` is called).

- [ ] **Step 3: Commit**

```bash
git add goldfish/models.py
git commit -m "Add OpenAICompatibleAdapter (serves OpenAI and open-weight providers)"
```

### Task 10: Real smoke test, then decide open-weight provider and run the multi-model matrix

- [ ] **Step 1:** Once the user confirms `OPENAI_API_KEY` is in `.env`, run one real episode (`full_history`, seed=0) exactly like the Anthropic smoke test in M1 — expect this to surface at least one real formatting bug, same as every prior real-model integration in this project has. Fix at the source, add a regression test, re-verify offline suite, re-run, repeat until clean.
- [ ] **Step 2:** Ask the user which open-weight provider/model to target (Together AI and Groq both offer OpenAI-compatible endpoints and commonly host Llama/Qwen/DeepSeek open-weight models) and get that API key before writing anything provider-specific — do not guess a provider.
- [ ] **Step 3:** Once both real adapters pass a smoke test, decide with the user how much of the M1/M2 matrix to re-run multi-model (full 6-strategy x 3-battery x 3-seed x 3-model sweep is 3x the cost of the M2 run, i.e. roughly $45-50 based on the M1/M2 actuals — confirm budget before launching, do not assume).

---

## Self-Review Notes

- **Spec coverage:** PRD section 5 item 6 (structured notes) → Phase 1. Section 6 "fidelity as a function of compaction generation" → Phase 2. Section 7 "one Anthropic, one OpenAI, one open weight" → Phase 3. Section 6's cache-on/off-as-a-flag requirement was already satisfied by `cost_usd`/`cost_usd_no_cache` in M2 and needs no new work here.
- **Discrepancy resolved explicitly:** masking is not re-implemented (already exists); noted in the header so a future reader doesn't wonder why Phase 1 only has one strategy when the PRD says two.
- **Type consistency check:** `Probe.generation_at_ask` (Task 5) matches the `getattr(strategy, "generation", 0)` pattern already used for `EpisodeResult.compaction_generations`, so both fields stay consistent with each other rather than diverging. `recall_by_generation()`'s input shape `list[tuple[int, bool]]` intentionally matches `half_life()`'s existing shape so report code can share the same query-building logic for both.
- **Budget separation:** Task 7 explicitly uses a new results file (`results_generation_study.jsonl`) rather than appending budget=200 data into the existing budget=700 file, to avoid silently confounding the M2 report if someone runs it against the wrong file later.
