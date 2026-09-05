"""Cross-model coverage check (PRD section 7: one Anthropic, one OpenAI, one
open weight). This is deliberately NOT a full re-run of the M1/M2 matrix
(6 strategies x 3 batteries x 3 seeds x 3 models, ~$45-50) -- that answers "do
these curves generalize in every detail," which this project doesn't need to
claim. This answers the cheaper, still-real question: does the strategy
*ranking* on the two headline findings (cache inversion, per-class half life)
hold up on a second and third model family. 3 representative strategies
(control + the two with the clearest M2 findings) x 2 seeds x 1 battery
(budget=700, matching results_full_matrix_real.jsonl's "mid" battery) x 2 new
models = 12 episodes, written to results_multimodel.jsonl (kept separate from
the Anthropic-only files so nothing silently mixes budgets or batteries).

Requires OPENAI_API_KEY and GROQ_API_KEY in .env. Run a single episode first
(Ctrl-C after the first printed line) as a smoke test before letting this run
to completion -- same discipline as every other real-model run in this
project: real API integration has surfaced a bug on the first live call every
single time so far.
"""
import json
import sys
import time

sys.path.insert(0, ".")

from goldfish import strategies, runner
from goldfish.models import OpenAICompatibleAdapter
from goldfish.probes import default_probe_suite

STRATEGY_NAMES = ["full_history", "summarization", "sliding_window"]
SEEDS = [0, 1]
BUDGET = 700
OUT = "results_multimodel.jsonl"

MODELS = {
    "openai-gpt-5-mini": lambda: OpenAICompatibleAdapter(
        model="gpt-5-mini",
        name="openai-gpt-5-mini",
        token_param="max_completion_tokens",
        max_tokens=2048,  # higher than the Anthropic default: this budget also pays for hidden reasoning tokens
        reasoning_effort="low",  # this task needs tool-call discipline, not deep reasoning -- keep it cheap
    ),
    "groq-llama-3.3-70b": lambda: OpenAICompatibleAdapter(
        model="llama-3.3-70b-versatile",
        name="groq-llama-3.3-70b",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    ),
}

done = set()
try:
    for line in open(OUT):
        d = json.loads(line)
        done.add((d["model"], d["strategy"], d["seed"]))
except FileNotFoundError:
    pass

todo = [
    (mname, sname, seed)
    for mname in MODELS
    for sname in STRATEGY_NAMES
    for seed in SEEDS
    if (mname, sname, seed) not in done
]
print(f"{len(todo)} episodes remaining: {todo}")

t0 = time.time()
with open(OUT, "a") as f:
    for mname, sname, seed in todo:
        model = MODELS[mname]()
        r = runner.run_episode(
            strategies.REGISTRY[sname](),
            model,
            seed=seed,
            budget=BUDGET,
            probes=default_probe_suite(),
        )
        f.write(json.dumps(r.to_dict()) + "\n")
        f.flush()
        print(f"[{time.time()-t0:6.0f}s] {mname:<20} {sname:<16} seed={seed} turns={r.turns} usage={r.usage}")

print(f"\ndone, wrote {OUT}")
