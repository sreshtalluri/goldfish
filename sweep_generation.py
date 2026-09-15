"""Compaction generation study: budget=200 to force multiple real
compactions within one ~60-turn episode (budget=700, used throughout
M1/M2, barely forces eviction at all -- M0 finding #3). Writes one JSON
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
