"""M0 smoke run: every model-free strategy, several seeds, offline simulator."""
import json, statistics
from collections import defaultdict
from goldfish import run_episode, strategies
from goldfish.models import ContextBoundAgent
from goldfish.probes import Outcome, default_probe_suite

SEEDS = [0, 1, 2, 3, 4]
BUDGET = 900

rows = []
for sname, factory in strategies.REGISTRY.items():
    for seed in SEEDS:
        r = run_episode(factory(), ContextBoundAgent(seed=seed), seed=seed,
                        budget=BUDGET, probes=default_probe_suite())
        rows.append(r)

by_strat = defaultdict(list)
for r in rows:
    by_strat[r.strategy].append(r)

print(f"{'strategy':<16}{'recall':>8}{'halluc':>8}{'admit':>8}{'viol':>7}{'retry':>7}{'compl':>8}{'redun':>7}{'peak':>8}")
for s, rs in by_strat.items():
    outs = [p.outcome for r in rs for p in r.probes]
    n = len(outs) or 1
    rec = sum(o is Outcome.RECALLED for o in outs)/n
    hal = sum(o is Outcome.HALLUCINATED for o in outs)/n
    adm = sum(o is Outcome.ADMITTED for o in outs)/n
    viol = statistics.mean(r.behavioural.constraint_violations for r in rs)
    retry = statistics.mean(r.behavioural.wire_retries_after_first for r in rs)
    compl = statistics.mean(r.behavioural.completion for r in rs)
    redun = statistics.mean(r.redundant_calls for r in rs)
    peak = statistics.mean(r.approx_context_peak for r in rs)
    print(f"{s:<16}{rec:>8.2f}{hal:>8.2f}{adm:>8.2f}{viol:>7.1f}{retry:>7.1f}{compl:>8.2f}{redun:>7.1f}{peak:>8.0f}")

print("\nper-class recall")
cls_rows = defaultdict(lambda: defaultdict(list))
for r in rows:
    for p in r.probes:
        cls_rows[p.cls.value][r.strategy].append(p.outcome is Outcome.RECALLED)
strats = list(by_strat)
print(f"{'class':<20}" + "".join(f"{s[:12]:>14}" for s in strats))
for c, d in cls_rows.items():
    print(f"{c:<20}" + "".join(f"{statistics.mean(d[s]) if d[s] else 0:>14.2f}" for s in strats))

json.dump([r.to_dict() for r in rows], open("results.json","w"), indent=1)
