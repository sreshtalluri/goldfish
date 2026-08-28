"""M2 report: recall-by-distance curves, context half life, cache-aware cost,
all with real n behind them. Reads a JSONL of EpisodeResult.to_dict() rows
(see sweep_battery.py / matrix_remainder.py) tagged with a "battery" field
per distance sweep.

    python3 report_m2.py results_full_matrix_real.jsonl
"""
import json
import sys
from collections import defaultdict

from goldfish import strategies
from goldfish.metrics import cost_usd, cost_usd_no_cache, half_life, wilson_ci

CLASSES = ["identifier", "constraint", "negative_knowledge", "goal", "artifact_state", "provenance"]


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def valid_points(rows: list[dict], strategy: str, cls: str) -> list[tuple[int, bool]]:
    """(distance, recalled) pairs for probes that were both planted and
    asked, with asked after planted. See probes.Probe.distance for why
    negative/never-planted cases are excluded rather than clamped.

    The >= 0 check here is not redundant with Probe.distance: episodes
    collected before that fix landed have negative distances already baked
    into their serialized JSON (to_dict() only stores the computed number,
    not the raw planted_turn/asked_turn), so old result files still need
    this filter applied on read.
    """
    return [
        (p["distance"], p["outcome"] == "recalled")
        for r in rows
        if r["strategy"] == strategy
        for p in r["probes"]
        if p["class"] == cls and p["distance"] is not None and p["distance"] >= 0
    ]


def report(rows: list[dict]) -> None:
    order = [s for s in strategies.REGISTRY if any(r["strategy"] == s for r in rows)]
    print(f"total episodes: {len(rows)}\n")

    print("=== recall-by-distance curve, per strategy x class (pooled over seeds) ===")
    for s in order:
        print(f"\n{s}:")
        for c in CLASSES:
            points = valid_points(rows, s, c)
            if not points:
                print(f"  {c:<20} no valid samples (never discovered, or only asked before planted)")
                continue
            by_dist: dict[int, list[bool]] = defaultdict(list)
            for d, recalled in points:
                by_dist[d].append(recalled)
            curve = "  ".join(f"d={d}:{sum(v)}/{len(v)}" for d, v in sorted(by_dist.items()))
            print(f"  {c:<20} {curve}")

    print("\n=== context half life (turns to 50% recall), per strategy x class ===")
    print(f"{'strategy':<16}" + "".join(f"{c[:14]:>16}" for c in CLASSES))
    for s in order:
        cells = []
        for c in CLASSES:
            points = valid_points(rows, s, c)
            if not points:
                cells.append("no data")
                continue
            hl = half_life(points)
            if hl.turns is None:
                cells.append("n/a")
            elif hl.censored == "right":
                cells.append(f">{hl.turns:.0f}")
            elif hl.censored == "left":
                cells.append(f"<{hl.turns:.0f}")
            else:
                cells.append(f"{hl.turns:.1f}")
        print(f"{s:<16}" + "".join(f"{c:>16}" for c in cells))

    print("\n=== overall recall with 95% Wilson CI ===")
    for s in order:
        rs = [r for r in rows if r["strategy"] == s]
        outs = [p["outcome"] == "recalled" for r in rs for p in r["probes"]]
        k, n = sum(outs), len(outs)
        lo, hi = wilson_ci(k, n)
        print(f"{s:<16} {k}/{n} = {k/n:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n=== cache-aware cost: actual vs no-cache counterfactual ===")
    print(f"{'strategy':<16}{'actual $':>12}{'no-cache $':>12}{'ratio':>8}{'$/episode':>12}")
    for s in order:
        rs = [r for r in rows if r["strategy"] == s]
        actual = sum(cost_usd(r["usage"]) for r in rs)
        no_cache = sum(cost_usd_no_cache(r["usage"]) for r in rs)
        print(f"{s:<16}{actual:>12.4f}{no_cache:>12.4f}{no_cache / actual:>8.2f}{actual / len(rs):>12.4f}")

    total_actual = sum(cost_usd(r["usage"]) for r in rows)
    total_no_cache = sum(cost_usd_no_cache(r["usage"]) for r in rows)
    print(f"\ntotal actual cost: ${total_actual:.2f}")
    print(f"total no-cache counterfactual: ${total_no_cache:.2f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_full_matrix_real.jsonl"
    report(load(path))
