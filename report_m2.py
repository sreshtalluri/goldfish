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
from goldfish.metrics import PRICES_BY_MODEL, SONNET_5_PRICES, cost_usd, cost_usd_no_cache, half_life, recall_by_generation, wilson_ci

# Strategies whose "generation" means what PRD section 6 means by it: a
# discrete recompaction event, so generation N+1 is built from generation N's
# already-compacted output ("summarizing a summary"). The oldest-first
# strategies (sliding_window, tool_masking, retrieval, scratchpad,
# structured_notes) also increment a generation counter every time reduce()
# evicts anything, but under sustained budget pressure that fires almost
# every turn -- their "generation" is much closer to a proxy for turns spent
# under pressure than a count of discrete compounding events, and reporting
# it next to summarization's without saying so would imply a comparability
# the data doesn't have.
RECOMPACTING_STRATEGIES = {"summarization"}

CLASSES = ["identifier", "constraint", "negative_knowledge", "goal", "artifact_state", "provenance"]


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def valid_points(rows: list[dict], model: str, strategy: str, cls: str) -> list[tuple[int, bool]]:
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
        if r["strategy"] == strategy and r["model"] == model
        for p in r["probes"]
        if p["class"] == cls and p["distance"] is not None and p["distance"] >= 0
    ]


def report(rows: list[dict]) -> None:
    # (model, strategy) rather than strategy alone: a single results file can
    # now hold more than one model (see results_multimodel.jsonl), and
    # pooling rows across models into one "strategy" bucket would silently
    # average away the exact cross-model comparison this report exists to make.
    order = sorted({(r["model"], r["strategy"]) for r in rows}, key=lambda ms: (ms[0], list(strategies.REGISTRY).index(ms[1])))
    models_present = len({m for m, _ in order})
    print(f"total episodes: {len(rows)}\n")

    print("=== recall-by-distance curve, per model x strategy x class (pooled over seeds) ===")
    for m, s in order:
        print(f"\n{m} / {s}:")
        for c in CLASSES:
            points = valid_points(rows, m, s, c)
            if not points:
                print(f"  {c:<20} no valid samples (never discovered, or only asked before planted)")
                continue
            by_dist: dict[int, list[bool]] = defaultdict(list)
            for d, recalled in points:
                by_dist[d].append(recalled)
            curve = "  ".join(f"d={d}:{sum(v)}/{len(v)}" for d, v in sorted(by_dist.items()))
            print(f"  {c:<20} {curve}")

    print("\n=== context half life (turns to 50% recall), per model x strategy x class ===")
    label_width = 16 + (10 if models_present > 1 else 0)
    print(f"{'model / strategy':<{label_width}}" + "".join(f"{c[:14]:>16}" for c in CLASSES))
    for m, s in order:
        label = f"{m}/{s}" if models_present > 1 else s
        cells = []
        for c in CLASSES:
            points = valid_points(rows, m, s, c)
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
        print(f"{label:<{label_width}}" + "".join(f"{c:>16}" for c in cells))

    print("\n=== overall recall with 95% Wilson CI ===")
    for m, s in order:
        rs = [r for r in rows if r["strategy"] == s and r["model"] == m]
        outs = [p["outcome"] == "recalled" for r in rs for p in r["probes"]]
        k, n = sum(outs), len(outs)
        lo, hi = wilson_ci(k, n)
        label = f"{m}/{s}" if models_present > 1 else s
        print(f"{label:<{label_width}} {k}/{n} = {k/n:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n=== cache-aware cost: actual vs no-cache counterfactual ===")
    print(f"{'model / strategy':<{label_width}}{'actual $':>12}{'no-cache $':>12}{'ratio':>8}{'$/episode':>12}")
    for m, s in order:
        rs = [r for r in rows if r["strategy"] == s and r["model"] == m]
        prices = PRICES_BY_MODEL.get(m, SONNET_5_PRICES)
        actual = sum(cost_usd(r["usage"], prices) for r in rs)
        no_cache = sum(cost_usd_no_cache(r["usage"], prices) for r in rs)
        label = f"{m}/{s}" if models_present > 1 else s
        ratio = no_cache / actual if actual else float("nan")
        print(f"{label:<{label_width}}{actual:>12.4f}{no_cache:>12.4f}{ratio:>8.2f}{actual / len(rs):>12.4f}")

    total_actual = sum(cost_usd(r["usage"], PRICES_BY_MODEL.get(r["model"], SONNET_5_PRICES)) for r in rows)
    total_no_cache = sum(cost_usd_no_cache(r["usage"], PRICES_BY_MODEL.get(r["model"], SONNET_5_PRICES)) for r in rows)
    print(f"\ntotal actual cost: ${total_actual:.2f}")
    print(f"total no-cache counterfactual: ${total_no_cache:.2f}")


def generation_report(rows: list[dict]) -> None:
    order = sorted({(r["model"], r["strategy"]) for r in rows}, key=lambda ms: (ms[0], list(strategies.REGISTRY).index(ms[1])))
    print("=== recall by compaction generation ===")
    print("(recompacting strategies only: generation is a discrete re-summarization")
    print(" count, comparable turn to turn. Others increment on every real eviction,")
    print(" which under a tight budget fires almost every turn -- see comment in")
    print(" report_m2.py. Reported for completeness, not as a compounding curve.)\n")
    for m, s in order:
        if s in RECOMPACTING_STRATEGIES:
            label = " (recompacting)"
        elif s == "full_history":
            label = " (never evicts)"
        else:
            label = " (near-continuous, see caveat)"
        rs = [r for r in rows if r["strategy"] == s and r["model"] == m]
        points = [
            (p["generation_at_ask"], p["outcome"] == "recalled")
            for r in rs
            for p in r["probes"]
            if p["generation_at_ask"] is not None
        ]
        buckets = recall_by_generation(points)
        curve = "  ".join(f"g{g}:{k}/{n}" for g, (k, n) in buckets.items())
        print(f"{m}/{s}{label}:\n  {curve}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results_full_matrix_real.jsonl"
    rows = load(path)
    report(rows)
    if any("generation_at_ask" in p for r in rows for p in r["probes"]):
        print()
        generation_report(rows)
