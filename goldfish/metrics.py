"""M2: confidence intervals, context half life, cache-aware cost accounting.

Per PRD section 6: everything reported with n, seeds, and confidence
intervals, or it is a blog post that gets dismantled in the comments. This
module is what turns the M1 point estimates into that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Normal-approximation CIs break exactly where this project's data lives:
    small n, and proportions that sit at or near 0 or 1 (many (strategy,
    class) cells are exactly 0.00 or 1.00 recall over 3 seeds). Wilson stays
    within [0, 1] and is well behaved there; a normal approximation would
    report a zero-width or out-of-range interval on those cells.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return (max(0.0, lo), min(1.0, hi))


# claude-sonnet-5, official pricing as of 2026-08-28 (platform.claude.com/docs/en/about-claude/pricing).
# 5-minute cache write rate, since AnthropicAdapter requests ephemeral caching
# without an explicit TTL, which defaults to the 5-minute cache.
SONNET_5_PRICES = {
    "input": 2.00,
    "output": 10.00,
    "cache_write": 2.50,
    "cache_read": 0.20,
}  # USD per million tokens


def cost_usd(usage: dict[str, int], prices: dict[str, float] = SONNET_5_PRICES) -> float:
    """Actual dollar cost of a usage dict, cache pricing as billed."""
    return (
        usage.get("input", 0) * prices["input"]
        + usage.get("output", 0) * prices["output"]
        + usage.get("cache_write", 0) * prices["cache_write"]
        + usage.get("cache_read", 0) * prices["cache_read"]
    ) / 1_000_000


def cost_usd_no_cache(usage: dict[str, int], prices: dict[str, float] = SONNET_5_PRICES) -> float:
    """Counterfactual cost if this exact conversation had been sent with
    caching turned off. cache_read tokens are content that caching let us
    reprocess at 10% of base price; without caching that content is ordinary
    input, billed at the full base rate. cache_write tokens are exactly the
    input tokens that were being cached in the first place, so without
    caching they are ordinary input too, and there is no write charge at all.
    This is the number the PRD section 6 cost-inversion hypothesis needs:
    whether a strategy that reduces token count can still cost *more* than
    the control once caching is accounted for, because eviction rewrites the
    prefix and invalidates the cache on every compaction.
    """
    equivalent_input = usage.get("input", 0) + usage.get("cache_read", 0) + usage.get("cache_write", 0)
    return (equivalent_input * prices["input"] + usage.get("output", 0) * prices["output"]) / 1_000_000


@dataclass
class HalfLife:
    """turns is the estimated distance at which recall crosses 50%, by linear
    interpolation between the two nearest observed points. censored explains
    when turns is not a real crossing: recall never dropped below 50% within
    the tested range ("right", half life is at least the max distance tested,
    possibly much more), or it started below 50% at the shortest distance
    tested ("left", half life is at most the min distance, possibly much
    less). Reporting a bare number in the censored cases would imply
    precision the data doesn't have.
    """

    turns: float | None
    censored: str | None  # None | "left" | "right"


def half_life(points: list[tuple[int, bool]]) -> HalfLife:
    """points: (distance, recalled) pairs, arbitrary order, arbitrary repeats
    per distance. Buckets by distance, computes empirical recall per bucket,
    and linearly interpolates the first crossing of 0.5 as distance
    increases. Assumes recall is roughly monotone non-increasing in distance,
    which is the thing this whole project is testing, not assuming away —
    with only a few distinct distances and few seeds, a single noisy bucket
    can violate it, so this takes the first crossing found rather than
    fitting a smoother monotone curve, which would be a stronger claim than
    3-seed data supports.
    """
    if not points:
        return HalfLife(None, None)
    by_distance: dict[int, list[bool]] = {}
    for d, recalled in points:
        by_distance.setdefault(d, []).append(recalled)
    xs = sorted(by_distance)
    ys = [sum(by_distance[x]) / len(by_distance[x]) for x in xs]

    if ys[0] < 0.5:
        return HalfLife(float(xs[0]), "left")
    if ys[-1] >= 0.5:
        return HalfLife(float(xs[-1]), "right")

    for i in range(len(xs) - 1):
        if ys[i] >= 0.5 > ys[i + 1]:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            frac = (y0 - 0.5) / (y0 - y1)
            return HalfLife(x0 + frac * (x1 - x0), None)
    return HalfLife(float(xs[-1]), "right")


def recall_by_generation(points: list[tuple[int, bool]]) -> dict[int, tuple[int, int]]:
    """points: (generation, recalled) pairs. Returns generation -> (recalled,
    total), for the section 6 question: does fidelity loss compound linearly
    across compaction generations, or does it cliff after some threshold.
    """
    buckets: dict[int, list[bool]] = {}
    for gen, recalled in points:
        buckets.setdefault(gen, []).append(recalled)
    return {gen: (sum(vals), len(vals)) for gen, vals in sorted(buckets.items())}
