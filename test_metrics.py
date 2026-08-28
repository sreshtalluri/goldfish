"""Tests for the M2 metrics: these must be right independent of any model or
environment, since a wrong confidence interval or half life formula would
misreport every downstream finding without ever failing a harness invariant.
"""
from goldfish.metrics import cost_usd, cost_usd_no_cache, half_life, wilson_ci


def test_wilson_ci_contains_point_estimate():
    for k, n in [(0, 3), (3, 3), (2, 3), (1, 10), (9, 10)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_ci_zero_n_is_maximally_uncertain():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_narrows_with_more_data():
    lo_small, hi_small = wilson_ci(1, 2)
    lo_big, hi_big = wilson_ci(50, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_ci_extreme_proportions_stay_in_bounds():
    # The normal approximation gives a zero-width or out-of-range interval
    # here; this is exactly the case Wilson exists for in this project,
    # since many (strategy, class) cells are exactly 0/3 or 3/3 recalled.
    lo, hi = wilson_ci(0, 3)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson_ci(3, 3)
    assert hi == 1.0 and lo < 1.0


def test_cost_no_cache_is_at_least_actual_cost():
    """Caching can only ever reduce or match cost relative to not caching:
    a cache read is always cheaper than the base input rate it stands in
    for. If this ever fails, either the pricing table or the formula is
    wrong."""
    usage = {"input": 1000, "output": 500, "cache_write": 2000, "cache_read": 8000}
    assert cost_usd_no_cache(usage) >= cost_usd(usage)


def test_cost_no_cache_equals_actual_when_nothing_cached():
    usage = {"input": 1000, "output": 500, "cache_write": 0, "cache_read": 0}
    assert cost_usd_no_cache(usage) == cost_usd(usage)


def test_cost_is_linear_and_matches_hand_calculation():
    usage = {"input": 1_000_000, "output": 1_000_000, "cache_write": 1_000_000, "cache_read": 1_000_000}
    # sonnet-5: $2 + $10 + $2.50 + $0.20 = $14.70
    assert abs(cost_usd(usage) - 14.70) < 1e-9


def test_half_life_interpolates_the_crossing():
    # 100% recall at distance 10, 0% at distance 30: crossing should land
    # at the midpoint under linear interpolation.
    points = [(10, True), (10, True), (30, False), (30, False)]
    hl = half_life(points)
    assert hl.censored is None
    assert abs(hl.turns - 20.0) < 1e-9


def test_half_life_right_censored_when_never_drops_below_half():
    points = [(10, True), (20, True), (30, True)]
    hl = half_life(points)
    assert hl.censored == "right"
    assert hl.turns == 30.0


def test_half_life_left_censored_when_already_below_half_at_shortest_distance():
    points = [(10, False), (20, False)]
    hl = half_life(points)
    assert hl.censored == "left"
    assert hl.turns == 10.0


def test_half_life_empty_input():
    hl = half_life([])
    assert hl.turns is None and hl.censored is None
