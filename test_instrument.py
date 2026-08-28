"""Instrument validity tests.

These do not test the science. They test that the measuring device does not
lie. Both invariants below have already caught real bugs in this harness.
"""
import pytest
from goldfish import run_episode, strategies
from goldfish.models import ContextBoundAgent
from goldfish.probes import Outcome, Probe, ProbeClass, default_probe_suite

UNBOUNDED = 10**9


def _run(strat, budget, seed=0):
    return run_episode(strat, ContextBoundAgent(seed=seed), seed=seed,
                       budget=budget, probes=default_probe_suite())


@pytest.mark.parametrize("seed", range(5))
def test_no_leak_under_full_history(seed):
    """Control must score a perfect ceiling. Anything less means the agent, the
    probes, or the environment is losing information the strategy did not."""
    r = _run(strategies.FullHistory(), UNBOUNDED, seed)
    assert all(p.outcome is Outcome.RECALLED for p in r.probes)


@pytest.mark.parametrize("name", list(strategies.REGISTRY))
def test_control_invariance(name):
    """With an unbounded budget every strategy must reduce to the control.
    If it does not, the strategy mutates history in ways unrelated to
    compaction, and every comparison against it is confounded."""
    base = _run(strategies.FullHistory(), UNBOUNDED).to_dict()
    got = _run(strategies.REGISTRY[name](), UNBOUNDED).to_dict()
    assert [p["outcome"] for p in got["probes"]] == [p["outcome"] for p in base["probes"]]
    assert got["behavioural"] == base["behavioural"]


def test_degradation_is_monotone_in_budget():
    """Tightening the budget must not improve recall. A non-monotonicity is a
    sign of an unstable environment or too few seeds, not a finding."""
    def recall(budget):
        rs = [_run(strategies.SlidingWindow(), budget, s) for s in range(5)]
        outs = [p.outcome for r in rs for p in r.probes]
        return sum(o is Outcome.RECALLED for o in outs) / len(outs)
    assert recall(150) <= recall(700) <= recall(UNBOUNDED)


def test_outcomes_are_three_valued():
    """Forgetting must be distinguishable from confident invention."""
    rs = [_run(strategies.Summarization(), 150, s) for s in range(5)]
    outs = {p.outcome for r in rs for p in r.probes}
    assert Outcome.HALLUCINATED in outs and Outcome.ADMITTED in outs


def test_short_numeric_truth_is_not_a_substring_false_positive():
    """"0" is a literal substring of "10", "20", "104"... A real model answers
    count probes in free text ("I've posted 10 transactions so far"), so plain
    substring matching would grade a confidently wrong count as recalled.
    Caught on the first live-model run, where this inflated every artifact
    state result until fixed."""
    p = Probe(id="x", cls=ProbeClass.ARTIFACT, question="how many?", expected=lambda env: "0", test_turn=0)
    assert p.grade("I have posted 10 transactions so far.", env=None) is Outcome.HALLUCINATED
    assert p.grade("0", env=None) is Outcome.RECALLED
    assert p.grade("I have posted 0 transactions so far.", env=None) is Outcome.RECALLED
