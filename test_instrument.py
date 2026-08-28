"""Instrument validity tests.

These do not test the science. They test that the measuring device does not
lie. Both invariants below have already caught real bugs in this harness.
"""
import pytest
from goldfish import run_episode, strategies
from goldfish.env import LedgerEnv
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


def test_file_report_rejects_non_numeric_total():
    """A real model once passed total_posted as a description
    ("$0.00 USD (0 of 16 posted)") instead of a bare number, which crashed
    score_behavioural downstream instead of failing the tool call. The
    environment must reject this at the boundary, not accept it silently."""
    env = LedgerEnv(seed=0)
    result = env.call("file_report", {"total_posted": "$0.00 USD (0 of 16 posted)", "accounts_touched": 0})
    assert not result.ok
    assert env.report_filed is None


def test_distance_is_none_when_asked_before_planted():
    """A probe asked before its fact was ever planted isn't testing recall,
    it's testing whether the trivial pre-dependency state happens to match
    the question (e.g. "0 transactions posted" before any posting has
    occurred). Surfaced in a real M2 distance sweep where an early test_turn
    landed before a late-planting artifact_state probe."""
    p = Probe(id="x", cls=ProbeClass.ARTIFACT, question="how many?", expected=lambda env: "0", test_turn=5)
    p.asked_turn = 5
    p.planted_turn = 12
    assert p.distance is None


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


def test_annotate_overwrites_not_accumulates():
    env = LedgerEnv(seed=0)
    env.call("annotate", {"note": "first"})
    env.call("annotate", {"note": "second"})
    assert env.annotation == "second"


def test_structured_notes_preserves_latest_annotation_under_pressure():
    messages = [{"role": "user", "kind": "goal", "content": "x" * 500}]
    messages.append({"role": "assistant", "kind": "action", "content": "annotate({'note': 'critical'})"})
    messages.append({"role": "user", "kind": "observation", "tool": "annotate", "content": '{"bytes": 8}'})
    for i in range(20):
        messages.append({"role": "assistant", "kind": "action", "content": f"noise {i}" * 20})
        messages.append({"role": "user", "kind": "observation", "tool": "list_pending", "content": "y" * 200})
    out = strategies.StructuredNotes().reduce(messages, budget=50)
    assert any(m.get("tool") == "annotate" for m in out)


def test_structured_notes_never_ends_on_assistant_when_note_is_last():
    """If the annotate call+observation pair is the most recent thing in
    history, excluding only the observation (not its paired call) leaves the
    unpaired assistant-role call as the new tail. The offline simulator never
    calls annotate, so this only ever showed up on a live model run."""
    messages = [{"role": "user", "kind": "goal", "content": "x" * 500}]
    for i in range(20):
        messages.append({"role": "assistant", "kind": "action", "content": f"noise {i}" * 20})
        messages.append({"role": "user", "kind": "observation", "tool": "list_pending", "content": "y" * 200})
    messages.append({"role": "assistant", "kind": "action", "content": "annotate({'note': 'critical'})"})
    messages.append({"role": "user", "kind": "observation", "tool": "annotate", "content": '{"bytes": 8}'})
    out = strategies.StructuredNotes().reduce(messages, budget=50)
    assert out[-1]["role"] != "assistant"


def test_generation_at_ask_increments_under_pressure():
    """A tight budget with a long episode should force multiple real
    compactions before the last probe is asked, and each probe's recorded
    generation should be non-decreasing over the episode."""
    r = run_episode(strategies.SlidingWindow(), ContextBoundAgent(seed=0), seed=0,
                     budget=150, probes=default_probe_suite())
    gens = [p.generation_at_ask for p in r.probes if p.asked_turn is not None]
    assert any(g and g > 0 for g in gens), "expected at least one probe asked after a real compaction"
    assert gens == sorted(gens)
