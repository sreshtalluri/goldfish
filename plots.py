"""Static charts for the README, generated from the same result files and
metrics functions report_m2.py uses -- these are report_m2.py's own numbers
rendered, not separately computed. Static PNGs for a markdown README: no
hover/interactivity/dark-mode variant, since GitHub renders a fixed image
regardless of viewer theme, unlike an in-app dashboard.

    python3 plots.py
"""
import json
import os

import matplotlib.pyplot as plt

from goldfish import strategies as strategies_mod
from goldfish.metrics import PRICES_BY_MODEL, SONNET_5_PRICES, cost_usd, half_life, wilson_ci

# dataviz skill's validated 3-slot categorical order (blue/orange/aqua):
# passes the all-pairs CVD + normal-vision floors in both light and dark
# modes, the strictest gate the skill defines. Aqua sits below the 3:1
# contrast floor against a light surface (a documented WARN, not a fail),
# so every aqua bar below carries a visible value label per the skill's
# relief rule instead of relying on color alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#0b0b0b"

CLASSES = ["identifier", "constraint", "negative_knowledge", "goal", "artifact_state", "provenance"]


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def valid_points(rows: list[dict], model: str, strategy: str, cls: str) -> list[tuple[int, bool]]:
    return [
        (p["distance"], p["outcome"] == "recalled")
        for r in rows
        if r["strategy"] == strategy and r["model"] == model
        for p in r["probes"]
        if p["class"] == cls and p["distance"] is not None and p["distance"] >= 0
    ]


def _strip_top_right(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_half_life_by_class(rows: list[dict], out: str) -> None:
    labels = [c.replace("_", " ") for c in CLASSES]
    values, censored = [], []
    for c in CLASSES:
        hl = half_life(valid_points(rows, "anthropic", "full_history", c))
        values.append(hl.turns)
        censored.append(hl.censored)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=BLUE, width=0.6)
    for bar, v, c in zip(bars, values, censored):
        symbol = {"right": "≥", "left": "≤"}.get(c, "")
        ax.annotate(
            f"{symbol}{v:.0f}",
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax.set_ylabel("turns until 50% recall (context half life)")
    ax.set_title(
        "Context half life by fact class — full_history control, claude-sonnet-5\n"
        "(no eviction at all: this is the task's own forgetting curve)"
    )
    _strip_top_right(ax)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_recall_by_strategy(rows: list[dict], out: str) -> None:
    order = [s for s in strategies_mod.REGISTRY if any(r["strategy"] == s for r in rows)]
    means, los, his = [], [], []
    for s in order:
        rs = [r for r in rows if r["strategy"] == s]
        outs = [p["outcome"] == "recalled" for r in rs for p in r["probes"]]
        k, n = sum(outs), len(outs)
        lo, hi = wilson_ci(k, n)
        means.append(k / n)
        los.append(k / n - lo)
        his.append(hi - k / n)

    order = sorted(zip(order, means, los, his), key=lambda t: -t[1])
    labels, means, los, his = zip(*order)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=[los, his], color=BLUE, capsize=4, width=0.6)
    for bar, m in zip(bars, means):
        ax.annotate(
            f"{m:.2f}",
            (bar.get_x() + bar.get_width() / 2, m),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("overall recall (95% Wilson CI)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall by compaction strategy — claude-sonnet-5, n=54 probes/strategy")
    _strip_top_right(ax)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_cost_by_model(anthropic_rows: list[dict], multimodel_rows: list[dict], out: str) -> None:
    strategies_common = ["full_history", "summarization", "sliding_window"]
    models = ["anthropic", "openrouter-gpt-5-mini", "openrouter-llama-3.3-70b"]
    model_labels = ["Claude Sonnet 5", "GPT-5-mini", "Llama-3.3-70B"]
    colors = [BLUE, ORANGE, AQUA]
    all_rows = anthropic_rows + multimodel_rows

    per_episode = {
        m: [
            sum(cost_usd(r["usage"], PRICES_BY_MODEL.get(m, SONNET_5_PRICES)) for r in all_rows if r["strategy"] == s and r["model"] == m)
            / len([r for r in all_rows if r["strategy"] == s and r["model"] == m])
            for s in strategies_common
        ]
        for m in models
    }

    x = list(range(len(models)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, s in enumerate(strategies_common):
        offsets = [xi + (i - 1) * width for xi in x]
        vals = [per_episode[m][i] for m in models]
        bars = ax.bar(offsets, vals, width=width, label=s, color=colors[i])
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"${v:.3f}",
                (bar.get_x() + bar.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("$ / episode (cache-aware actual cost)")
    ax.set_title(
        "Cache inversion is Claude-specific: full_history is cheapest on Claude,\n"
        "most expensive on both other models"
    )
    ax.legend(frameon=False)
    _strip_top_right(ax)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    anthropic_rows = load("results_full_matrix_real.jsonl")
    multimodel_rows = load("results_multimodel.jsonl")
    plot_half_life_by_class(anthropic_rows, "plots/half_life_by_class.png")
    plot_recall_by_strategy(anthropic_rows, "plots/recall_by_strategy.png")
    plot_cost_by_model(anthropic_rows, multimodel_rows, "plots/cost_by_model.png")
    print("wrote plots/half_life_by_class.png, plots/recall_by_strategy.png, plots/cost_by_model.png")
