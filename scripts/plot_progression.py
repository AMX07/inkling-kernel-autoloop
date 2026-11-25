"""Rebuild the progression figure from committed measurements; no GPU required.

uv run --no-project --with matplotlib==3.10.8 python scripts/plot_progression.py
"""
from pathlib import Path
import json
import statistics
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
fusion = json.loads((ROOT / "results/relative-bias.json").read_text())
routing = json.loads((ROOT / "results/tokenspeed-experiment.json").read_text())
run = next(r for r in routing["runs"] if r["id"] == "tokenspeed-20261115T193507Z")
assert len(run["cases"]) == 24
route_rows = []
for c in run["cases"]:
    assert c["eligible_for_performance_comparison"] and not c["instrumented"]
    groups = {}
    for sample in c["samples"]:
        groups.setdefault(sample["variant"], []).append(sample["wall_ms_per_invocation"])
    assert len(groups) == 2 and all(len(v) == 10 for v in groups.values())
    native = statistics.mean(groups["vllm-native-inkling2-route"])
    candidate = statistics.mean(groups["tokenspeed-pinned-native-consumer-adapter"])
    route_rows.append({**c["config"], "latency_change_percent": 100 * (candidate / native - 1)})

cases = {(c["phase"], c["length"]): c for c in fusion["cases"]}
for c in cases.values():
    assert c["verified"] and c["eligible_for_timing"] and not c["instrumented"]
    for variant in ("native", "fused"):
        mean = statistics.mean(o["wall_ms"] for o in c["observations"][variant])
        assert math.isclose(mean, c["latency_ms"][variant]["mean"], rel_tol=1e-10)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "svg.hashsalt": "inkling-progression"})
fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.12, 1]})
fig.subplots_adjust(left=.065, right=.97, top=.71, bottom=.26, wspace=.38)
fig.set_facecolor("#fafbfc")
fig.text(.065, .94, "From rejected routing to faster prefill", fontsize=24, weight="bold", color="#14293d")
fig.text(.065, .89, "Inkling kernel engineering · 2× B300 · measured November 15, 2026", fontsize=12, color="#536477")
fig.text(.065, .84, "Separate experiments and measurement boundaries — not a cumulative speedup curve.", fontsize=11)

ax = axes[0]
tokens = [1, 4, 16, 64, 256, 512]
styles = [("nvfp4", "realistic", "o", "#b84435", -.15),
          ("nvfp4", "imbalanced", "s", "#dc8b55", -.05),
          ("mxfp8", "realistic", "^", "#76518e", .05),
          ("mxfp8", "imbalanced", "D", "#ae85bd", .15)]
for quant, dist, marker, color, offset in styles:
    rows = sorted((r for r in route_rows if r["quantization"] == quant and r["routing_distribution"] == dist), key=lambda r: r["tokens"])
    ax.scatter([tokens.index(r["tokens"]) + offset for r in rows],
               [r["latency_change_percent"] for r in rows], marker=marker, color=color,
               s=38, label=f"{quant.upper()} / {dist}", zorder=3)
ax.set(title="1  Routing candidate: rejected", xlabel="Tokens per operation", ylabel="Operation latency change vs native (%)")
ax.set_xticks(range(6), tokens)
ax.set_ylim(-5, 110)
ax.axhline(0, color="#697888", linewidth=1)
ax.text(.03, .96, "+14.3% to +96.6% slower\nAll 24 eligible cases", transform=ax.transAxes, va="top", fontsize=10, color="#9c3025")
fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.065, .17), frameon=False, fontsize=8, ncol=4)

ax = axes[1]
lengths = [256, 1024, 4096]
native = [cases["prefill", n]["latency_ms"]["native"]["mean"] for n in lengths]
fused = [cases["prefill", n]["latency_ms"]["fused"]["mean"] for n in lengths]
for values, offset, color, label in [(native, -.19, "#8799ad", "Native"), (fused, .19, "#168475", "Fused")]:
    bars = ax.bar([i+offset for i in range(3)], values, width=.36, color=color, label=label, zorder=3)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
for i, n in enumerate(lengths):
    reduction = cases["prefill", n]["latency_reduction_percent"]
    ax.text(i, max(native[i], fused[i])+15, f"−{reduction:.1f}%", ha="center", weight="bold", color="#087567")
ax.set(title="2  Fusion: prefill improves", xlabel="Context tokens", ylabel="Complete eager-forward latency (ms)", ylim=(0, 175))
ax.set_xticks(range(3), ["256", "1,024", "4,096"])
ax.legend(frameon=False, loc="upper left")

ax = axes[2]
for i, n in enumerate(lengths):
    c = cases["decode", n]
    change = -c["latency_reduction_percent"]
    # Exported interval is native/fused speedup; invert endpoints for latency change.
    ci = c["paired_bootstrap95"]
    low, high = 100*(1/ci["upper"]-1), 100*(1/ci["lower"]-1)
    ax.errorbar(change, i, xerr=[[change-low], [high-change]], fmt="o", capsize=5,
                color="#b84435" if change > 0 else "#168475", markersize=7, zorder=3)
    ax.annotate(f"{change:+.2f}%", (change, i), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=10)
ax.axvline(0, color="#697888", linestyle="--", linewidth=1)
ax.set(title="3  Decode gate: not adopted", xlabel="Decode latency change vs native (%)", ylabel="Cached context tokens", xlim=(-3.3, 3.3), ylim=(-.65, 2.65))
ax.set_yticks(range(3), ["256", "1,024", "4,096"])
ax.invert_yaxis()
for ax in axes:
    ax.set_facecolor("#fafbfc")
    ax.grid(axis="y" if ax != axes[2] else "x", color="#e2e7ed", zorder=0)

fig.text(.065, .105, "Primary result: 4,096-token prefill 135.97 → 105.88 ms  |  22.1% lower latency  |  1.284× speedup", fontsize=12, weight="bold", color="#087567")
fig.text(.065, .06, "Decode bars: paired 95% bootstrap intervals (within-session uncertainty). All span parity; no causal decode regression established.\nThe strict mean no-regression gate still failed at 256 tokens. No serving-throughput improvement or adoption claimed.", fontsize=10, color="#536477")
for ext in ("png", "svg"):
    fig.savefig(ROOT / f"charts/progression-performance.{ext}", dpi=180, metadata={"Date": None} if ext == "svg" else None)
print("Verified plotted means against saved observations; wrote PNG and SVG.")
