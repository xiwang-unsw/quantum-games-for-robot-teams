"""Plot the exact stage-duration delivery experiments for manuscript Section 5."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator
from figure_style import apply_style, style_axes, BLUE, ORANGE, GREEN, QUIET

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "delivery_results" / "stage_experiments.json").read_text())
apply_style()


def finish(ax, stem):
    style_axes(ax)
    ax.figure.tight_layout(pad=0.5)
    for suffix in ("pdf", "png"):
        ax.figure.savefig(ROOT / f"{stem}.{suffix}")
    plt.close(ax.figure)


fig, ax = plt.subplots(figsize=(3.45, 2.65))
ax.axhline(104, color=QUIET, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
ax.text(0.06, 104.45, "Classical policy", fontsize=6.8, color=QUIET)
for tm, label, color, marker, linestyle in [
    (None, "No memory decay", BLUE, "o", "-"),
    (2400, r"$T_m=2400$ s", ORANGE, "s", "--"),
    (600, r"$T_m=600$ s", GREEN, "^", "-."),
]:
    rows = [r for r in DATA["inventory"] if r["tm"] == tm]
    ax.plot([r["inventory"] for r in rows], [r["completion_time_s"] for r in rows],
            color=color, marker=marker, linestyle=linestyle, markerfacecolor="white",
            markeredgewidth=0.8, label=label, zorder=3)
ax.set(xlabel=r"Initial pair inventory, $e_0$", ylabel="Expected completion time (s)",
       xlim=(-0.3, 14.3), ylim=(83, 106.5), xticks=range(0, 15, 2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86), frameon=False,
          handlelength=2.4, borderaxespad=0.2, labelspacing=0.35)
finish(ax, "fig_delivery_inventory")

fig, ax = plt.subplots(figsize=(3.45, 2.65))
rows = DATA["memory_sweep"]
for key, label, color, marker, linestyle in [
    ("refill_s", "3 initial + 3 fresh at stage 4", GREEN, "^", "-."),
    ("initial_six_s", "6 initial, no replacement", BLUE, "o", "-"),
]:
    ax.plot([r["tm"] for r in rows], [r[key] for r in rows],
            color=color, marker=marker, linestyle=linestyle, markerfacecolor="white",
            markeredgewidth=0.8, markevery=2, label=label, zorder=3)
classical_time = DATA["validation"]["classical_exact"][0]
ax.axhline(classical_time, color=QUIET, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
ax.text(34, classical_time + 0.45, "Classical benchmark: 104 s", fontsize=6.8, color=QUIET)
ax.set(xscale="log", xlabel=r"Memory time constant, $T_m$ (s)",
       ylabel="Expected completion time (s)", xlim=(25, 36000), ylim=(85, 107))
ax.xaxis.set_major_locator(FixedLocator([30, 100, 1000, 10000, 30000]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_locator(FixedLocator([85, 90, 95, 100, 105]))
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.81), frameon=False,
          handlelength=2.4,
          borderaxespad=0.2, labelspacing=0.35)
finish(ax, "fig_delivery_ageing")
print("Wrote fig_delivery_inventory.pdf/png and fig_delivery_ageing.pdf/png")
