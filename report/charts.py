#!/usr/bin/env python3
"""Figures for the technical report. All numbers are the measured values from
the Phase 3 and Phase 4 summaries (energy: mean +/- sample SD over 3 runs)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # validated slots 1-3

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 220,
})


def style(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


MODES = ["15W", "25W", "MAXN SUPER"]

# ------------------------------------------------------------ Figure 1
E = {  # J per decision, mean and SD over 3 runs
    "Clean, no defense":          ([10.196, 10.100, 9.889], [0.017, 0.066, 0.048]),
    "Attacked, no defense":       ([9.319, 9.166, 9.061], [0.007, 0.036, 0.046]),
    "Attacked + PatchBlock (15%)": ([15.545, 15.763, 14.755], [0.093, 0.061, 0.053]),
}
fig, ax = plt.subplots(figsize=(6.6, 2.9))
x = np.arange(len(MODES))
w = 0.22
for i, ((name, (m, sd)), col) in enumerate(zip(E.items(), (BLUE, ORANGE, AQUA))):
    pos = x + (i - 1) * (w + 0.02)
    ax.bar(pos, m, w, color=col, label=name, zorder=2)
    ax.errorbar(pos, m, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    if i == 2:                                   # label only the story series
        for p, v in zip(pos, m):
            ax.text(p, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    if i == 1:
        for p, v in zip(pos, m):
            ax.text(p, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(x, MODES)
ax.set_ylabel("Energy per decision (J)")
ax.set_ylim(0, 18.5)
ax.set_title("Energy per decision, by power mode")
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.13))
style(ax)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_energy.png")
plt.close(fig)

# ------------------------------------------------------------ Figure 2
idle_w = 5.98
over_j = np.array([6.227, 6.597, 5.694])
over_sd = np.array([0.099, 0.042, 0.034])
lat_nodef = np.array([0.80, 0.65, 0.56])
lat_pb = np.array([1.69, 1.66, 1.36])
extra_t = lat_pb - lat_nodef                        # median latency overhead (s)
base_j = idle_w * extra_t                            # estimate: baseline power x extra time
rest_j = over_j - base_j

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1]})
w = 0.42
a1.bar(x, base_j, w, color=BLUE, label="Idle power x extra time (est.)", zorder=2,
       edgecolor=SURF, linewidth=1.0)
a1.bar(x, rest_j, w, bottom=base_j, color=ORANGE, label="Rest of overhead (CPU work, est.)", zorder=2,
       edgecolor=SURF, linewidth=1.0)
a1.errorbar(x, over_j, yerr=over_sd, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
for xi, b, t in zip(x, base_j, over_j):
    a1.text(xi, b / 2, f"{100 * b / t:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
    a1.text(xi, t + 0.2, f"+{t:.2f} J", ha="center", va="bottom", fontsize=7.5, color=INK)
a1.set_xticks(x, MODES)
a1.set_ylim(0, 7.8)
a1.set_ylabel("Defense energy overhead (J)")
a1.set_title("(a) Energy overhead per decision")
a1.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=1)
style(a1)

w = 0.34
a2.bar(x - w / 2 - 0.01, lat_nodef, w, color=ORANGE, label="Attacked, no defense", zorder=2)
a2.bar(x + w / 2 + 0.01, lat_pb, w, color=AQUA, label="Attacked + PatchBlock (15%)", zorder=2)
for xi, a, b in zip(x, lat_nodef, lat_pb):
    a2.text(xi + w / 2 + 0.01, b + 0.03, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a2.text(xi - w / 2 - 0.01, a + 0.03, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
a2.set_xticks(x, MODES)
a2.set_ylim(0, 2.0)
a2.set_ylabel("Median latency per decision (s)")
a2.set_title("(b) Latency per decision")
a2.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=1)
style(a2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_overhead.png")
plt.close(fig)

# ------------------------------------------------------------ Figure 3
fr = np.array([5, 10, 15, 20])
pb_rec = np.array([10, 20, 40, 40])                   # % of 10 scenes
rnd_rec = np.array([20, 30, 50, 63])                  # % of 30 trials
pb_hold = np.array([80, 80, 40, 40])
rnd_hold = np.array([57, 53, 40, 10])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), sharex=True)
for ax, pbv, rv, title, ref, reflabel in (
        (a1, pb_rec, rnd_rec, "(a) Correct action recovered", 70, "Oracle location: 70%"),
        (a2, pb_hold, rnd_hold, "(b) Attack still succeeds", 0, "Oracle location: 0%")):
    ax.plot(fr, pbv, color=BLUE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5,
            label="PatchBlock detection (n=10)", zorder=3, solid_capstyle="round")
    ax.plot(fr, rv, color=ORANGE, lw=2, marker="o", ms=6, mec=SURF, mew=1.5,
            label="Random chunks, same count (n=30)", zorder=3, solid_capstyle="round")
    ax.axhline(ref, color=INK2, lw=1, zorder=1)
    ax.text(5, ref + (2.5 if ref < 90 else -6), reflabel, fontsize=7.5, color=INK2, va="bottom")
    ax.set_ylim(-4, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(fr, [f"{v}%" for v in fr])
    ax.set_xlabel("Share of chunks flagged and smoothed")
    ax.set_title(title)
    style(ax)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.06, 0.0), ncol=2)
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30, wspace=0.28)
fig.savefig(f"{OUT}/fig3_defense.png")
plt.close(fig)
print("figures written")
