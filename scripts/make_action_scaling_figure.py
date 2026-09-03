"""Main-text figure 2: single- and two-joint reacher schematics with scaling results.

Row 1: A = single-joint reacher schematic; B = final success rate vs N (standard RL vs SL embedding)
Row 2: C = two-joint reacher schematic;    D = final success rate vs N=k^2 (standard RL vs embedding)

DATA: panels B and D load real results via load_ring_scaling_data() / load_two_joint_scaling_data().
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from definitions import paper_fig_dir, revision_fig_dir

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "pdf.fonttype": 42,
})

COLOR = {"standard": "#A94850", "bottleneck": "#3B6FB0", "sl": "#2E8B57", "embedding": "#2E8B57"} # sl and embedding are both our model
LABEL = {"standard": "Standard RL", "bottleneck": "Fully-RL 2-D bottleneck",
         "sl": "RL with representations for actions", "embedding": "RL with representations for actions"}
ROW_TITLE = ["Single-joint reacher", "Two-joint reacher"]

TITLE_Y = 0.85  # baseline (data coords) for schematic titles

# Layout constants, in inches - figure size is derived from these (see build_figure)
MARGIN_LEFT_IN, MARGIN_RIGHT_IN = 0.2, 0.2
MARGIN_TOP_IN, MARGIN_BOTTOM_IN = 0.05, 0.2
ROW_HEIGHT_IN = 1.7             # height of one schematic+plot row
PLOT_WIDTH_IN = 4.4
PLOT_HEIGHT_IN = 1.0            # top-aligned within each row; below ~1.25in the rotated y-label
                                 # no longer fits inside the axes box and pokes into the panel letter
SCHEMATIC_PLOT_GAP_IN = 0.2     # minimum whitespace between a schematic and the plot's true left edge
DESIRED_UNIT_TO_IN = 0.9        # aesthetic cap on schematic scale, picked by eye

FIG_WIDTH_IN = 7.2
FIGSIZE_IN = (FIG_WIDTH_IN, MARGIN_TOP_IN + 2 * ROW_HEIGHT_IN + MARGIN_BOTTOM_IN)


# Generic drawing helpers (plain data-coordinate primitives, no per-element axes)
def draw_link(ax, p0, p1, lw=3.5, color="0.35"):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, solid_capstyle="round", zorder=1)


def draw_joint(ax, p, r=0.105, hollow=True, color="0.6", lw=1.4):
    face = "white" if hollow else color
    ax.add_patch(Circle(p, r, facecolor=face, edgecolor="0.15", lw=lw, zorder=3))


def draw_effector(ax, p, r=0.105, color=COLOR["sl"]):
    ax.add_patch(Circle(p, r, facecolor=color, edgecolor="0.15", lw=1.2, zorder=4))


def angle_label(ax, center, theta_deg, text, r_arc=0.28, style="italic"):
    arc_t = np.linspace(0, np.radians(theta_deg), 20)
    ax.plot(center[0] + r_arc * np.cos(arc_t), center[1] + r_arc * np.sin(arc_t),
            color="0.4", lw=1.0, zorder=2)
    tx = center[0] + (r_arc + 0.16) * np.cos(np.radians(theta_deg) / 2)
    ty = center[1] + (r_arc + 0.16) * np.sin(np.radians(theta_deg) / 2)
    ax.text(tx, ty, text, fontsize=11, ha="center", va="center", style=style)


def title_at(ax, x, text, style="normal", weight="normal", color="black", fontsize=9):
    ax.text(x, TITLE_Y, text, ha="center", va="bottom", style=style, weight=weight, color=color, fontsize=fontsize)


def panel_label_at(fig, x, y_top, label, dx=-0.012):
    fig.text(x + dx, y_top, label, fontsize=14, fontweight="bold", ha="left", va="top")


def label_group(ax, entries, loc="upper left", fontsize=10.5, line_gap_frac=None):
    if line_gap_frac is None:
        fig_h_in = ax.figure.get_size_inches()[1]
        axes_h_in = ax.get_position().height * fig_h_in
        line_gap_frac = (fontsize * 1.3 / 72) / axes_h_in
    x = 0.03 if "left" in loc else 0.97
    ha = "left" if "left" in loc else "right"
    y = 0.95 if "upper" in loc else 0.05
    va = "top" if "upper" in loc else "bottom"
    step = -line_gap_frac if "upper" in loc else line_gap_frac
    for i, (text, color) in enumerate(entries):
        ax.text(x, y + i * step, text, transform=ax.transAxes, color=color, fontsize=fontsize,
                 fontweight="bold", ha=ha, va=va)


# Schematics 
def draw_single_joint_schematic(ax): # single joint schematic - panel A
    theta_deg, link_len = 35, 0.96
    origin = (0.0, 0.0)
    tip = (link_len * np.cos(np.radians(theta_deg)), link_len * np.sin(np.radians(theta_deg)))
    draw_link(ax, origin, tip)
    draw_joint(ax, origin, hollow=True)
    draw_effector(ax, tip, color=COLOR["sl"])
    angle_label(ax, origin, theta_deg, r"$\theta$")
    ax.plot([0, 1.2], [0, 0], color="0.75", lw=1.0, ls="--", zorder=0)
    title_at(ax, 0.3, ROW_TITLE[0], fontsize=10, weight="bold", color="k")

    ax.set_xlim(-0.75, 1.35)
    ax.set_ylim(-0.3, TITLE_Y + 0.35)                      
    ax.axis("off")


def draw_two_joint_reacher_and_grid(ax): # two joint schematic - panel C
    t1_deg, t2_deg, l1, l2 = 40, -55, 0.66, 0.66
    origin = (0.0, 0.0)
    elbow = (l1 * np.cos(np.radians(t1_deg)), l1 * np.sin(np.radians(t1_deg)))
    tip = (elbow[0] + l2 * np.cos(np.radians(t1_deg + t2_deg)),
           elbow[1] + l2 * np.sin(np.radians(t1_deg + t2_deg)))
    draw_link(ax, origin, elbow)
    draw_link(ax, elbow, tip)
    draw_joint(ax, origin, hollow=True)
    draw_joint(ax, elbow, hollow=False, color=COLOR["bottleneck"])
    draw_effector(ax, tip, color=COLOR["sl"])
    angle_label(ax, origin, t1_deg, r"$\theta_1$")
    ax.text(elbow[0] + 0.12, elbow[1] + 0.18, r"$\theta_2$", fontsize=11, style="italic")
    ax.plot([0, 1.05], [0, 0], color="0.75", lw=1.0, ls="--", zorder=0)
    title_at(ax, 0.4, ROW_TITLE[1], fontsize=10, weight="bold", color="k")
    ax.set_xlim(-0.514, 1.45)
    ax.set_ylim(-0.3, TITLE_Y + 0.35)
    ax.axis("off")


# Physical-scale placement
def place_schematic_axes(fig, draw_fn, unit_to_in, left, vcenter):
    """Create a schematic axes, let draw_fn populate it, then size+position the axes' box so it
    renders at unit_to_in inches per data unit, left-anchored at `left` and vertically centered at
    `vcenter` (both figure-fraction)."""
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax = fig.add_axes([left, 0.01, 0.01, 0.01])  # placeholder box; resized below
    draw_fn(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = (x1 - x0) * unit_to_in / fig_w_in
    height = (y1 - y0) * unit_to_in / fig_h_in
    bottom = vcenter - height / 2
    ax.set_position([left, bottom, width, height])
    ax.set_aspect("equal", anchor="W")  # safety net
    return ax


def build_single_joint_schematic(fig, unit_to_in, left, vcenter):
    return place_schematic_axes(fig, draw_single_joint_schematic, unit_to_in, left, vcenter)


def build_two_joint_schematic(fig, unit_to_in, left, vcenter):
    return place_schematic_axes(fig, draw_two_joint_reacher_and_grid, unit_to_in, left, vcenter)


# Panel B: single-joint final success rate vs N
# DATA: figures/paper/single_joint_success_cont.json (or single_joint_success.json)
def load_single_joint_scaling_data():
    data_path = Path(paper_fig_dir) / "single_joint_success_cont.json"
    if not data_path.exists():
        data_path = Path(paper_fig_dir) / "single_joint_success.json"
    values = defaultdict(list)
    for row in json.loads(data_path.read_text()):
        values[(row["agent"], row["N"])].append(row["success"])
    return aggregate_scaling_data(values)


def aggregate_scaling_data(values):
    aggregated = {}
    for key, points in values.items():
        valid = [point for point in points if point is not None and not np.isnan(point)]
        if valid:
            aggregated[key] = (np.mean(valid), np.std(valid) / max(1, np.sqrt(len(valid))))
    return aggregated


def build_ring_scaling_plot(fig, left, bottom, width, height, plot_bottleneck=False):
    ax = fig.add_axes([left, bottom, width, height])
    stats = load_single_joint_scaling_data()
    agents = ["standard", "sl", "bottleneck"] if plot_bottleneck else ["standard", "sl"]
    labels = []
    for agent in agents:
        pts = sorted((N, m, s) for (a, N), (m, s) in stats.items() if a == agent)
        if not pts:
            continue
        Ns, ms, ss = (np.array(z) for z in zip(*pts))
        ax.plot(Ns, ms, color=COLOR[agent], lw=2)
        ax.fill_between(Ns, ms - ss, ms + ss, color=COLOR[agent], alpha=0.2, lw=0)
        labels.append((LABEL[agent], COLOR[agent]))
    ax.set_xscale("log")
    ax.set_xlabel("Number of actions N")
    ax.set_ylabel("Success rate")  
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    label_group(ax, labels, loc="lower left", fontsize=8.5)
    return ax


# Panel D: two-joint final success rate vs N=k^2
# DATA: figures/paper/prrfp_final_*_k*.json from proprio_reacher_rl.py
def load_two_joint_scaling_data():
    rows = []
    for fp in glob.glob(str(Path(paper_fig_dir) / "prrfp_final_*_k*.json")):
        rows += json.load(open(fp))
    d = defaultdict(list)
    for r in rows:
        d[(r["agent"], r["N"])].append(r["final_success"])
    return aggregate_scaling_data(d)


def build_two_joint_scaling_plot(fig, left, bottom, width, height):
    ax = fig.add_axes([left, bottom, width, height])
    stats = load_two_joint_scaling_data()
    for agent in ["standard", "embedding"]:
        pts = sorted((N, m, s) for (a, N), (m, s) in stats.items() if a == agent)
        if not pts:
            continue
        Ns, ms, ss = (np.array(z) for z in zip(*pts))
        ax.plot(Ns, ms, color=COLOR[agent], lw=2)
        ax.fill_between(Ns, ms - ss, ms + ss, color=COLOR[agent], alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of actions $N=k^2$")
    ax.set_ylabel("Success rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    return ax


# Assembly
def build_figure():
    fig = plt.figure(figsize=FIGSIZE_IN)
    fig_w_in, fig_h_in = fig.get_size_inches()

    margin_left = MARGIN_LEFT_IN / fig_w_in
    margin_right = MARGIN_RIGHT_IN / fig_w_in
    margin_top = MARGIN_TOP_IN / fig_h_in
    row_h = ROW_HEIGHT_IN / fig_h_in

    row1_top, row1_bot = 1 - margin_top, 1 - margin_top - row_h
    row2_top, row2_bot = row1_bot, row1_bot - row_h

    # 1) Plots are placed first, at a fixed width/right edge, top-aligned within their row
    plot_left = 1 - margin_right - PLOT_WIDTH_IN / fig_w_in
    plot_width = PLOT_WIDTH_IN / fig_w_in
    plot_h = PLOT_HEIGHT_IN / fig_h_in
    ax_b = build_ring_scaling_plot(fig, plot_left, row1_top - plot_h, plot_width, plot_h)
    ax_d = build_two_joint_scaling_plot(fig, plot_left, row2_top - plot_h, plot_width, plot_h)

    # 2) Draw once so tick-label extents are real, then measure each plot's true left edge
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    true_left_b = ax_b.get_tightbbox(renderer).transformed(inv).x0
    true_left_d = ax_d.get_tightbbox(renderer).transformed(inv).x0
    min_plot_left = min(true_left_b, true_left_d)

    # 3) Derive one shared UNIT_TO_IN from whichever row is more space-constrained
    avail_schematic_in = (min_plot_left - margin_left) * fig_w_in - SCHEMATIC_PLOT_GAP_IN
    xrange_single = 1.35 - (-0.75)       
    xrange_two_joint = 1.45 - (-0.514)   # draw_two_joint_reacher_and_grid's own xlim range
    yrange = (TITLE_Y + 0.35) - (-0.3)   # shared by both schematics
    unit_to_in = min(avail_schematic_in / xrange_single,
                      avail_schematic_in / xrange_two_joint,
                      ROW_HEIGHT_IN / yrange,
                      DESIRED_UNIT_TO_IN)
    if unit_to_in < 0.3:
        print(f"WARNING: schematics are being squeezed to UNIT_TO_IN={unit_to_in:.3f} -- "
              f"consider a wider PLOT_WIDTH_IN margin or a wider FIG_WIDTH_IN.")

    # 4) Place both schematics at that shared physical scale, vertically centered in their row.
    ax_a = build_single_joint_schematic(fig, unit_to_in, margin_left, (row1_top + row1_bot) / 2)
    ax_c = build_two_joint_schematic(fig, unit_to_in, margin_left, (row2_top + row2_bot) / 2)

    # 5) Panel labels share their row's top (row1_top/row2_top)
    panel_label_at(fig, margin_left, row1_top, "A")
    panel_label_at(fig, true_left_b, row1_top, "B")
    panel_label_at(fig, margin_left, row2_top, "C")
    panel_label_at(fig, true_left_d, row2_top, "D")

    return fig


def main():
    fig = build_figure()
    out = Path(revision_fig_dir) / "main_text_action_space_fig"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".svg", bbox_inches="tight")
    print("saved", out.name + ".pdf/.png/.svg")


if __name__ == "__main__":
    main()
