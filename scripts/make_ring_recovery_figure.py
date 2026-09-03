"""Supplementary figure 8: fully 2-D bottleneck control - 
    Looks at success rate on single joint reacher task with control, 
    also looks at recovered ring-alignment score 

A = single-joint reacher schematic (reused from make_action_scaling_figure)
B = final success rate vs N, now including the 2-D bottleneck baseline
C = recovered ring-alignment score vs N (RL-bottleneck vs SL embedding), reused from
    make_bottleneck_learning_figure so the two figures never drift on how that score is computed
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, re
from collections import defaultdict

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from definitions import paper_model_path, revision_fig_dir
import torch

from scripts.make_action_scaling_figure import (
    TITLE_Y, MARGIN_LEFT_IN, MARGIN_RIGHT_IN, MARGIN_TOP_IN, MARGIN_BOTTOM_IN, ROW_HEIGHT_IN,
    PLOT_HEIGHT_IN, SCHEMATIC_PLOT_GAP_IN, COLOR,
    build_single_joint_schematic, build_ring_scaling_plot, panel_label_at,
)

PLOT_WIDTH_IN = 2.6  
UNIT_TO_IN = 0.65 
PANEL_GAP_IN = 0.5     # gap between a plot's true right edge and the next panel


def load_rows():
    rows = []
    for fp in glob.glob(str(Path(paper_model_path) / "multitarget_*_seed*_nact*.pth")):
        m = re.search(r"multitarget_(sl|standard|bottleneck)_seed(\d+)_nact(\d+)\.pth", fp)
        if not m:
            continue
        ck = torch.load(fp, map_location="cpu")
        ring = ck.get("ring_score")
        rows.append({"agent": m.group(1), "seed": int(m.group(2)), "N": int(m.group(3)),
                     "final_err": ck.get("mean_greedy_err"), "hit_ep": ck.get("hit_ep"),
                     # ring chirality is arbitrary per seed (+/-0.99); |.| measures ring recovery
                     "ring": abs(ring) if ring is not None else None, "curve": ck.get("curve", [])})
    return rows


def agg(rows, key):
    d = defaultdict(list)
    for r in rows:
        v = r[key]
        if key == "hit_ep" and v is None:
            v = max((c["ep"] for c in r["curve"]), default=np.nan)
        d[(r["agent"], r["N"])].append(v)
    out = {}
    for k, vals in d.items():
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            out[k] = (np.mean(vals), np.std(vals) / max(1, np.sqrt(len(vals))))
    return out


def build_ring_alignment_plot(fig, left, bottom, width, height):
    ax = fig.add_axes([left, bottom, width, height])
    ring = agg(load_rows(), "ring")
    for agent in ("bottleneck", "sl"): # only bottleneck and SL embedding (our model) are relevant here
        pts = sorted((N, m, s) for (a, N), (m, s) in ring.items() if a == agent)
        if not pts:
            continue
        Ns, ms, ss = (np.array(z) for z in zip(*pts))
        ax.plot(Ns, ms, lw=2, color=COLOR[agent])
        ax.fill_between(Ns, ms - ss, ms + ss, color=COLOR[agent], alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Number of actions N")
    ax.set_ylabel("Alignment score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    return ax


def build_figure():
    xrange_single = 1.35 - (-0.75)      
    yrange = (TITLE_Y + 0.35) - (-0.3)  
    unit_to_in = min(UNIT_TO_IN, ROW_HEIGHT_IN / yrange)
    width_a_in = xrange_single * unit_to_in
    fig_h_in = MARGIN_TOP_IN + ROW_HEIGHT_IN + MARGIN_BOTTOM_IN

    left_a = MARGIN_LEFT_IN
    left_b = left_a + width_a_in + SCHEMATIC_PLOT_GAP_IN
    left_c = left_b + PLOT_WIDTH_IN + PANEL_GAP_IN
    fig_w_guess = left_c + PLOT_WIDTH_IN + MARGIN_RIGHT_IN

    fig = plt.figure(figsize=(fig_w_guess, fig_h_in))
    vcenter = 1 - (MARGIN_TOP_IN + ROW_HEIGHT_IN / 2) / fig_h_in
    plot_bottom = vcenter - (PLOT_HEIGHT_IN / fig_h_in) / 2

    ax_a = build_single_joint_schematic(fig, unit_to_in, left_a / fig_w_guess, vcenter)
    ax_b = build_ring_scaling_plot(fig, left_b / fig_w_guess, plot_bottom,
                                    PLOT_WIDTH_IN / fig_w_guess, PLOT_HEIGHT_IN / fig_h_in,
                                    plot_bottleneck=True)
    ax_c = build_ring_alignment_plot(fig, left_c / fig_w_guess, plot_bottom,
                                      PLOT_WIDTH_IN / fig_w_guess, PLOT_HEIGHT_IN / fig_h_in)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    overhang_b = max(0.0, left_b - ax_b.get_tightbbox(renderer).transformed(inv).x0 * fig_w_guess)
    overhang_c = max(0.0, left_c - ax_c.get_tightbbox(renderer).transformed(inv).x0 * fig_w_guess)

    left_b += overhang_b
    left_c = left_b + PLOT_WIDTH_IN + PANEL_GAP_IN + overhang_c
    fig_w_in = left_c + PLOT_WIDTH_IN + MARGIN_RIGHT_IN

    fig.set_size_inches(fig_w_in, fig_h_in)
    ax_a.set_position([left_a / fig_w_in, ax_a.get_position().y0, width_a_in / fig_w_in, ax_a.get_position().height])
    ax_b.set_position([left_b / fig_w_in, plot_bottom, PLOT_WIDTH_IN / fig_w_in, PLOT_HEIGHT_IN / fig_h_in])
    ax_c.set_position([left_c / fig_w_in, plot_bottom, PLOT_WIDTH_IN / fig_w_in, PLOT_HEIGHT_IN / fig_h_in])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    true_left_b = ax_b.get_tightbbox(renderer).transformed(inv).x0
    true_left_c = ax_c.get_tightbbox(renderer).transformed(inv).x0
    row_top = max(ax_a.get_position().y1, ax_b.get_position().y1, ax_c.get_position().y1)

    panel_label_at(fig, left_a / fig_w_in, row_top, "A", dx=-0.03)
    panel_label_at(fig, true_left_b, row_top, "B")
    panel_label_at(fig, true_left_c, row_top, "C")

    return fig


def main():
    fig = build_figure()
    out = Path(revision_fig_dir) / "supp_action_space_fig_2d_embedding"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out) + ".svg", bbox_inches="tight")
    print("saved", out.name + ".pdf/.png/.svg")


if __name__ == "__main__":
    main()
