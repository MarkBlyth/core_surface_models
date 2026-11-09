#!/usr/bin/env python3
"""Zoomed surface temperature under WLTP heating, at the highest h."""

import matplotlib.pyplot as plt

import coresurf
from coresurf.reference import solve_heat_equation

import common


def main():
    args = common.figure_args(__doc__)

    heatgen = common.wltp_heat_generation()
    ts = heatgen[0]
    h = common.TEST_HS[-1]
    params = common.cell(h)

    surf_truth = solve_heat_equation(params, heatgen, ts)["Temperature"](ts, 0)

    fig, (ax, ax2) = plt.subplots(2, layout="constrained", figsize=(6.5, 3.5), sharex=True)
    ax.plot(ts, surf_truth, color="k", linestyle=(0, (3, 3)), linewidth=2, zorder=1000,
            label="Full model")
    for key, builder in [
        ("small_biot", coresurf.small_biot_model),
        ("two_eigen", coresurf.two_eigenmode_model),
        ("eigen_ss", coresurf.eigenmode_steadystate_model),
    ]:
        style = common.MODEL_STYLES[key]
        surf = builder(params).simulate(heatgen, ts, max_step=common.MAX_STEP)["T_0"]
        ax.plot(ts, surf, style["color"], label=style["label"])
        ax2.plot(ts, surf - surf_truth, style["color"])

    ax.set_title(f"h = {h}, Biot = {params.biot:.3f}")
    ax2.set_xlabel("Time [s]")
    ax.set_ylabel("Surface\ntemperature [$^\\circ$C]")
    ax2.set_ylabel("Surface\nabsolute error [$^\\circ$C]")
    fig.align_ylabels([ax, ax2])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(labels), frameon=False)
    fig.get_layout_engine().set(rect=(0.01, 0.075, 0.99, 0.92))
    ax.set_xlim([1300, 1900])

    common.save_figure(fig, "wltp_zoom", args)


if __name__ == "__main__":
    main()
