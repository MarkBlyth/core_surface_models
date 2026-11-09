#!/usr/bin/env python3
"""Literature core-surface model vs. the full 1D model, constant heating."""

import matplotlib.pyplot as plt
import numpy as np

import coresurf
from coresurf.reference import symmetric_core_surface

import common


def main():
    args = common.figure_args(__doc__)

    fig, axarrs = plt.subplots(2, len(common.TEST_HS), figsize=(6, 3.75), layout="constrained")
    for [ax1, ax2], h in zip(axarrs.T, common.TEST_HS):
        t_eval = np.linspace(0, 5e4 / h, 200)
        if h == 1000:
            t_eval *= 3

        params = common.cell(h)
        core, surf = coresurf.literature_model(params).simulate(common.STANDARD_HEATGEN, t_eval)
        truth = symmetric_core_surface(params, common.STANDARD_HEATGEN, t_eval)

        ax1.plot(t_eval, core, label="Literature core-surface model", color="C4")
        ax1.plot(t_eval, truth["T_core"], color="k", label="1d model")
        ax2.plot(t_eval, surf, label="Core-surf. $T_s$", linestyle="--", color="C4")
        ax2.plot(t_eval, truth["T_surf"], color="k", linestyle="--")
        ax1.set_title(f"$h$={h}\nBiot = {params.biot:.3f}")

    axarrs[0][0].set_ylabel("Core temperature\nrise [$^\\circ$C]")
    axarrs[1][0].set_ylabel("Surface temperature\nrise [$^\\circ$C]")
    for ax in axarrs[-1]:
        ax.set_xlabel("Time [s]")
    handles, labels = axarrs[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), frameon=False, loc="outside lower center")

    common.save_figure(fig, "literature_model_comparison", args)


if __name__ == "__main__":
    main()
