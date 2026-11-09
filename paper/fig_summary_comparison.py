#!/usr/bin/env python3
"""Small-Biot, {phi_1, phi_2} and {phi_1, Psi} models vs. the full 1D
model, constant heating."""

import matplotlib.pyplot as plt
import numpy as np

import coresurf
from coresurf.reference import solve_heat_equation

import common


def main():
    args = common.figure_args(__doc__)

    fig, axarrs = plt.subplots(
        4, len(common.TEST_HS), layout="constrained", figsize=(6.5, 7), sharex="col"
    )

    for ax, h in zip(axarrs.T, common.TEST_HS):
        params = common.cell(h)
        q = common.STANDARD_HEATGEN

        tau = params.rho_cp * params.L / h
        if h == common.TEST_HS[-1]:
            tau *= 1.5
        ts = np.linspace(0, 4 * tau, 200)

        soln = solve_heat_equation(params, q, ts)
        surf_truth = soln["Temperature"](ts, 0)
        mean_truth = soln["Volume averaged temperature"](ts)

        ax[0].plot(ts, mean_truth, color="k", linestyle=(0, (2, 3)), linewidth=2,
                   label="Full model", zorder=1000)
        ax[2].plot(ts, surf_truth, color="k", linestyle=(0, (3, 3)), linewidth=2, zorder=1000)

        for key, builder in [
            ("small_biot", coresurf.small_biot_model),
            ("two_eigen", coresurf.two_eigenmode_model),
            ("eigen_ss", coresurf.eigenmode_steadystate_model),
        ]:
            mean, surf = builder(params).simulate(q, ts)
            common.plot_error_rows(ax, ts, key, mean, surf, mean_truth, surf_truth)

        ax[0].set_title(f"h = {h}\nBiot = {params.biot:.3f}")
        ax[3].set_xlabel("Time [s]")

    common.label_error_rows(axarrs)
    fig.align_ylabels()

    fig.get_layout_engine().set(rect=[0, 0.05, 1, 0.95])
    handles, labels = axarrs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.05), ncol=len(labels), frameon=False)

    common.save_figure(fig, "summary_comparison", args)


if __name__ == "__main__":
    main()
