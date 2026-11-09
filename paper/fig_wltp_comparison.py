#!/usr/bin/env python3
"""Small-Biot, {phi_1, phi_2} and {phi_1, Psi} models vs. the full 1D
model, WLTP drive-cycle heating."""

import matplotlib.pyplot as plt

import coresurf
from coresurf.reference import solve_heat_equation

import common


def main():
    args = common.figure_args(__doc__)

    heatgen = common.wltp_heat_generation()
    ts = heatgen[0]

    fig, axarrs = plt.subplots(5, len(common.TEST_HS), layout="constrained", figsize=(6.5, 8.5))

    for ax, h in zip(axarrs.T, common.TEST_HS):
        params = common.cell(h)

        soln = solve_heat_equation(params, heatgen, ts)
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
            mean, surf = builder(params).simulate(heatgen, ts, max_step=common.MAX_STEP)
            common.plot_error_rows(ax, ts, key, mean, surf, mean_truth, surf_truth)

        ax[0].set_title(f"h = {h}\nBiot = {params.biot:.3f}")
        ax[3].set_xlabel("Time [s]")
        for row in ax[:3]:
            row.set_xticklabels([])

    for ax in axarrs[-1, :]:
        ax.remove()
    gs = axarrs[0, 0].get_gridspec()
    ax_heatgen = fig.add_subplot(gs[-1, :])
    ax_heatgen.plot(*heatgen, color="C5")

    common.label_error_rows(axarrs)
    ax_heatgen.set_xlabel("Time [s]")
    ax_heatgen.set_ylabel("Heat generation\n[W m$^{-3}$]")
    fig.align_ylabels()

    fig.get_layout_engine().set(rect=[0, 0.05, 1, 0.95])
    handles, labels = axarrs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.05), ncol=len(labels), frameon=False)

    common.save_figure(fig, "wltp_comparison", args)


if __name__ == "__main__":
    main()
