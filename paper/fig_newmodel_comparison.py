#!/usr/bin/env python3
"""Our {phi_1, Psi} core-surface model vs. the literature model and the
full 1D model, constant heating."""

import matplotlib.pyplot as plt
import numpy as np

import coresurf
from coresurf.reference import symmetric_core_surface

import common


def main():
    args = common.figure_args(__doc__)

    fig, axarrs = plt.subplots(
        4, len(common.TEST_HS), layout="constrained", figsize=(6.5, 7), sharex="col"
    )

    for ax, h in zip(axarrs.T, common.TEST_HS):
        ts = np.linspace(0, 5e4 / h, 200)
        if h == 1000:
            ts *= 3

        params = common.cell(h)
        q = common.STANDARD_HEATGEN
        truth = symmetric_core_surface(params, q, ts)
        lit_core, lit_surf = coresurf.literature_model(params).simulate(q, ts)
        ours = coresurf.symmetric_core_surface_model(params).simulate(q, ts)
        our_core, our_surf = ours["T_core"], ours["T_surf"]

        ax[0].plot(ts, truth["T_core"], label="Full model", color="k", zorder=1000, linestyle=":")
        ax[2].plot(ts, truth["T_surf"], color="k", zorder=1000, linestyle=":")
        for key, core, surf in [
            ("eigen_ss", our_core, our_surf),
            ("literature", lit_core, lit_surf),
        ]:
            common.plot_error_rows(ax, ts, key, core, surf, truth["T_core"], truth["T_surf"])

        ax[0].set_title(f"h = {h}\nBiot = {params.biot:.3f}")
        ax[-1].set_xlabel("Time [s]")

    common.label_error_rows(axarrs)
    fig.align_ylabels()

    fig.get_layout_engine().set(rect=[0, 0.05, 1, 0.95])
    handles, labels = axarrs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.95, 0.05), ncol=len(labels), frameon=False)

    common.save_figure(fig, "newmodel_comparison", args)


if __name__ == "__main__":
    main()
