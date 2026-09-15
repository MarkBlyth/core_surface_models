#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import getthermals
import thermalparams


def main():
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    fig, axarrs = plt.subplots(
        4,
        len(getthermals.test_hs),
        layout="constrained",
        figsize=(6.5, 7),
        sharex="col",
    )

    for ax, h in zip(axarrs.T, getthermals.test_hs):
        literature_thermals = getthermals.get_symmetric_thermals(h)
        core_thermals = getthermals.get_half_domain_thermals(h)
        surf_thermals = getthermals.get_mirror_image_half_domain_thermals(h)

        tau = (
            literature_thermals.rho * literature_thermals.cp * literature_thermals.L / h
        )
        ts = np.linspace(0, 5e4 / h, 200)
        if h == 1000:
            ts *= 3

        pybamm_soln = getthermals.get_pybamm_soln(
            core_thermals, getthermals.standard_heatgen, ts
        )
        core_truth = pybamm_soln["Temperature"](ts, 0)
        surf_truth = pybamm_soln["Temperature"](ts, core_thermals.L)

        ode = getthermals.get_literature_core_surf_ode(
            literature_thermals, getthermals.standard_heatgen, 0
        )
        core, surf = getthermals.get_core_surf_soln(ode, ts, 0)

        eigen_steadystate_projection = (
            thermalparams.build_eigenmode_steadystatemode_model(core_thermals)
        )
        A, b = eigen_steadystate_projection.get_model_matrices()
        _, sscore = getthermals.solve_thermal(A, b, getthermals.standard_heatgen, 0, ts)

        eigen_steadystate_projection = (
            thermalparams.build_eigenmode_steadystatemode_model(surf_thermals)
        )
        A, b = eigen_steadystate_projection.get_model_matrices()
        _, sssurf = getthermals.solve_thermal(A, b, getthermals.standard_heatgen, 0, ts)

        ax[0].plot(
            ts,
            core_truth,
            label="Full model",
            color="k",
            zorder=1000,
            linestyle=":",
        )
        ax[2].plot(
            ts,
            surf_truth,
            color="k",
            zorder=1000,
            linestyle=":",
        )

        ax[0].plot(
            ts,
            sscore,
            "C2",
            label=r"$\{\phi_1,\Psi\}$ Galerkin",
        )
        ax[1].plot(ts, sscore - core_truth, "C2")
        ax[2].plot(
            ts,
            sssurf,
            "C2",
        )
        ax[3].plot(ts, sssurf - surf_truth, "C2")

        ax[0].plot(
            ts,
            core,
            "C4",
            label="Literature core-surface",
        )
        ax[1].plot(ts, core - core_truth, "C4")
        ax[2].plot(
            ts,
            surf,
            "C4",
        )
        ax[3].plot(ts, surf - surf_truth, "C4")

        ax[0].set_title(
            f"h = {h}\nBiot = {h*literature_thermals.L/literature_thermals.k:.3f}"
        )
        ax[-1].set_xlabel("Time [s]")

    axarrs[0][0].set_ylabel("Core temperature\nrise [$^\\circ$C]")
    axarrs[1][0].set_ylabel("Core absolute\nerror [$^\\circ$C]")
    axarrs[2][0].set_ylabel("Surface temperature\nrise [$^\\circ$C]")
    axarrs[3][0].set_ylabel("Surface absolute\nerror [$^\\circ$C]")
    fig.align_ylabels()

    fig.get_layout_engine().set(rect=[0, 0.05, 1, 0.95])
    handles, labels = axarrs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.95, 0.05),
        ncol=len(labels),
        frameon=False,
    )

    plt.savefig("newmodel_comparison.pgf")


if __name__ == "__main__":
    main()
