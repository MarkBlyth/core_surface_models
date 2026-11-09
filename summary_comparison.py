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
        thermals = getthermals.get_symmetric_thermals(h)

        tau = thermals.rho * thermals.cp * thermals.L / h
        if h == getthermals.test_hs[-1]:
            tau *= 1.5
        ts = np.linspace(0, 4 * tau, 200)

        pybamm_soln = getthermals.get_pybamm_soln(
            thermals, getthermals.standard_heatgen, ts
        )
        surf_truth = pybamm_soln["Temperature"](ts, 0)
        vol_truth = pybamm_soln["Volume averaged temperature"](ts)

        ax[0].plot(
            ts,
            vol_truth,
            color="k",
            linestyle=(0, (2, 3)),
            linewidth=2,
            label="Full model",
            zorder=1000,
        )
        ax[2].plot(
            ts,
            surf_truth,
            color="k",
            linestyle=(0, (3, 3)),
            linewidth=2,
            zorder=1000,
        )

        A, b = thermalparams.build_small_biot_model(thermals)
        smallbiot_vol_av_temp, smallbiot_surf_temp = getthermals.solve_thermal(
            A,
            b,
            getthermals.standard_heatgen,
            0,
            ts,
        )

        ax[0].plot(
            ts,
            smallbiot_vol_av_temp,
            "C0",
            label="Small Biot",
        )
        ax[1].plot(ts, smallbiot_vol_av_temp - vol_truth, "C0")
        ax[2].plot(ts, smallbiot_surf_temp, "C0")
        ax[3].plot(ts, smallbiot_surf_temp - surf_truth, "C0")

        two_eigens_projection = thermalparams.build_two_eigenmodes_model(thermals)
        A, b = two_eigens_projection.get_model_matrices()
        eigenscore, eigensurf = getthermals.solve_thermal(
            A, b, getthermals.standard_heatgen, 0, ts
        )
        ax[0].plot(
            ts,
            eigenscore,
            "C1",
            label=r"$\{\phi_1,\phi_2\}$ Galerkin",
        )
        ax[1].plot(ts, eigenscore - vol_truth, "C1")
        ax[2].plot(ts, eigensurf, "C1")
        ax[3].plot(ts, eigensurf - surf_truth, "C1")

        eigen_steadystate_projection = (
            thermalparams.build_eigenmode_steadystatemode_model(thermals)
        )
        A, b = eigen_steadystate_projection.get_model_matrices()
        sscore, sssurf = getthermals.solve_thermal(
            A, b, getthermals.standard_heatgen, 0, ts
        )
        ax[0].plot(
            ts,
            sscore,
            "C2",
            label=r"$\{\phi_1,\Psi\}$ Galerkin",
        )
        ax[1].plot(ts, sscore - vol_truth, "C2")
        ax[2].plot(ts, sssurf, "C2")
        ax[3].plot(ts, sssurf - surf_truth, "C2")

        ax[0].set_title(
            f"h = {h}\nBiot = {0.5*(thermals.h0 + thermals.hL)*thermals.L/thermals.k:.3f}"
        )
        ax[3].set_xlabel("Time [s]")

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
        bbox_to_anchor=(1.01, 0.05),
        ncol=len(labels),
        frameon=False,
    )

    plt.savefig("summary_comparison.pgf")


if __name__ == "__main__":
    main()
