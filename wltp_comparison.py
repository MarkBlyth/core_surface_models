#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
import getthermals
import thermalparams
import model

# Set a max. step to avoid skipping over rapid changes in the WLTP
# heat-gen profile
MAX_STEP = 0.5


def get_heating_func(
    drivecycle_file: str, param_file: str, ocv_file: str, capacity_Ah: float
):
    simulator = model.TheveninModel(ocv_file, param_file, capacity_Ah)

    df = pd.read_csv(drivecycle_file)
    currents = 10 * df["Current[A]"].to_numpy()
    times = df["Time[s]"].to_numpy()

    t0 = 1800
    t1 = 8000
    mask = np.logical_and(times > t0, times < t1)
    new_currents = currents[mask]
    new_ts = times[mask] - t0

    def currentfunc(t):
        return np.interp(t, new_ts, currents[mask], 0)

    ts, _, _, _, _, _, heat_gen_W = simulator.simulate(
        currentfunc,
        initial_soc=0.99,
        temp_inf=25,
        t_max=3100,
        max_step=MAX_STEP,
        atol=getthermals.TOL,
        rtol=getthermals.TOL,
    )
    heat_gen = heat_gen_W / (0.062 * 0.048 * 0.006)
    return ts, heat_gen


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

    heatgen_ts, heatgen_qs = get_heating_func(
        "wltp.csv", "MLP001_params.csv", "MLP001_ocv.csv", 2.132
    )
    heat_gen = pybamm.Interpolant(heatgen_ts, heatgen_qs, pybamm.t, "Heat generation")
    ts = heatgen_ts

    def heatgen_func(t):
        return np.interp(t, heatgen_ts, heatgen_qs)

    fig, axarrs = plt.subplots(
        5,
        len(getthermals.test_hs),
        layout="constrained",
        figsize=(6.5, 8.5),
    )

    for ax, h in zip(axarrs.T, getthermals.test_hs):
        thermals = getthermals.get_symmetric_thermals(h)

        pybamm_soln = getthermals.get_pybamm_soln(thermals, heat_gen, ts)
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
            heatgen_func,
            0,
            ts,
            max_step=MAX_STEP,
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
            A,
            b,
            heatgen_func,
            0,
            ts,
            max_step=MAX_STEP,
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
            A,
            b,
            heatgen_func,
            0,
            ts,
            max_step=MAX_STEP,
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

        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        ax[2].set_xticklabels([])

    for ax in axarrs[-1, :]:
        ax.remove()
    gs = axarrs[0, 0].get_gridspec()
    ax_heatgen = fig.add_subplot(gs[-1, :])
    ax_heatgen.plot(heatgen_ts, heatgen_qs, color="C5")

    axarrs[0][0].set_ylabel("Core temperature\nrise [$^\\circ$C]")
    axarrs[1][0].set_ylabel("Core absolute\nerror [$^\\circ$C]")
    axarrs[2][0].set_ylabel("Surface temperature\nrise [$^\\circ$C]")
    axarrs[3][0].set_ylabel("Surface absolute\nerror [$^\\circ$C]")
    ax_heatgen.set_xlabel("Time [s]")
    ax_heatgen.set_ylabel("Heat generation\n[W m$^{-3}$]")
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

    plt.savefig("wltp_comparison.pgf")


if __name__ == "__main__":
    main()
