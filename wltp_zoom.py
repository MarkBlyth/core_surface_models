#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
import getthermals
import thermalparams
import model

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

    h = getthermals.test_hs[-1]
    thermals = getthermals.get_symmetric_thermals(h)

    pybamm_soln = getthermals.get_pybamm_soln(thermals, heat_gen, ts)
    surf_truth = pybamm_soln["Temperature"](ts, 0)
    vol_truth = pybamm_soln["Volume averaged temperature"](ts)

    A, b = thermalparams.build_small_biot_model(thermals)
    smallbiot_vol_av_temp, smallbiot_surf_temp = getthermals.solve_thermal(
        A,
        b,
        heatgen_func,
        0,
        ts,
        max_step=MAX_STEP,
    )

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

    fig, (ax, ax2) = plt.subplots(
        2, layout="constrained", figsize=(6.5, 3.5), sharex=True
    )

    ax.plot(
        ts,
        surf_truth,
        color="k",
        linestyle=(0, (3, 3)),
        linewidth=2,
        zorder=1000,
        label="Full model",
    )
    ax.plot(ts, smallbiot_surf_temp, "C0", label="Small Biot")
    ax2.plot(ts, smallbiot_surf_temp - surf_truth, "C0")
    ax.plot(ts, eigensurf, "C1", label="$\\{\\phi_1, \\phi_2\\}$ Galerkin")
    ax2.plot(ts, eigensurf - surf_truth, "C1")
    ax.plot(ts, sssurf, "C2", label="$\\{\\phi_1, \\Psi\\}$ Galerkin")
    ax2.plot(ts, sssurf - surf_truth, "C2")

    ax.set_title(
        f"h = {h}, Biot = {0.5*(thermals.h0 + thermals.hL)*thermals.L/thermals.k:.3f}"
    )
    ax2.set_xlabel("Time [s]")
    ax.set_ylabel("Surface\ntemperature [$^\\circ$C]")
    ax2.set_ylabel("Surface\nabsolute error [$^\\circ$C]")
    fig.align_ylabels([ax, ax2])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        frameon=False,
    )
    fig.get_layout_engine().set(rect=(0.01, 0.075, 0.99, 0.92))

    ax.set_xlim([1300, 1900])

    plt.savefig("wltp_zoom.pgf")


if __name__ == "__main__":
    main()
