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

    T_inf = 0
    fig, axarrs = plt.subplots(
        2, len(getthermals.test_hs), figsize=(6, 3.75), layout="constrained"
    )
    for [ax1, ax2], h in zip(axarrs.T, getthermals.test_hs):
        t_eval = np.linspace(0, 5e4 / h, 200)
        if h == 1000:
            t_eval *= 3

        literature_thermals = getthermals.get_symmetric_thermals(h)
        ode = getthermals.get_literature_core_surf_ode(
            literature_thermals, getthermals.standard_heatgen, T_inf
        )
        core, surf = getthermals.get_core_surf_soln(ode, t_eval, T_inf)

        halfdomain_thermals = getthermals.get_half_domain_thermals(h)
        pybamm_soln = getthermals.get_pybamm_soln(
            halfdomain_thermals, getthermals.standard_heatgen, t_eval
        )

        ax1.plot(t_eval, core, label="Literature core-surface model", color="C4")
        ax1.plot(
            t_eval, pybamm_soln["Temperature"](t_eval, 0), color="k", label="1d model"
        )

        ax2.plot(t_eval, surf, label="Core-surf. $T_s$", linestyle="--", color="C4")
        ax2.plot(
            t_eval,
            pybamm_soln["Temperature"](t_eval, halfdomain_thermals.L),
            color="k",
            linestyle="--",
        )

    axarrs[0][0].set_ylabel("Core temperature\nrise [$^\\circ$C]")
    axarrs[1][0].set_ylabel("Surface temperature\nrise [$^\\circ$C]")
    for ax, h in zip(axarrs[0], getthermals.test_hs):
        ax.set_title(
            f"$h$={h}\nBiot = {h*literature_thermals.L/literature_thermals.k:.3f}"
        )
    for ax in axarrs[-1]:
        ax.set_xlabel("Time [s]")
    handles, labels = axarrs[0][1].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=len(labels), frameon=False, loc="outside lower center"
    )

    plt.savefig("literature_model_comparison.pgf")


if __name__ == "__main__":
    main()
