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
    heatgen_ts, heatgen_qs = get_heating_func(
        "wltp.csv", "MLP001_params.csv", "MLP001_ocv.csv", 2.132
    )
    heat_gen = pybamm.Interpolant(heatgen_ts, heatgen_qs, pybamm.t, "Heat generation")
    ts = heatgen_ts

    def heatgen_func(t):
        return np.interp(t, heatgen_ts, heatgen_qs)

    for h in getthermals.test_hs:
        literature_thermals = getthermals.get_symmetric_thermals(h)
        core_thermals = getthermals.get_half_domain_thermals(h)
        surf_thermals = getthermals.get_mirror_image_half_domain_thermals(h)

        pybamm_soln = getthermals.get_pybamm_soln(core_thermals, heat_gen, ts)
        core_truth = pybamm_soln["Temperature"](ts, 0)
        surf_truth = pybamm_soln["Temperature"](ts, core_thermals.L)

        ode = getthermals.get_literature_core_surf_ode(
            literature_thermals, heatgen_func, 0
        )
        lit_core, lit_surf = getthermals.get_core_surf_soln(ode, ts, 0)

        eigen_steadystate_projection = (
            thermalparams.build_eigenmode_steadystatemode_model(core_thermals)
        )
        A, b = eigen_steadystate_projection.get_model_matrices()
        _, our_core = getthermals.solve_thermal(A, b, heatgen_func, 0, ts)

        eigen_steadystate_projection = (
            thermalparams.build_eigenmode_steadystatemode_model(surf_thermals)
        )
        A, b = eigen_steadystate_projection.get_model_matrices()
        _, our_surf = getthermals.solve_thermal(A, b, heatgen_func, 0, ts)

        lit_core_error = np.sqrt(np.mean((lit_core - core_truth) ** 2))
        lit_surf_error = np.sqrt(np.mean((lit_surf - surf_truth) ** 2))

        our_core_error = np.sqrt(np.mean((our_core - core_truth) ** 2))
        our_surf_error = np.sqrt(np.mean((our_surf - surf_truth) ** 2))

        print(f"h={h}")
        print(f"Our core, surf RMS errors [degC]: {our_core_error}, {our_surf_error}")
        print(
            f"Literature core, surf RMS errors [degC]: {lit_core_error}, {lit_surf_error}"
        )
        print(
            f"Literature error / our error: core [degC]: {lit_core_error / our_core_error}"
        )
        print(
            f"Literature error / our error: surf [degC]: {lit_surf_error / our_surf_error}"
        )


if __name__ == "__main__":
    main()
