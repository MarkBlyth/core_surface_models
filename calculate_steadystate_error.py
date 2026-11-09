#!/usr/bin/env python3

import numpy as np
import getthermals
import thermalparams


def main():
    T_inf = 0
    t_eval = np.linspace(0, 1e3, 200)
    h = getthermals.test_hs[-1]

    literature_thermals = getthermals.get_symmetric_thermals(h)
    ode = getthermals.get_literature_core_surf_ode(
        literature_thermals, getthermals.standard_heatgen, T_inf
    )
    lit_core, _ = getthermals.get_core_surf_soln(ode, t_eval, T_inf)

    halfdomain_thermals = getthermals.get_half_domain_thermals(h)
    pybamm_soln = getthermals.get_pybamm_soln(
        halfdomain_thermals, getthermals.standard_heatgen, t_eval
    )
    pybamm_core = pybamm_soln["Temperature"](t_eval, 0)

    core_thermals = getthermals.get_half_domain_thermals(h)
    eigen_steadystate_projection = thermalparams.build_eigenmode_steadystatemode_model(
        core_thermals
    )
    A, b = eigen_steadystate_projection.get_model_matrices()
    _, our_core = getthermals.solve_thermal(
        A, b, getthermals.standard_heatgen, 0, t_eval
    )

    print(
        f"Ground-truth core temperature: {pybamm_core[-1]}, our model core temp: {our_core[-1]}, literature core temp: {lit_core[-1]}"
    )
    print(
        f"Literature steady-state error: {100 * (lit_core[-1] - pybamm_core[-1]) / pybamm_core[-1]} %"
    )
    print(
        f"Our steady-state error: {100 * (our_core[-1] - pybamm_core[-1]) / pybamm_core[-1]} %"
    )


if __name__ == "__main__":
    main()
