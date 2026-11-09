#!/usr/bin/env python3
"""RMS core and surface temperature errors of the literature model and
ours, under WLTP drive-cycle heating."""

import numpy as np

import coresurf
from coresurf.reference import symmetric_core_surface

import common


def rms(x):
    return np.sqrt(np.mean(x**2))


def main():
    heatgen = common.wltp_heat_generation()
    ts = heatgen[0]

    for h in common.TEST_HS:
        params = common.cell(h)
        truth = symmetric_core_surface(params, heatgen, ts)
        lit = coresurf.literature_model(params).simulate(heatgen, ts)
        ours = coresurf.symmetric_core_surface_model(params).simulate(heatgen, ts)

        errors = {
            (name, state): rms(soln[state] - truth[state])
            for name, soln in [("ours", ours), ("lit", lit)]
            for state in ("T_core", "T_surf")
        }

        print(f"h={h}")
        print(
            f"Our core, surf RMS errors [degC]: "
            f"{errors['ours', 'T_core']}, {errors['ours', 'T_surf']}"
        )
        print(
            f"Literature core, surf RMS errors [degC]: "
            f"{errors['lit', 'T_core']}, {errors['lit', 'T_surf']}"
        )
        print(
            f"Literature error / our error: core [degC]: "
            f"{errors['lit', 'T_core'] / errors['ours', 'T_core']}"
        )
        print(
            f"Literature error / our error: surf [degC]: "
            f"{errors['lit', 'T_surf'] / errors['ours', 'T_surf']}"
        )


if __name__ == "__main__":
    main()
