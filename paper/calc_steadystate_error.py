#!/usr/bin/env python3
"""Steady-state core temperature error of the literature model and ours,
at the highest h, under constant heating."""

import numpy as np

import coresurf
from coresurf.reference import symmetric_core_surface

import common


def main():
    t_eval = np.linspace(0, 1e3, 200)
    params = common.cell(common.TEST_HS[-1])
    q = common.STANDARD_HEATGEN

    truth = symmetric_core_surface(params, q, t_eval)["T_core"][-1]
    lit = coresurf.literature_model(params).simulate(q, t_eval)["T_core"][-1]
    ours = coresurf.symmetric_core_surface_model(params).simulate(q, t_eval)["T_core"][-1]

    print(
        f"Ground-truth core temperature: {truth}, our model core temp: {ours}, "
        f"literature core temp: {lit}"
    )
    print(f"Literature steady-state error: {100 * (lit - truth) / truth} %")
    print(f"Our steady-state error: {100 * (ours - truth) / truth} %")


if __name__ == "__main__":
    main()
