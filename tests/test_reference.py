"""Comparisons against the full 1D model. Skipped if pybamm is unavailable."""

import numpy as np
import pytest

import coresurf

pytest.importorskip("pybamm")
from coresurf.reference import symmetric_core_surface  # noqa: E402

Q = 20 / coresurf.parameters.KOKAM_POUCH_VOLUME


@pytest.mark.parametrize("h", [30, 1000])
def test_symmetric_model_tracks_full_model(h):
    params = coresurf.kokam_pouch(h)
    t = np.linspace(0, 3e3, 100)
    truth = symmetric_core_surface(params, Q, t)
    ours = coresurf.symmetric_core_surface_model(params).simulate(Q, t)
    lit = coresurf.literature_model(params).simulate(Q, t)

    for state in ("T_core", "T_surf"):
        our_err = np.max(np.abs(ours[state] - truth[state]))
        lit_err = np.max(np.abs(lit[state] - truth[state]))
        assert our_err < 0.05 * np.max(truth[state])
        assert our_err < lit_err
