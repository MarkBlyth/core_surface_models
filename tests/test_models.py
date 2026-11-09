import numpy as np
import pytest

import coresurf

HS = [30, 150, 1000]
Q = 3.5e5  # W/m3


def steady_state_profile(params, q):
    """Analytic steady state of the 1D heat equation under uniform
    heating q: returns T(x) as a function, relative to ambient."""
    k, h0, hL, L = params.k, params.h0, params.hL, params.L
    # T = -q x^2 / 2k + c1 x + c0, with k T'(0) = h0 T(0), -k T'(L) = hL T(L)
    M = np.array([[h0 / k, -1.0], [hL / k, hL * L / k + 1]])
    rhs = np.array([0.0, q * L / k + hL * q * L**2 / (2 * k**2)])
    c0, c1 = np.linalg.solve(M, rhs)
    return lambda x: -q * x**2 / (2 * k) + c1 * x + c0


def steady_state_of(model, q):
    return np.linalg.solve(model.A, -model.b * q)


@pytest.mark.parametrize("h", HS)
@pytest.mark.parametrize("asymmetric", [False, True])
def test_eigenmode_steadystate_model_is_exact_at_steady_state(h, asymmetric):
    params = coresurf.kokam_pouch(h, h / 3 if asymmetric else None)
    model = coresurf.eigenmode_steadystate_model(params)
    T_mean, T_0 = steady_state_of(model, Q)

    profile = steady_state_profile(params, Q)
    xs = np.linspace(0, params.L, 10001)
    np.testing.assert_allclose(T_0, profile(0), rtol=1e-8)
    np.testing.assert_allclose(T_mean, np.trapezoid(profile(xs), xs) / params.L, rtol=1e-6)


@pytest.mark.parametrize("h", HS)
def test_symmetric_model_core_and_surface_at_steady_state(h):
    params = coresurf.kokam_pouch(h)
    model = coresurf.symmetric_core_surface_model(params)
    profile = steady_state_profile(params, Q)

    t = np.linspace(0, 1e5, 50)
    core, surf = model.simulate(Q, t)
    np.testing.assert_allclose(core[-1], profile(params.L / 2), rtol=1e-5)
    np.testing.assert_allclose(surf[-1], profile(0), rtol=1e-5)


@pytest.mark.parametrize("h", HS)
def test_literature_model_matches_its_matrix_form(h):
    params = coresurf.kokam_pouch(h)
    lit = coresurf.literature_model(params)
    plain = coresurf.CoreSurfaceModel(lit.A, lit.b, lit.state_names)
    t = np.linspace(0, 2000, 50)
    np.testing.assert_allclose(lit.simulate(Q, t).y, plain.simulate(Q, t).y, rtol=1e-5)


def test_eigenvalues_satisfy_characteristic_equation():
    params = coresurf.kokam_pouch(150, 30)
    mus = coresurf.EigenmodeFamily(params).eigenvalues(5)
    assert np.all(np.diff(mus) > 0)
    p = params
    residual = np.sin(mus * p.L) * (p.h0 * p.hL - mus**2 * p.k**2) + mus * p.k * (
        p.h0 + p.hL
    ) * np.cos(mus * p.L)
    scale = mus**2 * p.k**2
    np.testing.assert_allclose(residual / scale, 0, atol=1e-8)


def test_heat_generation_inputs_agree():
    model = coresurf.eigenmode_steadystate_model(coresurf.kokam_pouch(150))
    t = np.linspace(0, 500, 20)
    const = model.simulate(Q, t).y
    func = model.simulate(lambda _: Q, t).y
    table = model.simulate(([0, 1000], [Q, Q]), t).y
    np.testing.assert_allclose(func, const, rtol=1e-6)
    np.testing.assert_allclose(table, const, rtol=1e-6)


def test_solution_access():
    model = coresurf.literature_model(coresurf.kokam_pouch(150))
    soln = model.simulate(Q, np.linspace(0, 10, 5))
    core, surf = soln
    np.testing.assert_array_equal(core, soln["T_core"])
    with pytest.raises(KeyError):
        soln["T_mean"]


def test_parameter_helpers():
    p = coresurf.kokam_pouch(100)
    half = p.half_domain()
    assert half.L == p.L / 2 and half.h0 == 0 and half.hL == 100
    assert half.mirrored().h0 == 100 and half.mirrored().hL == 0
    with pytest.raises(ValueError):
        coresurf.kokam_pouch(100, 50).half_domain()
    with pytest.raises(ValueError):
        coresurf.literature_model(coresurf.kokam_pouch(100, 50))


def test_custom_modes():
    """A user-supplied quadratic mode reproduces the built-in steady-state mode."""
    params = coresurf.kokam_pouch(150)
    builtin = coresurf.steadystate_mode(params)
    custom = coresurf.Mode(mode=builtin.mode, second_derivative=builtin.second_derivative)
    phi1 = coresurf.EigenmodeFamily(params).mode(1)
    a = coresurf.GalerkinProjection(params, phi1, custom).model()
    b = coresurf.eigenmode_steadystate_model(params)
    np.testing.assert_allclose(a.A, b.A)
    np.testing.assert_allclose(a.b, b.b)
