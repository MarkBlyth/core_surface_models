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
    core, surf, _ = model.simulate(Q, t)
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


def test_paper_listing_names_still_work():
    """The code listing in the paper's 'Software implementation' section."""
    from coresurf import ThermalParameters, build_eigenmode_steadystatemode_model

    thermals = ThermalParameters(0.0113, 0.916, 150, 150, 2408, 1030)
    projection = build_eigenmode_steadystatemode_model(thermals)
    A, b = projection.get_model_matrices()
    assert A.shape == (2, 2) and b.shape == (2,)
    A2, b2 = coresurf.build_small_biot_model(thermals)
    assert A2.shape == (2, 2)


def test_profile_reproduces_states():
    params = coresurf.kokam_pouch(300, 50)
    model = coresurf.eigenmode_steadystate_model(params)
    soln = model.simulate(Q, np.linspace(0, 2000, 40), T_ambient=20)
    xs = np.linspace(0, params.L, 2001)
    profile = model.temperature_profile(soln, xs, T_ambient=20)
    assert profile.shape == (len(xs), len(soln.t))
    np.testing.assert_allclose(profile[0], soln["T_0"], rtol=1e-10)
    np.testing.assert_allclose(np.trapezoid(profile, xs, axis=0) / params.L, soln["T_mean"], rtol=1e-6)


def test_symmetric_profile_is_symmetric_and_matches_states():
    params = coresurf.kokam_pouch(150)
    model = coresurf.symmetric_core_surface_model(params)
    soln = model.simulate(Q, np.linspace(0, 2000, 40))
    xs = np.linspace(0, params.L, 1001)
    profile = model.temperature_profile(soln, xs)
    np.testing.assert_allclose(profile, profile[::-1], rtol=1e-10)
    np.testing.assert_allclose(profile[500], soln["T_core"], rtol=1e-8)
    np.testing.assert_allclose(profile[0], soln["T_surf"], rtol=1e-6)
    np.testing.assert_allclose(np.trapezoid(profile, xs, axis=0) / params.L, soln["T_mean"], rtol=1e-5)


def test_timescales_and_describe():
    params = coresurf.kokam_pouch(150)
    tau = coresurf.modal_timescales(params, 3)
    assert np.all(np.diff(tau) < 0)
    # tau_1 for a lumped cell is rho cp L / (h0 + hL); Galerkin tau_1 is close but longer
    assert tau[0] > params.rho_cp * params.L / (2 * 150)
    text = coresurf.describe(params, show=False)
    assert "symmetric_core_surface_model" in text


def test_initial_temperature():
    params = coresurf.kokam_pouch(150)
    t = np.linspace(0, 1e5, 20)
    for model in (coresurf.symmetric_core_surface_model(params), coresurf.eigenmode_steadystate_model(params)):
        soln = model.simulate(0.0, t, T_ambient=20, T_initial=45)
        np.testing.assert_allclose(soln.y[:, 0], 45)
        np.testing.assert_allclose(soln.y[:, -1], 20, atol=1e-4)


def test_modes_must_be_one_at_origin():
    params = coresurf.kokam_pouch(150)
    builtin = coresurf.steadystate_mode(params)
    with pytest.raises(ValueError, match="phi\\(0\\) = 1"):
        coresurf.Mode(mode=lambda x: 2 * builtin.mode(x), second_derivative=builtin.second_derivative)
    with pytest.raises(ValueError, match="Cannot normalise"):
        coresurf.Mode.normalised(mode=lambda x: np.sin(x), second_derivative=lambda x: -np.sin(x))

    # A mode altered after creation is caught by GalerkinProjection
    altered = coresurf.Mode(mode=builtin.mode, second_derivative=builtin.second_derivative)
    altered.mode = lambda x: 3 + 0 * x
    phi1 = coresurf.EigenmodeFamily(params).mode(1)
    with pytest.raises(ValueError, match="phi\\(0\\) = 1"):
        coresurf.GalerkinProjection(params, phi1, altered)


def test_normalised_mode_matches_builtin():
    params = coresurf.kokam_pouch(150, 40)
    builtin = coresurf.steadystate_mode(params)
    scaled = coresurf.Mode.normalised(
        mode=lambda x: -4 * builtin.mode(x),
        second_derivative=lambda x: -4 * builtin.second_derivative(x),
    )
    phi1 = coresurf.EigenmodeFamily(params).mode(1)
    a = coresurf.GalerkinProjection(params, phi1, scaled).model()
    b = coresurf.eigenmode_steadystate_model(params)
    np.testing.assert_allclose(a.A, b.A, rtol=1e-8)
    np.testing.assert_allclose(a.b, b.b, rtol=1e-8)


def test_identical_modes_rejected():
    params = coresurf.kokam_pouch(150)
    phi1 = coresurf.EigenmodeFamily(params).mode(1)
    with pytest.raises(ValueError, match="different volume averages"):
        coresurf.GalerkinProjection(params, phi1, phi1).get_model_matrices()


@pytest.mark.parametrize(
    "build", [coresurf.eigenmode_steadystate_model, coresurf.symmetric_core_surface_model,
              coresurf.literature_model]
)
def test_ambient_must_be_constant(build):
    model = build(coresurf.kokam_pouch(150))
    t = np.linspace(0, 100, 5)
    for varying in (([0, 100], [20, 30]), lambda s: 20 + s, np.array([20, 30])):
        with pytest.raises(TypeError, match="constant"):
            model.simulate(Q, t, T_ambient=varying)
    np.testing.assert_allclose(
        model.simulate(Q, t, T_ambient=np.float64(20)).y, model.simulate(Q, t, T_ambient=20).y
    )


def test_ecm_lookup_methods():
    pytest.importorskip("pandas")
    from pathlib import Path

    from coresurf.ecm import TheveninModel

    data = Path(__file__).resolve().parent.parent / "data"
    ecm = TheveninModel(data / "MLP001_ocv.csv", data / "MLP001_params.csv", capacity_Ah=2.132)
    R = ecm.resistances(25, 0.5)
    assert R.shape == (ecm.n_rc + 1,) and np.all(R > 0)
    assert ecm.time_constants(25, 0.5).shape == (ecm.n_rc,)
    assert ecm.resistances(10, 0.5)[0] > ecm.resistances(40, 0.5)[0]
    assert 3.0 < float(np.squeeze(ecm.ocv(25, 0.5))) < 4.2


def test_modes_must_satisfy_boundary_conditions():
    params = coresurf.kokam_pouch(150)
    L = params.L
    # phi(0) = 1, but the slope at x = 0 and x = L ignores the cooling
    bad = coresurf.Mode(mode=lambda x: 1 + x / L - (x / L) ** 2,
                        second_derivative=lambda x: -2 / L**2 + 0 * x)
    phi1 = coresurf.EigenmodeFamily(params).mode(1)
    with pytest.raises(ValueError, match="boundary conditions"):
        coresurf.GalerkinProjection(params, phi1, bad)
    # Built-in modes from one cell don't suit a cell with different cooling
    other = coresurf.steadystate_mode(params.with_cooling(1000))
    with pytest.raises(ValueError, match="boundary conditions"):
        coresurf.GalerkinProjection(params, phi1, other)
