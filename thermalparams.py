from typing import Tuple, Callable, Optional
from dataclasses import dataclass
import scipy.optimize
import scipy.integrate
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ThermalParameters:
    L: float
    k: float
    h0: float
    hL: float
    rho: float
    cp: float

    @property
    def rho_cp(self) -> float:
        return self.rho * self.cp


@dataclass
class Mode:
    """
    A single Galerkin trial function, normalised so mode(0) = 1.
    `eigenvalue_squared`, if known, lets the projection use the shortcut
    second_derivative = -mu^2 * value instead of numerical differentiation
    or a hand-supplied second_derivative callable.
    """

    mode: Callable[[npt.ArrayLike], npt.ArrayLike]
    second_derivative: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None
    eigenvalue_squared: Optional[float] = None

    def __post_init__(self):
        mode0 = self.mode(0.0)
        if not np.isclose(mode0, 1.0):
            raise ValueError(f"mode(0) must equal 1, got {mode0}")
        if self.second_derivative is None:
            if self.eigenvalue_squared is None:
                raise ValueError(
                    "must supply second_derivative, or eigenvalue_squared "
                    "for an eigenfunction mode"
                )
            self.second_derivative = lambda x: -self.eigenvalue_squared * self.mode(x)


class GalerkinProjection:
    """
    Core-surface transformation of a two-mode Galerkin projection of
    the 1D heat equation onto span{mode1, mode2}, with both modes
    normalised to mode(0) = 1.

    Produces A, b for
        x = [T_core, T_surf],
        dx_dt = A (x - T_infty) + b * qdot
    """

    def __init__(self, params: ThermalParameters, mode1: Mode, mode2: Mode):
        self._params = params
        self.mode1 = mode1
        self.mode2 = mode2
        self._projection_constants = None
        self._matrices = None

    def _inner(self, f, g) -> float:
        soln, _ = scipy.integrate.quad(lambda x: f(x) * g(x), 0, self._params.L)
        return soln

    def _average(self, f) -> float:
        soln, _ = scipy.integrate.quad(f, 0, self._params.L)
        return soln / self._params.L

    def _compute_constants(self):
        if self._projection_constants is not None:
            return self._projection_constants

        m1, m2 = self.mode1.mode, self.mode2.mode
        d1, d2 = self.mode1.second_derivative, self.mode2.second_derivative

        m1_m1 = self._inner(m1, m1)
        m2_m2 = self._inner(m2, m2)
        m1_m2 = self._inner(m1, m2)
        m1_d1 = self._inner(m1, d1)
        m2_d1 = self._inner(m2, d1)
        m1_d2 = self._inner(m1, d2)
        m2_d2 = self._inner(m2, d2)
        int_m1 = self._inner(m1, lambda x: 1)
        int_m2 = self._inner(m2, lambda x: 1)

        c1 = m1_m1 * m2_m2 - m1_m2**2
        c2 = m2_m2 * m1_d1 - m1_m2 * m2_d1
        c3 = m2_m2 * m1_d2 - m1_m2 * m2_d2
        c4 = m2_m2 * int_m1 - m1_m2 * int_m2
        c5 = m1_m1 * m2_d1 - m1_m2 * m1_d1
        c6 = m1_m1 * m2_d2 - m1_m2 * m1_d2
        c7 = m1_m1 * int_m2 - m1_m2 * int_m1

        self._projection_constants = (c1, c2, c3, c4, c5, c6, c7)
        return self._projection_constants

    def get_model_matrices(self):
        if self._matrices is not None:
            return self._matrices

        c1, c2, c3, c4, c5, c6, c7 = self._compute_constants()
        m1av = self._average(self.mode1.mode)
        m2av = self._average(self.mode2.mode)
        D = self._params.k / self._params.rho_cp

        const = -D / (c1 * (m1av - m2av))
        A11 = const * (m1av * (c3 - c2) + m2av * (c6 - c5))
        A12 = const * (m2av * (m1av * c2 + m2av * c5) - m1av * (m1av * c3 + m2av * c6))
        A21 = const * (c3 + c6 - c2 - c5)
        A22 = const * (m2av * (c2 + c5) - m1av * (c3 + c6))
        A = np.array([[A11, A12], [A21, A22]])

        b_core = (c4 * m1av + c7 * m2av) / (self._params.rho_cp * c1)
        b_surf = (c4 + c7) / (self._params.rho_cp * c1)
        b = np.array([b_core, b_surf])

        self._matrices = (A, b)
        return A, b


class EigenmodeFamily:
    """
    The eigenfunctions phi_n of the Robin-Robin heat operator on [0, L].
    """

    def __init__(self, params: ThermalParameters):
        self.params = params
        self._eigenvalues: npt.NDArray = np.array([])

    def eigenvalues(self, n: int) -> npt.NDArray:
        if n > len(self._eigenvalues):
            self._eigenvalues = self.get_eigenvalues(n)
        return self._eigenvalues[:n]

    def get_eigenvalues(self, n_eigenvals: int) -> npt.NDArray:
        """
        Solve the eigenvalue equation for the first n eigenvalues, and
        return the result as an array. Numerical root finder needs a
        bracket to find the roots in.
        """
        ret = []

        def objective(mu):
            return np.sin(mu * self.params.L) * (
                self.params.h0 * self.params.hL - (mu**2) * (self.params.k**2)
            ) + mu * self.params.k * (self.params.h0 + self.params.hL) * np.cos(
                mu * self.params.L
            )

        for n in range(1, n_eigenvals + 1):
            lo = (n - 1) * np.pi / self.params.L + 1e-5
            hi = n * np.pi / self.params.L - 1e-5
            if objective(lo) * objective(hi) > 0:
                hi = (n - 0.5) * np.pi / self.params.L
            if objective(lo) * objective(hi) > 0:
                hi = (n + 0.5) * np.pi / self.params.L

            soln = scipy.optimize.root_scalar(objective, bracket=[lo, hi])
            ret.append(soln.root)

        return np.array(ret)

    def mode(self, index: int) -> Mode:
        """Return the index'th eigenfunction as a Mode (1-indexed)."""
        mu_squared = self.eigenvalues(index)[index - 1] ** 2
        return Mode(
            mode=lambda x, i=index: self._eigenfunction(x, i),
            eigenvalue_squared=mu_squared,
        )

    def _eigenfunction(self, x, index: int) -> npt.ArrayLike:
        mu = self.eigenvalues(index)[index - 1]
        if mu == 0:
            return np.ones_like(np.asarray(x, dtype=float))
        h0, k = self.params.h0, self.params.k
        return np.cos(mu * x) + h0 * np.sin(mu * x) / (mu * k)


def steadystate_mode(params: ThermalParameters) -> Mode:
    k, h0, hL, L = params.k, params.h0, params.hL, params.L
    numerator = k * (h0 + hL) + L * h0 * hL
    denominator = 2 * k**2 * L + hL * k * L**2
    frac = numerator / denominator  # coefficient of x^2

    def mode(x):
        return 1 + (h0 / k) * x - frac * x**2

    def second_derivative(x):
        return -2.0 * frac * np.ones_like(np.asarray(x, dtype=float))

    return Mode(mode=mode, second_derivative=second_derivative)


def build_two_eigenmodes_model(params: ThermalParameters) -> GalerkinProjection:
    eigenmodes = EigenmodeFamily(params)
    mode1 = eigenmodes.mode(1)
    mode2 = eigenmodes.mode(2)
    return GalerkinProjection(params, mode1, mode2)


def build_eigenmode_steadystatemode_model(
    params: ThermalParameters,
) -> GalerkinProjection:
    eigenmodes = EigenmodeFamily(params)
    mode1 = eigenmodes.mode(1)
    mode2 = steadystate_mode(params)
    return GalerkinProjection(params, mode1, mode2)


def build_small_biot_model(
    params: ThermalParameters,
) -> Tuple[npt.NDArray, npt.NDArray]:
    h0, hL = params.h0, params.hL
    rho_cp, L, k = params.rho_cp, params.L, params.k

    k1 = 3*hL*(h0-hL)/(L*(2*h0-hL))
    k23 = 2*(h0**2 - h0*hL + hL**2) / (L*(2*h0-hL))
    k4 = (4*h0**2 - h0*hL + hL**2)/(L*(2*h0-hL))

    A = np.array([[-k1, -k23], [k23, -k4]]) / rho_cp

    w2 = 1 - L * (2 * h0 - hL) / (6 * k)
    b = np.array([1, w2]) / rho_cp

    return A, b
