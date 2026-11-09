"""Linear two-state core-surface thermal models, and how to simulate them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, Union

import numpy as np
import numpy.typing as npt
import scipy.integrate

#: Default absolute and relative ODE solver tolerance.
DEFAULT_TOL = 1e-7

HeatGeneration = Union[
    float,
    Callable[[float], float],
    tuple[npt.ArrayLike, npt.ArrayLike],
]
"""
Volumetric heat generation q(t) [W m^-3]. One of
  - a constant,
  - a function of time,
  - a tuple (times, values), linearly interpolated.
"""


def as_heat_function(heatgen: HeatGeneration) -> Callable[[float], float]:
    """Convert any accepted heat-generation input to a function of time."""
    if callable(heatgen):
        return heatgen
    if isinstance(heatgen, tuple):
        times, values = (np.asarray(a, dtype=float) for a in heatgen)
        return lambda t: np.interp(t, times, values)
    value = float(heatgen)
    return lambda _: value


@dataclass
class Solution:
    """
    Simulated temperatures.

    `y` has one row per state. Rows can be accessed by name
    (`soln["T_core"]`) or by unpacking (`core, surf = soln`).
    """

    t: npt.NDArray
    y: npt.NDArray
    state_names: tuple[str, ...]

    def __getitem__(self, key: str | int) -> npt.NDArray:
        if isinstance(key, str):
            try:
                key = self.state_names.index(key)
            except ValueError:
                raise KeyError(
                    f"{key!r} is not a state; states are {self.state_names}"
                ) from None
        return self.y[key]

    def __iter__(self) -> Iterator[npt.NDArray]:
        return iter(self.y)

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class CoreSurfaceModel:
    """
    A linear two-state thermal model

        dx/dt = A (x - T_inf) + b q(t).

    `state_names` describes what each state represents; it depends on
    how the model was built (see the builder functions in
    `coresurf.builders`).
    """

    A: npt.NDArray
    b: npt.NDArray
    state_names: tuple[str, ...] = ("x0", "x1")

    def rhs(
        self, heatgen: HeatGeneration, T_ambient: float = 0.0
    ) -> Callable[[float, npt.NDArray], npt.NDArray]:
        """The ODE right-hand side f(t, x), for use with any ODE solver."""
        A, b = self.A, self.b
        if callable(heatgen) or isinstance(heatgen, tuple):
            q = as_heat_function(heatgen)
            return lambda t, x: np.matmul(A, x - T_ambient) + b * q(t)
        return lambda t, x: np.matmul(A, x - T_ambient) + b * heatgen

    def simulate(
        self,
        heatgen: HeatGeneration,
        t_eval: npt.ArrayLike,
        T_ambient: float = 0.0,
        x0: Sequence[float] | None = None,
        **solver_kwargs,
    ) -> Solution:
        """
        Simulate the model from t_eval[0] to t_eval[-1].

        Parameters
        ----------
        heatgen : volumetric heat generation [W m^-3]; a constant, a
            function of time, or a (times, values) tuple.
        t_eval : times at which to report the solution [s].
        T_ambient : ambient temperature.
        x0 : initial state; defaults to the ambient temperature.
        solver_kwargs : passed to scipy.integrate.solve_ivp. Tolerances
            default to DEFAULT_TOL. For rapidly varying heat generation,
            set `max_step` so that the solver does not step over it.
        """
        t_eval = np.asarray(t_eval, dtype=float)
        if x0 is None:
            x0 = [T_ambient] * len(self.b)
        kwargs = {"atol": DEFAULT_TOL, "rtol": DEFAULT_TOL, **solver_kwargs}
        soln = scipy.integrate.solve_ivp(
            self.rhs(heatgen, T_ambient),
            [t_eval[0], t_eval[-1]],
            x0,
            t_eval=t_eval,
            **kwargs,
        )
        if not soln.success:
            raise RuntimeError(f"ODE solver failed: {soln.message}")
        return Solution(soln.t, soln.y, self.state_names)
