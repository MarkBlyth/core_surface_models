"""
Thevenin equivalent-circuit model (ECM), used to generate realistic heat
generation profiles from a current profile, or to couple with a
core-surface thermal model (see examples/07_electrothermal_ecm.py).
Requires `pandas`.

The model and the MLP001 parameter set are from

    M. Blyth, A. Gupta and A. Hales, "How to parameterise an
    equivalent-circuit empirical battery model from time-domain data",
    MethodsX (2026), 103865.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.interpolate


@dataclass
class ECMSolution:
    t: npt.NDArray
    current: npt.NDArray
    voltage: npt.NDArray
    soc: npt.NDArray
    temperature: npt.NDArray
    v_rc: npt.NDArray
    heat_generation: npt.NDArray
    """Total heat generation [W]."""


class TheveninModel:
    """
    Thevenin ECM with N RC pairs and lookup-table parameters.

    Parameters
    ----------
    ocv_filename : CSV with columns SOC, OCV[V], Temperature_degC.
    params_filename : CSV with columns SOC, Temperature_degC, R0 [Ohm],
        and R{i} [Ohm], tau{i} [s] for each RC pair i = 1..N.
    capacity_Ah : cell capacity [Ah].
    thermal_mass : lumped thermal mass [J/K]; infinite for isothermal.
    convection_rate : lumped heat transfer coefficient [W/K].

    If the tables contain a single temperature, parameters are
    interpolated in SOC only; otherwise in SOC and temperature.
    """

    def __init__(
        self,
        ocv_filename: str,
        params_filename: str,
        capacity_Ah: float,
        thermal_mass: float = np.inf,
        convection_rate: float = 0,
    ):
        import pandas as pd

        self.thermal_mass = thermal_mass
        self.capacity_Ah = capacity_Ah
        self.convection_rate = convection_rate

        params_df = pd.read_csv(params_filename)
        self.n_rc = int((len(params_df.columns) - 3) / 2)

        if params_df["Temperature_degC"].nunique() < 2:
            interpolator = self._soc_lut
        else:
            interpolator = self._soc_temperature_lut
        self._r_luts = [interpolator(params_df, f"R{i} [Ohm]") for i in range(self.n_rc + 1)]
        self._tau_luts = [interpolator(params_df, f"tau{i} [s]") for i in range(1, self.n_rc + 1)]
        self._ocv_lut = interpolator(pd.read_csv(ocv_filename), "OCV[V]")

    @staticmethod
    def _soc_lut(df, param_header: str):
        socs = df["SOC"].to_numpy()
        targetdata = df[param_header].to_numpy()
        sort_idx = np.argsort(socs)
        interpolant = scipy.interpolate.PchipInterpolator(
            socs[sort_idx], targetdata[sort_idx], extrapolate=False
        )
        return lambda temp, soc: interpolant(soc)

    @staticmethod
    def _soc_temperature_lut(df, param_header: str):
        sort_df = df.sort_values(by=["Temperature_degC", "SOC"])
        temps = sort_df["Temperature_degC"].to_numpy()
        socs = sort_df["SOC"].to_numpy()
        return scipy.interpolate.LinearNDInterpolator(
            np.vstack((temps, socs)).T, sort_df[param_header].to_numpy()
        )

    def ocv(self, temperature, soc):
        """Open-circuit voltage [V] at a temperature [degC] and SOC [-]."""
        return self._ocv_lut(temperature, soc)

    def resistances(self, temperature, soc) -> npt.NDArray:
        """[R0, R1, ..., RN] [Ohm]: series resistance, then each RC pair."""
        return np.array([np.squeeze(r(temperature, soc)) for r in self._r_luts])

    def time_constants(self, temperature, soc) -> npt.NDArray:
        """[tau1, ..., tauN] [s], the time constant of each RC pair."""
        return np.array([np.squeeze(tau(temperature, soc)) for tau in self._tau_luts])

    def rhs(
        self, currentfunc: Callable[[float], float], temp_inf: float
    ) -> Callable[[float, npt.NDArray], npt.NDArray]:
        """ODE right-hand side for states [SOC, T, V_rc1, ..., V_rcN]."""

        def ode_rhs(t: float, x: npt.NDArray):
            soc, temp, v_rc = x[0], x[1], x[2:]
            current = currentfunc(t)

            rs = np.array([r_lut(temp, soc) for r_lut in self._r_luts])
            taus = np.array([tau_lut(temp, soc) for tau_lut in self._tau_luts])
            if any(np.isnan(rs)) or any(np.isnan(taus)):
                raise ValueError(
                    f"A parameter interpolator extrapolated at time {t} "
                    f"with SOC {soc} and temperature {temp}"
                )
            heat_gen = (
                current**2 * rs[0]
                + (v_rc**2 / rs[1:]).sum()
                - self.convection_rate * (temp - temp_inf)
            )
            dsoc_dt = -current / (self.capacity_Ah * 3600)
            dtemp_dt = heat_gen / self.thermal_mass
            dvrc_dt = (current * rs[1:] - v_rc) / taus
            return np.r_[dsoc_dt, dtemp_dt, dvrc_dt]

        return ode_rhs

    def simulate(
        self,
        currentfunc: Callable[[float], float],
        initial_soc: float,
        temp_inf: float,
        t_max: float,
        initial_temp: float | None = None,
        initial_vrcs: None | list | npt.NDArray = None,
        **solver_kwargs,
    ) -> ECMSolution:
        """
        Simulate from t=0 to t_max. `currentfunc(t)` gives the current
        [A], positive on discharge. `solver_kwargs` go to solve_ivp.
        """
        if initial_temp is None:
            initial_temp = temp_inf
        if initial_vrcs is None:
            initial_vrcs = [0] * self.n_rc
        initial_cond = np.r_[initial_soc, initial_temp, initial_vrcs]
        soln = scipy.integrate.solve_ivp(
            self.rhs(currentfunc, temp_inf), [0, t_max], initial_cond, **solver_kwargs
        )
        socs, temps, v_rcs = soln.y[0], soln.y[1], soln.y[2:]
        currents = np.array([currentfunc(t) for t in soln.t])
        ocv = self._ocv_lut(temps, socs)
        v_out = ocv - currents * self._r_luts[0](temps, socs) - v_rcs.sum(axis=0)
        heat_gen = -currents * (v_out - ocv)
        return ECMSolution(soln.t, currents, v_out, socs, temps, v_rcs, heat_gen)
