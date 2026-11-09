#!/usr/bin/env python3

from collections.abc import Callable
import numpy as np
import scipy.integrate
import scipy.interpolate
import pandas as pd


class TheveninModel:
    def __init__(
        self,
        ocv_filename: str,
        params_filename: str,
        capacity_Ah: float,
        thermal_mass: float = np.inf,
        convection_rate: float = 0,
    ):
        self.thermal_mass = thermal_mass
        self.capacity_Ah = capacity_Ah
        self.convection_rate = convection_rate

        params_df = pd.read_csv(params_filename)
        self._n_rc = int((len(params_df.columns) - 3) / 2)

        if params_df["Temperature_degC"].nunique() < 2:
            interpolator = self.get_soc_lut
        else:
            interpolator = self.get_soc_temperature_lut
        self._r_luts = [
            interpolator(params_df, f"R{i} [Ohm]") for i in range(self._n_rc + 1)
        ]
        self._tau_luts = [
            interpolator(params_df, f"tau{i} [s]") for i in range(1, self._n_rc + 1)
        ]
        self._ocv_lut = interpolator(pd.read_csv(ocv_filename), "OCV[V]")

    @staticmethod
    def get_soc_lut(
        df: pd.DataFrame, param_header: str
    ) -> scipy.interpolate.PchipInterpolator:
        socs = df["SOC"].to_numpy()
        targetdata = df[param_header].to_numpy()
        sort_idx = np.argsort(socs)
        interpolant = scipy.interpolate.PchipInterpolator(
            socs[sort_idx],
            targetdata[sort_idx],
            extrapolate=False,
        )
        return lambda temp, soc: interpolant(soc)

    @staticmethod
    def get_soc_temperature_lut(
        df: pd.DataFrame, param_header: str
    ) -> scipy.interpolate.CloughTocher2DInterpolator:
        sortings = ["Temperature_degC", "SOC"]
        sort_df = df.sort_values(by=sortings)
        temps = sort_df["Temperature_degC"].to_numpy()
        socs = sort_df["SOC"].to_numpy()
        targetdata = sort_df[param_header].to_numpy()
        preds = np.vstack((temps, socs)).T
#        return scipy.interpolate.CloughTocher2DInterpolator(
        return scipy.interpolate.LinearNDInterpolator(
            np.vstack((temps, socs)).T,
            targetdata,
        )

    def get_ode_rhs(
        self,
        currentfunc: Callable[[float], float],
        temp_inf: float,
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        def ode_rhs(t: float, x: np.ndarray):
            soc = x[0]
            temp = x[1]
            v_rc = x[2:]

            current = currentfunc(t)

            rs = np.array([r_lut(temp, soc) for r_lut in self._r_luts])
            taus = np.array([tau_lut(temp, soc) for tau_lut in self._tau_luts])

            if any(np.isnan(rs)) or any(np.isnan(taus)):
                raise ValueError(
                    f"A parameter interpolator extrapolated at time {t} with SOC {soc} and temperature {temp}"
                )
            # heat_gen = (
            #     current**2 * rs.sum()
            #     - self.convection_rate * (temp - temp_inf)
            # )
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
        initial_vrcs: None | list | np.ndarray = None,
        **solver_kwargs,
    ):
        if initial_temp is None:
            initial_temp = temp_inf
        if initial_vrcs is None:
            initial_vrcs = [0] * self._n_rc
        initial_cond = np.r_[initial_soc, initial_temp, initial_vrcs]
        ode_rhs = self.get_ode_rhs(currentfunc, temp_inf)
        soln = scipy.integrate.solve_ivp(
            ode_rhs,
            [0, t_max],
            initial_cond,
            **solver_kwargs,
        )
        socs, temps, v_rcs = soln.y[0], soln.y[1], soln.y[2:]
        currents = np.array([currentfunc(t) for t in soln.t])
        rs = np.array([r_lut(temps, socs) for r_lut in self._r_luts])
        v_out = (
            self._ocv_lut(temps, socs)
            - currents * self._r_luts[0](temps, socs)
            - v_rcs.sum(axis=0)
        )
        heat_gen = -currents*(v_out - self._ocv_lut(temps, socs))
        return soln.t, currents, v_out, socs, temps, v_rcs, heat_gen
