"""
Shared settings for reproducing the figures and numbers in the paper
'Why core-surface battery thermal models fail, and how to fix them'.

Every paper script uses the cell parameters, heat-transfer coefficients,
heat-generation profiles, tolerances and figure style defined here, so
that all results are consistent.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np

import coresurf
from coresurf.ecm import TheveninModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

#: Heat transfer coefficients [W m^-2 K^-1] compared throughout the paper.
TEST_HS = [30, 150, 1000]

#: 20 W generated uniformly in the Kokam pouch cell [W m^-3].
STANDARD_HEATGEN = 20 / coresurf.parameters.KOKAM_POUCH_VOLUME

#: Max. ODE step [s] for the WLTP simulations, so the solver does not
#: step over rapid changes in heat generation.
MAX_STEP = 0.5

cell = coresurf.kokam_pouch


@functools.cache
def wltp_heat_generation() -> tuple[np.ndarray, np.ndarray]:
    """
    Volumetric heat generation [W m^-3] over a section of a 10x-scaled
    WLTP drive cycle, computed with a Thevenin ECM of the MLP001 cell.
    Returns (times, heat_generation).
    """
    simulator = TheveninModel(
        DATA_DIR / "MLP001_ocv.csv", DATA_DIR / "MLP001_params.csv", capacity_Ah=2.132
    )

    import pandas as pd

    df = pd.read_csv(DATA_DIR / "wltp.csv")
    currents = 10 * df["Current[A]"].to_numpy()
    times = df["Time[s]"].to_numpy()

    t0, t1 = 1800, 8000
    mask = np.logical_and(times > t0, times < t1)
    window_ts = times[mask] - t0
    window_currents = currents[mask]

    def currentfunc(t):
        return np.interp(t, window_ts, window_currents, 0)

    soln = simulator.simulate(
        currentfunc,
        initial_soc=0.99,
        temp_inf=25,
        t_max=3100,
        max_step=MAX_STEP,
        atol=coresurf.DEFAULT_TOL,
        rtol=coresurf.DEFAULT_TOL,
    )
    cell_volume = 0.062 * 0.048 * 0.006  # MLP001 [m^3]
    return soln.t, soln.heat_generation / cell_volume


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

MODEL_STYLES = {
    "small_biot": dict(color="C0", label="Small Biot"),
    "two_eigen": dict(color="C1", label=r"$\{\phi_1,\phi_2\}$ Galerkin"),
    "eigen_ss": dict(color="C2", label=r"$\{\phi_1,\Psi\}$ Galerkin"),
    "literature": dict(color="C4", label="Literature core-surface"),
}


def figure_args(description: str) -> argparse.Namespace:
    """Parse the command-line options shared by all figure scripts, and
    configure matplotlib accordingly."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--outdir", type=Path, default=Path("figures"), help="output directory"
    )
    parser.add_argument(
        "--format",
        default="pgf",
        help="output format: pgf (as in the paper), pdf, png, ...",
    )
    parser.add_argument(
        "--show", action="store_true", help="show the figure instead of saving it"
    )
    args = parser.parse_args()
    setup_style(args.format, args.show)
    return args


def setup_style(fmt: str = "pgf", show: bool = False):
    import matplotlib

    if fmt == "pgf" and not show:
        matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def save_figure(fig, name: str, args: argparse.Namespace):
    import matplotlib.pyplot as plt

    if args.show:
        plt.show()
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    path = args.outdir / f"{name}.{args.format}"
    fig.savefig(path)
    print(f"Saved {path}")


def plot_error_rows(ax, ts, model_key, mean_or_core, surf, mean_or_core_truth, surf_truth):
    """Plot one model into a 4-row column: temperature, error,
    surface temperature, surface error."""
    style = MODEL_STYLES[model_key]
    ax[0].plot(ts, mean_or_core, style["color"], label=style["label"])
    ax[1].plot(ts, mean_or_core - mean_or_core_truth, style["color"])
    ax[2].plot(ts, surf, style["color"])
    ax[3].plot(ts, surf - surf_truth, style["color"])


def label_error_rows(axarrs):
    axarrs[0][0].set_ylabel("Core temperature\nrise [$^\\circ$C]")
    axarrs[1][0].set_ylabel("Core absolute\nerror [$^\\circ$C]")
    axarrs[2][0].set_ylabel("Surface temperature\nrise [$^\\circ$C]")
    axarrs[3][0].set_ylabel("Surface absolute\nerror [$^\\circ$C]")
