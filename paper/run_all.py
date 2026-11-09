#!/usr/bin/env python3
"""Regenerate every figure and number in the paper.

Extra arguments (e.g. --format pdf --outdir out) are passed to each
figure script.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FIGURES = [
    "fig_literature_comparison.py",
    "fig_newmodel_comparison.py",
    "fig_summary_comparison.py",
    "fig_wltp_comparison.py",
    "fig_wltp_zoom.py",
]
CALCULATIONS = ["calc_steadystate_error.py", "calc_wltp_error.py"]


def main():
    extra = sys.argv[1:]
    for script in FIGURES:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, HERE / script, *extra], check=True)
    for script in CALCULATIONS:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, HERE / script], check=True)


if __name__ == "__main__":
    main()
