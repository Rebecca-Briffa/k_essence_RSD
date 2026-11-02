"""fσ8 theory vs data at joint MAP of CC+PN+BAO+RSD → Results_0211."""

from __future__ import annotations

import importlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from paths import CHAINS, OUT, ROOT

Z_INT = np.linspace(0.0, 2.4, 800)

SPECS = {
    "LCDM": {
        "dir": ROOT / "LCDM",
        "chain": CHAINS["LCDM"]["CC+PN+BAO+RSD"]["chain"],
        "logprob": CHAINS["LCDM"]["CC+PN+BAO+RSD"]["logprob"],
        "color": "#1f77b4",
    },
    "Model_1": {
        "dir": ROOT / "Model_1",
        "chain": CHAINS["Model_1"]["CC+PN+BAO+RSD"]["chain"],
        "logprob": CHAINS["Model_1"]["CC+PN+BAO+RSD"]["logprob"],
        "color": "#2ca02c",
    },
    "Model_2": {
        "dir": ROOT / "Model_2",
        "chain": CHAINS["Model_2"]["CC+PN+BAO+RSD"]["chain"],
        "logprob": CHAINS["Model_2"]["CC+PN+BAO+RSD"]["logprob"],
        "color": "#ff7f0e",
    },
}


def load_fs8_data():
    df = pd.read_csv(ROOT / "Data" / "f_sigma8_updated.csv").sort_values("z")
    return (
        np.asarray(df.iloc[:, 0], float),
        np.asarray(df.iloc[:, 1], float),
        np.asarray(df.iloc[:, 2], float),
    )


def best_fit_theta(chain_path, logprob_path):
    samples = np.loadtxt(chain_path)
    lp = np.loadtxt(logprob_path).ravel()
    mask = np.isfinite(lp) & np.isfinite(samples).all(axis=1)
    samples, lp = samples[mask], lp[mask]
    return samples[np.argmax(lp)]


def D_diff(y, z, H_0, Om, hz_interp, dhz_interp):
    D, g = y
    hz, dhz = hz_interp(z), dhz_interp(z)
    part1 = -(dhz / hz - 1.0 / (1.0 + z))
    part2 = 1.5 * (H_0 / hz) ** 2 * Om * (1.0 + z)
    return [g, part1 * g + part2 * D]


def fs8_theory(z_data, H0, Om, s80, f0, Hz, dH_dz):
    hz_interp = interp1d(Z_INT, Hz, kind="cubic", bounds_error=False, fill_value="extrapolate")
    dhz_interp = interp1d(
        Z_INT, dH_dz, kind="cubic", bounds_error=False, fill_value="extrapolate"
    )
    sol = odeint(D_diff, [1.0, f0], Z_INT, args=(H0, Om, hz_interp, dhz_interp))
    dD = interp1d(Z_INT, sol[:, 1], kind="cubic", bounds_error=False, fill_value="extrapolate")(
        z_data
    )
    return -(1.0 + z_data) * s80 * dD


def import_solver(model_dir):
    for key in list(sys.modules):
        if key == "solver_new" or key.startswith("solver_new."):
            del sys.modules[key]
    sys.path.insert(0, str(model_dir))
    try:
        return importlib.import_module("solver_new")
    finally:
        if sys.path and sys.path[0] == str(model_dir):
            sys.path.pop(0)


def theory_for_model(name, theta, z_data):
    solver = import_solver(SPECS[name]["dir"])
    if name == "LCDM":
        H0, Om, s80, f0 = theta[:4]
        Hz, dH_dz = solver.H_dH_arrays(H0, Om, Z_INT)
        return fs8_theory(z_data, H0, Om, s80, f0, Hz, dH_dz)
    if name == "Model_1":
        H0, Om, B, s80, f0 = theta[:5]
        Hz, dH_dz = solver.H_dH_arrays(H0, B, Om, Z_INT)
        return fs8_theory(z_data, H0, Om, s80, f0, Hz, dH_dz)
    if name == "Model_2":
        H0, Om, x0, lambd, s80, f0 = theta[:6]
        Hz, dH_dz = solver.H_dH_arrays(H0, Om, x0, lambd, Z_INT)
        if not np.all(np.isfinite(Hz)):
            raise RuntimeError("Model 2 background failed at MAP")
        return fs8_theory(z_data, H0, Om, s80, f0, Hz, dH_dz)
    raise ValueError(name)


def main():
    z, fs8, err = load_fs8_data()
    z_dense = np.linspace(z.min(), z.max(), 200)
    theories = {}
    for name, spec in SPECS.items():
        theta = best_fit_theta(spec["chain"], spec["logprob"])
        print(f"{name} MAP: {theta}")
        theories[name] = (
            theory_for_model(name, theta, z_dense),
            theory_for_model(name, theta, z),
        )

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax0.errorbar(z, fs8, yerr=err, fmt="ko", ms=4, capsize=2, label="data", zorder=5)
    for name, spec in SPECS.items():
        y_d, _ = theories[name]
        label = r"$\Lambda$CDM" if name == "LCDM" else name.replace("_", " ")
        ax0.plot(z_dense, y_d, color=spec["color"], lw=1.8, label=label)
    ax0.set_ylabel(r"$f\sigma_8(z)$")
    ax0.legend(frameon=False)
    #ax0.set_title(r"Joint MAP from CC+PN+BAO+RSD (Results\_0211)")

    for name, spec in SPECS.items():
        _, y_z = theories[name]
        ax1.errorbar(
            z,
            fs8 - y_z,
            yerr=err,
            fmt="o",
            ms=3.5,
            color=spec["color"],
            alpha=0.85,
            label=name.replace("_", " "),
        )
    ax1.axhline(0.0, color="k", lw=0.8)
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel(r"data $-$ theory")
    fig.tight_layout()
    out = OUT / "fs8_residuals_BAO_RSD.pdf"
    try:
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(OUT / "fs8_residuals_BAO_RSD.png", dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    except PermissionError:
        out = OUT / "fs8_residuals_BAO_RSD_new.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(OUT / "fs8_residuals_BAO_RSD_new.png", dpi=200, bbox_inches="tight")
        print(f"Wrote {out} (original locked)")
    plt.close(fig)


if __name__ == "__main__":
    main()
