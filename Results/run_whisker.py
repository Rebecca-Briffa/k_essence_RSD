"""Whisker summary plot → Results_0211 (new chains)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paths import CHAINS, DATASETS, OUT, ROOT

MAX_SAMPLES = 250_000
FS_XLABEL, FS_TICK, FS_YGROUP, FS_LEGEND = 19, 17, 17, 17
LEGEND_MARKERSIZE, LEGEND_HANDLELENGTH = 9, 2.5

H0_SPA = (67.24, 0.35)
H0_DN = (73.50, 0.81)
S8_KIDS = (0.815, 0.016, 0.021)
S8_DESY6 = (0.789, 0.012, 0.012)
S8_PLANCK = (0.834, 0.016)

MODELS = ["LCDM", "Model 1", "Model 2"]
MODEL_KEY = {"LCDM": "LCDM", "Model 1": "Model_1", "Model 2": "Model_2"}

DATASET_STYLE = {
    "CC+PN": {"color": "#4c78a8", "marker": "o", "label": r"CC+PN"},
    "CC+PN+BAO": {"color": "#f58518", "marker": "D", "label": r"CC+PN+BAO"},
    "CC+PN+RSD": {"color": "#54a24b", "marker": "s", "label": r"CC+PN+RSD"},
    "CC+PN+BAO+RSD": {"color": "#e45756", "marker": "^", "label": r"CC+PN+BAO+RSD"},
}

PANEL_PARAMS = ["H0", "Om", "s80", "S8"]
PANEL_LABELS = {
    "H0": r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    "Om": r"$\Omega_{m,0}$",
    "s80": r"$\sigma_{8,0}$",
    "S8": r"$S_8$",
}


def peak_from_density(mcs, name):
    dens = mcs.get1DDensity(name)
    peak = float(dens.x[np.argmax(dens.P)])
    j = mcs.index[name]
    lo, hi = np.percentile(mcs.samples[:, j], [15.865, 84.135])
    return peak, float(hi - peak), float(peak - lo)


def load_constraints(path, names):
    # paths.py uses OM / rdrag; whisker historically used Om / r_drag
    rename = {"OM": "Om", "rdrag": "r_drag"}
    plot_in = [rename.get(n, n) for n in names]
    samples = np.loadtxt(path)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if len(samples) > MAX_SAMPLES:
        step = int(np.ceil(len(samples) / MAX_SAMPLES))
        samples = samples[::step]

    plot_names = list(plot_in)
    if "s80" in plot_names and "Om" in plot_names:
        i_om, i_s = plot_names.index("Om"), plot_names.index("s80")
        s8 = samples[:, i_s] * np.sqrt(samples[:, i_om] / 0.3)
        samples = np.column_stack([samples, s8])
        plot_names.append("S8")

    mcs = MCSamples(samples=samples, names=plot_names, labels=plot_names, label=path.stem)
    mcs.updateSettings(
        {"fine_bins_2D": 512, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.3}
    )
    return {n: peak_from_density(mcs, n) for n in ("H0", "Om", "s80", "S8") if n in plot_names}


def collect_all():
    data = {m: {} for m in MODELS}
    for label, key in MODEL_KEY.items():
        for dset in DATASETS:
            spec = CHAINS[key][dset]
            print(f"Loading {spec['chain'].relative_to(ROOT)} ...")
            data[label][dset] = load_constraints(spec["chain"], spec["names"])
    return data


def plot_whiskers(data):
    n_ds, n_models = len(DATASETS), len(MODELS)
    dy, gap = 1.0, 1.4
    block = n_ds * dy
    y_pos, model_centers = {}, {}
    y_cursor = 0.0
    for model in MODELS:
        ys = []
        for j, dset in enumerate(DATASETS):
            y = y_cursor + j * dy
            y_pos[(model, dset)] = y
            ys.append(y)
        model_centers[model] = 0.5 * (ys[0] + ys[-1])
        y_cursor += block + gap
    y_max, y_min = y_cursor - gap + 0.5, -0.5

    fig, axes = plt.subplots(1, len(PANEL_PARAMS), figsize=(3.2 * 4, 8.5), sharey=True)
    for ax, pname in zip(axes, PANEL_PARAMS):
        for k in range(n_models - 1):
            y_sep = (k + 1) * block + k * gap + gap / 2.0 - 0.5 * dy
            ax.axhline(y_sep, color="0.65", ls="--", lw=0.9, zorder=0)

        if pname == "H0":
            for (mu, sig), color in [(H0_SPA, "#9e9ac8"), (H0_DN, "#6baed6")]:
                ax.axvspan(mu - sig, mu + sig, color=color, alpha=0.35, zorder=1)
        if pname == "S8":
            for (mu, up, dn), color in [
                ((*S8_KIDS,), "#d95f0e"),
                ((*S8_DESY6,), "#e7298a"),
            ]:
                ax.axvspan(mu - dn, mu + up, color=color, alpha=0.28, zorder=1)
            ax.axvspan(
                S8_PLANCK[0] - S8_PLANCK[1],
                S8_PLANCK[0] + S8_PLANCK[1],
                color="#1b9e77",
                alpha=0.28,
                zorder=1,
            )

        for model in MODELS:
            for dset in DATASETS:
                cons = data[model].get(dset, {})
                if pname not in cons:
                    continue
                peak, up, down = cons[pname]
                st = DATASET_STYLE[dset]
                ax.errorbar(
                    peak,
                    y_pos[(model, dset)],
                    xerr=[[down], [up]],
                    fmt=st["marker"],
                    color=st["color"],
                    ecolor=st["color"],
                    elinewidth=1.5,
                    capsize=3,
                    markersize=7,
                    zorder=3,
                )

        ax.set_xlabel(PANEL_LABELS[pname], fontsize=FS_XLABEL)
        ax.set_ylim(y_max, y_min)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        if pname == "H0":
            ax.set_xlim(64.0, 78.0)
        if pname == "S8":
            ax.set_xlim(0.62, 0.88)

    axes[0].set_yticks([model_centers[m] for m in MODELS])
    axes[0].set_yticklabels(
        [r"$\mathbf{\Lambda}$CDM", "Model 1", "Model 2"],
        fontsize=FS_YGROUP,
        fontweight="bold",
    )
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    ds_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_STYLE[d]["color"],
            marker=DATASET_STYLE[d]["marker"],
            ls="",
            markersize=LEGEND_MARKERSIZE,
            label=DATASET_STYLE[d]["label"],
        )
        for d in DATASETS
    ]
    band_handles = [
        Patch(
            facecolor="#9e9ac8",
            alpha=0.45,
            label=rf"$H_0^{{\rm SPT+Planck+ACT}}={H0_SPA[0]:.2f}\pm{H0_SPA[1]:.2f}$",
        ),
        Patch(
            facecolor="#6baed6",
            alpha=0.45,
            label=rf"$H_0^{{\rm DN}}={H0_DN[0]:.2f}\pm{H0_DN[1]:.2f}$",
        ),
        Patch(
            facecolor="#d95f0e",
            alpha=0.40,
            label=rf"$S_8^{{\rm KiDS}}={S8_KIDS[0]:.3f}^{{+{S8_KIDS[1]:.3f}}}_{{-{S8_KIDS[2]:.3f}}}$",
        ),
        Patch(
            facecolor="#e7298a",
            alpha=0.40,
            label=rf"$S_8^{{\rm DES\,Y6}}={S8_DESY6[0]:.3f}^{{+{S8_DESY6[1]:.3f}}}_{{-{S8_DESY6[2]:.3f}}}$",
        ),
        Patch(
            facecolor="#1b9e77",
            alpha=0.40,
            label=rf"$S_8^{{\rm P18}}={S8_PLANCK[0]:.3f}\pm{S8_PLANCK[1]:.3f}$",
        ),
    ]
    fig.subplots_adjust(top=0.97, bottom=0.16, left=0.10, right=0.98, wspace=0.18)
    fig.legend(
        handles=ds_handles + band_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, -0.08),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.6",
        fontsize=FS_LEGEND,
        handlelength=LEGEND_HANDLELENGTH,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.4,
        labelspacing=0.55,
    )
    out_pdf = OUT / "whisker_summary.pdf"
    out_png = OUT / "whisker_summary.png"
    try:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.25)
        print(f"Wrote {out_pdf}")
    except PermissionError:
        out_pdf = OUT / "whisker_summary_new.pdf"
        out_png = OUT / "whisker_summary_new.png"
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.25)
        fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.25)
        print(f"Wrote {out_pdf} (original locked)")
    plt.close(fig)

    # latex table of whisker peaks
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Model & Dataset & $H_0$ & $\Omega_{m,0}$ & $\sigma_{8,0}$ & $S_8$ \\",
        r"\hline",
    ]
    for model in MODELS:
        for dset in DATASETS:
            c = data[model][dset]

            def f(name):
                if name not in c:
                    return "—"
                p, u, d = c[name]
                return rf"${p:.4g}^{{+{u:.3g}}}_{{-{d:.3g}}}$"

            mtex = r"$\Lambda$CDM" if model == "LCDM" else model
            lines.append(
                f"{mtex} & {dset} & {f('H0')} & {f('Om')} & {f('s80')} & {f('S8')} \\\\"
            )
    lines += [r"\hline", r"\end{tabular}", ""]
    (OUT / "whisker_constraints.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    data = collect_all()
    plot_whiskers(data)


if __name__ == "__main__":
    main()
