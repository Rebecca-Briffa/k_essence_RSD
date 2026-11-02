"""GetDist triangles + constraint tables for LCDM, Model 1, Model 2 → Results_0211."""

from __future__ import annotations

import numpy as np
from getdist import MCSamples, plots

from paths import CHAINS, DATASETS, LABELS, MODELS, OUT

MAX_SAMPLES = 250_000

PLOT_PARAMS = {
    "LCDM": ["H0", "OM", "s80"],
    "Model_1": ["H0", "OM", "B", "s80"],
    "Model_2": ["H0", "OM", "x0", "lambd", "s80"],
}

PARAM_LIMITS = {
    "Model_1": {"B": (None, 0.11)},
}

COLORS = {
    "CC+PN": ("#1f77b4", "--", 0.75),
    "CC+PN+BAO": ("#2ca02c", "-", 0.75),
    "CC+PN+RSD": ("#9467bd", "-.", 0.75),
    "CC+PN+BAO+RSD": ("#ff7f0e", "-", 0.45),
}

MODEL_LABEL = {
    "LCDM": r"$\Lambda$CDM",
    "Model_1": "Model 1",
    "Model_2": "Model 2",
}


def load_mcs(model: str, dset: str) -> MCSamples:
    spec = CHAINS[model][dset]
    samples = np.loadtxt(spec["chain"])
    samples = samples[np.isfinite(samples).all(axis=1)]
    names = spec["names"]
    if samples.ndim != 2 or samples.shape[1] != len(names):
        raise ValueError(f"{spec['chain'].name}: expected {len(names)} cols, got {samples.shape}")

    # Model 2 soft island cut (same as previous pipeline)
    if model == "Model_2" and "OM" in names and "x0" in names:
        i_om, i_x0 = names.index("OM"), names.index("x0")
        keep = (samples[:, i_om] < 0.5) & (samples[:, i_x0] > 0.6)
        samples = samples[keep]

    if len(samples) > MAX_SAMPLES:
        step = int(np.ceil(len(samples) / MAX_SAMPLES))
        samples = samples[::step]

    color, ls, alpha = COLORS[dset]
    mcs = MCSamples(
        samples=samples,
        names=names,
        labels=[LABELS[n] for n in names],
        label=dset.replace("+", " + "),
    )
    mcs.updateSettings(
        {"fine_bins_2D": 512, "smooth_scale_2D": 0.5, "smooth_scale_1D": 0.3}
    )
    mcs._r0211_style = (color, ls, alpha)  # stash for plotting
    return mcs


def peak_constraint(mcs: MCSamples, name: str):
    dens = mcs.get1DDensity(name)
    peak = float(dens.x[np.argmax(dens.P)])
    j = mcs.index[name]
    lo, hi = np.percentile(mcs.samples[:, j], [15.865, 84.135])
    return peak, hi - peak, peak - lo


def fmt(peak, up, down):
    return f"{peak:.4g}^{{+{up:.3g}}}_{{-{down:.3g}}}"


def write_constraints(model: str, mcs_list: list, dsets: list) -> None:
    all_names = []
    for d in dsets:
        for n in CHAINS[model][d]["names"]:
            if n not in all_names:
                all_names.append(n)

    lines = [f"# {model} constraints (peak ± 68% equal-tailed) Results_0211", ""]
    header = f"{'dataset':<22}" + "".join(f"{n:>22}" for n in all_names)
    lines += [header, "-" * len(header)]

    tex_rows = []
    for mcs, dset in zip(mcs_list, dsets):
        cells = {}
        for n in CHAINS[model][dset]["names"]:
            p, u, d = peak_constraint(mcs, n)
            cells[n] = fmt(p, u, d)
        line = f"{dset:<22}"
        for n in all_names:
            line += f"{cells.get(n, '—'):>22}"
        lines.append(line)
        vals = [f"${cells[n]}$" if n in cells else "—" for n in all_names]
        tex_rows.append(f"{dset} & " + " & ".join(vals) + r" \\")

    txt = OUT / f"{model}_constraints.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{tabular}{l" + "c" * len(all_names) + "}",
        r"\hline",
        "Dataset & " + " & ".join(f"${LABELS[n]}$" for n in all_names) + r" \\",
        r"\hline",
        *tex_rows,
        r"\hline",
        r"\end{tabular}",
        "",
    ]
    (OUT / f"{model}_constraints.tex").write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {txt}")


def plot_model(model: str) -> None:
    print(f"\n===== {model} =====")
    mcs_list = []
    for dset in DATASETS:
        print(f"  loading {dset} ...")
        mcs_list.append(load_mcs(model, dset))

    available = set()
    for mcs in mcs_list:
        available.update(mcs.getParamNames().list())
    params = [p for p in PLOT_PARAMS[model] if p in available]

    g = plots.get_subplot_plotter(width_inch=10 if model == "LCDM" else 12)
    g.settings.legend_fontsize = 11
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 13
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.7
    g.settings.linewidth_contour = 1.2
    g.settings.title_limit = 0

    line_args = []
    contour_colors, contour_ls, contour_args = [], [], []
    for mcs in mcs_list:
        color, ls, alpha = mcs._r0211_style
        line_args.append({"color": color, "ls": ls, "lw": 1.6, "alpha": alpha})
        contour_colors.append(color)
        contour_ls.append(ls)
        contour_args.append({"alpha": alpha, "color": color})

    kwargs = dict(
        params=params,
        filled=True,
        legend_labels=[s.replace("+", " + ") for s in DATASETS],
        legend_loc="upper right",
        contour_colors=contour_colors,
        contour_ls=contour_ls,
        contour_args=contour_args,
        line_args=line_args,
        title_limit=None,
    )
    if model in PARAM_LIMITS:
        kwargs["param_limits"] = PARAM_LIMITS[model]

    g.triangle_plot(mcs_list, **kwargs)
    out_pdf = OUT / f"{model}_all_combinations_getdist.pdf"
    try:
        g.export(str(out_pdf))
        print(f"Wrote {out_pdf}")
    except PermissionError:
        alt = OUT / f"{model}_all_combinations_getdist_new.pdf"
        g.export(str(alt))
        print(f"Wrote {alt} (original PDF locked)")

    write_constraints(model, mcs_list, DATASETS)


def main() -> None:
    for model in MODELS:
        plot_model(model)


if __name__ == "__main__":
    main()
