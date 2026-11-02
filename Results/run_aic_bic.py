"""AIC / BIC / ΔAIC / ΔBIC from the Results_0211 chain map."""

from __future__ import annotations

import numpy as np
import pandas as pd

from paths import CHAINS, DATASETS, MODELS, N_DATA, OUT, ROOT, assert_all_exist

REF = "LCDM"


def max_lnL(path) -> float | None:
    if not path.is_file():
        return None
    lp = np.asarray(np.loadtxt(path), dtype=float).ravel()
    lp = lp[np.isfinite(lp)]
    return float(np.max(lp)) if lp.size else None


def main() -> None:
    missing = assert_all_exist()
    if missing:
        print("MISSING FILES:")
        for m in missing:
            print(" ", m)
        raise SystemExit(1)

    rows = []
    for model in MODELS:
        for dset in DATASETS:
            spec = CHAINS[model][dset]
            names = spec["names"]
            k, n = len(names), N_DATA[dset]
            lnL = max_lnL(spec["logprob"])
            chi2 = -2.0 * lnL
            aic = 2.0 * k - 2.0 * lnL
            bic = k * np.log(n) - 2.0 * lnL
            rows.append(
                {
                    "model": model,
                    "dataset": dset,
                    "path": str(spec["logprob"].relative_to(ROOT)),
                    "params": ",".join(names),
                    "k": k,
                    "N": n,
                    "lnL_max": lnL,
                    "chi2_min": chi2,
                    "AIC": aic,
                    "BIC": bic,
                }
            )
    df = pd.DataFrame(rows)
    df["dAIC"] = np.nan
    df["dBIC"] = np.nan
    df["dchi2"] = np.nan
    for dset in DATASETS:
        ref = df[(df.model == REF) & (df.dataset == dset)].iloc[0]
        mask = df.dataset == dset
        df.loc[mask, "dAIC"] = df.loc[mask, "AIC"] - ref["AIC"]
        df.loc[mask, "dBIC"] = df.loc[mask, "BIC"] - ref["BIC"]
        df.loc[mask, "dchi2"] = df.loc[mask, "chi2_min"] - ref["chi2_min"]

    csv_path = OUT / "AIC_BIC_full.csv"
    df.to_csv(csv_path, index=False)

    # Absolute summary
    lines = [
        "Absolute AIC / BIC (Results_0211 reruns)",
        f"BIC = k ln N - 2 ln L_max; N_PN=1657 (fixed calibrator cut)",
        "",
    ]
    for dset in DATASETS:
        lines.append(f"=== {dset}  (N = {N_DATA[dset]}) ===")
        lines.append(
            f"{'model':<10} {'k':>2} {'lnL_max':>12} {'chi2':>10} {'AIC':>10} {'BIC':>10}"
        )
        for _, r in df[df.dataset == dset].iterrows():
            lines.append(
                f"{r.model:<10} {int(r.k):>2} {r.lnL_max:12.4f} {r.chi2_min:10.2f} "
                f"{r.AIC:10.2f} {r.BIC:10.2f}"
            )
        lines.append("")
    (OUT / "AIC_BIC_absolute.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Deltas
    dlines = ["dX = X_model - X_LCDM (Results_0211)", ""]
    for dset in DATASETS:
        dlines.append(f"=== {dset} ===")
        dlines.append(f"{'model':<10} {'dchi2':>10} {'dAIC':>10} {'dBIC':>10}")
        for _, r in df[df.dataset == dset].iterrows():
            dlines.append(
                f"{r.model:<10} {r.dchi2:10.4f} {r.dAIC:10.4f} {r.dBIC:10.4f}"
            )
        dlines.append("")
    (OUT / "AIC_BIC_delta_vs_LCDM.txt").write_text(
        "\n".join(dlines) + "\n", encoding="utf-8"
    )

    # Paper-style LaTeX (Model 1 / Model 2 only, with Δχ²)
    def row(model, dset):
        r = df[(df.model == model) & (df.dataset == dset)].iloc[0]
        return (
            f" & {dset} & {r.chi2_min:.2f} & {r.AIC:.2f} & {r.BIC:.2f} & "
            f"${r.dchi2:+.3f}$ & ${r.dAIC:+.2f}$ & ${r.dBIC:+.2f}$ \\\\"
        )

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Minimum $\chi^2$, AIC and BIC for Model~1 and Model~2 "
        r"(Results\_0211 reruns). $\Delta\chi^2$, $\Delta$AIC and $\Delta$BIC "
        r"are relative to $\Lambda$CDM for the same dataset "
        r"($\Delta X=X_{\mathrm{model}}-X_{\Lambda\mathrm{CDM}}$; "
        r"$N=1672/1685/1692/1705$). Negative values favour the alternative model.}",
        r"\label{tab:AIC_BIC_0211}",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"Model & Dataset & $\chi^2_{\min}$ & AIC & BIC & "
        r"$\Delta\chi^2$ & $\Delta$AIC & $\Delta$BIC \\",
        r"\hline",
        r"\multirow{4}{*}{Model~1}",
        *[row("Model_1", d) for d in DATASETS],
        r"\hline",
        r"\multirow{4}{*}{Model~2}",
        *[row("Model_2", d) for d in DATASETS],
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    # LCDM reference table
    tex += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{$\Lambda$CDM goodness-of-fit (Results\_0211).}",
        r"\label{tab:LCDM_AIC_0211}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Dataset & $\chi^2_{\min}$ & AIC & BIC \\",
        r"\hline",
    ]
    for d in DATASETS:
        r = df[(df.model == "LCDM") & (df.dataset == d)].iloc[0]
        tex.append(f"{d} & {r.chi2_min:.2f} & {r.AIC:.2f} & {r.BIC:.2f} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / "AIC_BIC_tables.tex").write_text("\n".join(tex), encoding="utf-8")

    def _safe(s: str) -> None:
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode("ascii"))

    _safe("\n".join(lines))
    _safe("\n".join(dlines))
    print(f"Wrote {csv_path}")
    print(f"Wrote {OUT / 'AIC_BIC_tables.tex'}")


if __name__ == "__main__":
    main()
