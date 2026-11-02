# Results_0211

All tables and figures regenerated from the Nov 2026 reruns.

## Chain map

See `paths.py` for the exact chain / log-prob files:

- **ΛCDM** → `LCDM/chains_new/*_2110*`
- **Model 1** → `Model_1/*_2710*` / `*_2810*`
- **Model 2** → `Model_2/*_2710*` / `*_2210_tight*`
  - **Caveat:** `Model_2/M2_CC_PN_chain_latest_tight.txt` is an 8-column
    BAO+RSD file misnamed as CC+PN. The CC+PN entry therefore uses the last
    true 5-column chain `Model_2/chains/M2_CC_PN_chain_latest_tight_new.txt`
    (Aug 2026, old Pantheon cut). Re-run Model 2 CC+PN for a fair AIC vs the
    Oct ΛCDM / Model 1 chains.

## Regenerate everything

```bash
python Results_0211/run_all.py
```

Or step by step:

```bash
python Results_0211/run_aic_bic.py
python Results_0211/run_getdist_triangles.py
python Results_0211/run_whisker.py
python Results_0211/run_fs8_residuals.py
```

## Outputs

| File | Content |
|------|---------|
| `AIC_BIC_full.csv` | Full χ² / AIC / BIC / deltas |
| `AIC_BIC_tables.tex` | Paper-ready LaTeX tables |
| `AIC_BIC_absolute.txt` / `AIC_BIC_delta_vs_LCDM.txt` | Text summaries |
| `LCDM_all_combinations_getdist.pdf` | ΛCDM triangle |
| `Model_1_all_combinations_getdist.pdf` | Model 1 triangle |
| `Model_2_all_combinations_getdist.pdf` | Model 2 triangle |
| `*_constraints.txt` / `.tex` | Peak ± 68% tables |
| `whisker_summary.pdf` / `.png` | H₀ / Ωm / σ₈ / S₈ whiskers |
| `whisker_constraints.tex` | Whisker peak table |
| `fs8_residuals_BAO_RSD.pdf` | fσ₈ MAP curves + residuals |

BIC uses \(N_{\rm PN}=1657\) (fixed calibrator cut): totals 1672 / 1685 / 1692 / 1705.
