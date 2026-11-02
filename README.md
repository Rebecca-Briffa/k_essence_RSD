# k-essence RSD

Late-time MCMC constraints for kinetic *k*-essence (Model 1 / Model 2) versus ΛCDM, using cosmic chronometers, Pantheon+SH0ES, DESI BAO (free \(r_{\rm drag}\)), and RSD/\(f\sigma_8\).

## Layout

| Path | Contents |
|------|----------|
| `LCDM/` | ΛCDM MCMC runners + `Data/` |
| `Model_1/` | Model 1 runners + `Data/` |
| `Model_2/` | Model 2 runners + `Data/` |
| `Results/` | AIC/BIC tables, GetDist triangles, whisker & \(f\sigma_8\) plots (Nov 2026 reruns) |
| `chains/` | Latest chain + log-prob files (**local only** — not on GitHub; see below) |

## Chains

The Nov 2026 rerun chains live in `chains/{LCDM,Model_1,Model_2}/` on the local copy. They are **not** pushed to GitHub (six files exceed the 100 MB limit; total ~2 GB). Keep them beside this repo or archive them on Zenodo / Git LFS if you need a remote copy.

## Results

Regenerate tables/figures (after placing chains where `Results/paths.py` expects them):

```bash
python Results/run_all.py
```

## Note

`Model_2/CC_PNp.py` is the corrected 5-parameter CC+PN run (`ndim=5`, outputs `M2_CC_PN_*_latest_0211.txt`).
