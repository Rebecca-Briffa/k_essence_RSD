# k-essence RSD

MCMC analysis of kinetic *k*-essence (Model 1 and Model 2) against flat ΛCDM, using cosmic chronometers (CC), Pantheon+SH0ES (PN), DESI BAO with free \(r_{\rm drag}\), and RSD/\(f\sigma_8\).

## Repository layout

```
k_essence_RSD/
├── LCDM/                 # flat ΛCDM
├── Model_1/              # k-essence Model 1 (α_phys)
├── Model_2/              # k-essence Model 2 (x₀, λ)
└── Results/              # post-processing tables and figures
```

Each model directory has the same structure:

| Path | Role |
|------|------|
| `CC_PNp.py` | MCMC: CC + PN |
| `CC_PNp_BAO.py` | MCMC: CC + PN + BAO |
| `CC_PNp_fs8.py` | MCMC: CC + PN + RSD |
| `CC_PNp_BAO_fs8.py` | MCMC: CC + PN + BAO + RSD |
| `Likelihoods.py` | Likelihoods (CC, PN, BAO, RSD) |
| `PNplus_data.py` | Pantheon+SH0ES data cut (incl. calibrators) |
| `solver_new.py` / `util.py` | Background / growth helpers |
| `Data/` | Observational inputs used by the likelihoods |

Scripts are meant to be run from inside the corresponding model directory (so relative `Data/` paths resolve).

## Dataset abbreviations

| Tag | Data |
|-----|------|
| CC | Cosmic chronometers |
| PN | Pantheon+SH0ES |
| BAO | DESI BAO (\(r_{\rm drag}\) free when BAO is included) |
| RSD | \(f\sigma_8\) / growth |

## Sampled parameters

### ΛCDM (`LCDM/`)

| Combination | Parameters |
|-------------|------------|
| CC+PN | \(H_0\), \(\Omega_{m,0}\), \(M\) |
| CC+PN+BAO | \(H_0\), \(\Omega_{m,0}\), \(M\), \(r_{\rm drag}\) |
| CC+PN+RSD | \(H_0\), \(\Omega_{m,0}\), \(\sigma_{8,0}\), \(D'(0)\), \(M\) |
| CC+PN+BAO+RSD | \(H_0\), \(\Omega_{m,0}\), \(\sigma_{8,0}\), \(D'(0)\), \(M\), \(r_{\rm drag}\) |

### Model 1 (`Model_1/`)

| Combination | Parameters |
|-------------|------------|
| CC+PN | \(H_0\), \(\Omega_{m,0}\), \(\alpha_{\rm phys}\), \(M\) |
| CC+PN+BAO | \(H_0\), \(\Omega_{m,0}\), \(\alpha_{\rm phys}\), \(M\), \(r_{\rm drag}\) |
| CC+PN+RSD | \(H_0\), \(\Omega_{m,0}\), \(\alpha_{\rm phys}\), \(\sigma_{8,0}\), \(D'(0)\), \(M\) |
| CC+PN+BAO+RSD | \(H_0\), \(\Omega_{m,0}\), \(\alpha_{\rm phys}\), \(\sigma_{8,0}\), \(D'(0)\), \(M\), \(r_{\rm drag}\) |

### Model 2 (`Model_2/`)

| Combination | Parameters |
|-------------|------------|
| CC+PN | \(H_0\), \(\Omega_{m,0}\), \(x_0\), \(\lambda\), \(M\) |
| CC+PN+BAO | \(H_0\), \(\Omega_{m,0}\), \(x_0\), \(\lambda\), \(M\), \(r_{\rm drag}\) |
| CC+PN+RSD | \(H_0\), \(\Omega_{m,0}\), \(x_0\), \(\lambda\), \(\sigma_{8,0}\), \(D'(0)\), \(M\) |
| CC+PN+BAO+RSD | \(H_0\), \(\Omega_{m,0}\), \(x_0\), \(\lambda\), \(\sigma_{8,0}\), \(D'(0)\), \(M\), \(r_{\rm drag}\) |

\(\sigma_{8,0}\) and \(D'(0)\) are sampled only when RSD is included. \(S_8=\sigma_{8,0}\sqrt{\Omega_{m,0}/0.3}\) is derived in post-processing.

## Results/

Precomputed tables and figures for the combinations above:

| File | Content |
|------|---------|
| `AIC_BIC_tables.tex` / `AIC_BIC_full.csv` | Absolute and ΔAIC / ΔBIC vs ΛCDM |
| `*_all_combinations_getdist.pdf` | Triangle plots (all four data combinations) |
| `*_constraints.txt` / `.tex` | Peak ± 68% equal-tailed constraints |
| `whisker_summary.pdf` | \(H_0\), \(\Omega_{m,0}\), \(\sigma_{8,0}\), \(S_8\) |
| `sigma8_S8_constraints.tex` | \(\sigma_{8,0}\) and \(S_8\) (RSD combinations only) |
| `fs8_residuals_BAO_RSD.pdf` | \(f\sigma_8\) theory vs data at the joint MAP |

BIC uses \(N_{\rm PN}=1657\) (fixed Pantheon+SH0ES selection): \(N=1672\) (CC+PN), \(1685\) (+BAO), \(1692\) (+RSD), \(1705\) (+BAO+RSD).
