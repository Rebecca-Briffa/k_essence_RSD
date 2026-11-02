"""
Central map of the Nov 2026 rerun chains / log-probs for Results_0211.
All update scripts import from here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Chains (samples) and matching log_prob files
# ---------------------------------------------------------------------------
CHAINS = {
    "LCDM": {
        "CC+PN": {
            "chain": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_chain_2110.txt",
            "logprob": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_log_prob_2110.txt",
            "names": ["H0", "OM", "M"],
        },
        "CC+PN+BAO": {
            "chain": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_BAO_chain_2110.txt",
            "logprob": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_BAO_log_prob_2110.txt",
            "names": ["H0", "OM", "M", "rdrag"],
        },
        "CC+PN+RSD": {
            "chain": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_fs8_chain_rd_2110.txt",
            "logprob": ROOT / "LCDM" / "chains_new" / "LCDM_CC_PN_fs8_log_prob_rd_2110.txt",
            "names": ["H0", "OM", "s80", "f0", "M"],
        },
        "CC+PN+BAO+RSD": {
            "chain": ROOT
            / "LCDM"
            / "chains_new"
            / "LCDM_CC_PN_BAO_fs8_chain_rd_2110.txt",
            "logprob": ROOT
            / "LCDM"
            / "chains_new"
            / "LCDM_CC_PN_BAO_fs8_log_prob_rd_2110.txt",
            "names": ["H0", "OM", "s80", "f0", "M", "rdrag"],
        },
    },
    "Model_1": {
        "CC+PN": {
            "chain": ROOT / "Model_1" / "CC_PN_chain_2710.txt",
            "logprob": ROOT / "Model_1" / "CC_PN_log_prob_2710.txt",
            "names": ["H0", "OM", "B", "M"],
        },
        "CC+PN+BAO": {
            "chain": ROOT / "Model_1" / "CC_PN_BAO_chain_rd_2710.txt",
            "logprob": ROOT / "Model_1" / "CC_PN_BAO_log_prob_rd_2710.txt",
            "names": ["H0", "OM", "B", "M", "rdrag"],
        },
        "CC+PN+RSD": {
            "chain": ROOT / "Model_1" / "CC_PN_fs8_chain_rd_2810.txt",
            "logprob": ROOT / "Model_1" / "CC_PN_fs8_log_prob_rd_2810.txt",
            "names": ["H0", "OM", "B", "s80", "f0", "M"],
        },
        "CC+PN+BAO+RSD": {
            "chain": ROOT / "Model_1" / "CC_PN_BAO_fs8_chain_rd_2810.txt",
            "logprob": ROOT / "Model_1" / "CC_PN_BAO_fs8_log_prob_rd_2810.txt",
            "names": ["H0", "OM", "B", "s80", "f0", "M", "rdrag"],
        },
    },
    "Model_2": {
        # True 5-param CC+PN re-run (after fixing CC_PNp.py). Until it exists,
        # assert_all_exist will flag it — then run Model_2/CC_PNp.py.
        "CC+PN": {
            "chain": ROOT / "Model_2" / "M2_CC_PN_chain_latest_0211.txt",
            "logprob": ROOT / "Model_2" / "M2_CC_PN_log_prob_latest_0211.txt",
            "names": ["H0", "OM", "x0", "lambd", "M"],
        },
        "CC+PN+BAO": {
            "chain": ROOT / "Model_2" / "M2_CC_PN_BAO_chain_latest_tight_2710.txt",
            "logprob": ROOT / "Model_2" / "M2_CC_PN_BAO_log_prob_latest_2710.txt",
            "names": ["H0", "OM", "x0", "lambd", "M", "rdrag"],
        },
        "CC+PN+RSD": {
            "chain": ROOT / "Model_2" / "M2_CC_PN_RSD_chain_latest_tight_2710.txt",
            "logprob": ROOT
            / "Model_2"
            / "M2_CC_PN_RSD_log_prob_latest_tight_2710.txt",
            "names": ["H0", "OM", "x0", "lambd", "s80", "f0", "M"],
        },
        "CC+PN+BAO+RSD": {
            "chain": ROOT
            / "Model_2"
            / "M2_CC_PN_BAO_RSD_chain_latest_2210_tight.txt",
            "logprob": ROOT
            / "Model_2"
            / "M2_CC_PN_BAO_RSD_log_prob_2210_tight.txt",
            "names": ["H0", "OM", "x0", "lambd", "s80", "f0", "M", "rdrag"],
        },
    },
}

DATASETS = ["CC+PN", "CC+PN+BAO", "CC+PN+RSD", "CC+PN+BAO+RSD"]
MODELS = ["LCDM", "Model_1", "Model_2"]

# Post-cut data counts (fixed Pantheon+SH0ES selection used in these reruns)
N_CC, N_PN, N_BAO, N_RSD = 15, 1657, 13, 20
N_DATA = {
    "CC+PN": N_CC + N_PN,
    "CC+PN+BAO": N_CC + N_PN + N_BAO,
    "CC+PN+RSD": N_CC + N_PN + N_RSD,
    "CC+PN+BAO+RSD": N_CC + N_PN + N_BAO + N_RSD,
}

LABELS = {
    "H0": r"H_0",
    "OM": r"\Omega_{m,0}",
    "B": r"\alpha_{\rm phys}",
    "x0": r"x_0",
    "lambd": r"\lambda",
    "s80": r"\sigma_{8,0}",
    "f0": r"D'(0)",
    "M": r"M",
    "rdrag": r"r_{\rm drag}",
}


def assert_all_exist() -> list[str]:
    missing = []
    for model, dsets in CHAINS.items():
        for dset, spec in dsets.items():
            for key in ("chain", "logprob"):
                p = spec[key]
                if not p.is_file():
                    missing.append(f"{model} {dset} {key}: {p}")
    return missing
