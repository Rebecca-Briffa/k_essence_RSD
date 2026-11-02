"""
Regenerate all Results_0211 tables and figures from the Nov 2026 reruns.

  python Results_0211/run_all.py
"""

from __future__ import annotations

import run_aic_bic
import run_fs8_residuals
import run_getdist_triangles
import run_whisker
from paths import assert_all_exist


def main() -> None:
    missing = assert_all_exist()
    if missing:
        print("Missing chain/logprob files:")
        for m in missing:
            print(" ", m)
        raise SystemExit(1)

    print("=== AIC / BIC ===")
    run_aic_bic.main()
    print("\n=== GetDist triangles ===")
    run_getdist_triangles.main()
    print("\n=== Whisker summary ===")
    run_whisker.main()
    print("\n=== fσ8 residuals ===")
    run_fs8_residuals.main()
    print("\nAll Results_0211 outputs written.")


if __name__ == "__main__":
    main()
