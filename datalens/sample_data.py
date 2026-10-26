"""Synthetic demo dataset with documented population coefficients.

The values are generated, not collected; the UI labels them as synthetic.
Because the true coefficients are known, users can check that the estimates
land near them (within sampling error).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "nilai_ujian"
TRUE_INTERCEPT = 20.0
TRUE_COEFFICIENTS = {
    "jam_belajar": 2.5,     # points per weekly study hour
    "jam_tidur": 1.2,       # points per nightly sleep hour
    "kehadiran_pct": 0.3,   # points per attendance percentage point
}
NOISE_SD = 5.0


def make_study_dataset(n: int = 200, seed: int = 42, missing_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kehadiran = np.clip(rng.normal(80, 10, n), 40, 100)
    # Study hours are mildly correlated with attendance, as one would expect.
    jam_belajar = np.clip(2 + 0.08 * kehadiran + rng.normal(0, 2, n), 0, None)
    jam_tidur = np.clip(rng.normal(7, 1, n), 4, 10)
    noise = rng.normal(0, NOISE_SD, n)

    nilai = (
        TRUE_INTERCEPT
        + TRUE_COEFFICIENTS["jam_belajar"] * jam_belajar
        + TRUE_COEFFICIENTS["jam_tidur"] * jam_tidur
        + TRUE_COEFFICIENTS["kehadiran_pct"] * kehadiran
        + noise
    )
    df = pd.DataFrame(
        {
            "program_studi": rng.choice(["Statistika", "Informatika", "Ekonomi"], n),
            "jam_belajar": jam_belajar.round(1),
            "jam_tidur": jam_tidur.round(1),
            "kehadiran_pct": kehadiran.round(1),
            TARGET: nilai.round(1),
        }
    )
    # Sprinkle a few missing values so the missing-data handling is visible.
    for col in ("jam_belajar", "jam_tidur"):
        mask = rng.random(n) < missing_rate
        df.loc[mask, col] = np.nan
    return df
