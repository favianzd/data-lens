import numpy as np
import pandas as pd
import pytest

TRUE_INTERCEPT = 2.0
TRUE_COEFS = {"x1": 3.0, "x2": -1.5}


def make_linear(n: int, noise_sd: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """y = 2 + 3·x1 − 1.5·x2 + ε, with ε ~ N(0, noise_sd²)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-2, 2, n)
    y = TRUE_INTERCEPT + TRUE_COEFS["x1"] * x1 + TRUE_COEFS["x2"] * x2 + rng.normal(0, noise_sd, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def known_df() -> pd.DataFrame:
    return make_linear(10_000)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return make_linear(40, seed=1)
