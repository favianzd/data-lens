"""Number formatting shared by the UI and the report."""

from __future__ import annotations

import math


def fmt_num(x: float, digits: int = 4) -> str:
    """Significant-digit formatting that stays readable for tiny and huge values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}g}"


def fmt_pvalue(p: float) -> str:
    """APA-style p-value: exact to 3 decimals, floored at < 0.001."""
    if p is None or math.isnan(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
