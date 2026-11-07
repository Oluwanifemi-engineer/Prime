"""Prime: a claim-first modelling core for engineering prediction.

The method is fixed and small. See ``prime.core`` for the records and the rules.
"""

from .core import (
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    Claim,
    Evidence,
    Quantity,
    Regime,
    Result,
    UnitMismatch,
    derive_confidence,
)

__all__ = [
    "Claim",
    "Evidence",
    "Quantity",
    "Regime",
    "Result",
    "UnitMismatch",
    "derive_confidence",
    "GROUNDED",
    "MODEL_BOUND",
    "EXTRAPOLATED",
]
