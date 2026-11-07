"""Prime: a claim-first modelling core for engineering prediction.

The method is fixed and small. See ``prime.core`` for the records and the rules.
"""

from .core import (
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    Claim,
    Evidence,
    ProvenanceMismatch,
    Quantity,
    Regime,
    Result,
    UnitMismatch,
    derive_confidence,
    weakest,
)

__all__ = [
    "Claim",
    "Evidence",
    "ProvenanceMismatch",
    "Quantity",
    "Regime",
    "Result",
    "UnitMismatch",
    "derive_confidence",
    "weakest",
    "GROUNDED",
    "MODEL_BOUND",
    "EXTRAPOLATED",
]
