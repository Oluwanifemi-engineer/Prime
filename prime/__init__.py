"""Prime: a claim-first modelling core for engineering prediction.

The method is fixed and small. See ``prime.core`` for the records and the rules.
"""

from .core import (
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    PARAMETER_FLOOR,
    READOUT_FLOOR,
    Claim,
    Evidence,
    Floor,
    ProvenanceMismatch,
    Quantity,
    Regime,
    Result,
    UnitMismatch,
    derive_confidence,
    dopant_floor,
    quadrature,
    weakest,
)

__all__ = [
    "Claim",
    "Evidence",
    "Floor",
    "PARAMETER_FLOOR",
    "ProvenanceMismatch",
    "Quantity",
    "READOUT_FLOOR",
    "Regime",
    "Result",
    "UnitMismatch",
    "derive_confidence",
    "dopant_floor",
    "quadrature",
    "weakest",
    "GROUNDED",
    "MODEL_BOUND",
    "EXTRAPOLATED",
]
