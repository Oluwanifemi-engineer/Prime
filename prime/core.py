"""The core records of Prime.

The method is fixed and small:

    claim -> regime -> parameters (with evidence) -> model -> result -> confidence

Every model in the system speaks these records and nothing else. The rules:

1. A result cannot exist without a declared regime.
2. Confidence is derived from the evidence and the gates, never asserted by hand.
3. Every intermediate worth auditing is kept in the result's trace.
4. A downstream number records which result it was derived from.

If a new capability cannot fill in these records, it does not belong yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Confidence -----------------------------------------------------------------
# Ordered from strongest to weakest. A prediction is never better than its weakest
# link, so downgrades only ever move down this list.

GROUNDED = "grounded"
"""The method is valid in the declared regime and every input is measured or specified."""

MODEL_BOUND = "model_bound"
"""The method is valid in the declared regime, but at least one input is an estimate
or a fitted value. The physics is fine; the numbers are soft."""

EXTRAPOLATED = "extrapolated"
"""The declared regime has been violated. The number may still be useful as an
indication, but it is not a prediction that can be relied on."""

CONFIDENCE_ORDER = (GROUNDED, MODEL_BOUND, EXTRAPOLATED)

# Evidence kinds that make an input soft. A datasheet value, a process spec or a
# measurement pins a number down; an estimate or a fitted value does not.
SOFT_EVIDENCE_KINDS = frozenset({"estimate", "fitted"})
KNOWN_EVIDENCE_KINDS = SOFT_EVIDENCE_KINDS | {"datasheet", "process_spec", "measurement"}


class UnitMismatch(ValueError):
    """Raised when a quantity is used in a relation that expects another unit."""


# --- Evidence and quantities -----------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """What a number rests on. Required for every input parameter."""

    kind: str
    relative_uncertainty: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KNOWN_EVIDENCE_KINDS:
            raise ValueError(
                f"unknown evidence kind {self.kind!r}; expected one of "
                f"{sorted(KNOWN_EVIDENCE_KINDS)}"
            )
        if self.relative_uncertainty < 0:
            raise ValueError("relative_uncertainty must be non-negative")

    @property
    def is_soft(self) -> bool:
        return self.kind in SOFT_EVIDENCE_KINDS


@dataclass(frozen=True)
class Quantity:
    """A number, its unit, and what it rests on.

    Units are strings on purpose. A full units system is a large commitment; a
    string plus an explicit ``require`` check at each model boundary catches the
    errors that actually happen (cm vs m, cm^-3 vs m^-3) without the machinery.
    """

    value: float
    unit: str
    evidence: Evidence | None = None

    def require(self, unit: str, role: str = "quantity") -> float:
        """Assert the unit of this quantity and return its value."""
        if self.unit != unit:
            raise UnitMismatch(f"{role} must be in {unit!r}, got {self.unit!r}")
        return self.value

    @property
    def is_soft(self) -> bool:
        return self.evidence is not None and self.evidence.is_soft


# --- Regime ----------------------------------------------------------------------


@dataclass(frozen=True)
class Regime:
    """The declared domain of validity.

    A regime is not documentation: it is executable. ``check`` returns the gates
    that fail, and a failed gate downgrades the result to EXTRAPOLATED.
    """

    name: str
    temperature_k: float
    bias_min_v: float
    bias_max_v: float
    approximations: tuple[str, ...]
    ignored: tuple[str, ...] = ()

    def check(self, bias_v: float) -> list[str]:
        """Return the list of violated gates for this operating point."""
        failed: list[str] = []
        if not (self.bias_min_v <= bias_v <= self.bias_max_v):
            failed.append(
                f"bias {bias_v:.3g} V outside modelled range "
                f"[{self.bias_min_v:.3g}, {self.bias_max_v:.3g}] V"
            )
        return failed


# --- Claim -----------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """What we expect, why, under which conditions, and what would refute it."""

    claim_id: str
    statement: str
    question: str
    conditions: dict[str, str]
    falsified_if: str


# --- Result ----------------------------------------------------------------------


@dataclass(frozen=True)
class Result:
    """A computed quantity, its basis, and how far it can be trusted."""

    quantity: str
    value: float
    unit: str
    basis: str
    gates_passed: tuple[str, ...]
    gates_failed: tuple[str, ...]
    confidence: str
    trace: dict[str, Quantity] = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()

    @property
    def reliable(self) -> bool:
        return self.confidence != EXTRAPOLATED

    def describe(self) -> str:
        head = f"{self.quantity} = {self.value:.4g} {self.unit} [{self.confidence}]"
        lines = [head, f"  basis: {self.basis}"]
        for gate in self.gates_passed:
            lines.append(f"  ok    : {gate}")
        for gate in self.gates_failed:
            lines.append(f"  FAILED: {gate}")
        for name, quantity in self.trace.items():
            lines.append(f"  trace : {name} = {quantity.value:.6g} {quantity.unit}")
        if self.derived_from:
            lines.append(f"  from  : {', '.join(self.derived_from)}")
        return "\n".join(lines)


def derive_confidence(
    gates_failed: list[str] | tuple[str, ...],
    inputs: tuple[Quantity, ...] | list[Quantity],
) -> str:
    """Derive confidence from the gates and the evidence behind the inputs.

    This is the only place confidence is decided. A violated gate always wins over
    soft evidence: an out-of-regime result is extrapolated regardless of how good
    the numbers were.
    """
    if gates_failed:
        return EXTRAPOLATED
    if any(q.is_soft for q in inputs):
        return MODEL_BOUND
    return GROUNDED
