"""Abrupt 1-D PN junction, full-depletion approximation.

Regime: uniform doping on both sides, abrupt metallurgical junction, complete
depletion of the space-charge region, no series resistance, no recombination,
low injection. Reverse bias and weak forward bias only.

The point of this model is not the junction: it is the shape. A small set of
relations, each with a gate, producing a result whose confidence is derived from
the evidence and from whether the gates held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import (
    ELEMENTARY_CHARGE_C,
    REFERENCE_TEMPERATURE_K,
    SILICON_INTRINSIC_CM3_AT_300K,
    SILICON_RELATIVE_PERMITTIVITY,
    THERMAL_VOLTAGE_PER_K,
    VACUUM_PERMITTIVITY_F_PER_CM,
)
from ..core import (
    CONFIDENCE_ORDER,
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    Evidence,
    Quantity,
    Regime,
    Result,
    derive_confidence,
)

# Doping above this is degenerate: the Boltzmann statistics behind V_bi break down.
DEGENERATE_DOPING_CM3 = 1.0e19
# The tabulated intrinsic concentration is only meaningful at its reference temperature.
TEMPERATURE_TOLERANCE_K = 1.0


def abrupt_pn_regime(temperature_k: float = REFERENCE_TEMPERATURE_K) -> Regime:
    """The declared regime for the where-abrupt-junction model."""
    return Regime(
        name="abrupt-junction, full-depletion, low-injection",
        temperature_k=temperature_k,
        bias_min_v=-100.0,
        bias_max_v=0.3,
        approximations=(
            "uniform doping on each side",
            "abrupt metallurgical junction",
            "complete depletion of the space-charge region",
            "Boltzmann statistics (non-degenerate doping)",
            "low injection: drift-diffusion current neglected",
            "no series resistance",
        ),
        ignored=(
            "recombination and generation current",
            "package and lead parasitics",
            "self-heating",
            "junction curvature and edge effects",
        ),
    )


@dataclass(frozen=True)
class AbruptJunction:
    """Parameters of one junction. Every field carries its provenance."""

    acceptor_density: Quantity  # cm^-3, p-side
    donor_density: Quantity  # cm^-3, n-side
    area: Quantity  # cm^2
    relative_permittivity: Quantity = Quantity(
        SILICON_RELATIVE_PERMITTIVITY,
        "1",
        Evidence("estimate", 0.03, "textbook value for Si; varies with doping/frequency"),
    )

    def check_units(self) -> None:
        self.acceptor_density.require("cm^-3", "acceptor_density")
        self.donor_density.require("cm^-3", "donor_density")
        self.area.require("cm^2", "area")
        self.relative_permittivity.require("1", "relative_permittivity")

    @property
    def inputs(self) -> tuple[Quantity, ...]:
        return (
            self.acceptor_density,
            self.donor_density,
            self.area,
            self.relative_permittivity,
        )

    @property
    def permittivity(self) -> float:
        """Absolute permittivity in F/cm."""
        return VACUUM_PERMITTIVITY_F_PER_CM * self.relative_permittivity.value


def built_in_potential(junction: AbruptJunction, regime: Regime) -> tuple[float, list[str]]:
    """V_bi = (kT/q) ln(N_a N_d / n_i^2). Returns (volts, failed gates)."""
    failed: list[str] = []
    if abs(regime.temperature_k - REFERENCE_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
        failed.append(
            f"n_i tabulated at {REFERENCE_TEMPERATURE_K:.0f} K only; "
            f"regime is at {regime.temperature_k:.0f} K"
        )
    thermal_voltage = THERMAL_VOLTAGE_PER_K * regime.temperature_k
    ratio = (
        junction.acceptor_density.value
        * junction.donor_density.value
        / SILICON_INTRINSIC_CM3_AT_300K**2
    )
    return thermal_voltage * math.log(ratio), failed


def junction_capacitance(
    junction: AbruptJunction, regime: Regime, bias_v: float
) -> Result:
    """Depletion capacitance at a given bias.

    C_j = eps * A / W, with W = sqrt(2 eps (V_bi - V) / q * (1/N_a + 1/N_d)).
    """
    junction.check_units()

    # Only checks that were actually performed are reported as gates. The
    # regime's approximations are assumptions, not verified facts, so they are
    # not listed as passing gates.
    passed: list[str] = ["units consistent (cm^-3, cm^2, 1)"]
    failed = regime.check(bias_v)
    if not failed:
        passed.append(
            f"bias within modelled range "
            f"[{regime.bias_min_v:.3g}, {regime.bias_max_v:.3g}] V"
        )

    v_bi, v_bi_failed = built_in_potential(junction, regime)
    failed.extend(v_bi_failed)
    if not v_bi_failed:
        passed.append(f"n_i used at its tabulated {REFERENCE_TEMPERATURE_K:.0f} K")

    max_doping = max(junction.acceptor_density.value, junction.donor_density.value)
    if max_doping >= DEGENERATE_DOPING_CM3:
        failed.append(
            f"degenerate doping (>= {DEGENERATE_DOPING_CM3:.0e} cm^-3): "
            "Boltzmann statistics and the depletion approximation fail"
        )
    else:
        passed.append(f"non-degenerate doping (< {DEGENERATE_DOPING_CM3:.0e} cm^-3)")

    if bias_v >= v_bi:
        failed.append(
            f"bias {bias_v:.3g} V at or beyond built-in potential "
            f"{v_bi:.4g} V: space-charge region collapses, model undefined"
        )
        width = 0.0
        capacitance = math.inf
    else:
        passed.append(f"bias below built-in potential ({v_bi:.4g} V)")
        width = math.sqrt(
            2.0
            * junction.permittivity
            * (v_bi - bias_v)
            / ELEMENTARY_CHARGE_C
            * (1.0 / junction.acceptor_density.value + 1.0 / junction.donor_density.value)
        )
        capacitance = junction.permittivity * junction.area.value / width

    return Result(
        quantity="junction_capacitance",
        value=capacitance,
        unit="F",
        basis="analytic: C = eps*A/W, full-depletion, abrupt junction",
        gates_passed=tuple(passed),
        gates_failed=tuple(failed),
        confidence=derive_confidence(failed, junction.inputs),
        trace={
            "built_in_potential": Quantity(v_bi, "V"),
            "depletion_width": Quantity(width, "cm"),
            "permittivity": Quantity(junction.permittivity, "F/cm"),
        },
    )


def single_pole_cutoff(resistance: Quantity, capacitance: Result) -> Result:
    """f_-3dB = 1 / (2 pi R C) for an ideal single-pole load.

    Included as the downstream link: the same capacitance that was predicted is
    the one used here, and the result records where it came from. If the
    capacitance was extrapolated, this is extrapolated too.
    """
    resistance.require("ohm", "resistance")
    if capacitance.unit != "F":
        raise ValueError(f"capacitance must be in 'F', got {capacitance.unit!r}")

    frequency = 1.0 / (2.0 * math.pi * resistance.value * capacitance.value)

    confidence = capacitance.confidence
    if resistance.is_soft and CONFIDENCE_ORDER.index(confidence) < CONFIDENCE_ORDER.index(
        MODEL_BOUND
    ):
        confidence = MODEL_BOUND
    if capacitance.gates_failed and confidence != EXTRAPOLATED:
        confidence = EXTRAPOLATED

    return Result(
        quantity="cutoff_frequency",
        value=frequency,
        unit="Hz",
        basis="analytic: f = 1/(2 pi R C), single pole, ideal R, no package parasitics",
        gates_passed=("ideal single-pole response",),
        gates_failed=capacitance.gates_failed,
        confidence=confidence,
        trace={
            "capacitance_used": Quantity(capacitance.value, "F"),
            "resistance": resistance,
        },
        derived_from=(capacitance.quantity,),
    )
