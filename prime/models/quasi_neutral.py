"""Ohmic conduction in a quasi-neutral semiconductor region.

This is the second model, and the reason it exists is architectural: it is valid
under assumptions that are *different* from the depletion model's. In the
space-charge region the carrier densities are not the equilibrium ones; in the
quasi-neutral region charge neutrality holds and conduction is ohmic. The two
cannot be merged into one relation, only composed.

Regime: uniform doping, constant mobility, low-level injection, no conductivity
modulation, no current crowding, no velocity saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import ELEMENTARY_CHARGE_C
from ..core import Quantity, Regime, Result, derive_confidence, dopant_floor

# Field at which drift velocity stops being proportional to field. Ohmic
# conduction requires staying well below this.
CRITICAL_FIELD_V_PER_CM = 1.0e4
# Non-degenerate limit, as in the junction model.
DEGENERATE_DOPING_CM3 = 1.0e19


def ohmic_regime(temperature_k: float = 300.0) -> Regime:
    """The declared regime for conduction in a quasi-neutral region."""
    return Regime(
        name="quasi-neutral ohmic conduction",
        temperature_k=temperature_k,
        bias_min_v=-100.0,
        bias_max_v=0.3,
        approximations=(
            "uniform doping along the region",
            "constant mobility (no field dependence)",
            "low-level injection: no conductivity modulation",
            "ohmic field below the velocity-saturation threshold",
            "no current crowding or lateral spreading",
            "quasi-neutral: charge neutrality holds",
        ),
        ignored=("contact resistance", "surface depletion", "self-heating"),
    )


@dataclass(frozen=True)
class QuasiNeutralRegion:
    """A doped region between the junction edge and the contact."""

    doping: Quantity  # cm^-3
    length: Quantity  # cm
    area: Quantity  # cm^2, cross-section perpendicular to current flow
    mobility: Quantity  # cm^2/(V s)

    def check_units(self) -> None:
        self.doping.require("cm^-3", "doping")
        self.length.require("cm", "length")
        self.area.require("cm^2", "area")
        self.mobility.require("cm^2/(V*s)", "mobility")

    @property
    def inputs(self) -> tuple[Quantity, ...]:
        return (self.doping, self.length, self.area, self.mobility)


def resistance(
    region: QuasiNeutralRegion,
    regime: Regime,
    device_id: str = "",
    mobility_reference_temperature_k: float = 300.0,
) -> Result:
    """R = L / (q mu N A) for a uniform doped bar.

    Resistance has no bias dependence, so the bias window is not gated here. The
    two things that do bound this relation are the temperature the mobility was
    quoted at, and the field (see ``check_ohmic_field``, which needs an operating
    current the caller may not have).
    """
    region.check_units()

    failed: list[str] = []
    passed = ["units consistent (cm^-3, cm, cm^2, cm^2/(V*s))"]

    if abs(regime.temperature_k - mobility_reference_temperature_k) > 1.0:
        failed.append(
            f"mobility quoted at {mobility_reference_temperature_k:.0f} K; "
            f"regime is at {regime.temperature_k:.0f} K (mobility is strongly "
            "temperature dependent)"
        )
    else:
        passed.append(
            f"mobility used at its quoted {mobility_reference_temperature_k:.0f} K"
        )

    if region.doping.value >= DEGENERATE_DOPING_CM3:
        failed.append(
            f"degenerate doping (>= {DEGENERATE_DOPING_CM3:.0e} cm^-3): "
            "constant-mobility ohmic conduction does not hold"
        )
    else:
        passed.append(f"non-degenerate doping (< {DEGENERATE_DOPING_CM3:.0e} cm^-3)")

    if region.length.value <= 0:
        raise ValueError("length must be positive")
    if region.area.value <= 0:
        raise ValueError("area must be positive")

    conductivity = (
        ELEMENTARY_CHARGE_C * region.mobility.value * region.doping.value
    )  # S/cm
    value = region.length.value / (conductivity * region.area.value)

    # The ohmic assumption, made checkable: E = J/(q mu N) is stated as an
    # assumption below because the operating current is not part of this call.
    passed.append("region length and area positive")

    return Result(
        quantity="series_resistance",
        value=value,
        unit="ohm",
        basis="analytic: R = L/(q mu N A), uniform quasi-neutral region",
        gates_passed=tuple(passed),
        gates_failed=tuple(failed),
        confidence=derive_confidence(failed, region.inputs),
        assumptions=regime.approximations,
        device_id=device_id,
        # Conductance averages over every dopant in the conducting volume, so a
        # long, fat region is far more predictable than a short, thin one.
        floors=(
            dopant_floor(
                "random dopant fluctuation in the conducting volume",
                region.doping.value * region.length.value * region.area.value,
                "1/sqrt(N) over the dopants in the region",
            ),
        ),
        trace={
            "conductivity": Quantity(conductivity, "S/cm"),
            "length": region.length,
            "area": region.area,
        },
    )


def check_ohmic_field(
    result: Result, region: QuasiNeutralRegion, current: Quantity
) -> list[str]:
    """Gate the ohmic assumption against a stated operating current.

    Separate from ``resistance`` on purpose: resistance does not depend on current,
    but its validity does. A caller who knows the operating point can test the
    assumption instead of inheriting it.
    """
    current.require("A", "current")
    if region.length.value <= 0:
        raise ValueError("length must be positive")
    current_density = current.value / region.area.value  # A/cm^2
    conductivity = ELEMENTARY_CHARGE_C * region.mobility.value * region.doping.value
    field = current_density / conductivity  # V/cm
    if field >= CRITICAL_FIELD_V_PER_CM:
        return [
            f"field {field:.3g} V/cm at or above the {CRITICAL_FIELD_V_PER_CM:.0e} V/cm "
            "velocity-saturation threshold: ohmic relation invalid"
        ]
    return []
