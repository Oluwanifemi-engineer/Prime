"""Forward-bias injection: the ideal diffusion diode.

This model exists because the previous one stops being true. The depletion model
is valid from reverse bias up to weak forward bias, where the current is a
nuisance. Here the current is the quantity of interest, and the capacitance model
has nothing to say about it.

It is a *replacement*, not an extension: a different set of relations, a different
regime, and a different set of things that can go wrong. Prime has no mechanism for
widening an existing model's range, and that is deliberate -- when a model's
assumptions fail, a new model is added and the old one keeps refusing.

Regime: low-level injection, no space-charge recombination (ideality factor 1),
long-base quasi-neutral regions, constant mobility, no series resistance, no
high-level injection, no breakdown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import (
    ELEMENTARY_CHARGE_C,
    REFERENCE_TEMPERATURE_K,
    SILICON_INTRINSIC_CM3_AT_300K,
    THERMAL_VOLTAGE_PER_K,
)
from ..core import (
    READOUT_FLOOR,
    Floor,
    Quantity,
    Regime,
    Result,
    derive_confidence,
)

# Non-degenerate limit, as in the other models.
DEGENERATE_DOPING_CM3 = 1.0e19
TEMPERATURE_TOLERANCE_K = 1.0


def forward_regime(temperature_k: float = REFERENCE_TEMPERATURE_K) -> Regime:
    """The declared regime for the ideal diffusion diode.

    The lower bound is not arbitrary: below ~0.1 V the generation-recombination
    current dominates the diffusion current and ideality factor 1 is wrong.
    """
    return Regime(
        name="ideal diffusion diode, low-level injection, long base",
        temperature_k=temperature_k,
        bias_min_v=0.1,
        bias_max_v=0.75,
        approximations=(
            "low-level injection: injected minority density below the majority density",
            "no recombination in the space-charge region (ideality factor 1)",
            "long-base diode: quasi-neutral regions longer than the diffusion length",
            "Einstein relation with constant mobility",
            "no series resistance",
            "no high-level injection, no breakdown",
        ),
        ignored=(
            "surface recombination",
            "contact resistance",
            "self-heating",
            "junction capacitance (this model is quasi-static)",
        ),
    )


@dataclass(frozen=True)
class DiffusionDiode:
    """Parameters of a junction for the forward-conduction model.

    Mobilities and lifetimes are the inputs that matter here, and they carry the
    provenance that decides whether the result is grounded. Lifetimes in particular
    are usually estimates, which is honest: they vary by an order of magnitude
    between processes.
    """

    acceptor_density: Quantity  # cm^-3, p-side
    donor_density: Quantity  # cm^-3, n-side
    area: Quantity  # cm^2
    hole_mobility: Quantity  # cm^2/(V s)
    electron_mobility: Quantity  # cm^2/(V s)
    hole_lifetime: Quantity  # s
    electron_lifetime: Quantity  # s

    def check_units(self) -> None:
        self.acceptor_density.require("cm^-3", "acceptor_density")
        self.donor_density.require("cm^-3", "donor_density")
        self.area.require("cm^2", "area")
        self.hole_mobility.require("cm^2/(V*s)", "hole_mobility")
        self.electron_mobility.require("cm^2/(V*s)", "electron_mobility")
        self.hole_lifetime.require("s", "hole_lifetime")
        self.electron_lifetime.require("s", "electron_lifetime")

    @property
    def inputs(self) -> tuple[Quantity, ...]:
        return (
            self.acceptor_density,
            self.donor_density,
            self.area,
            self.hole_mobility,
            self.electron_mobility,
            self.hole_lifetime,
            self.electron_lifetime,
        )


def diffusion_length(diffusivity: float, lifetime: float) -> float:
    """L = sqrt(D tau), in cm."""
    return math.sqrt(diffusivity * lifetime)


def saturation_current(
    diode: DiffusionDiode, regime: Regime
) -> tuple[float, dict[str, Quantity], list[str]]:
    """I_s = q A (D_p p_n0 / L_p + D_n n_p0 / L_n).

    Returns the current, the intermediate quantities worth auditing, and the gates
    that failed on the way. The trace matters here: I_s is exponentially sensitive
    to lifetime, so a result without L_p and L_n recorded is unauditable.
    """
    diode.check_units()
    failed: list[str] = []

    if abs(regime.temperature_k - REFERENCE_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
        failed.append(
            f"n_i tabulated at {REFERENCE_TEMPERATURE_K:.0f} K only; "
            f"regime is at {regime.temperature_k:.0f} K"
        )

    thermal_voltage = THERMAL_VOLTAGE_PER_K * regime.temperature_k
    n_i_sq = SILICON_INTRINSIC_CM3_AT_300K**2

    # Einstein relation: D = mu kT/q.
    d_p = diode.hole_mobility.value * thermal_voltage
    d_n = diode.electron_mobility.value * thermal_voltage
    l_p = diffusion_length(d_p, diode.hole_lifetime.value)
    l_n = diffusion_length(d_n, diode.electron_lifetime.value)

    p_n0 = n_i_sq / diode.donor_density.value  # minority holes in the n-region
    n_p0 = n_i_sq / diode.acceptor_density.value  # minority electrons in the p-region

    i_s = ELEMENTARY_CHARGE_C * diode.area.value * (
        d_p * p_n0 / l_p + d_n * n_p0 / l_n
    )

    trace = {
        "thermal_voltage": Quantity(thermal_voltage, "V"),
        "diffusivity_holes": Quantity(d_p, "cm^2/s"),
        "diffusivity_electrons": Quantity(d_n, "cm^2/s"),
        "diffusion_length_holes": Quantity(l_p, "cm"),
        "diffusion_length_electrons": Quantity(l_n, "cm"),
        "minority_holes_in_n": Quantity(p_n0, "cm^-3"),
        "minority_electrons_in_p": Quantity(n_p0, "cm^-3"),
    }
    return i_s, trace, failed


def forward_current(
    diode: DiffusionDiode,
    regime: Regime,
    bias_v: float,
    device_id: str = "",
    measurement_bandwidth_hz: float | None = None,
) -> Result:
    """I = I_s (exp(V / V_T) - 1) for an ideal diffusion diode.

    ``measurement_bandwidth_hz`` is optional on purpose. A shot-noise limit is a
    limit on *observing* the current, not on the current itself, so it is attached
    only when the caller says how the current will be measured. Without it the
    result carries no readout floor, which is a statement about the caller, not
    about the device.
    """
    diode.check_units()

    passed: list[str] = ["units consistent (cm^-3, cm^2, cm^2/(V*s), s)"]
    failed = regime.check(bias_v)
    if not failed:
        passed.append(
            f"bias within modelled range "
            f"[{regime.bias_min_v:.3g}, {regime.bias_max_v:.3g}] V"
        )

    i_s, trace, i_s_failed = saturation_current(diode, regime)
    failed.extend(i_s_failed)
    if not i_s_failed:
        passed.append(f"n_i used at its tabulated {REFERENCE_TEMPERATURE_K:.0f} K")

    max_doping = max(diode.acceptor_density.value, diode.donor_density.value)
    if max_doping >= DEGENERATE_DOPING_CM3:
        failed.append(
            f"degenerate doping (>= {DEGENERATE_DOPING_CM3:.0e} cm^-3): "
            "Boltzmann statistics and the minority-carrier relations fail"
        )
    else:
        passed.append(f"non-degenerate doping (< {DEGENERATE_DOPING_CM3:.0e} cm^-3)")

    thermal_voltage = THERMAL_VOLTAGE_PER_K * regime.temperature_k

    # The boundary that actually bites: as the forward bias rises, the injected
    # minority density at the junction edge approaches the majority density and the
    # whole low-level-injection derivation collapses.
    trace["excess_injection_factor"] = Quantity(
        math.expm1(bias_v / thermal_voltage), "1"
    )
    p_injected = boundary_injected_density(
        diode.donor_density.value, diode.acceptor_density.value, bias_v, thermal_voltage, "n"
    )
    n_injected = boundary_injected_density(
        diode.donor_density.value, diode.acceptor_density.value, bias_v, thermal_voltage, "p"
    )
    if p_injected >= diode.donor_density.value:
        failed.append(
            f"high-level injection: injected holes {p_injected:.3g} cm^-3 reach the "
            f"n-side doping {diode.donor_density.value:.3g} cm^-3 at {bias_v:.3g} V"
        )
    if n_injected >= diode.acceptor_density.value:
        failed.append(
            f"high-level injection: injected electrons {n_injected:.3g} cm^-3 reach the "
            f"p-side doping {diode.acceptor_density.value:.3g} cm^-3 at {bias_v:.3g} V"
        )
    if p_injected < diode.donor_density.value and n_injected < diode.acceptor_density.value:
        passed.append("low-level injection holds on both sides")

    current = i_s * math.expm1(bias_v / thermal_voltage) if bias_v >= 0.0 else -i_s

    floors: tuple[Floor, ...] = ()
    if measurement_bandwidth_hz is not None and current > 0.0:
        floors = (shot_noise_floor(current, measurement_bandwidth_hz),)

    return Result(
        quantity="forward_current",
        value=current,
        unit="A",
        basis="analytic: I = I_s (exp(V/V_T) - 1), ideal diffusion diode",
        gates_passed=tuple(passed),
        gates_failed=tuple(failed),
        confidence=derive_confidence(failed, diode.inputs),
        assumptions=regime.approximations,
        device_id=device_id,
        floors=floors,
        trace={"saturation_current": Quantity(i_s, "A"), **trace},
    )


def boundary_injected_density(
    donor_density: float, acceptor_density: float, bias_v: float, thermal_voltage: float, side: str
) -> float:
    """Minority density injected at the junction edge on one side, in cm^-3.

    This is p_n(0) = p_n0 exp(V/V_T) for the n-side and its mirror image for the
    p-side. It is what the low-level-injection gate compares against the majority
    doping.
    """
    n_i_sq = SILICON_INTRINSIC_CM3_AT_300K**2
    if side == "n":
        return n_i_sq / donor_density * math.exp(bias_v / thermal_voltage)
    return n_i_sq / acceptor_density * math.exp(bias_v / thermal_voltage)


def shot_noise_floor(current_a: float, bandwidth_hz: float) -> Floor:
    """The readout limit imposed by the discreteness of the charge being measured.

    sigma_I = sqrt(2 q I B). This is a limit on the measurement, not on the device,
    so it is a READOUT floor and never a substitute for a parameter floor.
    """
    if bandwidth_hz <= 0:
        raise ValueError("bandwidth must be positive")
    sigma = math.sqrt(2.0 * ELEMENTARY_CHARGE_C * current_a * bandwidth_hz)
    return Floor(
        mechanism="shot noise on the measured current",
        kind=READOUT_FLOOR,
        relative=sigma / current_a,
        basis=f"sqrt(2 q I B) over I, with B = {bandwidth_hz:.3g} Hz",
    )
