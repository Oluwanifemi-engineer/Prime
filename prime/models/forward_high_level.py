"""High-level injection: the diode above the crossover.

At low bias the injected minority density is far below the majority doping and the
diode conducts as exp(V/V_T). Once the injected density reaches the majority doping,
conductivity modulation sets in, both carrier types participate in an ambipolar
diffusion, and the exponential halves: exp(V / 2V_T), ideality factor 2.

This model fills the gap the previous two admitted. At 0.75 V the capacitance model
was out of range and the low-level conduction model had failed its own gate, so
Prime reported no model covering the bias. Now one does -- and it starts exactly
where the previous model stops, by construction rather than by a chosen constant.

That construction is the interesting part: the high-level saturation current is
*derived* from the low-level one at the crossover voltage, so the two models agree
to machine precision at the handover and can never drift apart. Neither model holds
a private opinion about where the boundary is.
"""

from __future__ import annotations

import math

from ..constants import (
    REFERENCE_TEMPERATURE_K,
    SILICON_INTRINSIC_CM3_AT_300K,
    THERMAL_VOLTAGE_PER_K,
)
from ..core import (
    ProvenanceMismatch,
    Quantity,
    Regime,
    Result,
    UnitMismatch,
    derive_confidence,
)
from .forward_injection import (
    DEGENERATE_DOPING_CM3,
    DiffusionDiode,
    crossover_voltage,
    saturation_current,
)

IDEALITY_FACTOR_HIGH_LEVEL = 2.0

# When the voltage dropped across the neglected series resistance reaches this
# fraction of V_T, ignoring it changes the current by more than ~e^0.5 and the
# model is no longer answering the question that was asked.
NEGLECTED_DROP_LIMIT_IN_VT = 0.5


def high_level_regime(temperature_k: float = REFERENCE_TEMPERATURE_K) -> Regime:
    """The declared regime above the high-level crossover.

    The bias bounds here are mechanical, not physical: the gates that carry meaning
    are the crossover (computed per-device) and the optional series-drop check. A
    fixed voltage ceiling would be device-independent, which is exactly wrong -- how
    far this model reaches depends on the resistance of the region in question.
    """
    return Regime(
        name="high-level injection, ambipolar diffusion, ideality factor 2",
        temperature_k=temperature_k,
        bias_min_v=0.6,
        bias_max_v=0.9,
        approximations=(
            "high-level injection: injected density at or above the majority doping",
            "ambipolar diffusion with ideality factor 2",
            "no recombination in the space-charge region",
            "long-base diode: quasi-neutral regions longer than the diffusion length",
            "no series resistance, no self-heating",
        ),
        ignored=(
            "series resistance (dominant above ~0.78 V at this geometry)",
            "Kirk effect and base push-out",
            "surface recombination",
        ),
    )


def resistance_in_ohms(series_resistance: Quantity | Result, device_id: str) -> float:
    """Accept a resistance as a bare quantity or as a result from another model.

    Passing a Result is the honest option -- it keeps where the number came from --
    so it is supported, but only for the same device. A resistance measured on a
    different die is not a bound on this one.
    """
    if isinstance(series_resistance, Result):
        if (
            series_resistance.device_id
            and device_id
            and series_resistance.device_id != device_id
        ):
            raise ProvenanceMismatch(
                f"resistance is for device {series_resistance.device_id!r}, "
                f"current is for {device_id!r}"
            )
        if series_resistance.unit != "ohm":
            raise UnitMismatch(
                f"series_resistance must be in 'ohm', got {series_resistance.unit!r}"
            )
        return series_resistance.value
    return series_resistance.require("ohm", "series_resistance")


def high_level_saturation_current(
    diode: DiffusionDiode, regime: Regime
) -> tuple[float, float, dict[str, Quantity], list[str]]:
    """I_2 = I_s exp(V_x / 2V_T), with V_x the crossover voltage.

    Returns (I_2, V_x, trace, failed gates). The form is chosen so that at V_x this
    model returns exactly I_s exp(V_x/V_T), the low-level model's answer: continuity
    is a consequence of the construction, not a fitted parameter.
    """
    i_s, trace, failed = saturation_current(diode, regime)
    thermal_voltage = THERMAL_VOLTAGE_PER_K * regime.temperature_k
    v_x = crossover_voltage(diode, thermal_voltage)
    i_2 = i_s * math.exp(v_x / (IDEALITY_FACTOR_HIGH_LEVEL * thermal_voltage))
    return i_2, v_x, trace, failed


def high_level_current(
    diode: DiffusionDiode,
    regime: Regime,
    bias_v: float,
    device_id: str = "",
    ideality_factor: float = IDEALITY_FACTOR_HIGH_LEVEL,
    series_resistance: Quantity | Result | None = None,
) -> Result:
    """I = I_2 exp(V / (n V_T)) with n = 2 in high-level injection.

    The gate that matters is the inverse of the previous model's: below the
    crossover this relation does not apply, and a caller who asks anyway is told so
    rather than handed a number.

    ``series_resistance`` is optional, and it is the difference between an assumption
    and a check. Supply the region's resistance and the neglected drop is computed
    and gated against a fraction of V_T; omit it and the assumption stands as stated,
    unverified -- which the result says plainly, as with every other assumption.
    """
    diode.check_units()

    passed: list[str] = ["units consistent (cm^-3, cm^2, cm^2/(V*s), s)"]
    failed = regime.check(bias_v)
    if not failed:
        passed.append(
            f"bias within modelled range "
            f"[{regime.bias_min_v:.3g}, {regime.bias_max_v:.3g}] V"
        )

    i_2, v_x, trace, i_2_failed = high_level_saturation_current(diode, regime)
    failed.extend(i_2_failed)
    if not i_2_failed:
        passed.append(f"n_i used at its tabulated {REFERENCE_TEMPERATURE_K:.0f} K")

    max_doping = max(diode.acceptor_density.value, diode.donor_density.value)
    if max_doping >= DEGENERATE_DOPING_CM3:
        failed.append(
            f"degenerate doping (>= {DEGENERATE_DOPING_CM3:.0e} cm^-3): "
            "Boltzmann statistics and the minority-carrier relations fail"
        )
    else:
        passed.append(f"non-degenerate doping (< {DEGENERATE_DOPING_CM3:.0e} cm^-3)")

    if bias_v < v_x:
        failed.append(
            f"biases below the high-level crossover {v_x:.4g} V are low-level "
            "injection: ideality factor 2 does not apply here"
        )
    else:
        passed.append(f"at or above the high-level crossover ({v_x:.4g} V)")

    thermal_voltage = THERMAL_VOLTAGE_PER_K * regime.temperature_k
    current = i_2 * math.exp(bias_v / (ideality_factor * thermal_voltage))

    trace = {
        "high_level_saturation_current": Quantity(i_2, "A"),
        "crossover_voltage": Quantity(v_x, "V"),
        "ideality_factor": Quantity(ideality_factor, "1"),
        **trace,
    }

    if series_resistance is not None:
        drop = current * resistance_in_ohms(series_resistance, device_id)
        drop_in_vt = drop / thermal_voltage
        trace["series_voltage_drop"] = Quantity(drop, "V")
        trace["series_drop_in_vt"] = Quantity(drop_in_vt, "1")
        if drop_in_vt > NEGLECTED_DROP_LIMIT_IN_VT:
            failed.append(
                f"neglected series resistance drops {drop * 1e3:.3g} mV "
                f"({drop_in_vt:.2g} V_T) at {current * 1e3:.3g} mA: above the "
                f"{NEGLECTED_DROP_LIMIT_IN_VT:g} V_T limit for ignoring it"
            )
        else:
            passed.append(
                f"neglected series drop is {drop_in_vt:.2g} V_T "
                f"(< {NEGLECTED_DROP_LIMIT_IN_VT:g} V_T)"
            )

    return Result(
        quantity="forward_current",
        value=current,
        unit="A",
        basis="analytic: I = I_2 exp(V/(nV_T)), n = 2, matched at the crossover",
        gates_passed=tuple(passed),
        gates_failed=tuple(failed),
        confidence=derive_confidence(failed, diode.inputs),
        assumptions=regime.approximations,
        device_id=device_id,
        trace=trace,
    )
