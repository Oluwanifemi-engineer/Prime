"""Tests for the high-level injection model.

The interesting properties are not the current itself but the handover: the two
forward models must agree exactly at the crossover, and neither may quietly claim
the other's side of it.
"""

import math

import pytest

from prime import EXTRAPOLATED, GROUNDED, MODEL_BOUND, Evidence, Quantity
from prime.constants import THERMAL_VOLTAGE_PER_K
from prime.models import (
    DiffusionDiode,
    crossover_voltage,
    forward_current,
    forward_regime,
    high_level_current,
    high_level_regime,
    high_level_saturation_current,
    saturation_current,
)

FIRM = Evidence("process_spec", 0.05)
MEAS = Evidence("measurement", 0.02)
SOFT = Evidence("estimate", 0.50, "process lifetime, +/- a decade")
V_T = THERMAL_VOLTAGE_PER_K * 300.0


def diode(lifetime_evidence: Evidence = FIRM) -> DiffusionDiode:
    return DiffusionDiode(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", MEAS),
        hole_mobility=Quantity(450.0, "cm^2/(V*s)", FIRM),
        electron_mobility=Quantity(1350.0, "cm^2/(V*s)", FIRM),
        hole_lifetime=Quantity(1.0e-6, "s", lifetime_evidence),
        electron_lifetime=Quantity(1.0e-6, "s", lifetime_evidence),
    )


# --- the crossover ------------------------------------------------------------------


def test_crossover_is_set_by_the_lightly_doped_side():
    """V_x = 2 V_T ln(N_d/n_i): 0.714 V for the 1e16 n-side."""
    v_x = crossover_voltage(diode(), V_T)
    assert v_x == pytest.approx(0.7143, rel=1e-3)
    assert v_x == pytest.approx(2 * V_T * math.log(1.0e16 / 1.0e10), rel=1e-12)


def test_both_models_agree_exactly_at_the_crossover():
    """Continuity is built in, not fitted: I_2 is derived from I_s at V_x."""
    v_x = crossover_voltage(diode(), V_T)
    low = forward_current(diode(), forward_regime(), v_x - 1e-9)
    high = high_level_current(diode(), high_level_regime(), v_x)
    assert high.value == pytest.approx(low.value, rel=1e-6)


def test_the_handover_has_no_double_claim():
    """At exactly V_x the low-level model refuses and the high-level one accepts."""
    v_x = crossover_voltage(diode(), V_T)
    assert forward_current(diode(), forward_regime(), v_x).confidence == EXTRAPOLATED
    assert high_level_current(diode(), high_level_regime(), v_x).confidence != EXTRAPOLATED


def test_high_level_saturation_current_is_derived_from_the_low_level_one():
    i_s, _, _ = saturation_current(diode(), forward_regime())
    i_2, v_x, _, _ = high_level_saturation_current(diode(), high_level_regime())
    assert i_2 == pytest.approx(i_s * math.exp(v_x / (2 * V_T)), rel=1e-12)
    assert i_2 == pytest.approx(6.41e-10, rel=0.01)


# --- the physics --------------------------------------------------------------------


def test_current_above_the_crossover():
    result = high_level_current(diode(), high_level_regime(), 0.75, device_id="D1")
    assert result.value == pytest.approx(1.278e-3, rel=0.01)
    assert result.confidence == GROUNDED


def test_the_exponential_halves():
    """Ideality factor 2: 20 mV doubles nothing, it multiplies by exp(dV/2V_T)."""
    high = high_level_current(diode(), high_level_regime(), 0.80).value
    low = high_level_current(diode(), high_level_regime(), 0.78).value
    assert high / low == pytest.approx(math.exp(0.02 / (2 * V_T)), rel=1e-9)


def test_ideality_factor_two_is_exactly_half_the_low_level_slope():
    """The same bias steps in both models differ by a factor of two in the exponent."""
    high = math.log(high_level_current(diode(), high_level_regime(), 0.80).value
                    / high_level_current(diode(), high_level_regime(), 0.78).value)
    low = math.log(forward_current(diode(), forward_regime(), 0.60).value
                   / forward_current(diode(), forward_regime(), 0.58).value)
    assert low / high == pytest.approx(2.0, rel=1e-6)


def test_current_scales_with_area():
    small = high_level_current(diode(), high_level_regime(), 0.75).value
    doubled = DiffusionDiode(
        acceptor_density=diode().acceptor_density,
        donor_density=diode().donor_density,
        area=Quantity(2.0e-4, "cm^2", MEAS),
        hole_mobility=diode().hole_mobility,
        electron_mobility=diode().electron_mobility,
        hole_lifetime=diode().hole_lifetime,
        electron_lifetime=diode().electron_lifetime,
    )
    big = high_level_current(doubled, high_level_regime(), 0.75).value
    assert big / small == pytest.approx(2.0, rel=1e-12)


def test_soft_lifetimes_make_it_model_bound():
    result = high_level_current(diode(SOFT), high_level_regime(), 0.75)
    assert result.confidence == MODEL_BOUND


# --- the boundaries -----------------------------------------------------------------


def test_below_the_crossover_the_model_refuses():
    for bias in (0.6, 0.70):
        assert high_level_current(diode(), high_level_regime(), bias).confidence == EXTRAPOLATED


def test_the_gap_is_closed():
    """0.75 V used to be covered by nothing. Now exactly one model covers it."""
    assert forward_current(diode(), forward_regime(), 0.75).confidence == EXTRAPOLATED
    assert high_level_current(diode(), high_level_regime(), 0.75).reliable


def test_series_resistance_gate_needs_the_resistance():
    """Omitted, the assumption stands unverified and no gate is invented."""
    without = high_level_current(diode(), high_level_regime(), 0.75)
    assert "series_voltage_drop" not in without.trace
    assert not any("series" in g for g in without.gates_passed)


def test_series_resistance_gate_passes_when_the_drop_is_small():
    result = high_level_current(
        diode(), high_level_regime(), 0.75, series_resistance=Quantity(6.935, "ohm")
    )
    assert result.reliable
    assert any("series drop" in g for g in result.gates_passed)
    assert result.trace["series_drop_in_vt"].value < 0.5


def test_series_resistance_gate_fails_when_the_drop_matters():
    """Above ~0.77 V the neglected drop exceeds half a thermal voltage."""
    result = high_level_current(
        diode(), high_level_regime(), 0.80, series_resistance=Quantity(6.935, "ohm")
    )
    assert result.confidence == EXTRAPOLATED
    assert any("neglected series resistance" in g for g in result.gates_failed)


def test_the_reachable_range_depends_on_the_device_not_on_a_fixed_voltage():
    """The same model reaches further on a device with less series resistance."""
    high_r = high_level_current(
        diode(), high_level_regime(), 0.80, series_resistance=Quantity(6.935, "ohm")
    )
    low_r = high_level_current(
        diode(), high_level_regime(), 0.80, series_resistance=Quantity(0.5, "ohm")
    )
    assert high_r.confidence == EXTRAPOLATED
    assert low_r.reliable
