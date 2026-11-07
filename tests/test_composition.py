"""Tests for the quasi-neutral model and the composition rule.

The composition tests are mostly about refusing to produce a number: results from
different devices, unnamed results, and extrapolated parts must not combine into
something that looks trustworthy.
"""

import math

import pytest

from prime import (
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    Evidence,
    ProvenanceMismatch,
    Quantity,
    Result,
    UnitMismatch,
)
from prime.models import (
    AbruptJunction,
    QuasiNeutralRegion,
    abrupt_pn_regime,
    check_ohmic_field,
    junction_capacitance,
    ohmic_regime,
    resistance,
    series_rc_bandwidth,
)

FIRM = Evidence("process_spec", 0.05, "process targeting")
SOFT = Evidence("estimate", 0.30, "typical mobility, not measured")

DEVICE = "D1"


def junction(device_id: str = DEVICE) -> AbruptJunction:
    return AbruptJunction(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
        relative_permittivity=Quantity(11.7, "1", Evidence("datasheet", 0.02)),
    )


def region(mobility_evidence: Evidence = FIRM) -> QuasiNeutralRegion:
    return QuasiNeutralRegion(
        doping=Quantity(1.0e17, "cm^-3", FIRM),
        length=Quantity(5.0e-3, "cm", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
        mobility=Quantity(450.0, "cm^2/(V*s)", mobility_evidence),
    )


def capacitance(device_id: str = DEVICE):
    return junction_capacitance(junction(), abrupt_pn_regime(), -5.0, device_id=device_id)


def series_resistance(device_id: str = DEVICE, mobility_evidence: Evidence = FIRM):
    return resistance(region(mobility_evidence), ohmic_regime(), device_id=device_id)


# --- the quasi-neutral model ------------------------------------------------------


def test_series_resistance_matches_hand_calculation():
    """R = L/(q mu N A) = 5e-3 / (1.602e-19 * 450 * 1e17 * 1e-4) = 6.94 ohm."""
    result = series_resistance()
    assert result.value == pytest.approx(6.94, rel=0.01)
    assert result.confidence == GROUNDED


def test_resistance_scales_linearly_with_length():
    regime = ohmic_regime()
    short = resistance(region(), regime)
    longer = resistance(
        QuasiNeutralRegion(
            doping=region().doping,
            length=Quantity(1.0e-2, "cm", FIRM),
            area=region().area,
            mobility=region().mobility,
        ),
        regime,
    )
    assert longer.value / short.value == pytest.approx(2.0, rel=1e-12)


def test_resistance_scales_inversely_with_area():
    regime = ohmic_regime()
    base = resistance(region(), regime)
    wider = resistance(
        QuasiNeutralRegion(
            doping=region().doping,
            length=region().length,
            area=Quantity(2.0e-4, "cm^2", FIRM),
            mobility=region().mobility,
        ),
        regime,
    )
    assert base.value / wider.value == pytest.approx(2.0, rel=1e-12)


def test_mobility_at_another_temperature_is_flagged():
    """Mobility is quoted at 300 K; using it at 400 K is an extrapolation."""
    result = resistance(region(), ohmic_regime(temperature_k=400.0))
    assert result.confidence == EXTRAPOLATED
    assert any("temperature" in g for g in result.gates_failed)


def test_ohmic_field_gate_passes_at_normal_current():
    failures = check_ohmic_field(
        series_resistance(), region(), Quantity(1.0e-3, "A", Evidence("measurement", 0.05))
    )
    assert failures == []


def test_ohmic_field_gate_fails_at_high_current():
    # 10 A through 1e-4 cm^2 gives J = 1e5 A/cm^2; with sigma = 7.2 S/cm that is
    # E = 1.4e4 V/cm, above the 1e4 V/cm saturation threshold.
    failures = check_ohmic_field(
        series_resistance(), region(), Quantity(10.0, "A", Evidence("measurement", 0.05))
    )
    assert failures and "velocity-saturation" in failures[0]


# --- composition ------------------------------------------------------------------


def test_composed_bandwidth_matches_hand_calculation():
    """tau = 6.94 ohm * 1.143 pF = 7.93 ps, f = 20.1 GHz."""
    result = series_rc_bandwidth(series_resistance(), capacitance())
    assert result.value == pytest.approx(20.1e9, rel=0.02)
    assert result.confidence == GROUNDED
    assert result.device_id == DEVICE


def test_device_rc_limit_dominates_the_external_load():
    """The point of the composed number: the device is not the bottleneck."""
    internal = series_rc_bandwidth(series_resistance(), capacitance())
    external = 1.0 / (2.0 * math.pi * 1.0e4 * capacitance().value)
    assert internal.value > 1000 * external


def test_composition_refuses_two_different_devices():
    with pytest.raises(ProvenanceMismatch):
        series_rc_bandwidth(series_resistance("D1"), capacitance("D2"))


def test_composition_refuses_unnamed_results():
    with pytest.raises(ProvenanceMismatch):
        series_rc_bandwidth(series_resistance(""), capacitance(""))


def test_composition_inherits_the_weaker_confidence():
    soft = series_rc_bandwidth(series_resistance(mobility_evidence=SOFT), capacitance())
    assert soft.confidence == MODEL_BOUND


def test_composition_propagates_a_failed_gate():
    bad_capacitance = junction_capacitance(junction(), abrupt_pn_regime(), 0.5, device_id=DEVICE)
    composed = series_rc_bandwidth(series_resistance(), bad_capacitance)
    assert composed.confidence == EXTRAPOLATED
    assert not composed.reliable
    assert composed.gates_failed


def test_composition_discharges_the_assumption_it_contradicts():
    """The junction assumes no series resistance; the composition models it.

    Inheriting that assumption would make the composed result contradict itself.
    """
    composed = series_rc_bandwidth(series_resistance(), capacitance())
    assert not any("no series resistance" in a for a in composed.assumptions)
    assert any("discharged: no series resistance" in g for g in composed.gates_passed)


def test_composition_records_both_parts_in_the_trace():
    composed = series_rc_bandwidth(series_resistance(), capacitance())
    assert composed.trace["series_resistance_used"].unit == "ohm"
    assert composed.trace["capacitance_used"].unit == "F"
    assert composed.trace["time_constant"].unit == "s"


def test_composition_rejects_wrong_units():
    wrong_unit = Result(
        quantity="series_resistance",
        value=6.94e-3,
        unit="kohm",
        basis="deliberately wrong unit, for the test",
        gates_passed=(),
        gates_failed=(),
        confidence=GROUNDED,
        device_id=DEVICE,
    )
    with pytest.raises(UnitMismatch):
        series_rc_bandwidth(wrong_unit, capacitance())
