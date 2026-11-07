"""Tests for the abrupt-junction model and the core guarantees.

Two kinds of test here, on purpose:

* golden numbers, which pin the implementation to the worked example;
* physics invariants, which would still pass if the constants were swapped but
  the algebra was wrong. These are the ones that matter.
"""

import math

import pytest

from prime import EXTRAPOLATED, GROUNDED, MODEL_BOUND, Evidence, Quantity, UnitMismatch
from prime.constants import SILICON_RELATIVE_PERMITTIVITY
from prime.models import (
    AbruptJunction,
    abrupt_pn_regime,
    built_in_potential,
    junction_capacitance,
    single_pole_cutoff,
)

FIRM = Evidence("process_spec", 0.05, "process targeting")
SOFT = Evidence("estimate", 0.20, "guessed from similar parts")


def junction(area_evidence: Evidence = FIRM) -> AbruptJunction:
    """A junction whose every input is pinned by a spec, so it can be GROUNDED."""
    return AbruptJunction(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", area_evidence),
        relative_permittivity=Quantity(
            SILICON_RELATIVE_PERMITTIVITY, "1", Evidence("datasheet", 0.02)
        ),
    )


# --- golden numbers ---------------------------------------------------------------


def test_built_in_potential_matches_hand_calculation():
    v_bi, failed = built_in_potential(junction(), abrupt_pn_regime())
    assert failed == []
    assert v_bi == pytest.approx(0.774, abs=5e-4)


def test_capacitance_at_five_volt_reverse_bias():
    result = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=-5.0)
    assert result.value == pytest.approx(1.14e-12, rel=0.01)
    assert result.confidence == GROUNDED


def test_depletion_width_at_five_volt_reverse_bias():
    result = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=-5.0)
    assert result.trace["depletion_width"].value == pytest.approx(9.06e-5, rel=0.01)


def test_downstream_cutoff_frequency():
    cj = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=-5.0)
    cutoff = single_pole_cutoff(Quantity(1.0e4, "ohm", Evidence("datasheet", 0.01)), cj)
    assert cutoff.value == pytest.approx(13.9e6, rel=0.02)
    assert cutoff.derived_from == ("junction_capacitance",)


# --- physics invariants -----------------------------------------------------------


def test_capacitance_scales_as_inverse_root_of_total_potential():
    """C_j(V) ~ (V_bi - V)^-1/2. Catches algebra errors that golden numbers can miss."""
    regime = abrupt_pn_regime()
    j = junction()
    cj_zero = junction_capacitance(j, regime, bias_v=0.0)
    cj_reverse = junction_capacitance(j, regime, bias_v=-5.0)
    v_bi = cj_zero.trace["built_in_potential"].value

    predicted_ratio = math.sqrt((v_bi + 5.0) / v_bi)
    assert cj_zero.value / cj_reverse.value == pytest.approx(predicted_ratio, rel=1e-9)


def test_capacitance_increases_with_reverse_bias_magnitude():
    regime = abrupt_pn_regime()
    values = [
        junction_capacitance(junction(), regime, bias_v=v).value for v in (0.0, -1.0, -5.0, -20.0)
    ]
    assert values == sorted(values, reverse=True)


def test_doping_asymmetry_shifts_width_toward_light_side():
    """W ~ 1/sqrt(N) for a strong asymmetry: the lightly doped side sets the width."""
    regime = abrupt_pn_regime()
    light = AbruptJunction(
        acceptor_density=Quantity(1.0e15, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
    )
    heavy = AbruptJunction(
        acceptor_density=Quantity(1.0e16, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
    )
    w_light = junction_capacitance(light, regime, bias_v=-5.0).trace["depletion_width"].value
    w_heavy = junction_capacitance(heavy, regime, bias_v=-5.0).trace["depletion_width"].value
    assert w_light > w_heavy


# --- regime enforcement -----------------------------------------------------------


def test_strong_forward_bias_is_extrapolated_not_silently_returned():
    result = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=0.5)
    assert result.confidence == EXTRAPOLATED
    assert not result.reliable
    assert result.gates_failed


def test_bias_beyond_built_in_potential_is_flagged():
    result = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=0.9)
    assert result.confidence == EXTRAPOLATED
    assert result.value == math.inf
    assert any("built-in potential" in g for g in result.gates_failed)


def test_off_reference_temperature_is_flagged():
    result = junction_capacitance(junction(), abrupt_pn_regime(temperature_k=350.0), bias_v=-5.0)
    assert result.confidence == EXTRAPOLATED
    assert any("300 K" in g for g in result.gates_failed)


def test_degenerate_doping_is_flagged():
    degenerate = AbruptJunction(
        acceptor_density=Quantity(5.0e19, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
    )
    result = junction_capacitance(degenerate, abrupt_pn_regime(), bias_v=-5.0)
    assert result.confidence == EXTRAPOLATED
    assert any("degenerate" in g for g in result.gates_failed)


# --- evidence and confidence ------------------------------------------------------


def test_soft_input_downgrades_to_model_bound():
    result = junction_capacitance(junction(SOFT), abrupt_pn_regime(), bias_v=-5.0)
    assert result.confidence == MODEL_BOUND


def test_default_permittivity_is_treated_as_an_estimate():
    """Silicon permittivity defaults to a textbook value, and the system says so."""
    bare = AbruptJunction(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
    )
    result = junction_capacitance(bare, abrupt_pn_regime(), bias_v=-5.0)
    assert result.confidence == MODEL_BOUND


def test_extrapolated_capacitance_poisons_the_downstream_number():
    bad = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=0.5)
    cutoff = single_pole_cutoff(Quantity(1.0e4, "ohm", Evidence("datasheet", 0.01)), bad)
    assert cutoff.confidence == EXTRAPOLATED


def test_soft_resistance_downgrades_downstream():
    cj = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=-5.0)
    cutoff = single_pole_cutoff(Quantity(1.0e4, "ohm", SOFT), cj)
    assert cutoff.confidence == MODEL_BOUND


# --- unit discipline --------------------------------------------------------------


def test_wrong_units_are_rejected_not_rescaled():
    wrong = AbruptJunction(
        acceptor_density=Quantity(1.0e23, "m^-3", FIRM),
        donor_density=Quantity(1.0e22, "m^-3", FIRM),
        area=Quantity(1.0e-8, "m^2", FIRM),
    )
    with pytest.raises(UnitMismatch):
        junction_capacitance(wrong, abrupt_pn_regime(), bias_v=-5.0)


def test_wrong_resistance_unit_rejected():
    cj = junction_capacitance(junction(), abrupt_pn_regime(), bias_v=-5.0)
    with pytest.raises(UnitMismatch):
        single_pole_cutoff(Quantity(1.0e4, "kohm", Evidence("datasheet", 0.01)), cj)
