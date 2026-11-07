"""Tests for the forward-injection model.

The most important tests here are not about the diode equation. They are about the
fact that this model and the depletion model have *disjoint* claims: each refuses
the other's territory, and neither is allowed to be the universally valid one.
"""

import math

import pytest

from prime import (
    EXTRAPOLATED,
    GROUNDED,
    MODEL_BOUND,
    READOUT_FLOOR,
    Evidence,
    Quantity,
)
from prime.constants import THERMAL_VOLTAGE_PER_K
from prime.models import (
    AbruptJunction,
    DiffusionDiode,
    abrupt_pn_regime,
    forward_current,
    forward_regime,
    junction_capacitance,
    saturation_current,
    shot_noise_floor,
)

FIRM = Evidence("process_spec", 0.05)
MEAS = Evidence("measurement", 0.02)
SOFT = Evidence("estimate", 0.30, "typical value")
V_T = THERMAL_VOLTAGE_PER_K * 300.0


def diode(mobility_evidence: Evidence = FIRM, lifetime_evidence: Evidence = FIRM):
    return DiffusionDiode(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", MEAS),
        hole_mobility=Quantity(450.0, "cm^2/(V*s)", mobility_evidence),
        electron_mobility=Quantity(1350.0, "cm^2/(V*s)", mobility_evidence),
        hole_lifetime=Quantity(1.0e-6, "s", lifetime_evidence),
        electron_lifetime=Quantity(1.0e-6, "s", lifetime_evidence),
    )


def junction():
    return AbruptJunction(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", MEAS),
        relative_permittivity=Quantity(11.7, "1", Evidence("datasheet", 0.02)),
    )


# --- the physics ------------------------------------------------------------------


def test_saturation_current_matches_hand_calculation():
    """I_s = qA(D_p p_n0/L_p + D_n n_p0/L_n) = 6.41e-16 A."""
    i_s, trace, failed = saturation_current(diode(), forward_regime())
    assert not failed
    assert i_s == pytest.approx(6.41e-16, rel=0.01)
    # Diffusion lengths are recorded because I_s is exponentially sensitive to them.
    assert trace["diffusion_length_holes"].value == pytest.approx(3.41e-3, rel=0.01)
    assert trace["diffusion_length_electrons"].value == pytest.approx(5.91e-3, rel=0.01)


def test_forward_current_at_0v6():
    result = forward_current(diode(), forward_regime(), 0.6, device_id="D1")
    assert result.value == pytest.approx(7.70e-6, rel=0.01)
    assert result.confidence == GROUNDED


def test_each_decade_of_bias_multiplies_current_by_exp_constant():
    """dV/V_T is the whole story while the -1 term is negligible."""
    low = forward_current(diode(), forward_regime(), 0.5).value
    high = forward_current(diode(), forward_regime(), 0.55).value
    assert high / low == pytest.approx(math.exp(0.05 / V_T), rel=1e-6)


def test_current_scales_with_area():
    small = forward_current(diode(), forward_regime(), 0.6).value
    doubled = DiffusionDiode(
        acceptor_density=diode().acceptor_density,
        donor_density=diode().donor_density,
        area=Quantity(2.0e-4, "cm^2", MEAS),
        hole_mobility=diode().hole_mobility,
        electron_mobility=diode().electron_mobility,
        hole_lifetime=diode().hole_lifetime,
        electron_lifetime=diode().electron_lifetime,
    )
    big = forward_current(doubled, forward_regime(), 0.6).value
    assert big / small == pytest.approx(2.0, rel=1e-12)


def test_soft_lifetimes_make_the_prediction_model_bound():
    result = forward_current(
        diode(lifetime_evidence=SOFT), forward_regime(), 0.6, device_id="D1"
    )
    assert result.confidence == MODEL_BOUND


# --- the boundaries -----------------------------------------------------------------


def test_high_level_injection_kills_the_model():
    """At 0.75 V the injected holes exceed the n-side doping: the derivation is void."""
    result = forward_current(diode(), forward_regime(), 0.75)
    assert result.confidence == EXTRAPOLATED
    assert any("high-level injection" in g for g in result.gates_failed)


def test_low_bias_is_outside_the_regime_where_ideality_factor_one_holds():
    result = forward_current(diode(), forward_regime(), 0.05)
    assert result.confidence == EXTRAPOLATED
    assert any("outside modelled range" in g for g in result.gates_failed)


def test_forward_model_refuses_reverse_bias():
    assert forward_current(diode(), forward_regime(), -5.0).confidence == EXTRAPOLATED


def test_depletion_model_refuses_forward_conduction():
    """The capacitance model's regime ends at +0.3 V; it cannot be stretched to 0.6 V."""
    result = junction_capacitance(junction(), abrupt_pn_regime(), 0.6, device_id="D1")
    assert result.confidence == EXTRAPOLATED


def test_the_models_legitimately_overlap_where_both_questions_make_sense():
    """Between 0.1 and 0.3 V both are valid: they answer different questions.

    Overlap is not a defect. The capacitance model says how much charge is stored;
    the conduction model says how much current flows. Different questions, same bias.
    """
    assert junction_capacitance(junction(), abrupt_pn_regime(), 0.2).reliable
    assert forward_current(diode(), forward_regime(), 0.2).reliable


def test_there_is_a_gap_in_the_map_and_the_tool_admits_it():
    """Above high-level injection, nothing applies -- and both models say so.

    The honest output of a boundary map is an empty region, not the nearest model
    stretched to cover it.
    """
    assert junction_capacitance(junction(), abrupt_pn_regime(), 0.75).confidence == EXTRAPOLATED
    assert forward_current(diode(), forward_regime(), 0.75).confidence == EXTRAPOLATED


# --- readout floors -----------------------------------------------------------------


def test_no_readout_floor_without_a_measurement_bandwidth():
    """Silence about the measurement is not evidence about the device."""
    result = forward_current(diode(), forward_regime(), 0.6)
    assert result.floors == ()


def test_shot_noise_floor_is_a_readout_floor():
    result = forward_current(
        diode(), forward_regime(), 0.6, measurement_bandwidth_hz=1.0e3
    )
    floor = result.floors[0]
    assert floor.kind == READOUT_FLOOR
    assert "shot noise" in floor.mechanism
    # sigma_I/I = sqrt(2qB/I); at 7.7 uA over 1 kHz that is 6.5e-6 relative.
    assert result.relative_floor == pytest.approx(6.45e-6, rel=0.01)


def test_shot_noise_worsens_as_current_falls():
    """The same instrument resolves a large current far better than a small one."""
    loud = shot_noise_floor(1.0e-3, 1.0e3)
    quiet = shot_noise_floor(1.0e-9, 1.0e3)
    # relative shot noise goes as 1/sqrt(I): six decades of current, three of noise.
    assert quiet.relative / loud.relative == pytest.approx(1000.0, rel=1e-9)


def test_shot_noise_floor_rejects_zero_bandwidth():
    with pytest.raises(ValueError):
        shot_noise_floor(1.0e-3, 0.0)
