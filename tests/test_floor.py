"""Tests for the physical floor.

The floor is the one quantity in Prime that confidence cannot touch: it is a fact
about nature, not about us. These tests pin down that independence, the scaling, and
the fact that floors survive composition.
"""

import math

import pytest

from prime import (
    GROUNDED,
    PARAMETER_FLOOR,
    READOUT_FLOOR,
    Evidence,
    Floor,
    Quantity,
    dopant_floor,
    quadrature,
)
from prime.models import (
    AbruptJunction,
    QuasiNeutralRegion,
    abrupt_pn_regime,
    junction_capacitance,
    ohmic_regime,
    resistance,
    series_rc_bandwidth,
)

FIRM = Evidence("process_spec", 0.05)


def junction() -> AbruptJunction:
    return AbruptJunction(
        acceptor_density=Quantity(1.0e17, "cm^-3", FIRM),
        donor_density=Quantity(1.0e16, "cm^-3", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
        relative_permittivity=Quantity(11.7, "1", Evidence("datasheet", 0.02)),
    )


def region() -> QuasiNeutralRegion:
    return QuasiNeutralRegion(
        doping=Quantity(1.0e17, "cm^-3", FIRM),
        length=Quantity(5.0e-3, "cm", FIRM),
        area=Quantity(1.0e-4, "cm^2", FIRM),
        mobility=Quantity(450.0, "cm^2/(V*s)", FIRM),
    )


def test_dopant_floor_is_one_over_root_n():
    floor = dopant_floor("test", 1.0e8, "8 orders of dopants")
    assert floor.relative == pytest.approx(1.0e-4)
    assert floor.kind == PARAMETER_FLOOR


def test_junction_floor_counts_the_ionised_charge_not_the_flattering_side():
    """Charge neutrality makes the count unambiguous: N_eff * W * A."""
    result = junction_capacitance(junction(), abrupt_pn_regime(), -5.0)
    width = result.trace["depletion_width"].value
    effective_doping = 1.0 / (1.0 / 1.0e17 + 1.0 / 1.0e16)
    expected = 1.0 / math.sqrt(effective_doping * width * 1.0e-4)
    assert result.relative_floor == pytest.approx(expected)
    assert result.relative_floor == pytest.approx(1.10e-4, rel=0.02)


def test_floor_survives_a_grounded_confidence():
    """Orthogonality: a perfect method still cannot beat a 1/sqrt(N) floor."""
    result = junction_capacitance(junction(), abrupt_pn_regime(), -5.0)
    assert result.confidence == GROUNDED
    assert result.relative_floor > 0.0


def test_floor_shrinks_as_sqrt_of_area():
    """Four times the area, half the irreducible spread."""
    regime = abrupt_pn_regime()
    small = junction_capacitance(junction(), regime, -5.0)
    big = junction_capacitance(
        AbruptJunction(
            acceptor_density=junction().acceptor_density,
            donor_density=junction().donor_density,
            area=Quantity(4.0e-4, "cm^2", FIRM),
            relative_permittivity=junction().relative_permittivity,
        ),
        regime,
        -5.0,
    )
    assert small.relative_floor / big.relative_floor == pytest.approx(2.0, rel=1e-2)


def test_resistance_floor_is_tiny_because_the_volume_is_large():
    result = resistance(region(), ohmic_regime())
    assert result.relative_floor == pytest.approx(1.0 / math.sqrt(1.0e17 * 5.0e-3 * 1.0e-4))


def test_floors_combine_in_quadrature_through_composition():
    capacitance = junction_capacitance(junction(), abrupt_pn_regime(), -5.0, device_id="D1")
    series = resistance(region(), ohmic_regime(), device_id="D1")
    composed = series_rc_bandwidth(series, capacitance)

    assert len(composed.floors) == 2
    assert composed.relative_floor == pytest.approx(
        quadrature(capacitance.relative_floor, series.relative_floor)
    )
    # The junction dominates: 110 ppm in quadrature with 4.5 ppm is still 110 ppm.
    assert composed.relative_floor == pytest.approx(capacitance.relative_floor, rel=1e-3)


def test_floor_travels_into_the_downstream_cutoff():
    from prime.models import single_pole_cutoff

    capacitance = junction_capacitance(junction(), abrupt_pn_regime(), -5.0, device_id="D1")
    cutoff = single_pole_cutoff(Quantity(1.0e4, "ohm", FIRM), capacitance)
    assert cutoff.relative_floor == pytest.approx(capacitance.relative_floor)


def test_floor_value_is_absolute_not_just_relative():
    result = junction_capacitance(junction(), abrupt_pn_regime(), -5.0)
    assert result.floor_value == pytest.approx(result.value * result.relative_floor)
    assert 1.0e-16 < result.floor_value < 1.5e-16  # about 0.13 fF on 1.143 pF


def test_floors_are_printed_with_their_mechanism():
    text = junction_capacitance(junction(), abrupt_pn_regime(), -5.0).describe()
    assert "floor :" in text
    assert "random dopant fluctuation" in text
    assert "irreducible" in text


def test_a_floor_cannot_be_negative_or_unknown():
    with pytest.raises(ValueError):
        Floor("test", PARAMETER_FLOOR, -1.0e-4, "nonsense")
    with pytest.raises(ValueError):
        Floor("test", "vibes", 1.0e-4, "nonsense")


def test_readout_floor_kind_exists_for_measurement_limits():
    """kT/C and instrument resolution are not facts about the device.

    They are kept as a separate kind so a readout limit is never mistaken for
    device variation.
    """
    floor = Floor("thermal noise on a 1 pF node", READOUT_FLOOR, 6.0e-5, "kT/C")
    assert floor.kind == READOUT_FLOOR
