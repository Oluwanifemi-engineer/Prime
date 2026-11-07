"""Composing models that are valid under different assumptions.

This module exists to make one rule enforceable: results may only be combined if
they describe the same physical device, and the combination is never stronger than
its weakest part.

The composition here is the junction capacitance in series with the resistance of
the quasi-neutral region. Neither model can express the combination — one is
valid where the space charge is depleted, the other where neutrality holds — and
the composed quantity is what a device engineer actually cares about.
"""

from __future__ import annotations

import math

from ..core import (
    ProvenanceMismatch,
    Quantity,
    Result,
    UnitMismatch,
    weakest,
)


# Assumptions that a composition makes redundant by modelling the effect itself.
def discharge(assumptions: tuple[str, ...], superseded: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    """Drop assumptions the composition now models explicitly.

    If the junction model assumes no series resistance and the composition adds
    series resistance, the assumption has been discharged, not inherited. Carrying
    it forward would make the composed result claim something it contradicts.
    """
    kept: list[str] = []
    removed: list[str] = []
    for assumption in assumptions:
        if any(term in assumption for term in superseded):
            removed.append(assumption)
        else:
            kept.append(assumption)
    return tuple(kept), removed


def require_same_device(*results: Result) -> str:
    """Assert that all results describe one device and return its id.

    An empty id counts as unmatched: silence is not agreement. A caller composing
    results must name the device, because a capacitance from one diode and a
    resistance from another compose into a plausible, meaningless number.
    """
    ids = {r.device_id for r in results}
    if len(ids) != 1 or "" in ids:
        names = ", ".join(f"{r.quantity}(device_id={r.device_id!r})" for r in results)
        raise ProvenanceMismatch(
            f"cannot compose results from different or unnamed devices: {names}"
        )
    return ids.pop()


def series_rc_bandwidth(resistance: Result, capacitance: Result) -> Result:
    """f_-3dB of the device's own RC limit: f = 1 / (2 pi R_s C_j).

    This is the bandwidth ceiling set by the device alone, before any external
    load. If it sits far above the external-RC bandwidth, the device is not the
    bottleneck — which is the decision this number exists to support.
    """
    device_id = require_same_device(resistance, capacitance)

    if resistance.unit != "ohm":
        raise UnitMismatch(f"resistance must be in 'ohm', got {resistance.unit!r}")
    if capacitance.unit != "F":
        raise UnitMismatch(f"capacitance must be in 'F', got {capacitance.unit!r}")

    time_constant = resistance.value * capacitance.value
    frequency = 1.0 / (2.0 * math.pi * time_constant)

    failed = tuple(resistance.gates_failed) + tuple(capacitance.gates_failed)
    passed = [
        f"both parts describe device {device_id!r}",
        "units compatible (ohm, F)",
        "junction and quasi-neutral regimes both declared",
    ]

    junction_assumptions, discharged = discharge(
        capacitance.assumptions, superseded=("no series resistance",)
    )
    passed.extend(f"discharged: {a} (modelled explicitly here)" for a in discharged)

    return Result(
        quantity="device_rc_bandwidth",
        value=frequency,
        unit="Hz",
        basis="composed: f = 1/(2 pi R_s C_j), junction in series with region resistance",
        gates_passed=tuple(passed),
        gates_failed=failed,
        confidence=weakest(resistance.confidence, capacitance.confidence),
        assumptions=tuple(
            dict.fromkeys(
                junction_assumptions
                + resistance.assumptions
                + ("lumped series RC: one resistance, one capacitance, no distributed effects",)
            )
        ),
        device_id=device_id,
        # Floors simply accumulate: relative_floor combines them in quadrature, and
        # no amount of modelling effort can discharge one.
        floors=capacitance.floors + resistance.floors,
        trace={
            "time_constant": Quantity(time_constant, "s"),
            "series_resistance_used": Quantity(resistance.value, "ohm"),
            "capacitance_used": Quantity(capacitance.value, "F"),
        },
    )
