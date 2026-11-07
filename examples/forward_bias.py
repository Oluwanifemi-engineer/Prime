"""Case 4: forward injection, where the previous model has to be replaced.

Claim FI-0001: at 0.6 V forward bias this diode carries ~7.7 uA, and that current
is limited by diffusion, not by the series resistance we can compute separately.

The point of this example is what the tool *refuses* to do: the depletion model
cannot be stretched to 0.6 V, and above about 0.72 V neither model applies at all.

Run with:  python -m examples.forward_bias
"""

from prime import Claim, Evidence, Quantity
from prime.models import (
    AbruptJunction,
    DiffusionDiode,
    QuasiNeutralRegion,
    abrupt_pn_regime,
    crossover_voltage,
    forward_current,
    forward_regime,
    high_level_current,
    high_level_regime,
    junction_capacitance,
    ohmic_regime,
    resistance,
)

DEVICE = "D1"

CLAIM = Claim(
    claim_id="FI-0001",
    statement="At V_F = 0.60 V the forward current of D1 is 7.7 uA.",
    question="Is D1's forward current set by diffusion, or is series resistance "
    "already limiting it at this operating point?",
    conditions={
        "temperature": "300 K",
        "bias": "+0.60 V",
        "area": "1.0e-4 cm^2",
        "n_side_doping": "1e16 cm^-3",
        "p_side_doping": "1e17 cm^-3",
        "ideality_factor": "1 (no space-charge recombination)",
    },
    falsified_if="the measured current differs from 7.7 uA by more than 2x",
)

DIODE = DiffusionDiode(
    acceptor_density=Quantity(1.0e17, "cm^-3", Evidence("process_spec", 0.05, "epi targeting")),
    donor_density=Quantity(1.0e16, "cm^-3", Evidence("process_spec", 0.05, "substrate spec")),
    area=Quantity(1.0e-4, "cm^2", Evidence("measurement", 0.02, "drawn layout area")),
    hole_mobility=Quantity(450.0, "cm^2/(V*s)", Evidence("estimate", 0.20, "bulk hole mobility")),
    electron_mobility=Quantity(1350.0, "cm^2/(V*s)", Evidence("estimate", 0.20, "bulk electron mobility")),
    hole_lifetime=Quantity(1.0e-6, "s", Evidence("estimate", 0.50, "process lifetime, +/- a decade")),
    electron_lifetime=Quantity(1.0e-6, "s", Evidence("estimate", 0.50, "process lifetime, +/- a decade")),
)

JUNCTION = AbruptJunction(
    acceptor_density=DIODE.acceptor_density,
    donor_density=DIODE.donor_density,
    area=DIODE.area,
)

P_REGION = QuasiNeutralRegion(
    doping=Quantity(1.0e17, "cm^-3", Evidence("process_spec", 0.05)),
    length=Quantity(5.0e-3, "cm", Evidence("process_spec", 0.10)),
    area=Quantity(1.0e-4, "cm^2", Evidence("measurement", 0.02)),
    mobility=Quantity(450.0, "cm^2/(V*s)", Evidence("estimate", 0.20, "bulk hole mobility")),
)


def main() -> None:
    current = forward_current(
        DIODE, forward_regime(), bias_v=0.60, device_id=DEVICE, measurement_bandwidth_hz=1.0e3
    )
    series = resistance(P_REGION, ohmic_regime(), device_id=DEVICE)

    print(f"claim {CLAIM.claim_id}: {CLAIM.statement}")
    print(f"question: {CLAIM.question}\n")
    print(current.describe())
    print()

    # What the previous model does with the same operating point.
    stretched = junction_capacitance(JUNCTION, abrupt_pn_regime(), 0.60, device_id=DEVICE)
    print(f"depletion model at the same bias: {stretched.confidence}")
    print(f"  reason: {stretched.gates_failed[0]}\n")

    # The handover. At 0.75 V the low-level model is out and the high-level model
    # takes over -- with the same boundary voltage, because both compute it from
    # the same function.
    gap_current = forward_current(DIODE, forward_regime(), bias_v=0.75, device_id=DEVICE)
    high_current = high_level_current(DIODE, high_level_regime(), bias_v=0.75, device_id=DEVICE)
    v_x = high_current.trace["crossover_voltage"].value
    print(f"at 0.75 V: low-level model {gap_current.confidence}, "
          f"high-level model {high_current.confidence} ({high_current.value * 1e3:.3g} mA)")
    print(f"  low-level model's reason: {gap_current.gates_failed[0]}")

    # Continuity is structural: I_2 is derived from I_s at the crossover, so the two
    # curves meet exactly rather than being joined by hand.
    below = forward_current(DIODE, forward_regime(), bias_v=v_x - 1e-9, device_id=DEVICE)
    above = high_level_current(DIODE, high_level_regime(), bias_v=v_x, device_id=DEVICE)
    print(f"  crossover at {v_x:.4f} V: low-level gives {below.value * 1e6:.6g} uA, "
          f"high-level gives {above.value * 1e6:.6g} uA")
    print(f"  relative jump at the handover: {abs(above.value - below.value) / below.value:.1e}")
    print("  the two models share one crossover function, so they cannot drift apart\n")

    # The assumption the forward model makes -- "no series resistance" -- tested
    # quantitatively rather than taken on trust.
    drop = current.value * series.value
    thermal_voltage = current.trace["thermal_voltage"].value
    print(f"series resistance: {series.value:.3g} ohm")
    print(f"  voltage dropped across it at this current: {drop * 1e6:.2f} uV")
    print(f"  that is {drop / thermal_voltage * 100:.2f}% of V_T ({thermal_voltage * 1e3:.2f} mV):")
    print("  the 'no series resistance' assumption is good to 0.2% here, so it is")
    print("  discharged by arithmetic rather than assumed away.")

    # The map ends where the neglected drop passes half a thermal voltage, and where
    # that happens depends on the device, not on a fixed voltage in the model.
    limit = None
    for millivolts in range(720, 950):
        bias = millivolts / 1000.0
        attempt = high_level_current(
            DIODE, high_level_regime(), bias_v=bias, device_id=DEVICE, series_resistance=series
        )
        if not attempt.reliable:
            limit = bias
            break
    print(f"high-level model reaches {limit:.3f} V on this device, then")
    print(f"  {attempt.gates_failed[0]}")
    print("  a lower-resistance device would reach further: the ceiling is a property")
    print("  of the device, so it is checked rather than hard-coded.\n")

    print(f"shot-noise readout floor at 1 kHz: {current.floor_value:.3g} A "
          f"({current.relative_floor * 1e6:.3g} ppm)")
    print(f"claim tolerance is 100% of 7.7 uA, the floor is {current.relative_floor*1e6:.3g} ppm:")
    print("  the claim is testable by any ordinary electrometer.")
    print(f"would be refuted if: {CLAIM.falsified_if}")


if __name__ == "__main__":
    main()
