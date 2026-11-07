"""End-to-end example: one claim, three models, two composed numbers.

Claim CJ-0001: at 5 V reverse bias this junction is 1.14 pF, so a 10 kohm
transimpedance front-end stays above 10 MHz. Follow-up question: is the device
itself, or the external resistor, the bandwidth bottleneck?

Run with:  python -m examples.junction_capacitance
"""

from prime import Claim, Evidence, Quantity
from prime.models import (
    AbruptJunction,
    QuasiNeutralRegion,
    abrupt_pn_regime,
    junction_capacitance,
    ohmic_regime,
    resistance,
    series_rc_bandwidth,
    single_pole_cutoff,
)

DEVICE = "D1"

CLAIM = Claim(
    claim_id="CJ-0001",
    statement="At V_R = 5.0 V, the junction capacitance of D1 is 1.14 pF.",
    question="Is D1's junction capacitance low enough that a 10 kohm "
    "transimpedance front-end keeps f_-3dB above 10 MHz, and is the device "
    "itself the bottleneck?",
    conditions={
        "temperature": "300 K",
        "bias": "-5.0 V (reverse)",
        "area": "1.0e-4 cm^2",
        "acceptor_density": "1e17 cm^-3",
        "donor_density": "1e16 cm^-3",
        "junction": "abrupt",
        "p_region_length": "50 um",
    },
    falsified_if="C_j > 1.5 pF, or the device RC limit falls below the external RC limit",
)

JUNCTION = AbruptJunction(
    acceptor_density=Quantity(1.0e17, "cm^-3", Evidence("process_spec", 0.05, "epi targeting")),
    donor_density=Quantity(1.0e16, "cm^-3", Evidence("process_spec", 0.05, "substrate spec")),
    area=Quantity(1.0e-4, "cm^2", Evidence("measurement", 0.02, "drawn layout area")),
)

P_REGION = QuasiNeutralRegion(
    doping=Quantity(1.0e17, "cm^-3", Evidence("process_spec", 0.05)),
    length=Quantity(5.0e-3, "cm", Evidence("process_spec", 0.10, "junction depth to contact")),
    area=Quantity(1.0e-4, "cm^2", Evidence("measurement", 0.02)),
    mobility=Quantity(450.0, "cm^2/(V*s)", Evidence("estimate", 0.20, "bulk hole mobility")),
)

EXTERNAL_R = Quantity(1.0e4, "ohm", Evidence("datasheet", 0.01, "10 kohm, 1% part"))

# The value above which CJ-0001 is refuted. Set by the decision, not by the model.
FALSIFICATION_LIMIT_F = 1.5e-12


def main() -> None:
    regime = abrupt_pn_regime()

    capacitance = junction_capacitance(JUNCTION, regime, bias_v=-5.0, device_id=DEVICE)
    series = resistance(P_REGION, ohmic_regime(), device_id=DEVICE)
    external_limit = single_pole_cutoff(EXTERNAL_R, capacitance)
    device_limit = series_rc_bandwidth(series, capacitance)

    print(f"claim {CLAIM.claim_id}: {CLAIM.statement}")
    print(f"question: {CLAIM.question}\n")
    for result in (capacitance, series, external_limit, device_limit):
        print(result.describe())
        print()

    holds = capacitance.value <= FALSIFICATION_LIMIT_F and external_limit.value >= 10e6
    device_is_bottleneck = device_limit.value < external_limit.value
    print(f"capacitance claim: {'HOLDS' if holds else 'REFUTED'}")
    print(
        "bottleneck: "
        + ("the device (R_s C_j)" if device_is_bottleneck else "the external resistor")
    )
    print(f"device limit is {device_limit.value / external_limit.value:.0f}x the external limit")
    print(f"would be refuted if: {CLAIM.falsified_if}")

    # The floor is what nature allows; the falsification band is what we chose to
    # care about. Their ratio says whether this claim can be settled by experiment
    # at all, and which instrument resolution would be required to settle it.
    room = FALSIFICATION_LIMIT_F - capacitance.value
    print(
        f"physical floor on C_j: {capacitance.floor_value:.3g} F "
        f"({capacitance.relative_floor * 1e6:.3g} ppm, irreducible)"
    )
    print(
        f"falsification band is {room / capacitance.floor_value:.0f}x the physical "
        "floor: testable, not merely believable"
    )


if __name__ == "__main__":
    main()
