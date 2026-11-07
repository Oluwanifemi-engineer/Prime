"""End-to-end example: one claim, one model, one downstream number.

Claim CJ-0001: at 5 V reverse bias, this junction is 1.14 pF, which keeps a
10 kohm transimpedance front-end above 10 MHz.

Run with:  python -m examples.junction_capacitance
"""

from prime import Claim, Evidence, Quantity
from prime.models import AbruptJunction, abrupt_pn_regime, junction_capacitance, single_pole_cutoff

CLAIM = Claim(
    claim_id="CJ-0001",
    statement="At V_R = 5.0 V, the junction capacitance of D1 is 1.14 pF.",
    question="Is D1's junction capacitance low enough that a 10 kohm "
    "transimpedance front-end keeps f_-3dB above 10 MHz?",
    conditions={
        "temperature": "300 K",
        "bias": "-5.0 V (reverse)",
        "area": "1.0e-4 cm^2",
        "acceptor_density": "1e17 cm^-3",
        "donor_density": "1e16 cm^-3",
        "junction": "abrupt",
    },
    falsified_if="C_j > 1.5 pF, which would push f_-3dB below 10.6 MHz",
)

JUNCTION = AbruptJunction(
    acceptor_density=Quantity(
        1.0e17, "cm^-3", Evidence("process_spec", 0.05, "epi targeting, +/-5%")
    ),
    donor_density=Quantity(
        1.0e16, "cm^-3", Evidence("process_spec", 0.05, "substrate spec, +/-5%")
    ),
    area=Quantity(1.0e-4, "cm^2", Evidence("measurement", 0.02, "layout area, drawn")),
)

RESISTANCE = Quantity(1.0e4, "ohm", Evidence("datasheet", 0.01, "10 kohm, 1% part"))


def main() -> None:
    regime = abrupt_pn_regime()
    capacitance = junction_capacitance(JUNCTION, regime, bias_v=-5.0)
    cutoff = single_pole_cutoff(RESISTANCE, capacitance)

    print(f"claim {CLAIM.claim_id}: {CLAIM.statement}")
    print(f"question: {CLAIM.question}\n")
    print(capacitance.describe())
    print()
    print(cutoff.describe())
    print()

    holds = capacitance.value <= 1.5e-12 and cutoff.value >= 10e6
    verdict = "HOLDS" if holds else "REFUTED"
    if not capacitance.reliable:
        verdict += " (but the prediction is extrapolated: do not rely on it)"
    print(f"verdict: {verdict}")
    print(f"would be refuted if: {CLAIM.falsified_if}")


if __name__ == "__main__":
    main()
