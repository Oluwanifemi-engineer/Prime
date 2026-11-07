# Prime

A claim-first modelling core for engineering prediction.

Prime is not a schematic editor, not a circuit simulator, and not a chat interface
over formulas. It is the small, strict layer underneath all of those: a place to
state what you expect a design to do, why you expect it, and under what
conditions — then compute inside that declared regime and report how far the
result can be trusted.

The first model, an abrupt silicon PN junction, is deliberately small. It exists
to fix the shape everything else will follow.

## The method

```
claim  ->  regime  ->  parameters (with evidence)  ->  model  ->  result  ->  confidence
```

Four rules hold this together:

1. **No result without a declared regime.** A model states where it applies; the
   engine checks the operating point against it. Outside the regime, the result is
   reported as *extrapolated*, not quietly returned.
2. **Confidence is derived, never asserted.** It comes from two things: whether the
   validity gates held, and whether the inputs are pinned down or guessed.
3. **Every number carries provenance.** A doping density is a process spec, a
   datasheet value, a measurement, an estimate or a fit — and the result says which.
4. **Downstream numbers are derived, not retyped.** A result that feeds a later
   calculation records what it came from, and inherits its confidence.

### The three confidence levels

| Level | Meaning |
|---|---|
| `grounded` | Method valid in regime; every input measured or specified. |
| `model_bound` | Method valid in regime, but at least one input is an estimate or fit. The physics is fine; the numbers are soft. |
| `extrapolated` | The declared regime was violated. Useful as an indication, not as a prediction. |

Extrapolation always wins over soft evidence: an out-of-regime result is
`extrapolated` no matter how good the inputs were.

## The records

All in `prime/core.py`:

- `Claim` — the assertion, the question it answers, its conditions, and what would refute it.
- `Regime` — the executable domain of validity: temperature, bias window, approximations, what is ignored.
- `Quantity` — a number, its unit, and its `Evidence` (kind, uncertainty, note).
- `Evidence` — what the number rests on: `datasheet`, `process_spec`, `measurement`, `estimate`, `fitted`.
- `Result` — value, basis, gates passed and failed, derived confidence, an auditable trace, and `derived_from`.

## Current model: abrupt PN junction

`prime/models/pn_junction.py` — full-depletion approximation, uniform doping,
abrupt junction, low injection, no series resistance. Reverse bias and weak
forward bias only (−100 V to +0.3 V, reference temperature 300 K).

```
V_bi = (kT/q) ln(N_a N_d / n_i^2)
W    = sqrt(2 eps (V_bi - V) / q * (1/N_a + 1/N_d))
C_j  = eps A / W
```

Gates actually checked: bias inside the modelled window, non-degenerate doping,
bias below the built-in potential, and intrinsic concentration used at its
tabulated temperature. Units are asserted at every model boundary: cm, cm^-3,
F, not metres.

The regime's approximations are listed as assumptions, not reported as passing
gates. Only checks that were performed are claimed.

## Usage

```bash
python -m pytest -q                        # 17 tests
python -m examples.junction_capacitance    # claim -> prediction -> verdict
```

```python
from prime import Claim, Evidence, Quantity
from prime.models import AbruptJunction, abrupt_pn_regime, junction_capacitance

junction = AbruptJunction(
    acceptor_density=Quantity(1e17, "cm^-3", Evidence("process_spec", 0.05)),
    donor_density=Quantity(1e16, "cm^-3", Evidence("process_spec", 0.05)),
    area=Quantity(1e-4, "cm^2", Evidence("measurement", 0.02)),
)

result = junction_capacitance(junction, abrupt_pn_regime(), bias_v=-5.0)
print(result.describe())   # C_j = 1.143e-12 F [model_bound]
```

## Extending it

An extension is admissible only if it fills in the same records and can state:

- what it assumes,
- where it applies,
- what it was calibrated against, if anything,
- what accuracy or confidence it can claim,
- what it depends on.

If a feature cannot say those things, it does not belong yet.

## Next regime

Series resistance. It is the first real test of the architecture, because it
cannot be added as a better formula: the depletion model stays valid for the
junction, while conduction in the quasi-neutral regions is valid under different
assumptions. The two must be composed with separate validity ranges rather than
merged into a single relation.
