# Prime

A claim-first modelling core for engineering prediction.

Prime is not a schematic editor, not a circuit simulator, and not a chat interface
over formulas. It is the small, strict layer underneath all of those: a place to
state what you expect a design to do, why you expect it, and under what
conditions — then compute inside that declared regime and report how far the
result can be trusted.

The models included so far are deliberately small. They exist to fix the shape
everything else will follow.

## The method

```
claim  ->  regime  ->  parameters (with evidence)  ->  model  ->  result  ->  confidence
```

Five rules hold this together:

1. **No result without a declared regime.** A model states where it applies; the
   engine checks the operating point against it. Outside the regime, the result is
   reported as *extrapolated*, not quietly returned.
2. **Confidence is derived, never asserted.** It comes from two things: whether the
   validity gates held, and whether the inputs are pinned down or guessed.
3. **Every number carries provenance.** A doping density is a process spec, a
   datasheet value, a measurement, an estimate or a fit — and the result says which.
4. **Downstream numbers are derived, not retyped.** A result that feeds a later
   calculation records what it came from, and inherits its confidence.
5. **Composition is never stronger than its weakest part.** Two results combine
   only if they describe the same device, and the combination takes the weaker
   confidence, inherits every failed gate, and *discharges* the assumptions it now
   models explicitly.

### The three confidence levels

| Level | Meaning |
|---|---|
| `grounded` | Method valid in regime; every input measured or specified. |
| `model_bound` | Method valid in regime, but at least one input is an estimate or fit. The physics is fine; the numbers are soft. |
| `extrapolated` | The declared regime was violated. Useful as an indication, not as a prediction. |

Extrapolation always wins: an out-of-regime result is `extrapolated` no matter how
good the inputs were.

## Confidence and floors are different axes

Confidence answers *do I believe my method*. A **floor** answers *what is the best
anyone could know*. They are orthogonal, and Prime keeps them apart:

| | Confidence | Floor |
|---|---|---|
| Source | our method, our data | nature |
| Improves with | better models, better measurements | nothing — only changing the device |
| Value | `grounded` / `model_bound` / `extrapolated` | a relative spread with a named mechanism |

A result can be `grounded` and still only knowable to 110 ppm, because the charge
in the space-charge region is made of 8.2 × 10⁷ countable, discrete dopants. A
theory that reported 1.143 pF to five significant figures would be claiming
precision nature does not permit.

That ratio is the tool's real output. For CJ-0001: the physical floor on C_j is
110 ppm (0.126 fF), while the chosen falsification band is 31% — **2,834× the
floor**. The claim is therefore testable rather than merely believable, and the
experiment that settles it needs only ~0.1% resolution. Had the band been *below*
the floor, no experiment could settle it and the claim would be a belief.

Floors come in two kinds, and conflating them is a category error:

- `parameter` — irreducible variation in the device itself: discrete dopants, thermal fluctuation.
- `readout` — a limit on how well the quantity can be observed (kT/C noise, instrument resolution), not on what it is.

Floors combine in quadrature through composition, because the mechanisms are
independent. No amount of evidence discharges one.

## The records

All in `prime/core.py`:

- `Claim` — the assertion, the question it answers, its conditions, and what would refute it.
- `Regime` — the executable domain of validity: temperature, bias window, approximations, what is ignored.
- `Quantity` — a number, its unit, and its `Evidence`.
- `Evidence` — what the number rests on: `datasheet`, `process_spec`, `measurement`, `estimate`, `fitted`.
- `Result` — value, basis, assumptions, gates passed and failed, derived confidence, floors, an auditable trace, `derived_from`, and `device_id`.
- `Floor` — a named irreducible mechanism and its relative spread, plus `dopant_floor` for the 1/√N case.

Units are strings, asserted at every model boundary (`Quantity.require("cm^-3")`).
A wrong unit raises `UnitMismatch`; it is never silently rescaled.

## Models

### `prime/models/pn_junction.py` — depletion capacitance

Full-depletion approximation, uniform doping, abrupt junction, low injection, no
series resistance. Reverse bias and weak forward bias (−100 V to +0.3 V, reference
temperature 300 K).

```
V_bi = (kT/q) ln(N_a N_d / n_i^2)
W    = sqrt(2 eps (V_bi - V) / q * (1/N_a + 1/N_d))
C_j  = eps A / W
```

Gates checked: bias inside the modelled window, non-degenerate doping, bias below
the built-in potential, intrinsic concentration used at its tabulated temperature.
The regime's approximations are listed as assumptions, not reported as passing
gates — only checks that were performed are claimed.

### `prime/models/quasi_neutral.py` — series resistance

Ohmic conduction in a uniformly doped region, valid where charge neutrality holds:
`R = L / (q mu N A)`. Gated on non-degenerate doping and on the temperature the
mobility was quoted at. The ohmic field assumption is made checkable separately via
`check_ohmic_field`, because it depends on the operating current, which the
resistance itself does not.

### `prime/models/forward_injection.py` — the ideal diffusion diode

`I = I_s (exp(V/V_T) − 1)` with `I_s = qA(D_p p_n0/L_p + D_n n_p0/L_n)`, valid from
0.1 V to 0.75 V at 300 K. Below 0.1 V the generation-recombination current dominates
and ideality factor 1 is wrong; above it, high-level injection breaks the derivation.

This model is a **replacement**, not an extension: a different set of relations, a
different regime, a different list of things that can go wrong. Prime has no
mechanism for widening an existing model's range, and that is the point — at 0.6 V
the depletion model reports `extrapolated` rather than being stretched, and nothing
inherits its answer.

The high-level-injection gate is quantitative: it compares the injected minority
density `p_n0 exp(V/V_T)` against the majority doping. For the example device it
trips at about 0.72 V, and above it **neither** model applies. A boundary map whose
honest output is an empty region is worth more than one that always has an answer.

It also carries the only readout floor so far: `measurement_bandwidth_hz` is optional,
because a shot-noise limit is a property of the measurement, not of the device.

### `prime/models/forward_high_level.py` — the crossover

`I = I_2 exp(V/(nV_T))` with `n = 2`, where `I_2 = I_s exp(V_x/(2V_T))` and `V_x` is
the bias at which the injected density reaches the majority doping. For the example
device `V_x = 0.7143 V`.

Two things make this more than a second formula:

- **Continuity is structural.** `I_2` is *derived* from the low-level `I_s` at `V_x`,
  so the two models meet to machine precision (relative jump ~4 × 10⁻⁸) and cannot
drift apart. `crossover_voltage` lives in one place and both models call it: the
  low-level model fails at exactly the voltage the high-level model begins.
- **The ceiling is checked, not hard-coded.** This model assumes no series
  resistance. Rather than pick a universal voltage bound — which would be a
  device-independent claim about a device-dependent effect — it accepts the region's
  resistance as a `Quantity` or a `Result` and gates on whether the neglected drop
  exceeds half a thermal voltage.

```
0.75 V, R_s = 6.94 ohm  -> model_bound
0.77 V, R_s = 6.94 ohm  -> extrapolated: drops 13.1 mV (0.5 V_T) at 1.88 mA
0.80 V, R_s = 0.5 ohm   -> model_bound   (a different device reaches further)
```

That last line is the point: how far a model reaches is a property of the device, so
it is computed rather than declared.

### `prime/models/composition.py` — junction in series with series resistance

The two models are valid under *different* assumptions, so they cannot be merged
into one relation — only composed. `series_rc_bandwidth` gives the device's own RC
ceiling, `f = 1 / (2 pi R_s C_j)`.

Composition enforces three things:

- `require_same_device` — results must name the same non-empty `device_id`; an
  empty id counts as unmatched, because silence is not agreement;
- confidence is `weakest(...)` of the parts, and failed gates propagate;
- the junction's "no series resistance" assumption is **discharged**, not inherited.
  Carrying it forward would make the composed result contradict itself.

## Usage

```bash
python -m pytest -q                        # 73 tests
python -m examples.junction_capacitance    # claim -> predictions -> verdict
python -m examples.forward_bias            # replacement, and the edge of the map
```

```python
from prime import Evidence, Quantity
from prime.models import AbruptJunction, abrupt_pn_regime, junction_capacitance

junction = AbruptJunction(
    acceptor_density=Quantity(1e17, "cm^-3", Evidence("process_spec", 0.05)),
    donor_density=Quantity(1e16, "cm^-3", Evidence("process_spec", 0.05)),
    area=Quantity(1e-4, "cm^2", Evidence("measurement", 0.02)),
)

result = junction_capacitance(junction, abrupt_pn_regime(), bias_v=-5.0, device_id="D1")
print(result.describe())   # C_j = 1.143e-12 F [model_bound]
```

The example runs one claim through the full chain and answers a real question:
at 5 V reverse bias the junction is 1.14 pF, giving 13.9 MHz into a 10 kΩ front
end, while the device's own `R_s C_j` limit is 20.1 GHz — so the external resistor
is the bottleneck, by a factor of ~1400.

## Extending it

An extension is admissible only if it fills in the same records and can state:

- what it assumes,
- where it applies,
- what it was calibrated against, if anything,
- what accuracy or confidence it can claim,
- what it depends on.

If a feature cannot say those things, it does not belong yet.

## The boundary map so far

| Case | Model | Dies at |
|---|---|---|
| 1 | Depletion capacitance | forward bias > 0.3 V, degenerate doping |
| 2 | Quasi-neutral conduction | field above velocity saturation, other temperatures |
| 3 | Composition of 1 and 2 | — (inherits both, adds lumped RC) |
| 4 | Forward injection (n = 1) | high-level injection at 0.714 V |
| 5 | High-level injection (n = 2) | neglected series drop > 0.5 V_T (~0.77 V here) |

The word that matters is *dies at*. The formulas are in every textbook; the
boundaries are where the engineering knowledge is.

Case 5 closed the gap case 4 opened, and immediately exposed the next one: at 0.77 V
the voltage dropped across the series resistance is already half a thermal voltage.
That is not a modelling failure, it is the honest end of the current model's reach —
and the next case has to include the resistance instead of neglecting it.

## Next regime

Series-corrected forward conduction, where the junction voltage is `V - I R_s` and
the current has to be solved self-consistently. It is the first composition that is
not a product of two results: an implicit relation, solved rather than multiplied.
After that, self-heating: every model here is isothermal, and that assumption has
never been tested.
