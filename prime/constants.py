"""Physical constants, in CGS (centimetre-gram-second), the working unit system
for semiconductor device work.

Every value here is a CODATA definition or a standard SI-to-CGS conversion, not a
fitted number. If a constant is not exact, say so in the comment.
"""

# Exact, by SI definition of the 2019 redefinition.
ELEMENTARY_CHARGE_C = 1.602_176_634e-19  # C, exact
BOLTZMANN_J_PER_K = 1.380_649e-23  # J/K, exact

# Derived: k/q in volts per kelvin. 8.617333262e-5 V/K, exact from the above.
THERMAL_VOLTAGE_PER_K = BOLTZMANN_J_PER_K / ELEMENTARY_CHARGE_C

# Exact (c = 299 792 458 m/s, mu_0 = 4*pi*1e-7 H/m), converted to F/cm.
VACUUM_PERMITTIVITY_F_PER_CM = 8.854_187_812_8e-14

# Relative permittivity of silicon. NOT a constant of nature: it varies with
# doping, temperature and frequency (a few percent). Treated as a parameter
# with provenance, defaulting to this textbook value.
SILICON_RELATIVE_PERMITTIVITY = 11.7

# Intrinsic carrier concentration of silicon at 300 K, in cm^-3. Strongly
# temperature dependent (~T^1.5), so only valid at 300 K. This restriction is
# enforced as a validity gate rather than silently ignored.
SILICON_INTRINSIC_CM3_AT_300K = 1.0e10

# The single temperature at which the above constant is defined.
REFERENCE_TEMPERATURE_K = 300.0
