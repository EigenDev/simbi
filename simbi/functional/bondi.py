# =============================================================================
# bondi.py
#
# the bondi accretion coefficient lambda_c(gamma) -- the ONE python
# transcription, mirroring the rust `symbi_ib::bondi::accretion_coefficient`
# branch for branch and tolerance for tolerance:
# - isothermal (|gamma - 1| < 1e-5): e^1.5 / 4 exactly
# - monoatomic edge (|gamma - 5/3| < 1e-5): 1/4 exactly (the general
#   exponent is 0/0 there)
# - general: 0.25 * (2/(5-3*gamma))^((5-3*gamma)/(2*(gamma-1)))
#
# six divergent copies preceded this one, three of them hardcoding the
# isothermal branch as 1.12 (0.04 percent off e^1.5/4) and disagreeing on the
# edge tolerance by seven orders -- in the normalization every accretion
# measurement is reported against.
#
# usage:
#   from simbi.functional.bondi import accretion_coefficient
#   lam = accretion_coefficient(gamma)
# =============================================================================
import math


def accretion_coefficient(gamma: float) -> float:
    """bondi accretion coefficient lambda_c(gamma)."""
    if abs(gamma - 1.0) < 1e-5:
        return math.e**1.5 / 4.0
    if abs(gamma - 5.0 / 3.0) < 1e-5:
        return 0.25
    num = 5.0 - 3.0 * gamma
    return 0.25 * (2.0 / num) ** (num / (2.0 * gamma - 2.0))
