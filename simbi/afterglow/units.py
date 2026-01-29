# =============================================================================
# units.py
#
# physical constants in cgs units from astropy.
# all values extracted from astropy.constants and astropy.units.
# =============================================================================

from astropy import constants as const
from astropy import units as u

# =============================================================================
# physical constants (cgs)
# =============================================================================

C_CGS = const.c.cgs.value  # 2.99792458e10 cm/s
H_CGS = const.h.cgs.value  # 6.62607015e-27 erg*s
K_B_CGS = const.k_B.cgs.value  # 1.380649e-16 erg/K
M_P_CGS = const.m_p.cgs.value  # 1.6726219e-24 g
M_E_CGS = const.m_e.cgs.value  # 9.1093837e-28 g
SIGMA_T_CGS = const.sigma_T.cgs.value  # 6.6524587e-25 cm^2
M_SUN_CGS = const.M_sun.cgs.value  # 1.98841e33 g
R_SUN_CGS = const.R_sun.cgs.value  # 6.96e10 cm
PC_CGS = const.pc.cgs.value  # 3.0857e18 cm
AU_CGS = const.au.cgs.value  # 1.496e13 cm

# =============================================================================
# unit conversions
# =============================================================================

JANSKY_CGS = u.Jy.to(u.erg / u.cm**2 / u.s / u.Hz)  # 1e-23 erg/cm^2/s/Hz
DAY_CGS = u.day.to(u.s)  # 86400 s
YEAR_CGS = u.year.to(u.s)  # 3.154e7 s
