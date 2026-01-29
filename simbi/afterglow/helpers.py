# =============================================================================
# helpers.py
#
# utility functions for afterglow calculations.
# =============================================================================

from astropy import units
from astropy.cosmology import FlatLambdaCDM

_cosmo = FlatLambdaCDM(
    H0=70 * units.km / units.s / units.Mpc, Tcmb0=2.725 * units.K, Om0=0.3
)


def get_dL(z: float):
    """
    compute luminosity distance for given redshift.

    args:
        z: redshift (z=0 returns 1e28 cm as placeholder)

    returns:
        luminosity distance in cgs units
    """
    if z > 0:
        return _cosmo.luminosity_distance(z).cgs
    else:
        return 1e28 * units.cm
