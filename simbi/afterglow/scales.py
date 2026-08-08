from astropy import units, constants
from typing import TypeVar, Type
from math import pi

T = TypeVar("T")
USER_SCALES = {}


def user_scale(cls: Type[T]) -> None:
    class_name = "".join(
        ["-" + c.lower() if c.isupper() else c for c in cls.__name__]
    ).lstrip("-")
    USER_SCALES[class_name] = cls


@user_scale
class Solar:
    length_scale = constants.R_sun.cgs
    rho_scale = (constants.M_sun / (4.0 / 3.0 * pi * length_scale**3)).cgs
    e_scale = (constants.M_sun * constants.c**2).cgs
    pre_scale = (e_scale / (4.0 / 3.0 * pi * length_scale**3)).cgs
    time_scale = (length_scale / constants.c).cgs


@user_scale
class BlandfordMckee:
    e_scale = 1e53 * units.erg
    rho_scale = 1.0 * constants.m_p.cgs / units.cm**3
    length_scale = ((e_scale / (rho_scale * constants.c.cgs**2)) ** (1 / 3)).cgs
    time_scale = length_scale / constants.c.cgs
    pre_scale = e_scale / length_scale**3


@user_scale
class Gw170817:
    """a GW170817-like nearby merger, for redeploying a scale-free blast to VLBI scales.

    the hydrodynamics of a blandford-mckee sector is SCALE-FREE: nothing in the code-unit
    solution knows E or n. they enter only here, through the length scale, so one
    simulation can be read out as a cosmological GRB or a nearby merger with no new
    compute -- which is the whole reason a synthetic-image prediction can be compared
    against an actual VLBI measurement without re-running anything.

    E and n are the two most uncertain numbers for GW170817. these are the fiducial
    afterglow-fit values (E_iso ~ 1e52 erg into n ~ 1e-2 cm^-3); the inferred density
    ranges over ~1e-4 to 1e-2, and since every length goes as (E/n)^(1/3) the low-density
    end stretches the source by ~4.6x. vary them deliberately rather than trusting this
    one entry -- and note the DIRECTION: a lower density makes the image BIGGER and its
    evolution SLOWER, both of which help detectability.
    """

    e_scale = 1e52 * units.erg
    rho_scale = 1e-2 * constants.m_p.cgs / units.cm**3
    length_scale = ((e_scale / (rho_scale * constants.c.cgs**2)) ** (1 / 3)).cgs
    time_scale = length_scale / constants.c.cgs
    pre_scale = e_scale / length_scale**3


# ---------------------------
# Rest of user scales here
# ---------------------------


def get_scale_model(name: str) -> T:
    try:
        return USER_SCALES[name]
    except KeyError:
        valid_scales = "".join([f"> {a}\n" for a in USER_SCALES.keys()])
        raise ValueError(
            f"{name} is not a valid scale model. Available scale models are:\n{valid_scales}"
        )
