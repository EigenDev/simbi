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
