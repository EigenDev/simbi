# =============================================================================
# simbi/analysis/__init__.py
#
# pure physics/math analysis functions with no viz dependency.
# reusable from notebooks, scripts, or the viz pipeline.
# =============================================================================
from .bondi import accretion_coefficient, bondi_profiles
from .radial_profiles import (
    mass_flux_profile,
    momentum_equation_terms,
    radial_velocity_profile,
    reynolds_delta_v_profile,
    sound_speed_profile,
    spherical_profile,
    stitch_leaf_cells,
    time_average_profiles,
    turbulent_velocity_sq_profile,
)
from .spectrum import (
    angular_power_spectrum,
    angular_velocity_power_spectrum,
    composite_angular_power_spectrum,
    composite_angular_velocity_power_spectrum,
    composite_shell_averaged_scalar_spectrum,
    composite_shell_averaged_spectrum,
    lomb_scargle_fap_levels,
    lomb_scargle_psd,
    shell_averaged_scalar_spectrum,
    shell_averaged_spectrum,
    welch_lomb_scargle_psd,
)

__all__ = [
    "accretion_coefficient",
    "angular_power_spectrum",
    "angular_velocity_power_spectrum",
    "bondi_profiles",
    "composite_angular_power_spectrum",
    "composite_angular_velocity_power_spectrum",
    "composite_shell_averaged_spectrum",
    "composite_shell_averaged_scalar_spectrum",
    "lomb_scargle_fap_levels",
    "lomb_scargle_psd",
    "mass_flux_profile",
    "momentum_equation_terms",
    "radial_velocity_profile",
    "reynolds_delta_v_profile",
    "shell_averaged_scalar_spectrum",
    "shell_averaged_spectrum",
    "sound_speed_profile",
    "spherical_profile",
    "stitch_leaf_cells",
    "time_average_profiles",
    "turbulent_velocity_sq_profile",
    "welch_lomb_scargle_psd",
]
