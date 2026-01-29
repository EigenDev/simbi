# =============================================================================
# afterglow module
#
# synchrotron radiation calculations for relativistic hydrodynamic simulations.
# =============================================================================

from .generate import generate_from_files, read_events, read_metadata
from .mesh_expansion import expand_to_3d
from .plotting import (
    plot_lightcurve,
    plot_polarization,
    plot_skymap,
    plot_skymap_animation,
    plot_spectrum,
)
from .postprocess import (
    compute_lightcurve,
    compute_polarization,
    compute_skymap,
    compute_spectrum,
    lightcurve_t,
    metadata_t,
    photon_events_t,
    polarization_t,
    read_photon_events,
    skymap_t,
    spectrum_t,
)
from .scale_config import (
    STANDARD_SCALES,
    list_standard_scales,
    load_scale_config,
    make_blandford_mckee_scale,
    save_scale_config,
    scale_config_t,
)

__all__ = [
    # generation
    "generate_from_files",
    "read_events",
    "read_metadata",
    # postprocessing
    "read_photon_events",
    "compute_lightcurve",
    "compute_skymap",
    "compute_polarization",
    "compute_spectrum",
    # data structures
    "photon_events_t",
    "metadata_t",
    "lightcurve_t",
    "skymap_t",
    "polarization_t",
    "spectrum_t",
    # plotting
    "plot_lightcurve",
    "plot_skymap",
    "plot_skymap_animation",
    "plot_polarization",
    "plot_spectrum",
    # scale configuration
    "scale_config_t",
    "load_scale_config",
    "save_scale_config",
    "list_standard_scales",
    "make_blandford_mckee_scale",
    "STANDARD_SCALES",
    # mesh expansion
    "expand_to_3d",
]
