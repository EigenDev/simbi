# =============================================================================
# power_spectrum.py
#
# thin wrapper around simbi.analysis spectrum functions.
# unpacks SimData, calls the pure function, packages into PlotData.
# supports both vector fields (kinetic energy spectrum) and
# scalar fields (e.g., entropy power spectrum).
#
# modes:
#   - default: uses base level (level 0) — reliable up to k_nyquist(L0)
#   - composite: uses finest FMR level — extends to k_nyquist(L_finest)
# =============================================================================
from typing import Optional, Sequence

from simbi.analysis import (
    shell_averaged_scalar_spectrum,
    shell_averaged_spectrum,
)
from simbi.analysis.spectrum import (
    composite_angular_power_spectrum,
    composite_angular_velocity_power_spectrum,
    composite_shell_averaged_scalar_spectrum,
    composite_shell_averaged_spectrum,
)
from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..types import FieldData, PlotData

_SCALAR_LABELS = {
    "entropy-measure": r"$P_\kappa(k)$",
    "entropy-gradient": r"$P_{|\nabla\kappa|}(k)$",
    "rho": r"$P_\rho(k)$",
    "p": r"$P_p(k)$",
}


def _scalar_name(field: str) -> str:
    return _SCALAR_LABELS.get(field, rf"$P_{{\mathrm{{{field}}}}}(k)$")


def create_power_spectrum_data(
    data: SimData,
    config: VisualizationConfig,
    fields: Sequence[str] = ("v1", "v2", "v3"),
    subtract_radial_mean: bool = False,
    use_composite: bool = False,
) -> PlotData:
    """
    compute power spectrum from simulation checkpoint.

    if 3 fields are given, computes kinetic energy spectrum E(k).
    if 1 field is given, computes scalar power spectrum P(k).

    modes:
      - default: level 0 only, reliable up to k_nyquist = pi / dx_coarse
      - composite: finest FMR level, extends to k_nyquist = pi / dx_fine
    """
    is_vector = len(fields) >= 3

    if use_composite and data.num_levels > 1:
        if is_vector:
            k_centers, spectrum, _ = composite_shell_averaged_spectrum(
                data, fields, subtract_mean=subtract_radial_mean
            )
            name = r"$E(k)$"
        else:
            k_centers, spectrum, _ = composite_shell_averaged_scalar_spectrum(
                data, fields[0], subtract_mean=subtract_radial_mean
            )
            name = _scalar_name(fields[0])
    else:
        mesh = data.mesh
        dx = float(mesh.x1v[1] - mesh.x1v[0])
        if is_vector:
            vx = data.get_field(fields[0], level=0)
            vy = data.get_field(fields[1], level=0)
            vz = data.get_field(fields[2], level=0)
            k_centers, spectrum = shell_averaged_spectrum(
                vx, vy, vz, dx, subtract_mean=subtract_radial_mean
            )
            name = r"$E(k)$"
        else:
            scalar = data.get_field(fields[0], level=0)
            k_centers, spectrum = shell_averaged_scalar_spectrum(
                scalar, dx, subtract_mean=subtract_radial_mean
            )
            name = _scalar_name(fields[0])

    xlabel = r"$k\;[\mathrm{rad}/a]$"

    spectrum_field = FieldData(
        name=name,
        values=spectrum,
        domain=[k_centers],
        time=data.metadata.time,
        axis_names=[xlabel],
    )

    return PlotData(
        fields=[spectrum_field],
        time=data.metadata.time,
        dimensions=1,
    )


_ANGULAR_SCALAR_LABELS = {
    "entropy-measure": r"$C_\ell^{(\kappa)}$",
    "entropy-gradient": r"$C_\ell^{(|\nabla\kappa|)}$",
    "rho": r"$C_\ell^{(\rho)}$",
    "p": r"$C_\ell^{(p)}$",
}


def create_angular_spectrum_data(
    data: SimData,
    config: VisualizationConfig,
    field: str = "rho",
    fields: Optional[Sequence[str]] = None,
    radii: Optional[Sequence[float]] = None,
    n_shells: int = 5,
    n_theta: int = 64,
    n_phi: int = 128,
    subtract_mean: bool = False,
) -> PlotData:
    """compute angular power spectrum C_l from simulation checkpoint.

    uses spherical harmonic decomposition (via 2D FFT on equirectangular
    grid) on radial shells of stitched leaf-cell data.

    if fields has 3 entries (velocity components), computes the non-radial
    velocity spectrum C_l(v_theta) + C_l(v_phi). otherwise computes the
    scalar angular spectrum of the single field.

    if radii is None, auto-selects n_shells log-spaced shells.
    """
    is_vector = fields is not None and len(fields) >= 3

    if is_vector:
        ell, c_ell, r_mean = composite_angular_velocity_power_spectrum(
            data, velocity_fields=fields,
            radii=radii, n_shells=n_shells,
            n_theta=n_theta, n_phi=n_phi,
            subtract_mean=subtract_mean,
        )
        name = r"$C_\ell^{(v_\perp)}$"
    else:
        ell, c_ell, r_mean = composite_angular_power_spectrum(
            data, field,
            radii=radii, n_shells=n_shells,
            n_theta=n_theta, n_phi=n_phi,
            subtract_mean=subtract_mean,
        )
        name = _ANGULAR_SCALAR_LABELS.get(
            field, rf"$C_\ell^{{(\mathrm{{{field}}})}}$"
        )

    # convert ell -> k = ell / r_mean so the x-axis matches Cartesian spectra
    k = ell / r_mean
    xlabel = r"$k\;[\mathrm{rad}/a]$"

    spectrum_field = FieldData(
        name=name,
        values=c_ell,
        domain=[k],
        time=data.metadata.time,
        axis_names=[xlabel],
    )

    return PlotData(
        fields=[spectrum_field],
        time=data.metadata.time,
        dimensions=1,
    )
