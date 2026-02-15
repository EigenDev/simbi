# =============================================================================
# power_spectrum.py
#
# thin wrapper around simbi.analysis spectrum functions.
# unpacks SimData, calls the pure function, packages into PlotData.
# supports both vector fields (kinetic energy spectrum) and
# scalar fields (e.g., entropy power spectrum).
# uses base level (level 0) only — the spectrum is smooth and correct
# up to the coarsest-level nyquist frequency.
# =============================================================================
from typing import Sequence

from simbi.analysis import (
    shell_averaged_scalar_spectrum,
    shell_averaged_spectrum,
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
) -> PlotData:
    """
    compute power spectrum from simulation checkpoint.

    if 3 fields are given, computes kinetic energy spectrum E(k).
    if 1 field is given, computes scalar power spectrum P(k).

    uses base level (level 0) only. the spectrum is reliable up to
    k_nyquist = pi / dx_coarse.
    """
    mesh = data.mesh
    dx = float(mesh.x1v[1] - mesh.x1v[0])
    is_vector = len(fields) >= 3

    if is_vector:
        vx = data.get_field(fields[0], level=0)
        vy = data.get_field(fields[1], level=0)
        vz = data.get_field(fields[2], level=0)
        k_centers, spectrum = shell_averaged_spectrum(vx, vy, vz, dx)
        name = r"$E(k)$"
    else:
        scalar = data.get_field(fields[0], level=0)
        k_centers, spectrum = shell_averaged_scalar_spectrum(scalar, dx)
        name = _scalar_name(fields[0])

    spectrum_field = FieldData(
        name=name,
        values=spectrum,
        domain=[k_centers],
        time=data.metadata.time,
        axis_names=[r"$k$"],
    )

    return PlotData(
        fields=[spectrum_field],
        time=data.metadata.time,
        dimensions=1,
    )
