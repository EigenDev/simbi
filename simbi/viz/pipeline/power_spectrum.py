# =============================================================================
# power_spectrum.py
#
# thin wrapper around simbi.analysis.shell_averaged_spectrum.
# unpacks SimData, calls the pure function, packages into PlotData.
# =============================================================================
from typing import Sequence

from simbi.analysis import shell_averaged_spectrum
from simbi.reader.adapter import SimData

from ..config import VisualizationConfig
from ..types import FieldData, PlotData


def create_power_spectrum_data(
    data: SimData,
    config: VisualizationConfig,
    fields: Sequence[str] = ("v1", "v2", "v3"),
) -> PlotData:
    """
    compute kinetic energy power spectrum from simulation checkpoint.

    uses base level (level 0) only. for AMR data, the spectrum reflects
    the coarsest resolution — fine-level structure is not captured.
    """
    vx = data.get_field(fields[0], level=0)
    vy = data.get_field(fields[1], level=0)
    vz = data.get_field(fields[2], level=0)

    mesh = data.mesh
    x1v = mesh.x1v
    dx = float(x1v[1] - x1v[0])

    k_centers, e_k = shell_averaged_spectrum(vx, vy, vz, dx)

    spectrum_field = FieldData(
        name=r"$E(k)$",
        values=e_k,
        domain=[k_centers],
        time=data.metadata.time,
        axis_names=[r"$k$"],
    )

    return PlotData(
        fields=[spectrum_field],
        time=data.metadata.time,
        dimensions=1,
    )
