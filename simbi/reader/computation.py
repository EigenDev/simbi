# =============================================================================
# computation.py
#
# derived field computation for simulation data.
# contains all physics calculations needed for post-processing.
# =============================================================================
from enum import Enum, IntEnum
from typing import Any, Callable, Sequence

import numpy as np

from ..types import Array
from .io import Checkpoint

ComputeFunc = Callable[[dict[str, Any]], Any]


class FieldComputationError(ValueError):
    """a derived field could not be computed from the checkpoint's stored
    datasets (e.g. a pressure-dependent field on a file with no pressure).
    deliberately not a KeyError: callers that skip fields absent from a
    level must still surface this as a hard failure."""


# =============================================================================
# physics primitives
# =============================================================================
class VectorComponent(IntEnum):
    X1 = 0
    X2 = 1
    X3 = 2


class VectorMode(Enum):
    Magnitude = "magnitude"
    All = "all"


def _dot_product(a: Sequence[Array], b: Sequence[Array]) -> Array:
    """dot product of two vector fields."""
    return np.sum([a[ii] * b[ii] for ii in range(len(a))], axis=0)


def _vector_dof(regime: str, ndim: int) -> int:
    """number of stored velocity / magnetic vector components. an mhd field is a 3-vector no
    matter the spatial dimensionality: a 2.5D (D=2) or 1.75D (D=1) run still evolves and stores the
    out-of-plane components (the toroidal v_phi and b_phi), so reading only the ndim in-plane ones
    drops real data — zeroing the magnetic pressure of a purely toroidal field and corrupting the
    lorentz factor and v.B of any rotating flow. a hydro velocity has one component per axis."""
    return 3 if "mhd" in regime else ndim


def lorentz_factor(
    velocity: Sequence[Array], regime: str, using_gamma_beta: bool = False
) -> Array | float:
    """compute lorentz factor from velocity."""
    vsquared = _dot_product(velocity, velocity)
    if regime != "newtonian" and np.any(vsquared >= 1.0):
        raise ValueError("velocity exceeds speed of light")

    if regime == "newtonian":
        return 1.0
    elif not using_gamma_beta:
        return np.asarray(1.0 / np.sqrt(1.0 - vsquared))
    else:
        return np.asarray(np.sqrt(1.0 + vsquared))


def four_velocity(
    velocity: Sequence[Array], regime: str, component: int
) -> Array:
    """compute four-velocity component."""
    W = lorentz_factor(velocity, regime)
    return np.asarray(velocity[component] * W)


def spec_enthalpy(
    adiabatic_index: float, rho: Array, pressure: Array, regime: str
) -> Array | float:
    """compute specific enthalpy."""
    if regime == "newtonian":
        if adiabatic_index == 1.0:
            return 1.0 + pressure / rho
        return 1.0
    return 1.0 + adiabatic_index * pressure / (rho * (adiabatic_index - 1.0))


def _enthalpy(
    rho: Array, pre: Array, gamma: float, regime: str
) -> Array | float:
    """compute enthalpy per particle."""
    if regime == "newtonian":
        return 1.0
    return 1.0 + gamma * pre / (rho * (gamma - 1.0))


def labframe_density(
    rho: Array, velocity: Sequence[Array], regime: str
) -> Array:
    """compute lab-frame density."""
    return rho * lorentz_factor(velocity, regime)


def labframe_energy_density(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "newtonian",
) -> Array:
    """compute lab-frame energy density."""
    vsq = sum(v**2 for v in vel)

    if regime == "newtonian":
        if gamma == 1.0:
            return pre / rho
        return pre / (gamma - 1.0) + 0.5 * rho * vsq

    W = 1.0 / np.sqrt(1.0 - vsq)
    h = _enthalpy(rho, pre, gamma, regime)

    if regime == "rhd":
        return np.asarray(rho * W**2 * h - pre - rho * W)

    if regime == "rmhd":
        bsq = sum(b**2 for b in bfield)
        vdb = _dot_product(vel, bfield)
        return np.asarray(
            rho * W**2 * h - pre - rho * W + 0.5 * (bsq + vsq * bsq - vdb**2)
        )

    raise NotImplementedError(f"regime '{regime}' not implemented")


def labframe_momentum(
    rho: Array,
    pre: Array,
    vel: Sequence[Array],
    bfield: Sequence[Array],
    gamma: float,
    regime: str = "newtonian",
    mode: VectorMode | VectorComponent = VectorMode.All,
) -> Array:
    """compute lab-frame momentum."""
    if regime == "newtonian":
        mom_vec = np.asarray(rho) * np.asarray(vel)
    elif regime in ("rhd", "rmhd"):
        h = _enthalpy(rho, pre, gamma, regime)
        D = labframe_density(rho, vel, regime).squeeze()
        W = lorentz_factor(vel, regime)
        if isinstance(h, np.ndarray):
            h = h.squeeze()
        if isinstance(D, np.ndarray):
            D = D.squeeze()
        if isinstance(W, np.ndarray):
            W = W.squeeze()

        magnetic_part: Array | float = 0.0
        vel = [v.squeeze() for v in vel]
        if regime == "rmhd":
            bsq = np.array(sum(b**2 for b in bfield), dtype=float)
            vdb = _dot_product(vel, bfield)
            magnetic_part = np.array(
                [bsq * v - vdb * b for v, b in zip(vel, bfield)]
            ).squeeze()

        mom_vec = D * h * W * np.asarray(vel) + magnetic_part
    else:
        raise NotImplementedError(f"regime '{regime}' not implemented")

    if mode == VectorMode.Magnitude:
        return np.asarray(np.sqrt(np.sum(mom_vec**2, axis=0)))
    elif mode == VectorMode.All:
        return mom_vec
    else:
        return np.asarray(mom_vec[int(mode.value)])


def magnetic_pressure(
    bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """compute magnetic pressure."""
    bsq: Array = np.sum([b**2 for b in bfields], axis=0)
    if regime == "rmhd":
        W = 1.0 / np.sqrt(1.0 - _dot_product(velocity, velocity))
        vdb = _dot_product(velocity, bfields)
        bsq = bsq / W**2 + vdb**2
    return 0.5 * bsq


def total_pressure(
    pre: Array, bfields: Sequence[Array], velocity: Sequence[Array], regime: str
) -> Array:
    """compute total pressure (gas + magnetic)."""
    if "mhd" not in regime:
        return pre
    return pre + magnetic_pressure(bfields, velocity, regime)


def enthalpy_density(
    rho: Array,
    pre: Array,
    bfields: Sequence[Array],
    velocity: Sequence[Array],
    adiabatic_index: float,
    regime: str = "newtonian",
) -> Array:
    """compute enthalpy density."""
    if regime == "newtonian":
        return rho
    elif regime == "rhd":
        return rho * _enthalpy(rho, pre, adiabatic_index, regime)
    elif regime == "rmhd":
        return rho * _enthalpy(
            rho, pre, adiabatic_index, regime
        ) + 2.0 * magnetic_pressure(bfields, velocity, regime)
    raise NotImplementedError(f"regime '{regime}' not implemented")


def magnetization(rho: Array, bfields: Sequence[Array]) -> Array:
    """compute magnetization sigma = b^2 / rho."""
    return np.asarray(_dot_product(bfields, bfields) / rho)


# =============================================================================
# computation pipeline factory
# =============================================================================
def _broadcast_cell_centers(mesh: Any, ndim: int) -> tuple[Array, ...]:
    """cell-center coordinate arrays reshaped to broadcast along their storage axis.

    field arrays are stored in (x3, x2, x1) = (z, y, x) order, so x1 varies along
    the last array axis, x2 the second-to-last, and x3 the first. a bare 1d
    coordinate array broadcasts against the last axis only: correct for x1, but it
    transposes x2/x3 against the data — silent on a square grid, a shape error on a
    non-square one. each returned array carries singleton axes so it multiplies
    against its own storage axis."""
    x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
    if ndim == 1:
        return (x,)
    y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])
    if ndim == 2:
        return x[None, :], y[:, None]
    z = 0.5 * (mesh.x3v[1:] + mesh.x3v[:-1])
    return x[None, None, :], y[None, :, None], z[:, None, None]


def create_computation_pipeline(data: Checkpoint) -> dict[str, Any]:
    """
    create a pipeline of derived field computations.

    returns a dict mapping field names to derived-field objects.
    each derived-field exposes evaluate(level) -> array.
    """
    ndim = data.metadata.dimensions
    regime = data.metadata.regime
    gamma = data.metadata.gamma

    # level context object passed into compute functions.
    # it behaves like a mapping for field access and exposes mesh as attribute.
    class level_context_t:
        def __init__(self, fields: dict[str, Array], mesh):
            self._fields = fields
            self.mesh = mesh

        def __getitem__(self, key: str) -> Array:
            return self._fields[key]

        def get(self, key: str, default=None):
            return self._fields.get(key, default)

        def __contains__(self, key: str) -> bool:
            # without this, `key in ctx` falls back to integer __getitem__
            # iteration and raises KeyError: 0 — which broke every b*_mean
            # derived field on MHD checkpoints.
            return key in self._fields

        def __iter__(self):
            return iter(self._fields)

        def __getattr__(self, name: str):
            # forward attribute access to mesh for convenience
            if hasattr(self.mesh, name):
                return getattr(self.mesh, name)
            raise AttributeError(name)

    # simple wrapper representing a derived field
    class derived_field_t:
        def __init__(
            self,
            name: str,
            func: Callable[..., Array],
            requires_composite: bool = False,
        ):
            self._name = name
            self._func = func
            self._requires_composite = requires_composite

        def evaluate(self, level: int) -> Array:
            # check if this field requires composite/base-level-only computation
            if level > 0 and self._requires_composite:
                raise ValueError(
                    f"field '{self._func.__name__}' requires base level computation. "
                    f"use level=0 or enable composite view"
                )

            # assemble fields and mesh for the requested level
            from simbi.reader.adapter import MeshAdapter

            level_data = data.levels[level]

            # derived fields are computed on a single contiguous partition
            if level_data.num_partitions != 1:
                raise NotImplementedError(
                    "multi-partition support not implemented"
                )

            partition = level_data.partitions[0]

            # create mesh adapter with coordinate generation
            owned_domain = None
            if level > 0:
                owned_domain = (
                    partition.owned_domain.start,
                    partition.owned_domain.fin,
                )

            mesh = MeshAdapter(level_data.mesh, owned_domain=owned_domain)
            halo = mesh.halo_radius

            # build base field mapping from FieldData objects:
            # - primitives: use interior(halo).data to remove ghost/halo cells
            # - magnetic: use raw face-centered .data
            mapping: dict[str, Array] = {}

            # primitives (cell-centered)
            for name, field in partition.hydro.primitives.items():
                mapping[name] = field.interior(halo).data

            # magnetic fields (face-centered) - keep raw data; viz will handle centering/averaging
            # magnetic fields (face-centered) - guard against missing magnetic dict
            if partition.hydro.magnetic is not None:
                for name, field in partition.hydro.magnetic.items():
                    mapping[name] = field.data

            # the isothermal regime carries no pressure dataset; its eos
            # closes with p = cs^2 rho at the constant metadata sound speed,
            # so p-dependent derived fields stay computable.
            if "p" not in mapping and regime == "isothermal":
                cs = data.metadata.sound_speed
                if cs is not None:
                    mapping["p"] = cs * cs * mapping["rho"]

            # provide context to compute function
            ctx = level_context_t(mapping, mesh)
            try:
                result = self._func(ctx)
            except KeyError as exc:
                missing = exc.args[0] if exc.args else "?"
                hint = ""
                if missing == "p" and regime == "isothermal":
                    hint = (
                        " (this isothermal checkpoint predates the"
                        " 'sound_speed' metadata attr, so p = cs^2 rho"
                        " cannot be reconstructed)"
                    )
                raise FieldComputationError(
                    f"derived field '{self._name}' needs base field"
                    f" '{missing}', which this {regime} checkpoint does not"
                    f" store; available: {sorted(mapping)}{hint}"
                ) from exc
            return np.asarray(result)

    def get_velocities(fields: dict[str, Array]) -> list[Array]:
        # gather the full 3-component vector DOF for mhd: the out-of-plane v_phi is a stored,
        # evolved field in 2.5D / 1.75D runs and enters the lorentz factor and v.B. callers that
        # need a fixed 3-vector zero-pad the (genuinely absent) tail.
        return [fields[f"v{ii}"] for ii in range(1, _vector_dof(regime, ndim) + 1)]

    def get_b_fields(fields: dict[str, Array]) -> list[Array]:
        """
        convert face-centered magnetic fields to cell-centered components.

        input:
          fields: mapping that may contain 'b1', 'b2', 'b3' which are face-centered.
        returns:
          list of cell-centered arrays [b1c, b2c, b3c] each shaped like fields['rho'].
        note:
          face-centered convention:
            b1 shape differs from rho by +1 along the x1-normal axis (fastest varying / last axis)
            b2 differs by +1 along x2-normal (middle axis)
            b3 differs by +1 along x3-normal (slowest / first axis)
          the field is a 3-vector: in a 2.5D / 1.75D run the out-of-plane component (the toroidal
          b_phi) is cell-centered (no face in the missing axis) and is accepted as-is below.
        """
        # gather the full 3-component magnetic vector DOF for mhd: missing entries resolve to
        # zeros. reading only b1..b_ndim drops the out-of-plane b_phi, which zeros the magnetic
        # pressure of a purely toroidal field.
        raw = [fields.get(f"b{ii}", None) for ii in range(1, _vector_dof(regime, ndim) + 1)]
        rho_shape = tuple(np.asarray(fields["rho"]).shape)

        def _average_faces_to_cells(face_arr: Array, axis: int) -> Array:
            """
            average neighboring face values along `axis` to produce cell-centered values.
            expects face_arr to have length rho_shape[axis] + 1 along `axis`.
            returns array with one fewer element along that axis (i.e., matches rho_shape).
            """
            # build slicers for left and right neighbors
            slicer_left = [slice(None)] * face_arr.ndim
            slicer_right = [slice(None)] * face_arr.ndim
            slicer_left[axis] = slice(0, -1)
            slicer_right[axis] = slice(1, None)
            left = face_arr[tuple(slicer_left)]
            right = face_arr[tuple(slicer_right)]
            return 0.5 * (left + right)

        out: list[Array] = []
        for comp, view in enumerate(raw):
            # missing component -> zeros
            if view is None:
                out.append(np.zeros(rho_shape, dtype=float))
                continue

            arr = np.asarray(view)

            # if already cell-centered, accept as-is when shapes match
            if arr.shape == rho_shape:
                out.append(arr)
                continue

            # compute expected axis for this component:
            # b1 normal -> last axis, b2 -> second-to-last, b3 -> third-to-last
            axis = arr.ndim - 1 - comp
            if axis < 0 or axis >= arr.ndim:
                out.append(np.zeros(rho_shape, dtype=float))
                continue

            # if face dimension equals rho dimension + 1 along axis, average
            if arr.shape[axis] == rho_shape[axis] + 1:
                try:
                    cell = _average_faces_to_cells(arr, axis)
                    # sometimes result may have singleton dimensions; attempt to squeeze safely
                    cell = np.squeeze(cell)
                    # if shape matches, accept, otherwise try to broadcast/reshape fallback
                    if tuple(cell.shape) == rho_shape:
                        out.append(cell)
                        continue
                    # try to pad/truncate to exact shape if possible (best-effort)
                    # fallthrough to general attempt below
                except Exception:
                    # fall back to zeros for this component
                    out.append(np.zeros(rho_shape, dtype=float))
                    continue

            # best-effort: try averaging along the computed axis even if sizes don't match exactly
            try:
                slicer_left = [slice(None)] * arr.ndim
                slicer_right = [slice(None)] * arr.ndim
                slicer_left[axis] = slice(0, -1)
                slicer_right[axis] = slice(1, None)
                left = arr[tuple(slicer_left)]
                right = arr[tuple(slicer_right)]
                cell = 0.5 * (left + right)
                if tuple(cell.shape) == rho_shape:
                    out.append(cell)
                else:
                    # cannot reconcile shapes: return zeros to avoid silent mis-shapes
                    out.append(np.zeros(rho_shape, dtype=float))
            except Exception:
                out.append(np.zeros(rho_shape, dtype=float))

        return [b.squeeze() for b in out]

    # basic derived fields (functions unchanged, they accept a mapping-like ctx)
    def compute_W(fields: dict[str, Array]) -> Array | float:
        return lorentz_factor(get_velocities(fields), regime)

    def compute_D(fields: dict[str, Array]) -> Array:
        return fields["rho"] * lorentz_factor(get_velocities(fields), regime)

    def compute_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return fields["v1"]
        return np.sqrt(sum(fields[f"v{ii}"] ** 2 for ii in range(1, ndim + 1)))

    def compute_four_velocity_magnitude(fields: dict[str, Array]) -> Array:
        if ndim == 1:
            return four_velocity(get_velocities(fields), regime, 0)
        v_mag = compute_velocity_magnitude(fields)
        W = lorentz_factor(get_velocities(fields), regime)
        return v_mag * W

    def compute_u_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            return four_velocity(get_velocities(fields), regime, component)

        return _compute

    def compute_momentum_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        def _compute(fields: dict[str, Array]) -> Array:
            return labframe_momentum(
                fields["rho"],
                fields["p"],
                get_velocities(fields),
                get_b_fields(fields),
                gamma,
                regime,
                VectorComponent(component),
            )

        return _compute

    def compute_b_labframe_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        """cell-centered lab-frame magnetic field component b_i. face-centered values are
        averaged to the cell center by get_b_fields, which also accepts the out-of-plane
        component of a 2.5d / 1.75d run as-is (it carries no face in the missing axis)."""

        def _compute(fields: dict[str, Array]) -> Array:
            return np.asarray(get_b_fields(fields)[component])

        return _compute

    def compute_b_four_component(
        component: int,
    ) -> Callable[[dict[str, Array]], Array]:
        """spatial component of the comoving magnetic four-vector,
        b^i = B^i / W + W (v.B) v^i, with B the cell-centered lab-frame field, v the
        three-velocity, and W the lorentz factor. its norm b_mu b^mu = |B|^2 / W^2 +
        (v.B)^2 equals twice the relativistic magnetic pressure."""

        def _compute(fields: dict[str, Array]) -> Array:
            vel = get_velocities(fields)
            b = get_b_fields(fields)
            W = lorentz_factor(vel, regime)
            vdb = _dot_product(vel, b)
            return np.asarray(b[component] / W + W * vdb * vel[component])

        return _compute

    def compute_energy(fields: dict[str, Array]) -> Array:
        return labframe_energy_density(
            fields["rho"],
            fields["p"],
            get_velocities(fields),
            get_b_fields(fields),
            gamma,
            regime,
        )

    def compute_enthalpy(fields: dict[str, Array]) -> Array | float:
        return spec_enthalpy(gamma, fields["rho"], fields["p"], regime)

    def compute_enthalpy_density(fields: dict[str, Array]) -> Array:
        return enthalpy_density(
            fields["rho"],
            fields["p"],
            get_b_fields(fields),
            get_velocities(fields),
            gamma,
            regime,
        )

    def compute_magnetization(fields: dict[str, Array]) -> Array:
        return magnetization(fields["rho"], get_b_fields(fields))

    def compute_total_pressure(fields: dict[str, Array]) -> Array:
        return total_pressure(
            fields["p"], get_b_fields(fields), get_velocities(fields), regime
        )

    def compute_magnetic_pressure(fields: dict[str, Array]) -> Array:
        return magnetic_pressure(
            get_b_fields(fields), get_velocities(fields), regime
        )

    def compute_magnetic_energy(fields: dict[str, Array]) -> Array:
        """lab-frame magnetic energy density 1/2 |B|^2 (cell-centered B). distinct from `pmag`,
        which carries the relativistic (comoving) magnetic pressure for rmhd."""
        b = get_b_fields(fields)
        return np.asarray(0.5 * _dot_product(b, b))

    def compute_mach_number(fields: dict[str, Array]) -> Array:
        v_mag = compute_velocity_magnitude(fields)
        cs = np.sqrt(gamma * fields["p"] / fields["rho"])
        return np.asanyarray(v_mag / cs)

    def compute_chi_density(fields: dict[str, Array]) -> Array:
        return (
            fields["rho"]
            * fields["chi"]
            * lorentz_factor(get_velocities(fields), regime)
        )

    def angular_momentum_density(level_data: dict[str, Array]) -> Array:
        mesh = getattr(level_data, "mesh")
        coords = _broadcast_cell_centers(mesh, ndim)
        x, y = coords[0], coords[1]
        Sx = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(0),
        )
        Sy = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(1),
        )
        return np.asarray(x * Sy - y * Sx)

    def specific_angular_momentum(level_data: dict[str, Array]) -> Array:
        mesh = getattr(level_data, "mesh")
        coords = _broadcast_cell_centers(mesh, ndim)
        x, y = coords[0], coords[1]
        Sx = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(0),
        )
        Sy = labframe_momentum(
            level_data["rho"],
            level_data["p"],
            get_velocities(level_data),
            get_b_fields(level_data),
            gamma,
            regime,
            VectorComponent(1),
        )
        Lz = x * Sy - y * Sx
        den = level_data["rho"]

        if den.ndim == 3:
            dz = (mesh.x3v[1:] - mesh.x3v[:-1])[:, None, None]
            Sigma = np.sum(den * dz, axis=0)
            Lz_int = np.sum(Lz * dz, axis=0)
            return np.asarray(Lz_int / (Sigma + np.finfo(float).tiny))
        return np.asarray(Lz / (den + np.finfo(float).tiny))

    def surface_density(level_data: dict[str, Any]) -> Array:
        den = level_data["rho"]
        if den.ndim == 3:
            mesh = getattr(level_data, "mesh")
            dz = (mesh.x3v[1:] - mesh.x3v[:-1])[:, None, None]
            return np.asarray(np.sum(den * dz, axis=0))
        return np.asarray(den)

    def mass_flux(level_data: dict[str, Any]) -> Array:
        mesh = getattr(level_data, "mesh")
        coords = _broadcast_cell_centers(mesh, ndim)
        vx, vy = level_data["v1"], level_data["v2"]

        if vx.ndim == 3:
            x, y, z = coords
            r = np.sqrt(x**2 + y**2 + z**2)
            vz = level_data["v3"]
            vr = (x * vx + y * vy + z * vz) / (r + np.finfo(float).tiny)
        else:
            x, y = coords[0], coords[1]
            r = np.sqrt(x**2 + y**2)
            vr = (x * vx + y * vy) / (r + np.finfo(float).tiny)

        return np.asarray(4.0 * np.pi * r**2 * level_data["rho"] * vr)

    def compute_divergence(level_data: dict[str, Array]) -> Array:
        mesh = getattr(level_data, "mesh")
        vx, vy = level_data["v1"], level_data["v2"]
        x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
        y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])

        dvx_dx = np.gradient(vx, x, axis=ndim - 1)
        dvy_dy = np.gradient(vy, y, axis=ndim - 2)

        if ndim == 3:
            vz = level_data["v3"]
            z = 0.5 * (mesh.x3v[1:] + mesh.x3v[:-1])
            dvz_dz = np.gradient(vz, z, axis=0)
            return np.asarray(dvx_dx + dvy_dy + dvz_dz)
        return np.asarray(dvx_dx + dvy_dy)

    def compute_vorticity_z(level_data: dict[str, Array]) -> Array:
        mesh = getattr(level_data, "mesh")
        vx, vy = level_data["v1"], level_data["v2"]
        x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
        y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])

        dvy_dx = np.gradient(vy, x, axis=ndim - 1)
        dvx_dy = np.gradient(vx, y, axis=ndim - 2)
        return np.asarray(dvy_dx - dvx_dy)

    def compute_vorticity_magnitude(level_data: dict[str, Array]) -> Array:
        """
        vorticity magnitude: |\nabla \times \\mathbf{v}|

        computes all three components:
        \\omega_x = \\partial v_z / \\partial y - \\partial v_y / \\partial z
        \\omega_y = \\partial v_x / \\partial z - \\partial v_z / \\partial x
        \\omega_z = \\partial v_y / \\partial x - \\partial v_x / \\partial y

        returns: sqrt(\\omega_x^2 + \\omega_y^2 + \\omega_z^2)
        """
        mesh = getattr(level_data, "mesh")
        # get_velocities returns ndim entries: pad the out-of-plane components
        # with zeros so the 2d (and 1d) branches below unpack a full 3-vector;
        # unpacking fewer than three would raise a bare ValueError.
        vels = get_velocities(level_data)
        while len(vels) < 3:
            vels.append(np.zeros_like(vels[0]))
        vx, vy, vz = vels

        # cell-centered coordinates
        coords = []
        if ndim >= 1:
            x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
            coords.append(x)
        if ndim >= 2:
            y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])
            coords.append(y)
        if ndim >= 3:
            z = 0.5 * (mesh.x3v[1:] + mesh.x3v[:-1])
            coords.append(z)

        # compute vorticity components based on dimensionality
        omega_x = np.zeros_like(vx)
        omega_y = np.zeros_like(vx)
        omega_z = np.zeros_like(vx)

        if ndim == 2:
            # 2d case: only omega_z is non-zero
            # omega_z = dv_y/dx - dv_x/dy
            dvy_dx = np.gradient(vy, coords[0], axis=ndim - 1)
            dvx_dy = np.gradient(vx, coords[1], axis=ndim - 2)
            omega_z = dvy_dx - dvx_dy

        elif ndim == 3:
            # 3d case: all three components
            # omega_x = dv_z/dy - dv_y/dz
            dvz_dy = np.gradient(vz, coords[1], axis=ndim - 2)
            dvy_dz = np.gradient(vy, coords[2], axis=ndim - 3)
            omega_x = dvz_dy - dvy_dz

            # omega_y = dv_x/dz - dv_z/dx
            dvx_dz = np.gradient(vx, coords[2], axis=ndim - 3)
            dvz_dx = np.gradient(vz, coords[0], axis=ndim - 1)
            omega_y = dvx_dz - dvz_dx

            # omega_z = dv_y/dx - dv_x/dy
            dvy_dx = np.gradient(vy, coords[0], axis=ndim - 1)
            dvx_dy = np.gradient(vx, coords[1], axis=ndim - 2)
            omega_z = dvy_dx - dvx_dy

        # magnitude
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        return np.asarray(omega_mag)

    def compute_q_criterion(level_data: dict[str, Array]) -> Array:
        """
        q-criterion for vortex identification.

        q = 0.5 * (||\\Omega||^2 - ||S||^2)

        where:
        \\Omega = antisymmetric part of \nabla v = (\nabla v - \nabla v^T) / 2
        S = symmetric part of \nabla v = (\nabla v + \nabla v^T) / 2

        q > 0: rotation-dominated regions (vortex cores)
        q < 0: strain-dominated regions (shear layers)
        """
        mesh = getattr(level_data, "mesh")
        # get_velocities returns ndim entries: pad the out-of-plane components
        # with zeros so the 2d (and 1d) branches below unpack a full 3-vector;
        # unpacking fewer than three would raise a bare ValueError.
        vels = get_velocities(level_data)
        while len(vels) < 3:
            vels.append(np.zeros_like(vels[0]))
        vx, vy, vz = vels

        # cell-centered coordinates
        coords = []
        if ndim >= 1:
            x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
            coords.append(x)
        if ndim >= 2:
            y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])
            coords.append(y)
        if ndim >= 3:
            z = 0.5 * (mesh.x3v[1:] + mesh.x3v[:-1])
            coords.append(z)

        if ndim == 2:
            # 2d velocity gradient tensor components
            # note: axis indices are reversed (storage order)
            dvx_dx = np.gradient(vx, coords[0], axis=1)  # x is axis 1
            dvx_dy = np.gradient(vx, coords[1], axis=0)  # y is axis 0
            dvy_dx = np.gradient(vy, coords[0], axis=1)
            dvy_dy = np.gradient(vy, coords[1], axis=0)

            # symmetric part (strain rate tensor)
            s11 = dvx_dx
            s12 = 0.5 * (dvx_dy + dvy_dx)
            s22 = dvy_dy

            # antisymmetric part (rotation rate tensor)
            omega12 = 0.5 * (dvy_dx - dvx_dy)

            # frobenius norms
            s_norm_sq = s11**2 + 2 * s12**2 + s22**2
            omega_norm_sq = 2 * omega12**2

            q = 0.5 * (omega_norm_sq - s_norm_sq)

        elif ndim == 3:
            # 3d velocity gradient tensor (all 9 components)
            dvx_dx = np.gradient(vx, coords[0], axis=2)  # x is axis 2
            dvx_dy = np.gradient(vx, coords[1], axis=1)  # y is axis 1
            dvx_dz = np.gradient(vx, coords[2], axis=0)  # z is axis 0

            dvy_dx = np.gradient(vy, coords[0], axis=2)
            dvy_dy = np.gradient(vy, coords[1], axis=1)
            dvy_dz = np.gradient(vy, coords[2], axis=0)

            dvz_dx = np.gradient(vz, coords[0], axis=2)
            dvz_dy = np.gradient(vz, coords[1], axis=1)
            dvz_dz = np.gradient(vz, coords[2], axis=0)

            # symmetric part (strain rate tensor)
            s11 = dvx_dx
            s12 = 0.5 * (dvx_dy + dvy_dx)
            s13 = 0.5 * (dvx_dz + dvz_dx)
            s22 = dvy_dy
            s23 = 0.5 * (dvy_dz + dvz_dy)
            s33 = dvz_dz

            # antisymmetric part (rotation rate tensor)
            omega12 = 0.5 * (dvy_dx - dvx_dy)
            omega13 = 0.5 * (dvz_dx - dvx_dz)
            omega23 = 0.5 * (dvz_dy - dvy_dz)

            # frobenius norms
            s_norm_sq = (
                s11**2 + s22**2 + s33**2 + 2 * (s12**2 + s13**2 + s23**2)
            )
            omega_norm_sq = 2 * (omega12**2 + omega13**2 + omega23**2)

            q = 0.5 * (omega_norm_sq - s_norm_sq)
        else:
            # 1d: no vorticity or shear
            q = np.zeros_like(vx)

        return np.asarray(q)

    def compute_okubo_weiss(level_data: dict[str, Array]) -> Array:
        """
        okubo-weiss parameter for 2d flow structure identification.

        ow = s_n^2 + s_s^2 - \\omega^2

        where:
        s_n = \\partial v_x / \\partial x - \\partial v_y / \\partial y  (normal strain)
        s_s = \\partial v_x / \\partial y + \\partial v_y / \\partial x  (shear strain)
        \\omega =\\partial v_y /\\partial x -\\partial v_x /\\partial y  (vorticity)

        ow < 0: vortex-dominated (rotation exceeds strain)
        ow > 0: strain-dominated (hyperbolic/elliptic regions)

        commonly used in 2d turbulence and geophysical flows.
        for 3d data, computes ow on each z-slice.
        """
        mesh = getattr(level_data, "mesh")
        vx, vy = level_data["v1"], level_data["v2"]

        # cell-centered coordinates
        x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
        y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])

        # velocity gradients (note: axis order reversed for storage)
        dvx_dx = np.gradient(vx, x, axis=ndim - 1)
        dvx_dy = np.gradient(vx, y, axis=ndim - 2)
        dvy_dx = np.gradient(vy, x, axis=ndim - 1)
        dvy_dy = np.gradient(vy, y, axis=ndim - 2)

        # strain components
        s_n = dvx_dx - dvy_dy  # normal strain
        s_s = dvx_dy + dvy_dx  # shear strain

        # vorticity
        omega_z = dvy_dx - dvx_dy

        # okubo-weiss parameter
        ow = s_n**2 + s_s**2 - omega_z**2

        return np.asarray(ow)

    def compute_schlieren(level_data: dict[str, Array]) -> Array:
        """
        numerical schlieren: gradient magnitude of log(density).
        highlights shocks and contact discontinuities.
        """
        mesh = getattr(level_data, "mesh")
        rho = level_data["rho"]

        # work with log(density) for better shock sensitivity
        log_rho = np.log(rho + 1e-20)  # avoid log(0)

        # compute gradient components
        grads = []
        coords = []

        if ndim >= 1:
            x = 0.5 * (mesh.x1v[1:] + mesh.x1v[:-1])
            coords.append(x)
        if ndim >= 2:
            y = 0.5 * (mesh.x2v[1:] + mesh.x2v[:-1])
            coords.append(y)
        if ndim >= 3:
            z = 0.5 * (mesh.x3v[1:] + mesh.x3v[:-1])
            coords.append(z)

        # gradient along each active dimension
        for axis in range(ndim):
            grad = np.gradient(log_rho, coords[axis], axis=ndim - 1 - axis)
            grads.append(grad)

        # magnitude: sqrt(sum of squares)
        grad_mag = np.sqrt(sum(g**2 for g in grads))

        return np.asarray(grad_mag)

    # build base pipeline of functions
    base_pipeline: dict[str, Callable[..., Any]] = {
        "W": compute_W,
        "D": compute_D,
        "v": compute_velocity_magnitude,
        "u": compute_four_velocity_magnitude,
        "energy": compute_energy,
        "enthalpy": compute_enthalpy,
        "enthalpy_density": compute_enthalpy_density,
        "sigma": compute_magnetization,
        "ptot": compute_total_pressure,
        "pmag": compute_magnetic_pressure,
        "emag": compute_magnetic_energy,
        "mach": compute_mach_number,
        "chi_dens": compute_chi_density,
        "j": angular_momentum_density,
        "mass_flux": mass_flux,
        "j_spec": specific_angular_momentum,
        "Sigma": surface_density,
        "vorticity": compute_vorticity_z,
        "vorticity_magnitude": compute_vorticity_magnitude,
        "q_criterion": compute_q_criterion,
        "okubo_weiss": compute_okubo_weiss,
        "div_v": compute_divergence,
        "schlieren": compute_schlieren,
    }

    # add component fields. an mhd velocity / momentum is a 3-vector regardless of the spatial
    # dimensionality (a 2.5d / 1.75d run still evolves the out-of-plane component), so the
    # component count follows the stored vector dof: 3 for mhd, ndim for hydro.
    vector_dof = _vector_dof(regime, ndim)
    for ii in range(1, vector_dof + 1):
        base_pipeline[f"u{ii}"] = compute_u_component(ii - 1)
        base_pipeline[f"m{ii}"] = compute_momentum_component(ii - 1)

    if data.metadata.is_mhd:
        # cell-centered lab-frame magnetic field b1_mean / b2_mean / b3_mean (the plain b1/b2/b3
        # names resolve to the raw face-centered fields, so the cell-centered form carries the
        # _mean suffix).
        for ii in range(1, vector_dof + 1):
            base_pipeline[f"b{ii}_mean"] = compute_b_labframe_component(ii - 1)
        # spatial components of the comoving magnetic four-vector, defined only for the
        # relativistic mhd regime.
        if regime == "rmhd":
            for ii in range(1, vector_dof + 1):
                base_pipeline[f"bmu{ii}"] = compute_b_four_component(ii - 1)

    # fields requiring composite/base-level computation
    # these involve spatial integration or coordinate transformations that are
    # meaningless when computed on partial refined domains
    composite_required = {
        "Sigma",  # surface_density: integrates over z-column
        "j",  # angular_momentum_density: uses cross(r, v) with coordinates
        "j_spec",  # specific_angular_momentum: same
        "mass_flux",  # uses coordinate radius
    }

    # wrap functions into derived-field objects exposing evaluate(level)
    pipeline: dict[str, Any] = {
        name: derived_field_t(
            name, func, requires_composite=(name in composite_required)
        )
        for name, func in base_pipeline.items()
    }

    return pipeline
