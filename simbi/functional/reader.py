import h5py
import numpy as np
from types import TracebackType
from enum import IntFlag
from typing import Any, Callable, Optional, Union, Sequence
from numpy.typing import NDArray
from ..physics.calculations import (
    VectorComponent,
    enthalpy_density,
    four_velocity,
    labframe_energy_density,
    labframe_momentum,
    spec_enthalpy,
    lorentz_factor,
    magnetization,
    total_pressure,
    magnetic_pressure,
)

Array = NDArray[np.floating[Any]]


class BodyCapability(IntFlag):
    NONE = 0
    GRAVITATIONAL = 1 << 0
    ACCRETION = 1 << 1
    ELASTIC = 1 << 2
    DEFORMABLE = 1 << 3
    RIGID = 1 << 4


def has_capability(body_type: BodyCapability, capability: BodyCapability) -> bool:
    return bool(body_type & capability)


def vector(ndim: int, name: str) -> Sequence[str]:
    return [f"{name}{i}" for i in range(1, ndim + 1)]


def vec_from_dict(the_dict: dict[str, Any], ndim: int, name: str) -> Sequence[Array]:
    try:
        return [the_dict[f"{name}{i}"] for i in range(1, ndim + 1)]
    except KeyError:
        return [np.zeros(1)] * ndim


class LazySimulationReader:
    """
    Lazily reads simulation data from HDF5 files, including fields, metadata, and mesh.
    """

    def __init__(self, filename: str, unpad: bool = True):
        """Initialize with a filename but don't load any data yet."""
        self.filename = filename
        self.file = None
        self._metadata_cache: dict[str, Any] = {}
        self._mesh_cache: dict[str, Any] = {}
        self._immersed_bodies_cache: dict[str, Any] = {}
        self._field_pipeline = None
        self._iteration_index = 0
        self.unpad_mode = unpad

    def _create_lazy_fields_dict(self) -> dict[str, Callable[[], Array]]:
        """create a dictionary that lazily loads fields when accessed."""
        pipeline = self.create_field_pipeline()

        dimensions = self.metadata["dimensions"]

        # create a dictionary-like object that loads data when accessed
        class LazyFieldDict(dict[str, Any]):
            def __getitem__(self, key: str) -> Array:
                if key in pipeline:
                    return pipeline[key]()
                elif key == "v":
                    # if I am in 1D, just get v1
                    # but if I am in 2D or 3D, get
                    # the velocity magnitude
                    if dimensions == 1:
                        return pipeline["v1"]()
                    else:
                        # get the velocity magnitude
                        return np.sqrt(
                            sum(pipeline[f"v{i}"]() ** 2 for i in range(1, 4))
                        )
                raise KeyError(f"Field '{key}' not found")

            def __contains__(self, key: object) -> bool:
                return key in pipeline

            def keys(self) -> Any:
                return pipeline.keys()

        return LazyFieldDict()

    def __iter__(self) -> "LazySimulationReader":
        """Make this object iterable."""
        self._iteration_index = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next component during iteration."""
        components: Sequence[Union[dict[str, Any], Callable[[], dict[str, Any]]]] = [
            self._create_lazy_fields_dict,  # LazyFieldDict
            self.metadata,  # Metadata dictionary
            self.get_mesh,  # Mesh dictionary,
            self.immersed_bodies,  # Immersed bodies dictionary
        ]

        if self._iteration_index < len(components):
            component = components[self._iteration_index]
            self._iteration_index += 1
            # If it's a callable, execute it
            if callable(component):
                return component()
            return component
        else:
            raise StopIteration

    def __getitem__(self, key: Union[int, str]) -> Any:
        """
        Access components by index or key.

        - Integer index: 0=fields, 1=metadata, 2=mesh
        - String key: Access a specific field directly
        """
        if isinstance(key, int):
            components = [
                self._create_lazy_fields_dict,  # LazyFields dictionary
                self.metadata,  # Metadata dictionary
                self.get_mesh,  # Mesh dictionary
            ]

            if 0 <= key < len(components):
                component = components[key]
                # If it's a callable, execute it
                if callable(component):
                    return component()
                return component
            raise IndexError(f"Index {key} out of range")

        elif isinstance(key, str):
            # Get a specific field by name
            pipeline = self.create_field_pipeline()
            if key in pipeline:
                return pipeline[key]()
            elif key in self.metadata:
                return self.metadata[key]
            elif key in self.get_mesh():
                return self.get_mesh()[key]
            else:
                raise KeyError(f"Key '{key}' not found in fields, metadata, or mesh")

    def get_field_dict(self) -> dict[str, Array]:
        """
        Get a dictionary with all fields pre-loaded.
        (Eager loading of all fields, :P)
        """
        pipeline = self.create_field_pipeline()
        return {name: loader() for name, loader in pipeline.items()}

    def __enter__(self) -> object:
        """Open the file when entering a context manager."""
        self.file = h5py.File(self.filename, "r")
        return self

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file when exiting the context."""
        if self.file:
            self.file.close()
            self.file = None

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get simulation metadata - cached after first access.
        Metadata is usually small, so caching it is efficient.
        """
        if not self._metadata_cache:
            if not self.file:
                with h5py.File(self.filename, "r") as f:
                    self._load_metadata(f)
            else:
                self._load_metadata(self.file)
        return self._metadata_cache

    @property
    def immersed_bodies(self) -> dict[str, Any]:
        """Get immersed bodies data - cached after first access."""
        if not self._immersed_bodies_cache:
            if not self.file:
                with h5py.File(self.filename, "r") as f:
                    self._load_immersed_bodies(f)
            else:
                self._load_immersed_bodies(self.file)

        return self._immersed_bodies_cache

    def _load_immersed_bodies(self, file_obj: h5py.File) -> None:
        """load immersed bodies data from file if any exist."""
        if "immersed_bodies" not in file_obj:
            return None

        self._immersed_bodies_cache = {}
        # load immersed bodies data
        ib_group = file_obj["immersed_bodies"]

        # get the number of bodies
        body_count: int = int(ib_group.attrs["count"])

        # Read data for each body
        for i in range(body_count):
            body_group = ib_group[f"body_{i}"]

            # Read properties
            mass = body_group["mass"][...]
            radius = body_group["radius"][...]
            position = body_group["position"][...]
            velocity = body_group["velocity"][...]
            force = body_group["force"][...]
            body_type = BodyCapability(int(body_group.attrs["capabilities"]))
            # Check body type
            self._immersed_bodies_cache[f"body_{i}"] = {
                "mass": mass,
                "radius": radius,
                "position": position,
                "velocity": velocity,
                "force": force,
                "type": body_type,
            }

            # For accretors, read additional properties
            if has_capability(body_type, BodyCapability.ACCRETION):
                self._immersed_bodies_cache[f"body_{i}"].update(
                    {
                        # "accretion_rate": body_group["accretion_rate"][...],
                        "accretion_radius": body_group["accretion_radius"][...],
                        "total_accreted_mass": body_group["total_accreted_mass"][...],
                    }
                )

    def _load_metadata(self, file_obj: h5py.File) -> None:
        """Load and process metadata from file."""

        raw_metadata = dict(file_obj["sim_info"].attrs)
        # decode byte strings if present
        self._metadata_cache = {
            k: v.decode("utf-8") if isinstance(v, bytes) else v
            for k, v in raw_metadata.items()
        }
        # turn numpy uint8 into bool
        self._metadata_cache = {
            k: bool(v) if isinstance(v, np.uint8) else v
            for k, v in self._metadata_cache.items()
        }

        # derived metadata
        self._metadata_cache.update(
            {
                "is_cartesian": self._metadata_cache["geometry"]
                in ["cartesian", "axis_cylindrical", "cylindrical"],
                "coord_system": self._metadata_cache.pop("geometry"),
                "time": self._metadata_cache.pop("current_time", 0.0),
            }
        )

    def get_mesh(self, force_reload: bool = False) -> dict[str, Array]:
        """
        Get mesh data - lazily loaded and cached.
        Mesh data is computed based on metadata and is typically small enough to cache.

        Args:
            force_reload: If True, reload mesh even if cached

        Returns:
            Dictionary with mesh data
        """
        if not self._mesh_cache or force_reload:
            # Ensure metadata is loaded
            metadata = self.metadata
            ndim = metadata["dimensions"]

            # Create mesh data based on metadata
            mesh = {}

            def grid_length_str(i: int) -> str:
                if i == 0:
                    return "nx"
                elif i == 1:
                    return "ny"
                else:
                    return "nz"

            effective_dimensions = 0
            for i in range(1, ndim + 1):
                # Determine spacing function based on metadata
                spacing_func = (
                    np.linspace
                    if metadata[f"x{i}_spacing"] == "linear"
                    else np.geomspace
                )

                # Calculate number of active zones
                nghosts = 1 + (metadata["spatial_order"] == "plm")
                active_zones = max(
                    metadata[f"{grid_length_str(i - 1)}"] - 2 * nghosts, 1
                )

                # Generate mesh points
                mesh[f"x{i}v"] = spacing_func(
                    metadata[f"x{i}min"], metadata[f"x{i}max"], active_zones + 1
                )
                effective_dimensions += active_zones != 1

            mesh["effective_dimensions"] = effective_dimensions
            if self._metadata_cache is not None:
                self._metadata_cache["effective_dimensions"] = effective_dimensions

            self._mesh_cache = mesh

        return self._mesh_cache

    def get_field(self, field_name: str) -> Any:
        """
        Create a lazy accessor for a field.
        The field data is only loaded when the returned function is called.

        Args:
            field_name: Name of the field to access

        Returns:
            Function that loads and returns the field data when called
        """

        # Create a function that will load the data when called
        def field_loader(indices: Sequence[slice] | None = None) -> Array:
            # Open file if needed
            if not self.file:
                with h5py.File(self.filename, "r") as f:
                    return self._load_field_data(f, field_name, indices)
            else:
                return self._load_field_data(self.file, field_name, indices)

        return field_loader

    def _is_gas_variable(self, name: str) -> bool:
        return name in ["rho", "p", "v1", "v2", "v3", "chi"]

    def _average_field(self, data: Array, field_name: str) -> Array:
        if self._is_gas_variable(field_name):
            return data

        if field_name == "b1":
            bview = data[1:-1, 1:-1] if self.unpad_mode else data
            res = 0.5 * (bview[..., 1:] + bview[..., :-1])
            if not self.unpad_mode:
                pad = 1 if self.metadata["spatial_order"] == "plm" else 0
                npad = ((pad, pad), (pad, pad), (1 + pad, 1 + pad))
                return np.pad(res, pad_width=npad, mode="constant")

            return res
        elif field_name == "b2":
            bview = data[1:-1, :, 1:-1] if self.unpad_mode else data
            res = 0.5 * (bview[:, 1:, :] + bview[:, :-1, :])
            if not self.unpad_mode:
                pad = 1 if self.metadata["spatial_order"] == "plm" else 0
                npad = ((pad, pad), (1 + pad, 1 + pad), (pad, pad))
                return np.pad(res, pad_width=npad, mode="constant")
            return res
        else:
            bview = data[:, 1:-1, 1:-1] if self.unpad_mode else data
            res = 0.5 * (bview[1:, :, :] + bview[:-1, :, :])
            if not self.unpad_mode:
                pad = 1 if self.metadata["spatial_order"] == "plm" else 0
                npad = ((1 + pad, 1 + pad), (pad, pad), (pad, pad))
                return np.pad(res, pad_width=npad, mode="constant")
            return res

    def _get_centered_fields(self, bfields: Sequence[Array]) -> Sequence[Array]:
        return [self._average_field(d, f"b{i + 1}") for i, d in enumerate(bfields)]

    def _load_field_data(
        self, file_obj: h5py.File, field_name: str, indices: Optional[Any] = None
    ) -> Array:
        """
        Load field data from file, applying any necessary processing.

        Args:
            file_obj: Open h5py File object
            field_name: Name of the field to load
            indices: Optional indices for partial loading

        Returns:
            Processed field data
        """

        if field_name not in file_obj:
            if field_name in ["b1", "b2", "b3"]:
                return np.zeros(1)
            raise KeyError(f"Field '{field_name}' not found in file")

        # Get metadata for processing
        metadata = self.metadata
        ndim = metadata["dimensions"]

        # Load data (fully or partially)
        padwidth = (metadata["spatial_order"] != "pcm") + 1
        if indices is None:
            data: Array = np.asarray(file_obj[field_name][:])
            # Reshape based on dimensions
            if self._is_gas_variable(field_name):
                data = data.reshape(metadata["nz"], metadata["ny"], metadata["nx"])
            else:
                xactive_zones = metadata["nx"] - 2 * padwidth
                yactive_zones = metadata["ny"] - 2 * padwidth
                zactive_zones = metadata["nz"] - 2 * padwidth
                if field_name == "b1":
                    bshape = (zactive_zones + 2, yactive_zones + 2, xactive_zones + 1)
                elif field_name == "b2":
                    bshape = (zactive_zones + 2, yactive_zones + 1, xactive_zones + 2)
                else:
                    bshape = (zactive_zones + 1, yactive_zones + 2, xactive_zones + 2)

                data = data.reshape(bshape)
        else:
            # Load only requested slice
            data = np.asarray(file_obj[field_name][indices])

        # Apply padding/unpadding based on metadata
        if self.unpad_mode and self._is_gas_variable(field_name):
            npad = tuple((padwidth, padwidth) for _ in range(ndim))
            # if ndim < 3, we get everything from the other axes
            npad = ((0, 0),) * (3 - ndim) + npad

            # Unpad the data if needed
            if padwidth > 0 and indices is None:
                data = self._unpad(data, npad)

        if self.unpad_mode:
            if any(s == 1 for s in data.shape):
                data = data.reshape(tuple(s for s in data.shape if s != 1))

        return data

    def _unpad(self, arr: Array, pad_width: tuple[tuple[int, int], ...]) -> Array:
        """Remove padding from array."""
        slices = []
        for c in pad_width:
            e = None if c[1] == 0 else -c[1]
            slices.append(slice(c[0], e))
        return arr[tuple(slices)]

    def create_field_pipeline(self) -> dict[str, Callable[[], Array]]:
        """
        Create a comprehensive pipeline of field processors.

        Returns:
            Dictionary of field processors that load and process data on demand
        """
        # metadata to determine available fields
        metadata = self.metadata

        # prim fields present in all simulations
        basic_fields = ["rho", "p", "chi"]

        # get the veloicty field
        ndim = metadata["dimensions"]
        for i in range(1, ndim + 1):
            basic_fields.append(f"v{i}")
            if "mhd" in metadata["regime"]:
                basic_fields.append(f"b{i}")

        # Create processors for basic fields
        pipeline = {field: self.get_field(field) for field in basic_fields}

        # now processors for derived fields
        pipeline["W"] = self.get_derived_field(
            "W",
            [f"v{i}" for i in range(1, ndim + 1)],
            lambda fields: lorentz_factor(
                [fields[f"v{i}"] for i in range(1, ndim + 1)], metadata["regime"]
            ),
        )
        pipeline["D"] = self.get_derived_field(
            "D",
            ["rho", *[f"v{i}" for i in range(1, ndim + 1)]],
            lambda fields: fields["rho"]
            * lorentz_factor(
                [fields[f"v{i}"] for i in range(1, ndim + 1)], metadata["regime"]
            ),
        )
        for i in range(1, ndim + 1):
            pipeline[f"u{i}"] = self.get_derived_field(
                f"u{i}",
                vector(ndim, "v"),
                lambda fields, i=i: four_velocity(
                    vec_from_dict(fields, ndim, "v"), metadata["regime"], i - 1
                ),
            )
            pipeline[f"m{i}"] = self.get_derived_field(
                f"m{i}",
                ["rho", *vector(ndim, "v"), *vector(ndim, "b"), "p"],
                lambda fields, i=i: labframe_momentum(
                    fields["rho"],
                    fields["p"],
                    vec_from_dict(fields, ndim, "v"),
                    self._get_centered_fields(vec_from_dict(fields, ndim, "b")),
                    metadata["adiabatic_index"],
                    metadata["regime"],
                    VectorComponent(i - 1),
                ),
            )
            pipeline[f"b{i}_mean"] = self.get_derived_field(
                f"b{i}_mean",
                [f"b{i}"],
                lambda fields, i=i: self._average_field(fields[f"b{i}"], f"b{i}"),
            )

        pipeline["energy"] = self.get_derived_field(
            "energy",
            ["rho", *vector(ndim, "v"), *vector(ndim, "b"), "p"],
            lambda fields: labframe_energy_density(
                fields["rho"],
                fields["p"],
                vec_from_dict(fields, ndim, "v"),
                self._get_centered_fields(vec_from_dict(fields, ndim, "b")),
                metadata["adiabatic_index"],
                metadata["regime"],
            ),
        )
        pipeline["enthalpy"] = self.get_derived_field(
            "enthalpy",
            ["rho", "p"],
            lambda fields: spec_enthalpy(
                metadata["adiabatic_index"],
                fields["rho"],
                fields["p"],
                metadata["regime"],
            ),
        )
        pipeline["enthalpy_density"] = self.get_derived_field(
            "enthalpy_density",
            ["rho", "p", *vector(ndim, "b")],
            lambda fields: enthalpy_density(
                fields["rho"],
                fields["p"],
                self._get_centered_fields(vec_from_dict(fields, ndim, "b")),
                metadata["adiabatic_index"],
                metadata["regime"],
            ),
        )
        pipeline["sigma"] = self.get_derived_field(
            "sigma",
            ["rho", *vector(ndim, "b")],
            lambda fields: magnetization(
                fields["rho"],
                self._get_centered_fields(vec_from_dict(fields, ndim, "b")),
            ),
        )
        pipeline["ptot"] = self.get_derived_field(
            "ptot",
            ["p", *vector(ndim, "b")],
            lambda fields: total_pressure(
                fields["p"], self._get_centered_fields(vec_from_dict(fields, ndim, "b"))
            ),
        )
        pipeline["pmag"] = self.get_derived_field(
            "pmag",
            [*vector(ndim, "b")],
            lambda fields: magnetic_pressure(
                self._get_centered_fields(vec_from_dict(fields, ndim, "b"))
            ),
        )
        return pipeline

    def get_derived_field(
        self,
        field_name: str,
        dependencies: Sequence[str],
        compute_func: Union[Callable[..., Array | float], Callable[..., Array]],
    ) -> Union[Callable[..., Array | float], Callable[..., Array]]:
        """
        Create a lazy accessor for a derived field.

        Args:
            field_name: Name for the derived field
            dependencies: List of fields this derived field depends on
            compute_func: Function that computes the derived field from dependencies

        Returns:
            Function that computes and returns the derived field when called
        """

        def derived_field_loader(
            indices: Sequence[slice] | None = None,
        ) -> Array | float:
            # Create a dictionary to hold dependency data
            dep_data = {}

            # Load all dependencies
            if not self.file:
                with h5py.File(self.filename, "r") as f:
                    for dep in dependencies:
                        dep_data[dep] = self._load_field_data(f, dep, indices)
            else:
                for dep in dependencies:
                    dep_data[dep] = self._load_field_data(self.file, dep, indices)

            # Compute derived field
            return compute_func(dep_data)

        return derived_field_loader
