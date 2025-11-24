# =============================================================================
# io.py
#
# low-level hdf5 file operations.
# handles opening files and extracting raw data structures.
# =============================================================================
from typing import Any

import h5py
import numpy as np

from ..functional.result import Result
from ..types import Array, RawHDF5


def open_file(filename: str) -> Result[h5py.File]:
    try:
        return Result.ok(h5py.File(filename, "r"))
    except Exception as e:
        return Result.err(e)


def read_group_recursively(group: h5py.Group) -> dict[str, Any]:
    """Recursively read group data and nested subgroups"""
    result: dict[str, Any] = {}

    # read attributes
    for attr_key, attr_value in group.attrs.items():
        if isinstance(attr_value, bytes):
            result[attr_key] = attr_value.decode("utf-8")
        else:
            result[attr_key] = attr_value

    # read datasets and subgroups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = np.asarray(item[...])
        elif isinstance(item, h5py.Group):
            result[key] = read_group_recursively(item)

    return result


def detect_format_version(file: h5py.File) -> str:
    """Detect checkpoint format version"""
    if "format_version" in file.attrs:
        version = file.attrs["format_version"]
        if isinstance(version, bytes):
            return version.decode("utf-8")
        return str(version)
    # old format didn't have version attribute
    return "1.0"


def read_raw_data_v2(file: h5py.File) -> Result[RawHDF5]:
    """Read new format (v2.0) checkpoint data"""
    try:
        fields: dict[str, Array] = {}
        attributes: dict[str, str | float | int | bool] = {}
        groups: dict[str, dict[str, Any]] = {}

        # read metadata group
        if "metadata" in file:
            metadata_group = file["metadata"]
            for key, value in metadata_group.attrs.items():
                if isinstance(value, bytes):
                    attributes[key] = value.decode("utf-8")
                else:
                    attributes[key] = value

            # read resolution dataset
            if "resolution" in metadata_group:
                attributes["resolution"] = tuple(
                    int(x) for x in metadata_group["resolution"][...]
                )

            # read boundary conditions group
            if "boundary_conditions" in metadata_group:
                bc_group = metadata_group["boundary_conditions"]
                bcs = []
                idx = 0
                while f"bc_{idx}" in bc_group.attrs:
                    bc_val = bc_group.attrs[f"bc_{idx}"]
                    if isinstance(bc_val, bytes):
                        bcs.append(bc_val.decode("utf-8"))
                    else:
                        bcs.append(str(bc_val))
                    idx += 1
                attributes["boundary_conditions"] = tuple(bcs)

            # read level_dts if present
            if "level_dts" in metadata_group:
                groups["level_dts"] = np.asarray(
                    metadata_group["level_dts"][...]
                )

        # read hierarchy info
        if "hierarchy" in file:
            groups["hierarchy"] = read_group_recursively(file["hierarchy"])

        # read level_0 data (base level)
        if "level_0" in file:
            level_group = file["level_0"]
            groups["mesh_config"] = {}

            # read mesh config
            if "mesh" in level_group:
                mesh_group = level_group["mesh"]
                groups["mesh_config"] = read_group_recursively(mesh_group)

            # read partition_0 hydro data (primary partition)
            if "partition_0" in level_group:
                part_group = level_group["partition_0"]

                # read partition topology
                if "owned_start" in part_group:
                    groups["partition_info"] = {
                        "owned_start": np.asarray(
                            part_group["owned_start"][...]
                        ),
                        "owned_fin": np.asarray(part_group["owned_fin"][...]),
                    }

                # read hydro/primitives
                if "hydro" in part_group:
                    hydro_group = part_group["hydro"]
                    if "primitives" in hydro_group:
                        prim_group = hydro_group["primitives"]

                        # read domain info
                        if (
                            "domain" in prim_group.attrs
                            or "start" in prim_group.attrs
                        ):
                            pass  # domain info available

                        # read field datasets
                        field_mapping = {
                            "rho": "rho",
                            "pre": "p",
                            "chi": "chi",
                            "v1": "v1",
                            "v2": "v2",
                            "v3": "v3",
                            "b1_mean": "b1",
                            "b2_mean": "b2",
                            "b3_mean": "b3",
                        }

                        for cpp_name, py_name in field_mapping.items():
                            if cpp_name in prim_group:
                                fields[py_name] = np.asarray(
                                    prim_group[cpp_name][...]
                                )

        # read additional levels for FMR
        level_idx = 1
        while f"level_{level_idx}" in file:
            level_key = f"level_{level_idx}"
            level_group = file[level_key]
            groups[level_key] = read_group_recursively(level_group)
            level_idx += 1

        # read bodies if present
        if "bodies" in file:
            groups["bodies"] = read_group_recursively(file["bodies"])

        return Result.ok(
            RawHDF5(fields=fields, attributes=attributes, groups=groups)
        )
    except Exception as e:
        return Result.err(e)


def read_raw_data_v1(file: h5py.File) -> Result[RawHDF5]:
    """Read old format (v1.0) checkpoint data - legacy support"""
    try:
        fields: dict[str, Array] = {}
        attributes: dict[str, str | float | int | bool] = {}
        groups: dict[str, dict[str, Any]] = {}

        # read metadata group
        if "metadata" in file:
            metadata_group = file["metadata"]
            for key, value in metadata_group.attrs.items():
                if isinstance(value, bytes):
                    attributes[key] = value.decode("utf-8")
                else:
                    attributes[key] = value

        # read level_0 group for fields (old format had flat structure)
        if "level_0" in file:
            level_group = file["level_0"]
            for key in level_group.keys():
                if key in [
                    "rho",
                    "v1",
                    "v2",
                    "v3",
                    "p",
                    "chi",
                    "b1",
                    "b2",
                    "b3",
                ]:
                    fields[key] = np.asarray(level_group[key][...])
                elif key == "mesh":
                    groups["mesh_config"] = {}
                    for mkey, value in level_group["mesh"].attrs.items():
                        if isinstance(value, bytes):
                            groups["mesh_config"][mkey] = value.decode("utf-8")
                        else:
                            groups["mesh_config"][mkey] = value

        # read hierarchy info if present
        if "hierarchy" in file:
            groups["hierarchy"] = read_group_recursively(file["hierarchy"])
            level = 1
            while f"level_{level}" in file:
                level_group = file[f"level_{level}"]
                groups[f"level_{level}"] = read_group_recursively(level_group)
                level += 1

        if "bodies" in file:
            groups["bodies"] = read_group_recursively(file["bodies"])

        return Result.ok(
            RawHDF5(fields=fields, attributes=attributes, groups=groups)
        )
    except Exception as e:
        return Result.err(e)


def read_raw_data(file: h5py.File) -> Result[RawHDF5]:
    """Read checkpoint data, auto-detecting format version"""
    version = detect_format_version(file)

    if version.startswith("2"):
        return read_raw_data_v2(file)
    else:
        return read_raw_data_v1(file)
