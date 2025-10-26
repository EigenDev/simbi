from typing import Any

import h5py
import numpy as np

from ..core.types import Array, RawHDF5
from ..functional.result import Result


def open_file(filename: str) -> Result[h5py.File]:
    try:
        return Result.ok(h5py.File(filename, "r"))
    except Exception as e:
        return Result.err(e)


def read_group_recursively(group: h5py.Group) -> dict[str, Any]:
    """Recursively read group data and nested subgroups"""
    result: dict[str, Any] = {}

    # Read attributes
    for attr_key, attr_value in group.attrs.items():
        if isinstance(attr_value, bytes):
            result[attr_key] = attr_value.decode("utf-8")
        else:
            result[attr_key] = attr_value

    # Read datasets and subgroups
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = np.asarray(item[...])
        elif isinstance(item, h5py.Group):
            result[key] = read_group_recursively(item)

    return result


def read_raw_data(file: h5py.File) -> Result[RawHDF5]:
    try:
        fields: dict[str, Array] = {}
        attributes: dict[str, str | float | int | bool] = {}
        groups: dict[str, dict[str, str | float | int | bool | Array]] = {}

        #  read metadata group
        if "metadata" in file:
            metadata_group = file["metadata"]
            for key, value in metadata_group.attrs.items():
                if isinstance(value, bytes):
                    attributes[key] = value.decode("utf-8")
                else:
                    attributes[key] = value

        # read level_0 group for fields
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
                # add mesh data to groups['mesh']
                elif key == "mesh":
                    groups["mesh_config"] = {}
                    for mkey, value in level_group["mesh"].attrs.items():
                        if isinstance(value, bytes):
                            groups["mesh_config"][mkey] = value.decode("utf-8")
                        else:
                            groups["mesh_config"][mkey] = value

        # read hierarchy info if present (for FMR)
        if "hierarchy" in file:
            groups["hierarchy"] = read_group_recursively(file["hierarchy"])
            # Read additional levels
            level = 1
            while f"level_{level}" in file:
                level_group = file[f"level_{level}"]
                level_data = read_group_recursively(level_group)
                groups[f"level_{level}"] = level_data
                level += 1

        if "bodies" in file:
            groups["bodies"] = read_group_recursively(file["bodies"])

        return Result.ok(
            RawHDF5(fields=fields, attributes=attributes, groups=groups)
        )
    except Exception as e:
        return Result.err(e)
