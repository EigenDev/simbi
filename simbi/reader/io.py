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
            result[key] = np.asarray(item[:])
        elif isinstance(item, h5py.Group):
            result[key] = read_group_recursively(item)

    return result


def read_raw_data(file: h5py.File) -> Result[RawHDF5]:
    try:
        fields: dict[str, Array] = {}
        attributes: dict[str, str | float | int | bool] = {}
        groups: dict[str, dict[str, str | float | int | bool | Array]] = {}

        for key in file.keys():
            item = file[key]
            if isinstance(item, h5py.Dataset):
                if key == "sim_info":
                    # Read metadata attributes
                    for attr_key, attr_value in item.attrs.items():
                        if isinstance(attr_value, bytes):
                            attributes[attr_key] = attr_value.decode("utf-8")
                        else:
                            attributes[attr_key] = attr_value
                else:
                    # Regular field data
                    fields[key] = np.asarray(item[:])
            elif isinstance(item, h5py.Group):
                groups[key] = read_group_recursively(item)

        return Result.ok(
            RawHDF5(fields=fields, attributes=attributes, groups=groups)
        )
    except Exception as e:
        return Result.err(e)
