cimport numpy as np
import numpy as np
from hydro_classes cimport *
from enum import Enum

cdef ConfigDict convert_python_to_config_dict(py_dict):
    """Convert a Python dictionary to ConfigDict with intelligent type handling."""
    cdef ConfigDict result

    for key, value in py_dict.items():
        if value is None:
            continue

        cpp_key: string = key.encode("utf-8")

        # Handle common vector keys specially
        if key in ["position", "velocity", "force"] and isinstance(value, (list, tuple, np.ndarray)):
            result[cpp_key] = convert_to_vector_of_doubles(value)

        # Basic scalar types
        elif isinstance(value, bool):
            result[cpp_key] = ConfigValue(<cbool>value)
        elif isinstance(value, int) or isinstance(value, np.integer):
            if key == "body_type":
                result[cpp_key] = ConfigValue(<BodyCapability>value)
            else:
                result[cpp_key] = ConfigValue(<int>value)
        elif isinstance(value, float) or isinstance(value, np.floating):
            result[cpp_key] = ConfigValue(<double>value)
        elif isinstance(value, str):
            result[cpp_key] = ConfigValue(<string>value.encode("utf-8"))
        elif isinstance(value, type):
            # For string-based enums like Regime, CoordSystem, etc.
            if isinstance(value.value, str):
                result[cpp_key] = ConfigValue(<string>value.value.encode("utf-8"))
            # For integer-based enums like BodyCapability
            elif isinstance(value.value, int):
                result[cpp_key] = ConfigValue(<int>value.value)
            else:
                raise ValueError(f"Unsupported enum type: {type(value)} for key: {key}")
        # Special case for bounds
        elif "bounds" in key and isinstance(value, (list, tuple)) and len(value) == 2:
            result[cpp_key] = ConfigValue(pair[double, double](
                <double>value[0], <double>value[1]))

        # Collections
        elif isinstance(value, (list, tuple, np.ndarray)):
            result[cpp_key] = convert_collection(value)
        elif isinstance(value, dict):
            result[cpp_key] = ConfigValue(convert_python_to_config_dict(value))
        elif callable(value):
            # Skip callable objects
            pass
        else:
            raise ValueError(f"Unsupported type: {type(value)} for key: {key}")

    return result

cdef ConfigValue convert_to_vector_of_doubles(value):
    """Convert a Python sequence to a vector of doubles."""
    cdef vector[double] vec
    for x in value:
        vec.push_back(<double>x)
    return ConfigValue(vec)

cdef ConfigValue convert_collection(collection):
    """Convert a Python collection to the appropriate ConfigValue."""
    # Empty collection
    if len(collection) == 0:
        return ConfigValue(vector[double]())

    # Check first item to determine collection type
    first_item = collection[0]
    cdef vector[vector[double]] nested_vec
    cdef cpplist[ConfigDict] dict_list
    cdef vector[int] int_vec
    cdef vector[string] str_vec
    cdef vector[double] double_vec

    # Nested lists/arrays
    if isinstance(first_item, (list, tuple, np.ndarray, np.flatiter)):
        for item in collection:
            nested_vec.push_back([<double>x for x in item])
        return ConfigValue(nested_vec)

    # Sequence of dictionaries (bodies)
    elif isinstance(first_item, dict):
        for item in collection:
            dict_list.push_back(convert_python_to_config_dict(item))
        return ConfigValue(dict_list)

    # Sequence of same type
    else:
        # Integer list
        if isinstance(first_item, int) or isinstance(first_item, np.integer):
            for item in collection:
                int_vec.push_back(<int>item)
            return ConfigValue(int_vec)

        # Double list
        elif isinstance(first_item, float) or isinstance(first_item, np.floating):
            for item in collection:
                double_vec.push_back(<double>item)
            return ConfigValue(double_vec)

        # String list
        elif isinstance(first_item, str):
            for item in collection:
                str_vec.push_back(str(item).encode("utf-8"))
            return ConfigValue(str_vec)
        # Fall back to double vector
        else:
            try:
                return convert_to_vector_of_doubles(collection)
            except:
                raise ValueError(f"Unable to convert collection with items of type {type(first_item)}")


cdef class SimState:
    cdef Driver driver_state

    def __cinit__(self):
        self.driver_state = Driver()

    def run(
        self,
        *,
        np.ndarray[np.double_t, ndim=2] state,
        dict sim_info,
        a: callable[[float], float],
        adot: callable[[float], float]
    ):
        # comvert the Python dictionary to a ConfigDict
        cdef ConfigDict config_dict = convert_python_to_config_dict(sim_info)
        cdef InitialConditions sim_cond = InitialConditions.create(config_dict)

        cdef PyObjWrapper a_cpp = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)
        self.driver_state.run(
            state,
            sim_info["dimensionality"],
            sim_info["regime"].encode("utf-8"),
            sim_cond,
            a_cpp,
            adot_cpp
        )
