cimport numpy as np
import numpy as np
from hydro_classes cimport *

cdef ConfigDict convert_python_to_config_dict(py_dict):
    """Convert a Python dictionary to ConfigDict."""
    cdef ConfigDict result
    cdef vector[double] vec_of_floating
    cdef vector[string] vec_of_strings
    cdef vector[vector[double]] vec_of_vec_floating
    cdef cpplist[ConfigDict] vec_dict
    cdef pair[double, double] tuple_type

    for key, value in py_dict.items():
        if value is None:
            continue

        cpp_key: string = key.encode("utf-8")
        if isinstance(value, bool):
            result[cpp_key] = ConfigValue(<cbool>value)
        elif isinstance(value, int):
            result[cpp_key] = ConfigValue(<int>value)
        elif isinstance(value, float):
            result[cpp_key] = ConfigValue(<double>value)
        elif isinstance(value, str):
            result[cpp_key] = ConfigValue(<string>value.encode("utf-8"))
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            tuple_type = pair[double, double](value[0], value[1])
            result[cpp_key] = ConfigValue(tuple_type)
        elif isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(x, (list, tuple, np.ndarray)) for x in value):
            # Convert nested lists
            for x in value:
                vec_of_vec_floating.push_back([<double>y for y in x])
            result[cpp_key] = ConfigValue(vec_of_vec_floating)
        elif isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(x, (int, float)) for x in value):
            # Convert numeric lists
            for x in value:
                vec_of_floating.push_back(<double>x)
            result[cpp_key] = ConfigValue(vec_of_floating)
        elif isinstance(value, (list, tuple, np.ndarray)) and all(isinstance(x, str) for x in value):
            # Convert string lists
            for x in value:
                vec_of_strings.push_back(<string>x.encode("utf-8"))
            result[cpp_key] = ConfigValue(vec_of_strings)
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            result[cpp_key] = ConfigValue(convert_python_to_config_dict(value))
        elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
            for x in value:
                vec_dict.push_back(convert_python_to_config_dict(x))
            result[cpp_key] = ConfigValue(vec_dict)
        else:
            raise ValueError(f"Unsupported type: {type(value)} for variable: {key}")

    return result

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
        # Convert the Python dictionary to a ConfigDict
        cdef ConfigDict config_dict = convert_python_to_config_dict(sim_info)
        # Create InitialConditions using the builder pattern
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
