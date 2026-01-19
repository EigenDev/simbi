// =============================================================================
// config_converter.hpp
//
// converts python dictionaries to and from the c++ `config_dict_t` type.
// this is a key part of the python-c++ boundary, allowing simulation
// parameters to be defined in python and passed to the c++ backend.
//
// usage:
//   // python -> c++
//   config_dict_t cpp_config = dict_to_config(python_dict);
//
//   // c++ -> python
//   py::dict python_dict = config_to_dict(cpp_config);
// =============================================================================
#pragma once

#include "utility/config_dict.hpp"

#include <pybind11/pybind11.h>

namespace simbi {
    namespace py = pybind11;

    // Convert Python dict to config_dict_t
    config_dict_t dict_to_config(const py::dict& dict);

    // Convert config_dict_t to Python dict (for results)
    py::dict config_to_dict(const config_dict_t& config);

    // Register these converters with pybind11
    void register_config_converters(py::module_& m);

} // namespace simbi
