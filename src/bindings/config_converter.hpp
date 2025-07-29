#ifndef CONFIG_CONVERTER_HPP
#define CONFIG_CONVERTER_HPP

#include "core/utility/config_dict.hpp"
#include <pybind11/pybind11.h>

namespace simbi {
    namespace py = pybind11;

    // Convert Python dict to config_dict_t
    config_dict_t dict_to_config(const py::dict& dict);

    // Convert config_dict_t to Python dict (for results)
    py::dict config_to_dict(const config_dict_t& config);

    // Register these converters with pybind11
    void register_config_converters(py::module_& m);

}   // namespace simbi
#endif
