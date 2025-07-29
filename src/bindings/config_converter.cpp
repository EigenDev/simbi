#include "config_converter.hpp"
#include "pybind11/pytypes.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace simbi {
    namespace py = pybind11;

    bool is_enum(const py::object& obj) { return py::hasattr(obj, "value"); }

    // helper function for safely accessing sequence items
    py::object get_item(const py::object& seq, size_t idx)
    {
        return seq.attr("__getitem__")(idx);
    }

    // convert python sequence to vector of doubles
    config_value_t convert_to_vector_of_doubles(const py::object& sequence)
    {
        std::vector<real> vec;
        for (size_t i = 0; i < py::len(sequence); ++i) {
            vec.push_back(py::cast<real>(get_item(sequence, i)));
        }
        return config_value_t(std::move(vec));
    }

    // convert Python collection to appropriate config_value_t
    config_value_t convert_collection(const py::object& collection)
    {
        // empty collection
        if (py::len(collection) == 0) {
            return config_value_t(std::vector<real>());
        }

        // check first item to determine collection type
        py::object first_item = get_item(collection, 0);

        // nested lists/arrays
        if (py::isinstance<py::sequence>(first_item)) {
            // if nested sequence is sequence of sequences
            // of enums, we flatten it to a vector of strings
            // otherwise we keep it as a vector of doubles
            if (py::isinstance<py::str>(get_item(first_item, 0))) {
                std::vector<std::string> str_vec;
                for (size_t ii = 0; ii < py::len(collection); ++ii) {
                    str_vec.push_back(
                        py::cast<std::string>(get_item(collection, ii))
                    );
                }

                return config_value_t(std::move(str_vec));
            }

            std::vector<std::vector<real>> nested_vec;
            for (size_t i = 0; i < py::len(collection); ++i) {
                py::object item = get_item(collection, i);
                std::vector<real> inner_vec;
                for (size_t j = 0; j < py::len(item); ++j) {
                    inner_vec.push_back(py::cast<real>(get_item(item, j)));
                }
                nested_vec.push_back(std::move(inner_vec));
            }
            return config_value_t(std::move(nested_vec));
        }

        // sequence of dictionaries (bodies)
        else if (py::isinstance<py::dict>(first_item)) {
            std::list<config_dict_t> dict_list;
            for (size_t i = 0; i < py::len(collection); ++i) {
                dict_list.push_back(
                    dict_to_config(py::cast<py::dict>(get_item(collection, i)))
                );
            }
            return config_value_t(std::move(dict_list));
        }

        // sequence of same type
        else {
            // Integer list
            if (py::isinstance<py::int_>(first_item)) {
                std::vector<std::int64_t> int_vec;
                for (size_t i = 0; i < py::len(collection); ++i) {
                    int_vec.push_back(
                        py::cast<std::int64_t>(get_item(collection, i))
                    );
                }
                return config_value_t(std::move(int_vec));
            }

            // double list
            else if (py::isinstance<py::float_>(first_item)) {
                std::vector<real> real_vec;
                for (size_t i = 0; i < py::len(collection); ++i) {
                    real_vec.push_back(py::cast<real>(get_item(collection, i)));
                }
                return config_value_t(std::move(real_vec));
            }

            // enum list
            else if (py::hasattr(first_item, "value")) {
                py::object enum_value = first_item.attr("value");
                if (py::isinstance<py::str>(enum_value)) {
                    std::vector<std::string> str_vec;
                    for (size_t i = 0; i < py::len(collection); ++i) {
                        py::object item = get_item(collection, i);
                        str_vec.push_back(
                            py::cast<std::string>(item.attr("value"))
                        );
                    }
                    return config_value_t(std::move(str_vec));
                }
                else if (py::isinstance<py::int_>(enum_value)) {
                    std::vector<std::int64_t> int_vec;
                    for (size_t i = 0; i < py::len(collection); ++i) {
                        py::object item = get_item(collection, i);
                        int_vec.push_back(
                            py::cast<std::int64_t>(item.attr("value"))
                        );
                    }
                    return config_value_t(std::move(int_vec));
                }
            }

            // string list
            else if (py::isinstance<py::str>(first_item)) {
                std::vector<std::string> str_vec;
                for (size_t i = 0; i < py::len(collection); ++i) {
                    str_vec.push_back(
                        py::cast<std::string>(get_item(collection, i))
                    );
                }
                return config_value_t(std::move(str_vec));
            }

            // fall back to double vector
            try {
                return convert_to_vector_of_doubles(collection);
            }
            catch (const py::cast_error&) {
                throw py::value_error(
                    "Unable to convert collection with items of type " +
                    py::str(py::type::of(first_item)).cast<std::string>()
                );
            }
        }
    }

    // Main dictionary conversion function
    config_dict_t dict_to_config(const py::dict& dict)
    {
        config_dict_t result;

        for (auto item : dict) {
            // Get key as string
            std::string key  = py::cast<std::string>(py::str(item.first));
            py::object value = py::reinterpret_borrow<py::object>(item.second);

            // Skip None values
            if (value.is_none()) {
                continue;
            }

            // Handle common vector keys specially
            if ((key == "position" || key == "velocity" || key == "force") &&
                py::isinstance<py::sequence>(value)) {
                result[key] = convert_to_vector_of_doubles(value);
            }
            // Basic scalar types
            else if (py::isinstance<py::bool_>(value)) {
                result[key] = config_value_t(py::cast<bool>(value));
            }
            else if (py::isinstance<py::int_>(value)) {
                if (key == "capability") {
                    result[key] = config_value_t(
                        static_cast<BodyCapability>(
                            py::cast<std::int64_t>(value)
                        )
                    );
                }
                else {
                    result[key] = config_value_t(py::cast<std::int64_t>(value));
                }
            }
            else if (py::isinstance<py::float_>(value)) {
                result[key] = config_value_t(py::cast<real>(value));
            }
            else if (py::isinstance<py::str>(value)) {
                result[key] = config_value_t(py::cast<std::string>(value));
            }
            // Type object with value attribute (enum)
            else if (py::hasattr(value, "value")) {
                py::object enum_value = value.attr("value");
                if (py::isinstance<py::str>(enum_value)) {
                    result[key] =
                        config_value_t(py::cast<std::string>(enum_value));
                }
                else if (py::isinstance<py::int_>(enum_value)) {
                    result[key] =
                        config_value_t(py::cast<std::int64_t>(enum_value));
                }
                else {
                    throw py::value_error(
                        "Unsupported enum type for key: " + key
                    );
                }
            }
            // Special case for bounds
            else if (key.find("bounds") != std::string::npos &&
                     py::isinstance<py::sequence>(value) &&
                     py::len(value) == 2) {
                result[key] = config_value_t(
                    std::pair<real, real>(
                        py::cast<real>(get_item(value, 0)),
                        py::cast<real>(get_item(value, 1))
                    )
                );
            }
            // Collections
            else if (py::isinstance<py::sequence>(value)) {
                result[key] = convert_collection(value);
            }
            // Dictionaries
            else if (py::isinstance<py::dict>(value)) {
                result[key] =
                    config_value_t(dict_to_config(py::cast<py::dict>(value)));
            }
            // Callable objects
            else if (py::isinstance<py::function>(value)) {
                // Skip callable objects
                continue;
            }
            else {
                throw py::value_error("Unsupported type for key: " + key);
            }
        }

        return result;
    }

}   // namespace simbi
