#ifndef CONFIG_DICT_HPP
#define CONFIG_DICT_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include <list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace simbi {
    // Forward declaration for recursive definition
    struct ConfigValue;

    // Dictionary type
    using ConfigDict = std::unordered_map<std::string, ConfigValue>;

    // Variant to hold different value types
    struct ConfigValue {
        // The variant type that can hold various data types including nested
        // dictionary
        using ValueType = std::variant<
            std::monostate,         // For empty/null values
            bool,                   // For boolean values
            int,                    // For integer values
            double,                 // For floating point
            std::string,            // For string values
            std::vector<double>,    // For numeric arrays
            ConfigDict,             // For nested dictionaries
            std::list<ConfigDict>   // For list of dictionaries
            >;

        ValueType value;

        // Default constructor
        ConfigValue() : value(std::monostate{}) {}

        // Constructors for different types
        ConfigValue(bool v) : value(v) {}
        ConfigValue(int v) : value(v) {}
        ConfigValue(double v) : value(v) {}
        ConfigValue(const char* v) : value(std::string(v)) {}
        ConfigValue(std::string v) : value(std::move(v)) {}
        ConfigValue(std::vector<double> v) : value(std::move(v)) {}
        ConfigValue(ConfigDict v) : value(std::move(v)) {}
        ConfigValue(std::list<ConfigDict> v) : value(std::move(v)) {}

        // Helper for spatial vectors
        template <typename T, size_type Dims>
        ConfigValue(spatial_vector_t<T, Dims> v)
        {
            std::vector<double> vec_values;
            for (size_type i = 0; i < Dims; ++i) {
                vec_values.push_back(static_cast<double>(v[i]));
            }
            value = std::move(vec_values);
        }

        // Type checking
        bool is_null() const
        {
            return std::holds_alternative<std::monostate>(value);
        }
        bool is_bool() const { return std::holds_alternative<bool>(value); }
        bool is_int() const { return std::holds_alternative<int>(value); }
        bool is_double() const { return std::holds_alternative<double>(value); }
        bool is_number() const { return is_int() || is_double(); }
        bool is_string() const
        {
            return std::holds_alternative<std::string>(value);
        }
        bool is_array() const
        {
            return std::holds_alternative<std::vector<double>>(value);
        }
        bool is_dict() const
        {
            return std::holds_alternative<ConfigDict>(value);
        }
        bool is_list() const
        {
            return std::holds_alternative<std::list<ConfigDict>>(value);
        }

        // Value access with type checking
        template <typename T>
        T get() const
        {
            if constexpr (std::is_same_v<T, bool>) {
                if (!is_bool()) {
                    throw std::runtime_error("Not a boolean value");
                }
                return std::get<bool>(value);
            }
            else if constexpr (std::is_same_v<T, int>) {
                if (is_int()) {
                    return std::get<int>(value);
                }
                if (is_double()) {
                    return static_cast<int>(std::get<double>(value));
                }
                throw std::runtime_error("Not an integer value");
            }
            else if constexpr (std::is_same_v<T, double>) {
                if (is_double()) {
                    return std::get<double>(value);
                }
                if (is_int()) {
                    return static_cast<double>(std::get<int>(value));
                }
                throw std::runtime_error("Not a numeric value");
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                if (!is_string()) {
                    throw std::runtime_error("Not a string value");
                }
                return std::get<std::string>(value);
            }
            else if constexpr (std::is_same_v<T, std::vector<double>>) {
                if (!is_array()) {
                    throw std::runtime_error("Not an array value");
                }
                return std::get<std::vector<double>>(value);
            }
            else if constexpr (std::is_same_v<T, ConfigDict>) {
                if (!is_dict()) {
                    throw std::runtime_error("Not a dictionary value");
                }
                return std::get<ConfigDict>(value);
            }
            else if constexpr (std::is_same_v<T, std::list<ConfigDict>>) {
                if (!is_list()) {
                    throw std::runtime_error("Not a list of dictionaries");
                }
                return std::get<std::list<ConfigDict>>(value);
            }
            else {
                static_assert(always_false<T>::value, "Unsupported type");
            }
        }

        // Conversion to spatial vector
        template <typename T, size_type Dims>
        spatial_vector_t<T, Dims> to_spatial_vector() const
        {
            if (!is_array()) {
                throw std::runtime_error("Not an array value");
            }

            const auto& array = std::get<std::vector<double>>(value);
            if (array.size() < Dims) {
                throw std::runtime_error("Array too small for spatial vector");
            }

            spatial_vector_t<T, Dims> result;
            for (size_type i = 0; i < Dims; ++i) {
                result[i] = static_cast<T>(array[i]);
            }
            return result;
        }

      private:
        // Helper for static_assert failure
        template <typename>
        struct always_false : std::false_type {
        };
    };

    // Helper function to create a spatial vector from ConfigValue
    template <typename T, size_type Dims>
    spatial_vector_t<T, Dims> to_spatial_vector(const ConfigValue& value)
    {
        return value.to_spatial_vector<T, Dims>();
    }
}   // namespace simbi
#endif
