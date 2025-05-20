#ifndef CONFIG_DICT_HPP
#define CONFIG_DICT_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use
#include "core/types/containers/vector.hpp"   // for spatial_vector_t
#include "enums.hpp"
#include <cstdint>
#include <exception>
#include <list>
#include <stdexcept>
#include <string>
#include <type_traits>
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
            std::monostate,                   // For empty/null values
            bool,                             // For boolean values
            int,                              // For integer values
            luint,                            // For unsigned integer values
            real,                             // For floating point
            std::string,                      // For string values
            std::vector<real>,                // For numeric arrays
            std::vector<std::vector<real>>,   // For 2D arrays
            std::vector<std::string>,         // For string arrays
            std::vector<int>,                 // For integer arrays
            std::pair<real, real>,            // For pairs of real values
            ConfigDict,                       // For nested dictionaries
            std::list<ConfigDict>,            // For list of dictionaries
            BodyCapability                    // For BodyCapabilities
            >;

        ValueType value;

        // Default constructor
        ConfigValue() : value(std::monostate{}) {}

        // Constructors for different types
        ConfigValue(bool v) : value(v) {}
        ConfigValue(int v) : value(v) {}
        ConfigValue(luint v) : value(v) {}
        ConfigValue(real v) : value(v) {}
        ConfigValue(const char* v) : value(std::string(v)) {}
        ConfigValue(std::string v) : value(std::move(v)) {}
        ConfigValue(std::vector<real> v) : value(std::move(v)) {}
        ConfigValue(std::vector<std::string> v) : value(std::move(v)) {}
        ConfigValue(std::vector<int> v) : value(std::move(v)) {}
        ConfigValue(std::vector<std::vector<real>> v) : value(std::move(v)) {}
        ConfigValue(ConfigDict v) : value(std::move(v)) {}
        ConfigValue(std::list<ConfigDict> v) : value(std::move(v)) {}
        ConfigValue(std::pair<real, real> v) : value(std::move(v)) {}
        ConfigValue(BodyCapability v) : value(v) {}

        // Helper for spatial vectors
        template <typename T, size_type Dims>
        ConfigValue(spatial_vector_t<T, Dims> v)
        {
            std::vector<real> vec_values;
            for (size_type i = 0; i < Dims; ++i) {
                vec_values.push_back(static_cast<real>(v[i]));
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
        bool is_uint() const { return std::holds_alternative<luint>(value); }
        bool is_real_number() const
        {
            return std::holds_alternative<real>(value);
        }
        bool is_number() const { return is_int() || is_real_number(); }
        bool is_string() const
        {
            return std::holds_alternative<std::string>(value);
        }
        bool is_array() const
        {
            return std::holds_alternative<std::vector<real>>(value);
        }
        bool is_nested_array_of_floats() const
        {
            return std::holds_alternative<std::vector<std::vector<real>>>(value
            );
        }
        bool is_dict() const
        {
            return std::holds_alternative<ConfigDict>(value);
        }
        bool is_list() const
        {
            return std::holds_alternative<std::list<ConfigDict>>(value);
        }
        bool is_array_of_strings() const
        {
            return std::holds_alternative<std::vector<std::string>>(value);
        }
        bool is_pair() const
        {
            return std::holds_alternative<std::pair<real, real>>(value);
        }
        bool is_array_of_ints() const
        {
            return std::holds_alternative<std::vector<int>>(value);
        }

        bool is_body_cap() const
        {
            return std::holds_alternative<BodyCapability>(value);
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
                if (is_real_number()) {
                    return static_cast<int>(std::get<real>(value));
                }
                throw std::runtime_error("Not an integer value");
            }
            else if constexpr (std::is_same_v<T, real>) {
                if (is_real_number()) {
                    return std::get<real>(value);
                }
                if (is_int()) {
                    return static_cast<real>(std::get<int>(value));
                }
                throw std::runtime_error("Not a numeric value");
            }
            else if constexpr (std::is_same_v<T, std::string>) {
                if (!is_string()) {
                    throw std::runtime_error("Not a string value");
                }
                return std::get<std::string>(value);
            }
            else if constexpr (std::is_same_v<T, std::vector<real>>) {
                if (!is_array()) {
                    throw std::runtime_error("Not an array value");
                }
                return std::get<std::vector<real>>(value);
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
            else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                if (!is_array_of_strings()) {
                    throw std::runtime_error("Not an array of strings");
                }
                return std::get<std::vector<std::string>>(value);
            }
            else if constexpr (std::is_same_v<T, std::pair<real, real>>) {
                if (!is_pair()) {
                    throw std::runtime_error("Not a pair of real values");
                }
                return std::get<std::pair<real, real>>(value);
            }
            else if constexpr (std::is_same_v<T, std::vector<int>>) {
                if (!is_array_of_ints()) {
                    throw std::runtime_error("Not an array of integers");
                }
                return std::get<std::vector<int>>(value);
            }
            else if constexpr (std::is_same_v<
                                   T,
                                   std::vector<std::vector<real>>>) {
                if (!is_nested_array_of_floats()) {
                    throw std::runtime_error("Not a nested array of floats");
                }
                return std::get<std::vector<std::vector<real>>>(value);
            }
            else if constexpr (std::is_same_v<T, luint>) {
                if (is_uint()) {
                    return std::get<luint>(value);
                }
                if (is_int()) {
                    return static_cast<luint>(std::get<int>(value));
                }
                throw std::runtime_error("Not an unsigned integer value");
            }
            else if constexpr (std::is_same_v<T, BodyCapability>) {
                return std::get<BodyCapability>(value);
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

            const auto& array = std::get<std::vector<real>>(value);
            if (array.size() < Dims) {
                throw std::runtime_error("Array too small for spatial vector");
            }

            spatial_vector_t<T, Dims> result;
            for (size_type ii = 0; ii < Dims; ++ii) {
                result[ii] = static_cast<T>(array[ii]);
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

    // helper functions for property extraction
    namespace config {
        template <typename T>
        Maybe<T> try_read(const ConfigDict& dict, const std::string& key)
        {
            if (!dict.contains(key)) {
                return Nothing;
            }

            try {
                return Maybe<T>(dict.at(key).template get<T>());
            }
            catch (const std::exception&) {
                return Nothing;
            }
        }

        template <typename T, size_type Dims>
        Maybe<spatial_vector_t<T, Dims>>
        try_read_vec(const ConfigDict& dict, const std::string& key)
        {
            if (!dict.contains(key)) {
                return Nothing;
            }

            try {
                return Maybe<spatial_vector_t<T, Dims>>(
                    dict.at(key).template to_spatial_vector<T, Dims>()
                );
            }
            catch (const std::exception&) {
                return Nothing;
            }
        }

        // predicate for checking if a property exists and is of type T
        template <typename T>
        auto has_property_of_type(const std::string& key)
        {
            return [key](const ConfigDict& props) {
                return try_read<T>(props, key).has_value();
            };
        }

        // predicate for checking if a property exists and equals a specific
        // value
        template <typename T>
        auto property_equals(const std::string& key, T expected_value)
        {
            return [key, expected_value](const ConfigDict& props) {
                auto maybe_value = try_read<T>(props, key);
                return maybe_value.has_value() &&
                       *maybe_value == expected_value;
            };
        }
    }   // namespace config
}   // namespace simbi
#endif
