#ifndef SIMBI_CORE_GRAPH_COORDINATE_HPP
#define SIMBI_CORE_GRAPH_COORDINATE_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>

namespace simbi::base {
    using namespace simbi::concepts;
    // templated coordinate for any dimensionality
    template <std::uint64_t Dims>
        requires valid_dimension<Dims>
    struct coordinate_t {
        std::array<std::int64_t, Dims> coords;
        using value_type = std::int64_t;

        // constructors
        coordinate_t() : coords{} {}

        // 1d constructor
        coordinate_t(std::int64_t x)
            requires(Dims == 1)
            : coords{x}
        {
        }

        // 2d constructor
        coordinate_t(std::int64_t x, std::int64_t y)
            requires(Dims == 2)
            : coords{x, y}
        {
        }

        // 3d constructor
        coordinate_t(std::int64_t x, std::int64_t y, std::int64_t z)
            requires(Dims == 3)
            : coords{x, y, z}
        {
        }

        // array constructor
        coordinate_t(const std::array<std::int64_t, Dims>& arr) : coords(arr) {}

        DUAL constexpr std::array<std::uint64_t, Dims> to_domain_point() const
        {
            std::array<std::uint64_t, Dims> result;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                assert(
                    coords[i] >= 0 && "Coordinate must be non-negative for "
                                      "domain point conversion"
                );
                result[i] = static_cast<std::uint64_t>(coords[i]);
            }
            return result;
        }

        // check if this coordinate can be converted to a valid domain point
        DUAL constexpr bool is_valid_domain_point() const
        {
            for (std::uint64_t i = 0; i < Dims; ++i) {
                if (coords[i] < 0) {
                    return false;
                }
            }
            return true;
        }

        // create a coordinate from a domain point
        DUAL static constexpr coordinate_t<Dims>
        from_domain_point(const std::array<std::uint64_t, Dims>& point)
        {
            coordinate_t<Dims> result;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                result[i] = static_cast<std::int64_t>(point[i]);
            }
            return result;
        }

        // accessors
        std::int64_t& operator[](std::uint64_t i) { return coords[i]; }
        const std::int64_t& operator[](std::uint64_t i) const
        {
            return coords[i];
        }

        // named accessors for convenience
        std::int64_t& x()
            requires(Dims >= 1)
        {
            return coords[0];
        }
        const std::int64_t& x() const
            requires(Dims >= 1)
        {
            return coords[0];
        }

        std::int64_t& y()
            requires(Dims >= 2)
        {
            return coords[1];
        }
        const std::int64_t& y() const
            requires(Dims >= 2)
        {
            return coords[1];
        }

        std::int64_t& z()
            requires(Dims >= 3)
        {
            return coords[2];
        }
        const std::int64_t& z() const
            requires(Dims >= 3)
        {
            return coords[2];
        }

        // arithmetic operations
        coordinate_t operator+(const coordinate_t& other) const
        {
            coordinate_t result;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                result[i] = coords[i] + other[i];
            }
            return result;
        }

        coordinate_t operator-(const coordinate_t& other) const
        {
            coordinate_t result;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                result[i] = coords[i] - other[i];
            }
            return result;
        }

        // comparison operators
        bool operator==(const coordinate_t& other) const
        {
            return coords == other.coords;
        }

        auto operator<=>(const coordinate_t& other) const
        {
            return coords <=> other.coords;
        }

        void print() const
        {
            std::cout << "(";
            for (std::uint64_t i = 0; i < Dims; ++i) {
                if (i > 0) {
                    std::cout << ",";
                }
                std::cout << coords[i];
            }
            std::cout << ")";
        }

        constexpr std::uint64_t size() const { return Dims; }
        auto begin() { return coords.begin(); }
        auto end() { return coords.end(); }
        auto begin() const { return coords.begin(); }
        auto end() const { return coords.end(); }
    };

    // hash function for coordinate_t to use in unordered_map
    template <std::uint64_t Dims>
    struct coordinate_hash_t {
        std::uint64_t operator()(const coordinate_t<Dims>& coord) const
        {
            std::uint64_t hash_val = 0;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                hash_val ^= std::hash<std::int64_t>()(coord[i]) << i;
            }
            return hash_val;
        }
    };
}   // namespace simbi::base

#endif   // SIMBI_CORE_GRAPH_COORDINATE_HPP
