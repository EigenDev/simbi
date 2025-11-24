#ifndef GEOMETRY_COORDINATE_MAP_HPP
#define GEOMETRY_COORDINATE_MAP_HPP

#include "compat.hpp"

#include <cmath>
#include <cstdint>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // cell geometry info
    // return type for coordinate queries
    // -------------------------------------------------------------------------
    struct cell_interval_t {
        real start;   // left face x_i-1/2
        real end;     // right face x_i+1/2
        // centroid (geometric center, not necessarily arithmetic
        // avg)
        real center;
        real width;   // dx
    };

    // -------------------------------------------------------------------------
    // uniform map: x(i) = start + i * dx
    // -------------------------------------------------------------------------
    struct uniform_map_t {
        real start_;
        real dx_;

        DUAL constexpr uniform_map_t(real start, real dx)
            : start_(start), dx_(dx)
        {
        }

        // get full cell info at index i
        DUAL constexpr cell_interval_t operator()(std::int64_t ii) const
        {
            real x_l = start_ + static_cast<real>(ii) * dx_;
            real x_r = x_l + dx_;
            // for linear spacing, centroid is arithmetic mean
            return {x_l, x_r, 0.5 * (x_l + x_r), dx_};
        }

        // specific queries for efficiency
        DUAL constexpr real face(std::int64_t ii) const
        {
            return start_ + static_cast<real>(ii) * dx_;
        }

        DUAL constexpr real center(std::int64_t ii) const
        {
            return start_ + (static_cast<real>(ii) + 0.5) * dx_;
        }

        // map physical x -> index (for lookups)
        DUAL constexpr std::int64_t index_at(real x) const
        {
            return static_cast<std::int64_t>(std::floor((x - start_) / dx_));
        }
    };

    // -------------------------------------------------------------------------
    // log map: x(i) = start * (ratio ^ i)
    // often used for radial coordinates in astrophysics
    // -------------------------------------------------------------------------
    struct log_map_t {
        real start_;       // x_min
        real log_slope_;   // log10(x_max / x_min) / N

        DUAL log_map_t(real start, real log_slope)
            : start_(start), log_slope_(log_slope)
        {
        }

        DUAL constexpr cell_interval_t operator()(std::int64_t ii) const
        {
            // calculate faces using powers
            // x_i = start * 10^(i * slope)
            real x_l =
                start_ * std::pow(10.0, static_cast<real>(ii) * log_slope_);
            real x_r =
                start_ * std::pow(10.0, static_cast<real>(ii + 1) * log_slope_);

            // centroid for log spacing is often geometric mean: sqrt(a*b)
            // or volume-weighted centroid?
            // let's use geometric mean as the default "center" for log maps
            real x_c = std::sqrt(x_l * x_r);

            return {x_l, x_r, x_c, x_r - x_l};
        }

        DUAL constexpr real face(std::int64_t ii) const
        {
            return start_ * std::pow(10.0, static_cast<real>(ii) * log_slope_);
        }

        DUAL constexpr real center(std::int64_t ii) const
        {
            real x_l = face(ii);
            real x_r = face(ii + 1);
            return std::sqrt(x_l * x_r);
        }

        DUAL constexpr std::int64_t index_at(real x) const
        {
            // i = log10(x / start) / slope
            return static_cast<std::int64_t>(
                std::floor(std::log10(x / start_) / log_slope_)
            );
        }
    };

}   // namespace simbi::geometry

#endif   // GRID_GEOMETRY_COORDINATE_MAP_HPP
