/**
 * pattern.hpp
 * hardware-agnostic description of stencil patterns
 */
#ifndef SIMBI_PARALLEL_PATTERN_HPP
#define SIMBI_PARALLEL_PATTERN_HPP

#include "core/containers/array.hpp"
#include "core/types/alias/alias.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>

namespace simbi::parallel {

    /**
     * represents a relative offset in a stencil pattern
     */
    template <size_type Dims>
    struct offset_t {
        array_t<int, Dims> indices;

        // comparison operators for containers
        bool operator==(const offset_t<Dims>& other) const
        {
            return indices == other.indices;
        }

        bool operator<(const offset_t<Dims>& other) const
        {
            return indices < other.indices;
        }

        // common offsets
        static constexpr offset_t zero()
        {
            array_t<int, Dims> zeros{};
            return {zeros};
        }

        static constexpr offset_t unit(size_type dim, int direction = 1)
        {
            array_t<int, Dims> indices{};
            if (dim < Dims) {
                indices[dim] = direction;
            }
            return {indices};
        }

        // manhattan distance from origin
        constexpr int manhattan_distance() const
        {
            int dist = 0;
            for (size_type i = 0; i < Dims; ++i) {
                dist += std::abs(indices[i]);
            }
            return dist;
        }

        // euclidean distance squared from origin
        constexpr int euclidean_distance_squared() const
        {
            int dist_sq = 0;
            for (size_type i = 0; i < Dims; ++i) {
                dist_sq += indices[i] * indices[i];
            }
            return dist_sq;
        }
    };

    // direction helpers for readability
    namespace direction {
        template <size_type Dims>
        constexpr auto center()
        {
            return offset_t<Dims>::zero();
        }

        template <size_type Dims>
        constexpr auto east()
        {
            return offset_t<Dims>::unit(0, 1);
        }

        template <size_type Dims>
        constexpr auto west()
        {
            return offset_t<Dims>::unit(0, -1);
        }

        template <size_type Dims>
        constexpr auto north()
        {
            return offset_t<Dims>::unit(1, 1);
        }

        template <size_type Dims>
        constexpr auto south()
        {
            return offset_t<Dims>::unit(1, -1);
        }

        template <size_type Dims>
        constexpr auto up()
        {
            return offset_t<Dims>::unit(2, 1);
        }

        template <size_type Dims>
        constexpr auto down()
        {
            return offset_t<Dims>::unit(2, -1);
        }
    }   // namespace direction

    /**
     * collection of offsets that define a stencil pattern
     */
    template <size_type Dims>
    class pattern_t
    {
      public:
        using offset_type = offset_t<Dims>;

        // default constructor - just center point
        constexpr pattern_t() : offsets_{offset_type::zero()} {}

        // construct from initializer_list
        constexpr pattern_t(std::initializer_list<offset_type> offsets)
            : offsets_(offsets)
        {
        }

        // construct from vector
        constexpr pattern_t(const std::vector<offset_type>& offsets)
            : offsets_(offsets)
        {
        }

        // access the offsets
        constexpr const auto& offsets() const { return offsets_; }

        // get number of points in the pattern
        constexpr size_type size() const { return offsets_.size(); }

        // get required halo size for this pattern
        constexpr array_t<size_type, Dims> halo_size() const
        {
            array_t<size_type, Dims> halo{};
            for (const auto& off : offsets_) {
                for (size_type i = 0; i < Dims; ++i) {
                    halo[i] = std::max(
                        halo[i],
                        static_cast<size_type>(std::abs(off.indices[i]))
                    );
                }
            }
            return halo;
        }

        // get maximum halo size across all dimensions
        constexpr size_type max_halo_size() const
        {
            size_type max_halo = 0;
            for (const auto& off : offsets_) {
                for (size_type i = 0; i < Dims; ++i) {
                    max_halo = std::max(
                        max_halo,
                        static_cast<size_type>(std::abs(off.indices[i]))
                    );
                }
            }
            return max_halo;
        }

        // basic patterns

        // center point only
        static constexpr pattern_t identity()
        {
            return pattern_t{offset_type::zero()};
        }

        // von Neumann neighborhood (center + direct neighbors)
        static constexpr pattern_t von_neumann(size_type radius = 1)
        {
            std::vector<offset_type> offsets;
            offsets.push_back(offset_type::zero());   // center point

            for (size_type dim = 0; dim < Dims; ++dim) {
                for (int r = 1; r <= static_cast<int>(radius); ++r) {
                    auto pos = offset_type::zero();
                    auto neg = offset_type::zero();

                    pos.indices[dim] = r;
                    neg.indices[dim] = -r;

                    offsets.push_back(pos);
                    offsets.push_back(neg);
                }
            }

            return pattern_t(offsets);
        }

        // Moore neighborhood (all points within a given radius)
        static pattern_t moore(size_type radius = 1)
        {
            std::vector<offset_type> offsets;

            // This is more complex for arbitrary dimensions
            // For simplicity, we'll just enumerate all points within the radius
            // and filter by Euclidean distance

            // Helper function to recursively generate points
            std::function<void(array_t<int, Dims>&, size_type)> generate;
            generate = [&](array_t<int, Dims>& point, size_type dim) {
                if (dim == Dims) {
                    offset_type off{point};
                    if (off.euclidean_distance_squared() <=
                        static_cast<int>(radius * radius)) {
                        offsets.push_back(off);
                    }
                    return;
                }

                for (int i = -static_cast<int>(radius);
                     i <= static_cast<int>(radius);
                     ++i) {
                    point[dim] = i;
                    generate(point, dim + 1);
                }
            };

            array_t<int, Dims> point{};
            generate(point, 0);

            return pattern_t(offsets);
        }

        // pattern for upwind schemes
        static pattern_t upwind(size_type dim, bool positive = true)
        {
            std::vector<offset_type> offsets;

            // center point
            offsets.push_back(offset_type::zero());

            // upwind direction
            auto upwind_offset         = offset_type::zero();
            upwind_offset.indices[dim] = positive ? -1 : 1;
            offsets.push_back(upwind_offset);

            return pattern_t(offsets);
        }

        // centered difference pattern
        static pattern_t centered(size_type dim)
        {
            std::vector<offset_type> offsets;

            // center point
            offsets.push_back(offset_type::zero());

            // left and right neighbors
            auto left          = offset_type::zero();
            auto right         = offset_type::zero();
            left.indices[dim]  = -1;
            right.indices[dim] = 1;
            offsets.push_back(left);
            offsets.push_back(right);

            return pattern_t(offsets);
        }

        // combine two patterns
        pattern_t operator+(const pattern_t& other) const
        {
            std::vector<offset_type> combined = offsets_;
            combined.insert(
                combined.end(),
                other.offsets().begin(),
                other.offsets().end()
            );
            return pattern_t(combined);
        }

      private:
        std::vector<offset_type> offsets_;
    };

}   // namespace simbi::parallel

#endif   // SIMBI_PARALLEL_PATTERN_HPP
