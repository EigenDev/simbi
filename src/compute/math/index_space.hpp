#ifndef SIMBI_INDEX_SPACE_HPP
#define SIMBI_INDEX_SPACE_HPP

#include "compute/math/cfd_expressions.hpp"
#include "index_types.hpp"
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>

namespace simbi {

    template <typename T, std::uint64_t Dims>
    struct vector_t;

    template <std::uint64_t Dims>
    using uarray = vector_t<std::uint64_t, Dims>;

    using ulist = std::initializer_list<std::uint64_t>;

    template <std::uint64_t Dims>
    struct index_space_t {
        static constexpr std::uint64_t dimensions = Dims;
        using value_type                          = std::uint64_t;
        uarray<Dims> start_, end_;

        // calculate total number of coordinates in this space
        std::uint64_t size() const
        {
            std::uint64_t total = 1;
            for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                total *= (end_[dim] - start_[dim]);
            }
            return total;
        }

        // get domain extents (CRITICAL for field integration)
        uarray<Dims> shape() const
        {
            uarray<Dims> extents;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                extents[i] = end_[i] - start_[i];
            }
            return extents;
        }

        std::uint64_t coord_to_linear_index(const uarray<Dims>& coord) const
        {
            uarray<Dims> domain_shape = shape();

            // convert coordinate relative to domain start
            uarray<Dims> original_coord;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                original_coord[ii] = coord[ii] + start_[ii];
            }

            // row-major linear index calculation
            std::uint64_t index = original_coord[0];
            for (std::uint64_t dim = 1; dim < Dims; ++dim) {
                index = index * domain_shape[dim] + original_coord[dim];
            }
            return index;
        }

        // convert linear index to coordinate within this space
        uarray<Dims> index_to_coord(std::uint64_t linear_idx) const
        {
            uarray<Dims> coord;
            std::uint64_t remaining = linear_idx;

            for (int dim = Dims - 1; dim >= 0; --dim) {
                std::uint64_t extent = end_[dim] - start_[dim];
                coord[dim]           = start_[dim] + (remaining % extent);
                remaining /= extent;
            }

            return coord;
        }

        // access a subdomain of this space
        index_space_t
        subdomain(const uarray<Dims>& start, const uarray<Dims>& end) const
        {
            uarray<Dims> new_start, new_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] = std::max(start_[ii], start[ii]);
                new_end[ii]   = std::min(end_[ii], end[ii]);
            }
            return index_space_t{new_start, new_end};
        }

        // subdomain access operator
        index_space_t operator[](const index_space_t& coord) const
        {
            uarray<Dims> new_start, new_end, radii;
            // compute the radii for each dimensions. this is the
            // difference between the length of the original
            // coordinate space and the new coordinate space.
            // If my original coordinate space is [0, 104] (1D) and
            // the new coordinate space is [0, 100], then the
            // radii is 4/2 = 2. This allows us to shift the
            // index space and perform affine transformations
            // correctly between active and global dims.
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                const auto new_length = coord.end_[ii] - coord.start_[ii];
                const auto old_length = end_[ii] - start_[ii];
                if (old_length > new_length) {
                    radii[ii] = (old_length - new_length) / 2;
                }
                else {
                    radii[ii] = 0;
                }
            }

            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] =
                    std::max(start_[ii], coord.start_[ii]) + radii[ii];
                new_end[ii] = std::min(end_[ii], coord.end_[ii]) + radii[ii];
            }
            return index_space_t{new_start, new_end};
        }

        index_space_t intersection(const index_space_t& other) const
        {
            uarray<Dims> new_start, new_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] = std::max(start_[ii], other.start_[ii]);
                new_end[ii]   = std::min(end_[ii], other.end_[ii]);
            }
            return index_space_t{new_start, new_end};
        }

        // uniform expansion/contraction
        index_space_t expand(std::uint64_t radius) const
        {
            uarray<Dims> radii;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                radii[ii] = radius;
            }
            return expand(radii);
        }

        index_space_t contract(std::uint64_t radius) const
        {
            uarray<Dims> radii;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                radii[ii] = radius;
            }
            return contract(radii);
        }

        // firectional expansion/contraction
        index_space_t expand(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_start, new_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] =
                    (start_[ii] >= radii[ii]) ? start_[ii] - radii[ii] : 0;
                new_end[ii] = end_[ii] + radii[ii];
            }
            return index_space_t{new_start, new_end};
        }

        index_space_t contract(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_start, new_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] = start_[ii] + radii[ii];
                new_end[ii] =
                    (end_[ii] >= radii[ii]) ? end_[ii] - radii[ii] : start_[ii];
            }
            return index_space_t{new_start, new_end};
        }

        // initializer list overloads
        index_space_t contract(const ulist& list) const
        {
            uarray<Dims> radii;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (ii < list.size()) {
                    radii[ii] = *(std::begin(list) + ii);
                }
                else {
                    radii[ii] = 0;
                }
            }
            return contract(radii);
        }

        index_space_t expand(const ulist& list) const
        {
            uarray<Dims> radii;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (ii < list.size()) {
                    radii[ii] = *(std::begin(list) + ii);
                }
                else {
                    radii[ii] = 0;
                }
            }
            return expand(radii);
        }

        // FINE-GRAINED BOUNDARY CONTROL
        index_space_t expand_start(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_start = start_;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] =
                    (start_[ii] >= radii[ii]) ? start_[ii] - radii[ii] : 0;
            }
            return index_space_t{new_start, end_};
        }

        index_space_t expand_start(const ulist& radii) const
        {
            uarray<Dims> new_start = start_;
            for (std::uint64_t ii = 0; ii < Dims && ii < radii.size(); ++ii) {
                const auto rad = *(std::begin(radii) + ii);
                new_start[ii]  = (start_[ii] >= rad) ? start_[ii] - rad : 0;
            }
            return index_space_t{new_start, end_};
        }

        index_space_t expand_end(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_end = end_;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_end[ii] = end_[ii] + radii[ii];
            }
            return index_space_t{start_, new_end};
        }

        index_space_t expand_end(const ulist& radii) const
        {
            uarray<Dims> new_end = end_;
            for (std::uint64_t ii = 0; ii < Dims && ii < radii.size(); ++ii) {
                new_end[ii] = end_[ii] + *(std::begin(radii) + ii);
            }
            return index_space_t{start_, new_end};
        }

        index_space_t contract_start(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_start = start_;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] = start_[ii] + radii[ii];
            }
            return index_space_t{new_start, end_};
        }

        index_space_t contract_start(const ulist& radii) const
        {
            uarray<Dims> new_start = start_;
            for (std::uint64_t ii = 0; ii < Dims && ii < radii.size(); ++ii) {
                new_start[ii] = start_[ii] + *(std::begin(radii) + ii);
            }
            return index_space_t{new_start, end_};
        }

        index_space_t contract_end(const uarray<Dims>& radii) const
        {
            uarray<Dims> new_end = end_;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_end[ii] =
                    (end_[ii] >= radii[ii]) ? end_[ii] - radii[ii] : start_[ii];
            }
            return index_space_t{start_, new_end};
        }

        index_space_t contract_end(const ulist& radii) const
        {
            uarray<Dims> new_end = end_;
            for (std::uint64_t ii = 0; ii < Dims && ii < radii.size(); ++ii) {
                const auto rad = *(std::begin(radii) + ii);
                new_end[ii] = (end_[ii] >= rad) ? end_[ii] - rad : start_[ii];
            }
            return index_space_t{start_, new_end};
        }

        // UTILITY OPERATIONS

        // bounds checking and safety
        index_space_t clamp_to(const index_space_t& bounds) const
        {
            uarray<Dims> new_start, new_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                new_start[ii] = std::max(start_[ii], bounds.start_[ii]);
                new_end[ii]   = std::min(end_[ii], bounds.end_[ii]);
            }
            return index_space_t{new_start, new_end};
        }

        // check if coordinate is within this space
        bool contains(const uarray<Dims>& coord) const
        {
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                if (coord[ii] < start_[ii] || coord[ii] >= end_[ii]) {
                    return false;
                }
            }
            return true;
        }

        // alias for more intuitive usage
        bool is_valid_coord(const uarray<Dims>& coord) const
        {
            return contains(coord);
        }

        // ACCESSORS
        const uarray<Dims>& start() const { return start_; }
        const uarray<Dims>& end() const { return end_; }

        // SEMANTIC TYPE CONVERSION
        template <index_semantic_t Semantic>
        auto as_semantic() const
        {
            return semantic_space_t<Dims, Semantic>{start_, end_};
        }

        // enable pipeline syntax from domain
        template <typename Op>
        auto operator|(Op&& op) const
        {
            return cfd::make_domain_expr(*this) | std::forward<Op>(op);
        }
    };

    // FACTORY FUNCTIONS
    template <std::uint64_t Dims>
    auto make_space(const uarray<Dims>& start, const uarray<Dims>& end)
    {
        return index_space_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
    auto make_space(const uarray<Dims>& end)
    {
        return index_space_t<Dims>{uarray<Dims>{}, end};
    }

    template <std::uint64_t Dims>
    auto make_space(const ulist& lstart, const ulist& lend)
    {
        uarray<Dims> start, end;
        for (std::uint64_t ii = 0; ii < Dims; ii++) {
            start[ii] = *(std::begin(lstart) + ii);
            end[ii]   = *(std::begin(lend) + ii);
        }
        return index_space_t<Dims>{start, end};
    }

}   // namespace simbi

#endif   // SIMBI_INDEX_SPACE_HPP
