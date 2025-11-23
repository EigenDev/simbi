
#ifndef DOMAIN_ALGEBRA_HPP
#define DOMAIN_ALGEBRA_HPP

#include "containers/vector.hpp"
#include "domain.hpp"
#include "functional/fp.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::grid::domain_algebra {

    // set intersection - needed for overlap detection
    template <std::uint64_t Rank>
    constexpr auto
    intersection(const domain_t<Rank>& a, const domain_t<Rank>& b)
    {
        iarray<Rank> new_start, new_end;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            new_start[ii] = std::max(a.start[ii], b.start[ii]);
            new_end[ii]   = std::min(a.fin[ii], b.fin[ii]);
        }
        return domain_t<Rank>{new_start, new_end};
    }

    // set union - needed for domain merging
    template <std::uint64_t Rank>
    constexpr auto union_of(const domain_t<Rank>& a, const domain_t<Rank>& b)
    {
        iarray<Rank> new_start, new_end;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            new_start[ii] = std::min(a.start[ii], b.start[ii]);
            new_end[ii]   = std::max(a.fin[ii], b.fin[ii]);
        }
        return domain_t<Rank>{new_start, new_end};
    }

    // expand domain by amount - needed for ghost regions
    template <std::uint64_t Rank>
    constexpr auto expand(const domain_t<Rank>& d, const iarray<Rank>& amount)
    {
        return domain_t<Rank>{d.start - amount, d.fin + amount};
    }

    // expand end only
    template <std::uint64_t Rank>
    constexpr auto
    expand_end(const domain_t<Rank>& d, const iarray<Rank>& amount)
    {
        iarray<Rank> new_end = d.fin + amount;
        return domain_t<Rank>{d.start, new_end};
    }

    // contract domain by amount - needed for active region extraction
    template <std::uint64_t Rank>
    constexpr auto contract(const domain_t<Rank>& d, const iarray<Rank>& amount)
    {
        return domain_t<Rank>{d.start + amount, d.fin - amount};
    }

    // containment queries
    template <std::uint64_t Rank>
    constexpr bool
    contains(const domain_t<Rank>& container, const iarray<Rank>& point)
    {
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            if (point[ii] < container.start[ii] ||
                point[ii] >= container.fin[ii]) {
                return false;
            }
        }
        return true;
    }

    template <std::uint64_t Rank>
    constexpr bool
    contains(const domain_t<Rank>& container, const domain_t<Rank>& contained)
    {
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            if (contained.start[ii] < container.start[ii] ||
                contained.fin[ii] > container.fin[ii]) {
                return false;
            }
        }
        return true;
    }

    // overlap detection - faster than computing full intersection
    template <std::uint64_t Rank>
    constexpr bool overlaps(const domain_t<Rank>& a, const domain_t<Rank>& b)
    {
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            if (a.fin[ii] <= b.start[ii] || b.fin[ii] <= a.start[ii]) {
                return false;
            }
        }
        return true;
    }

    // adjacency detection - domains touch but don't overlap
    template <std::uint64_t Rank>
    constexpr bool adjacent(const domain_t<Rank>& a, const domain_t<Rank>& b)
    {
        // must touch in exactly one dimension, overlap in all others
        std::uint64_t touching_dims = 0;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            if (a.fin[ii] == b.start[ii] || b.fin[ii] == a.start[ii]) {
                touching_dims++;
            }
            else if (a.fin[ii] <= b.start[ii] || b.fin[ii] <= a.start[ii]) {
                // separated in this dimension
                return false;
            }
        }
        return touching_dims == 1;
    }

    // set difference container - holds non-overlapping result regions
    template <std::uint64_t Rank>
    struct difference_set_t {
        static constexpr std::size_t max_regions = []() {
            std::size_t total = 1;
            for (std::uint64_t i = 0; i < Rank; ++i) {
                total *= 3;   // {before, inside, after} for each dim
            }
            return total - 1;   // minus the center "inside" region
        }();

        vector_t<domain_t<Rank>, max_regions> regions;
        std::size_t count = 0;

        auto begin() { return regions.begin(); }
        auto end() { return regions.begin() + count; }
        auto begin() const { return regions.begin(); }
        auto end() const { return regions.begin() + count; }

        bool empty() const { return count == 0; }
    };

    template <std::uint64_t Rank>
    constexpr bool increment_base3_coord(iarray<Rank>& coord)
    {
        for (std::uint64_t dim = 0; dim < Rank; ++dim) {
            if (coord[dim] < 2) {
                coord[dim]++;
                return true;
            }
            coord[dim] = 0;   // carry to next dimension
        }
        return false;   // overflow - we're done
    }

    template <std::uint64_t Rank>
    constexpr auto difference(const domain_t<Rank>& a, const domain_t<Rank>& b)
    {
        auto overlap = intersection(a, b);
        difference_set_t<Rank> result;

        if (overlap.empty()) {
            result.regions[0] = a;
            result.count      = 1;
            return result;
        }

        if (overlap.start == a.start && overlap.fin == a.fin) {
            result.count = 0;
            return result;
        }

        // define interval type
        struct interval_t {
            std::int64_t start, end;
            bool valid;
        };

        // for each dimension, generate exactly 3 intervals: {before, overlap,
        // after}
        vector_t<vector_t<interval_t, 3>, Rank> interval_sets;

        for (std::uint64_t dim = 0; dim < Rank; ++dim) {
            interval_sets[dim] = {
              {// before: [a.start, overlap.start)
               {a.start[dim],
                overlap.start[dim],
                a.start[dim] < overlap.start[dim]},
               // overlap: [overlap.start, overlap.fin)
               {overlap.start[dim], overlap.fin[dim], true},
               // after: [overlap.fin, a.fin)
               {overlap.fin[dim], a.fin[dim], overlap.fin[dim] < a.fin[dim]}
              }
            };
        }

        // generate all 3^Rank combinations using base-3 increment
        iarray<Rank> indices{};
        do {
            // skip center region (all overlap intervals, i.e., all indices ==
            // 1)
            bool is_center = fp::range(Rank) | fp::all_of([&](auto dim) {
                                 return indices[dim] == 1;
                             });

            if (!is_center) {
                // check if all intervals in this combination are valid
                bool valid_combination =
                    fp::range(Rank) | fp::all_of([&](auto dim) {
                        return interval_sets[dim][indices[dim]].valid;
                    });

                if (valid_combination) {
                    domain_t<Rank> region;
                    for (std::uint64_t dim = 0; dim < Rank; ++dim) {
                        auto interval     = interval_sets[dim][indices[dim]];
                        region.start[dim] = interval.start;
                        region.fin[dim]   = interval.end;
                    }

                    if (!region.empty()) {
                        result.regions[result.count++] = region;
                    }
                }
            }

        } while (increment_base3_coord(indices));

        return result;
    }

    // get lower boundary region of specified width in dimension d
    template <std::uint64_t Rank>
    constexpr auto get_lower_boundary(
        const domain_t<Rank>& d,
        std::uint64_t dim,
        std::int64_t width
    )
    {
        auto boundary     = d;
        boundary.fin[dim] = boundary.start[dim] + width;
        return boundary;
    }

    // get upper boundary region of specified width in dimension d
    template <std::uint64_t Rank>
    constexpr auto get_upper_boundary(
        const domain_t<Rank>& d,
        std::uint64_t dim,
        std::int64_t width
    )
    {
        auto boundary       = d;
        boundary.start[dim] = boundary.fin[dim] - width;
        return boundary;
    }

    // shift a domain by a vector (v)
    // useful for "moving" a neighbor's domain to check connectivity
    template <std::uint64_t Rank>
    constexpr domain_t<Rank>
    shift(const domain_t<Rank>& d, const iarray<Rank>& v)
    {
        return domain_t<Rank>{d.start + v, d.fin + v};
    }

    // "wrap" a coordinate into the global domain [0, shape)
    // e.g. -1 -> 99 (if shape is 100)
    template <std::uint64_t Rank>
    constexpr iarray<Rank>
    wrap_coord(iarray<Rank> coord, const iarray<Rank>& global_shape)
    {
        iarray<Rank> wrapped;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            auto len = global_shape[ii];
            auto val = coord[ii] % len;
            if (val < 0) {
                val += len;
            }
            wrapped[ii] = val;
        }
        return wrapped;
    }

    // create a periodic image of a domain relative to the global box
    // direction: -1 (left image), +1 (right image), 0 (no shift)
    template <std::uint64_t Rank>
    constexpr domain_t<Rank> periodic_image(
        const domain_t<Rank>& d,
        const domain_t<Rank>& global_box,
        std::uint64_t dim,
        int direction
    )
    {
        iarray<Rank> shift_vec{};   // all zeros
        std::int64_t len = global_box.fin[dim] - global_box.start[dim];

        shift_vec[dim] = (direction * len);
        return shift(d, shift_vec);
    }

    // template <std::uint64_t Rank>
    // auto subdivide(const domain_t<Rank>& domain, const iarray<Rank>&
    // divisions)
    // {
    //     // reasonable max for most cases
    //     vector_t<domain_t<Rank>, 64> subdomains;
    //     std::size_t count = 0;

    //     auto chunk_sizes = domain.shape();
    //     for (std::uint64_t ii = 0; ii < Rank; ++ii) {
    //         chunk_sizes[ii] =
    //             (chunk_sizes[ii] + divisions[ii] - 1) / divisions[ii];
    //     }

    //     // generate all subdivision combinations
    //     iarray<Rank> div_coord{};
    //     do {
    //         iarray<Rank> sub_start, sub_end;
    //         for (std::uint64_t ii = 0; ii < Rank; ++ii) {
    //             sub_start[ii] =
    //                 domain.start[ii] + div_coord[ii] * chunk_sizes[ii];
    //             sub_end[ii] =
    //                 std::min(sub_start[ii] + chunk_sizes[ii],
    //                 domain.fin[ii]);
    //         }

    //         if (sub_start != sub_end) {   // non-empty subdomain
    //             subdomains[count++] = domain_t<Rank>{sub_start, sub_end};
    //         }

    //     } while (increment_coord(div_coord, divisions));

    //     return std::pair{subdomains, count};
    // }a

}   // namespace simbi::grid::domain_algebra

#endif
