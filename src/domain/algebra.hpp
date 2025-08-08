#ifndef DOMAIN_ALGEBRA_HPP
#define DOMAIN_ALGEBRA_HPP

#include "containers/vector.hpp"
#include "domain.hpp"
#include "functional/fp.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace simbi::domain_algebra {

    // set intersection - needed for overlap detection
    template <std::uint64_t Dims>
    constexpr auto
    intersection(const domain_t<Dims>& a, const domain_t<Dims>& b)
    {
        iarray<Dims> new_start, new_end;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            new_start[ii] = std::max(a.start[ii], b.start[ii]);
            new_end[ii]   = std::min(a.end[ii], b.end[ii]);
        }
        return domain_t<Dims>{new_start, new_end};
    }

    // set union - needed for domain merging
    template <std::uint64_t Dims>
    constexpr auto
    domain_union(const domain_t<Dims>& a, const domain_t<Dims>& b)
    {
        iarray<Dims> new_start, new_end;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            new_start[ii] = std::min(a.start[ii], b.start[ii]);
            new_end[ii]   = std::max(a.end[ii], b.end[ii]);
        }
        return domain_t<Dims>{new_start, new_end};
    }

    // expand domain by amount - needed for ghost regions
    template <std::uint64_t Dims>
    constexpr auto expand(const domain_t<Dims>& d, const iarray<Dims>& amount)
    {
        return domain_t<Dims>{d.start - amount, d.end + amount};
    }

    // expand end only
    template <std::uint64_t Dims>
    constexpr auto
    expand_end(const domain_t<Dims>& d, const iarray<Dims>& amount)
    {
        iarray<Dims> new_end = d.end + amount;
        return domain_t<Dims>{d.start, new_end};
    }

    // contract domain by amount - needed for active region extraction
    template <std::uint64_t Dims>
    constexpr auto contract(const domain_t<Dims>& d, const iarray<Dims>& amount)
    {
        return domain_t<Dims>{d.start + amount, d.end - amount};
    }

    // containment queries
    template <std::uint64_t Dims>
    constexpr bool
    contains(const domain_t<Dims>& container, const iarray<Dims>& point)
    {
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            if (point[ii] < container.start[ii] ||
                point[ii] >= container.end[ii]) {
                return false;
            }
        }
        return true;
    }

    template <std::uint64_t Dims>
    constexpr bool
    contains(const domain_t<Dims>& container, const domain_t<Dims>& contained)
    {
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            if (contained.start[ii] < container.start[ii] ||
                contained.end[ii] > container.end[ii]) {
                return false;
            }
        }
        return true;
    }

    // overlap detection - faster than computing full intersection
    template <std::uint64_t Dims>
    constexpr bool overlaps(const domain_t<Dims>& a, const domain_t<Dims>& b)
    {
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            if (a.end[ii] <= b.start[ii] || b.end[ii] <= a.start[ii]) {
                return false;
            }
        }
        return true;
    }

    // adjacency detection - domains touch but don't overlap
    template <std::uint64_t Dims>
    constexpr bool adjacent(const domain_t<Dims>& a, const domain_t<Dims>& b)
    {
        // must touch in exactly one dimension, overlap in all others
        std::uint64_t touching_dims = 0;
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            if (a.end[ii] == b.start[ii] || b.end[ii] == a.start[ii]) {
                touching_dims++;
            }
            else if (a.end[ii] <= b.start[ii] || b.end[ii] <= a.start[ii]) {
                // separated in this dimension
                return false;
            }
        }
        return touching_dims == 1;
    }

    // set difference container - holds non-overlapping result regions
    template <std::uint64_t Dims>
    struct difference_set_t {
        static constexpr std::size_t max_regions = []() {
            std::size_t total = 1;
            for (std::uint64_t i = 0; i < Dims; ++i) {
                total *= 3;   // {before, inside, after} for each dim
            }
            return total - 1;   // minus the center "inside" region
        }();

        vector_t<domain_t<Dims>, max_regions> regions;
        std::size_t count = 0;

        auto begin() { return regions.begin(); }
        auto end() { return regions.begin() + count; }
        auto begin() const { return regions.begin(); }
        auto end() const { return regions.begin() + count; }

        bool empty() const { return count == 0; }
    };

    template <std::uint64_t Dims>
    constexpr bool increment_base3_coord(iarray<Dims>& coord)
    {
        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            if (coord[dim] < 2) {
                coord[dim]++;
                return true;
            }
            coord[dim] = 0;   // carry to next dimension
        }
        return false;   // overflow - we're done
    }

    template <std::uint64_t Dims>
    constexpr auto difference(const domain_t<Dims>& a, const domain_t<Dims>& b)
    {
        auto overlap = intersection(a, b);
        difference_set_t<Dims> result;

        if (overlap.empty()) {
            result.regions[0] = a;
            result.count      = 1;
            return result;
        }

        if (overlap.start == a.start && overlap.end == a.end) {
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
        vector_t<vector_t<interval_t, 3>, Dims> interval_sets;

        for (std::uint64_t dim = 0; dim < Dims; ++dim) {
            interval_sets[dim] = {
              {// before: [a.start, overlap.start)
               {a.start[dim],
                overlap.start[dim],
                a.start[dim] < overlap.start[dim]},
               // overlap: [overlap.start, overlap.end)
               {overlap.start[dim], overlap.end[dim], true},
               // after: [overlap.end, a.end)
               {overlap.end[dim], a.end[dim], overlap.end[dim] < a.end[dim]}
              }
            };
        }

        // generate all 3^Dims combinations using base-3 increment
        iarray<Dims> indices{};
        do {
            // skip center region (all overlap intervals, i.e., all indices ==
            // 1)
            bool is_center = fp::range(Dims) | fp::all_of([&](auto dim) {
                                 return indices[dim] == 1;
                             });

            if (!is_center) {
                // check if all intervals in this combination are valid
                bool valid_combination =
                    fp::range(Dims) | fp::all_of([&](auto dim) {
                        return interval_sets[dim][indices[dim]].valid;
                    });

                if (valid_combination) {
                    domain_t<Dims> region;
                    for (std::uint64_t dim = 0; dim < Dims; ++dim) {
                        auto interval     = interval_sets[dim][indices[dim]];
                        region.start[dim] = interval.start;
                        region.end[dim]   = interval.end;
                    }

                    if (!region.empty()) {
                        result.regions[result.count++] = region;
                    }
                }
            }

        } while (increment_base3_coord(indices));

        return result;
    }

    template <std::uint64_t Dims>
    auto subdivide(const domain_t<Dims>& domain, const iarray<Dims>& divisions)
    {
        // reasonable max for most cases
        vector_t<domain_t<Dims>, 64> subdomains;
        std::size_t count = 0;

        auto chunk_sizes = domain.shape();
        for (std::uint64_t ii = 0; ii < Dims; ++ii) {
            chunk_sizes[ii] =
                (chunk_sizes[ii] + divisions[ii] - 1) / divisions[ii];
        }

        // generate all subdivision combinations
        iarray<Dims> div_coord{};
        do {
            iarray<Dims> sub_start, sub_end;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                sub_start[ii] =
                    domain.start[ii] + div_coord[ii] * chunk_sizes[ii];
                sub_end[ii] =
                    std::min(sub_start[ii] + chunk_sizes[ii], domain.end[ii]);
            }

            if (sub_start != sub_end) {   // non-empty subdomain
                subdomains[count++] = domain_t<Dims>{sub_start, sub_end};
            }

        } while (increment_coord(div_coord, divisions));

        return std::pair{subdomains, count};
    }

}   // namespace simbi::domain_algebra

#endif
