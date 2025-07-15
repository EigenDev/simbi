#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "compute/functional/fp.hpp"
#include "data/containers/vector.hpp"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>

namespace simbi {
    // forward declarations
    template <std::uint64_t Dims>
    struct domain_t;

    namespace set_ops {

        template <std::uint64_t Dims>
        constexpr auto
        difference(domain_t<Dims> full_domain, domain_t<Dims> active_domain);

        template <std::uint64_t Dims>
        constexpr auto intersection(domain_t<Dims> a, domain_t<Dims> b);

        template <std::uint64_t Dims>
        struct set_t {
            static constexpr std::size_t max_regions = []() {
                std::size_t total = 1;
                for (std::uint64_t i = 0; i < Dims; ++i) {
                    total *= 3;   // {before, inside, after} for each dim
                }
                return total - 1;   // minus the center "inside" region
            }();

            vector_t<domain_t<Dims>, max_regions> regions;
            std::size_t count = 0;   // actual number of non-empty regions

            // iterator support for FP toolkit
            auto begin() { return regions.begin(); }
            auto end() { return regions.begin() + count; }
            auto begin() const { return regions.begin(); }
            auto end() const { return regions.begin() + count; }
        };

        enum class ghost_type_t : std::uint8_t {
            face,       // ghost region touches active domain on one face
            edge,       // ghost region touches active domain on one edge
            corner,     // ghost region touches active domain on one corner
            interior,   // ghost region is fully inside the active domain
        };

        enum class direction_t : std::uint8_t {
            minus,   // ghost region is before the active domain
            plus,    // ghost region is after the active domain
            none,    // ghost doesn't touch this dim
        };

        // pure mathematical operations as free functions
        template <std::uint64_t Dims>
        constexpr auto expand(domain_t<Dims> d, const iarray<Dims>& amount)
        {
            return domain_t<Dims>{d.start - amount, d.fin + amount};
        }

        template <std::uint64_t Dims>
        constexpr auto contract(domain_t<Dims> d, const iarray<Dims>& amount)
        {
            return domain_t<Dims>{d.start + amount, d.fin - amount};
        }

        template <std::uint64_t Dims>
        constexpr auto
        center(domain_t<Dims> relative_domain, domain_t<Dims> base_domain)
        {
            // calculate the total padding required on each dimension
            auto total_padding = base_domain.shape() - relative_domain.shape();

            // calculate half padding (for each side)
            auto half_padding = total_padding / 2;

            // create new domain with equal padding on each side
            return domain_t<Dims>{
              base_domain.start + half_padding,
              base_domain.fin - half_padding
            };
        }

        template <std::uint64_t Dims>
        constexpr auto intersection(domain_t<Dims> a, domain_t<Dims> b)
        {
            const auto start =
                fp::zip(a.start, b.start) |
                fp::unpack_map([](auto x, auto y) { return std::max(x, y); }) |
                fp::collect<iarray<Dims>>;
            const auto end =
                fp::zip(a.fin, b.fin) |
                fp::unpack_map([](auto x, auto y) { return std::min(x, y); }) |
                fp::collect<iarray<Dims>>;

            return domain_t<Dims>{start, end};
        }

        // core ghost identification
        template <std::uint64_t Dims>
        auto identify_ghost_regions(
            domain_t<Dims> full_domain,
            domain_t<Dims> active_domain
        )
        {
            return difference(full_domain, active_domain);
        }

        // classify ghost regions by contact type
        template <std::uint64_t Dims>
        auto classify_ghost_contact(
            const domain_t<Dims>& ghost_region,
            const domain_t<Dims>& active_domain
        )
        {
            const auto gr = fp::zip(ghost_region.start, ghost_region.fin);
            const auto ad = fp::zip(active_domain.start, active_domain.fin);
            // how many dimensions does the ghost region touch the active
            // domain?
            auto contact_dims =
                fp::zip(gr, ad) | fp::unpack_map([](auto ghost, auto active) {
                    const auto [gs, ge] = ghost;
                    const auto [as, ae] = active;
                    return (gs == ae) ||
                           (ge == as);   // touching on this dimension
                }) |
                fp::collect<vector_t<bool, Dims>>;

            auto contact_count = contact_dims |
                                 fp::map([](auto b) { return b ? 1 : 0; }) |
                                 fp::sum;

            if (contact_count == 1) {
                return ghost_type_t::face;
            }
            if (contact_count == 2) {
                return ghost_type_t::edge;
            }
            if (contact_count == 3) {
                return ghost_type_t::corner;
            }
            return ghost_type_t::interior;   // shouldn't happen for proper
                                             // ghosts
        }

        // directional queries
        template <std::uint64_t Dims>
        auto ghost_direction(
            domain_t<Dims> ghost_region,
            domain_t<Dims> active_domain
        )
        {
            const auto gr = fp::zip(ghost_region.start, ghost_region.fin);
            const auto ad = fp::zip(active_domain.start, active_domain.fin);
            return fp::zip(gr, ad) |
                   fp::unpack_map([](auto ghost, auto active) {
                       const auto [gs, ge] = ghost;
                       const auto [as, ae] = active;
                       if (ge == as) {
                           return direction_t::minus;   // ghost is before
                                                        // active
                       }
                       if (gs == ae) {
                           return direction_t::plus;   // ghost is after active
                       }
                       return direction_t::none;   // ghost spans active in this
                                                   // dim
                   }) |
                   fp::collect<vector_t<direction_t, Dims>>;
        }

        // template <std::uint64_t Dims>
        // constexpr auto
        // difference(domain_t<Dims> full_domain, domain_t<Dims> active_domain)
        // {
        //     auto overlap = intersection(full_domain, active_domain);
        //     set_t<Dims> result;

        //     // no overlap - return original space
        //     if (overlap.empty()) {
        //         result.regions[0] = full_domain;
        //         result.count      = 1;
        //         return result;
        //     }

        //     // completely contained - return empty
        //     if (overlap.start == full_domain.start &&
        //         overlap.fin == full_domain.fin) {
        //         result.count = 0;
        //         return result;
        //     }

        //     // we'll generate 3^dims combinations (before/inside/after for
        //     each
        //     // dimension) and filter out invalid ones
        //     constexpr std::uint64_t total_combinations = []() {
        //         std::uint64_t total = 1;
        //         for (std::uint64_t i = 0; i < Dims; ++i) {
        //             total *= 3;
        //         }
        //         return total;
        //     }();

        //     // generate all combinations of before/inside/after for each
        //     // dimension
        //     fp::range(
        //         total_combinations
        //     ) | fp::filter([&](std::uint64_t combo) {
        //         // filter out the "all inside" case (active domain)
        //         std::uint64_t temp = combo;
        //         for (std::uint64_t dim = 0; dim < Dims; ++dim) {
        //             if (temp % 3 != 1) {   // not "inside" for this dimension
        //                 return true;
        //             }
        //             temp /= 3;
        //         }
        //         return false;   // all dimensions were "inside"
        //     }) | fp::for_each([&](std::uint64_t combo) {
        //         // for each valid combination, create a ghost region
        //         auto ghost_start  = full_domain.start;
        //         auto ghost_end    = full_domain.fin;
        //         bool valid_region = true;

        //         // decode the combination into positions
        //         std::uint64_t temp = combo;
        //         for (std::uint64_t dim = 0; dim < Dims && valid_region;
        //         ++dim) {
        //             const uint8_t pos =
        //                 temp % 3;   // 0=before, 1=inside, 2=after
        //             temp /= 3;

        //             switch (pos) {
        //                 case 0:   // before
        //                     if (full_domain.start[dim] >= overlap.start[dim])
        //                     {
        //                         valid_region = false;
        //                         break;
        //                     }
        //                     ghost_start[dim] = full_domain.start[dim];
        //                     ghost_end[dim]   = overlap.start[dim];
        //                     break;

        //                 case 1:   // inside
        //                     ghost_start[dim] = overlap.start[dim];
        //                     ghost_end[dim]   = overlap.fin[dim];
        //                     break;

        //                 case 2:   // after
        //                     if (overlap.fin[dim] >= full_domain.fin[dim]) {
        //                         valid_region = false;
        //                         break;
        //                     }
        //                     ghost_start[dim] = overlap.fin[dim];
        //                     ghost_end[dim]   = full_domain.fin[dim];
        //                     break;
        //             }
        //         }

        //         // add valid ghost region to the result
        //         if (valid_region && result.count < set_t<Dims>::max_regions)
        //         {
        //             result.regions[result.count++] =
        //                 domain_t<Dims>{ghost_start, ghost_end};
        //         }
        //     });

        //     return result;
        // }

        template <std::uint64_t Dims>
        constexpr auto
        difference(domain_t<Dims> full_domain, domain_t<Dims> active_domain)
        {
            auto overlap = intersection(full_domain, active_domain);
            set_t<Dims> result;

            // no overlap - return original space
            if (overlap.empty()) {
                result.regions[0] = full_domain;
                result.count      = 1;
                return result;
            }

            // completely contained - return empty
            if (overlap.start == full_domain.start &&
                overlap.fin == full_domain.fin) {
                result.count = 0;
                return result;
            }

            // create non-overlapping slabs by partitioning space
            for (std::uint64_t array_dim = 0; array_dim < Dims; ++array_dim) {
                // slab before overlap in this dimension
                if (full_domain.start[array_dim] < overlap.start[array_dim]) {
                    auto slab_start     = full_domain.start;
                    auto slab_end       = full_domain.fin;
                    slab_end[array_dim] = overlap.start[array_dim];
                    result.regions[result.count++] =
                        domain_t<Dims>{slab_start, slab_end};
                }

                // slab after overlap in this dimension
                if (overlap.fin[array_dim] < full_domain.fin[array_dim]) {
                    auto slab_start       = full_domain.start;
                    auto slab_end         = full_domain.fin;
                    slab_start[array_dim] = overlap.fin[array_dim];
                    result.regions[result.count++] =
                        domain_t<Dims>{slab_start, slab_end};
                }

                // constrain remaining dimensions to overlap range only
                if (array_dim < Dims - 1) {
                    full_domain.start[array_dim] = overlap.start[array_dim];
                    full_domain.fin[array_dim]   = overlap.fin[array_dim];
                }
            }

            return result;
        }

        template <std::uint64_t Dims>
        auto active_staggered_domain(
            domain_t<Dims> active_domain,
            std::uint64_t stag_dim
        )
        {
            auto staggered = active_domain;
            // staggered domain is one larger in the staggered direction
            staggered.fin[stag_dim] += 1;
            return staggered;
        }
    }   // namespace set_ops

    // factory functions for clean construction
    template <std::uint64_t Dims>
    auto make_domain(const iarray<Dims>& start, const iarray<Dims>& end)
    {
        return domain_t<Dims>{start, end};
    }

    template <std::uint64_t Dims>
    auto make_domain(const iarray<Dims>& shape)
    {
        return domain_t<Dims>{iarray<Dims>{0}, shape};
    }

    template <std::uint64_t Dims>
    struct domain_t {
        iarray<Dims> start, fin;

        // iterator for coordinate traversal with memory-order optimization
        class iterator
        {
          private:
            std::int64_t linear_index_;
            const domain_t* domain_;
            std::int64_t total_size_;

          public:
            using iterator_category = std::forward_iterator_tag;
            using value_type        = iarray<Dims>;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const value_type*;
            using reference         = value_type;

            constexpr iterator()
                : linear_index_(0), domain_(nullptr), total_size_(0)
            {
            }

            constexpr iterator(std::int64_t index, const domain_t* dom)
                : linear_index_(index), domain_(dom), total_size_(dom->size())
            {
            }

            constexpr reference operator*() const
            {
                return domain_->linear_to_coord(linear_index_);
            }

            constexpr iterator& operator++()
            {
                ++linear_index_;
                return *this;
            }

            constexpr iterator operator++(int)
            {
                auto tmp = *this;
                ++(*this);
                return tmp;
            }

            constexpr bool operator==(const iterator& other) const
            {
                return linear_index_ == other.linear_index_;
            }

            constexpr bool operator!=(const iterator& other) const
            {
                return !(*this == other);
            }
        };

        // basic queries
        constexpr std::uint64_t size() const noexcept
        {
            return static_cast<std::uint64_t>(shape() | fp::product);
        }

        constexpr auto shape() const noexcept { return fin - start; }

        constexpr bool empty() const noexcept
        {
            return shape() | fp::any_of([](auto dim) { return dim <= 0; });
        }

        constexpr iterator begin() const { return iterator{0, this}; }

        constexpr iterator end() const
        {
            return iterator{static_cast<std::int64_t>(size()), this};
        }

        // english-like queries
        constexpr bool contains(const iarray<Dims>& coord) const noexcept
        {
            auto in_bounds =
                fp::zip(coord, start) |
                fp::unpack_map([](auto c, auto s) { return c >= s; }) |
                fp::collect<vector_t<bool, Dims>>;
            auto below_end =
                fp::zip(coord, fin) |
                fp::unpack_map([](auto c, auto e) { return c < e; }) |
                fp::collect<vector_t<bool, Dims>>;

            return in_bounds | fp::all_of([](auto b) { return b; }) &&
                   below_end | fp::all_of([](auto b) { return b; });
        }

        constexpr bool contains(const domain_t& other) const noexcept
        {
            auto startok =
                fp::zip(other.start, start) |
                fp::unpack_map([](auto os, auto s) { return os >= s; }) |
                fp::collect<vector_t<bool, Dims>>;
            auto endok =
                fp::zip(other.fin, fin) |
                fp::unpack_map([](auto oe, auto e) { return oe <= e; }) |
                fp::collect<vector_t<bool, Dims>>;

            return startok | fp::all_of([](auto b) { return b; }) &&
                   endok | fp::all_of([](auto b) { return b; });
        }

        constexpr bool overlaps(const domain_t& other) const noexcept
        {
            return !intersection(*this, other).empty();
        }

        constexpr auto coord_to_linear(const iarray<Dims>& coord) const
        {
            auto local_coord  = coord - start;   // convert to local coordinates
            auto domain_shape = shape();

            std::int64_t index  = 0;
            std::int64_t stride = 1;

            // row-major: rightmost dimension changes fastest
            for (std::int64_t dim = Dims - 1; dim >= 0; --dim) {
                index += local_coord[dim] * stride;
                stride *= domain_shape[dim];
            }
            return index;
        }

        constexpr auto linear_to_coord(std::int64_t linear_idx) const
        {
            iarray<Dims> local_coord;
            auto domain_shape = shape();

            // unravel index in row-major order
            for (std::int64_t dim = Dims - 1; dim >= 0; --dim) {
                local_coord[dim] = linear_idx % domain_shape[dim];
                linear_idx /= domain_shape[dim];
            }

            return local_coord + start;   // convert back to global coordinates
        }

        template <typename Func>
        constexpr auto operator|(Func&& func) const
        {
            return std::forward<Func>(func)(*this);
        }
    };

    // ostream operator for easy printing
    template <std::uint64_t Dims>
    std::ostream& operator<<(std::ostream& os, const domain_t<Dims>& d)
    {
        os << "Domain(";
        for (std::uint64_t i = 0; i < Dims; ++i) {
            os << d.start[i] << ":" << d.fin[i];
            if (i < Dims - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
}   // namespace simbi

#endif
