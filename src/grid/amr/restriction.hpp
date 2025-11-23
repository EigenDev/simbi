#ifndef GRID_AMR_RESTRICTION_HPP
#define GRID_AMR_RESTRICTION_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // conservative restriction kernel
    // averages r^rank fine cells to produce one coarse cell value
    // -------------------------------------------------------------------------
    template <typename FineComp, std::uint64_t Rank>
    struct restrict_average_t {
        using vec_type = iarray<Rank>;
        using U        = typename FineComp::value_type;
        using T        = std::remove_cv_t<std::remove_reference_t<U>>;

        FineComp fine_comp_;
        vec_type ratio_;
        double inv_volume_;

        DUAL restrict_average_t(FineComp fine, vec_type ratio)
            : fine_comp_(std::move(fine)), ratio_(ratio), inv_volume_(1.0)
        {
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                inv_volume_ /= static_cast<double>(ratio[ii]);
            }
        }

        // input: coarse coordinate
        // action: loops over fine children and averages
        DUAL T operator()(const vec_type& coarse_coord) const
        {
            vec_type fine_base;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                fine_base[d] = coarse_coord[d] * ratio_[d];
            }

            T sum{};

            // manual loop unrolling for typical dimensions
            // we cannot use recursion easily in a device lambda/functor
            // without more template machinery
            if constexpr (Rank == 1) {
                for (std::int64_t i = 0; i < ratio_[0]; ++i) {
                    sum = sum + fine_comp_({fine_base[0] + i});
                }
            }
            else if constexpr (Rank == 2) {
                for (std::int64_t j = 0; j < ratio_[1]; ++j) {
                    for (std::int64_t i = 0; i < ratio_[0]; ++i) {
                        sum = sum +
                              fine_comp_({fine_base[0] + i, fine_base[1] + j});
                    }
                }
            }
            else if constexpr (Rank == 3) {
                for (std::int64_t k = 0; k < ratio_[2]; ++k) {
                    for (std::int64_t j = 0; j < ratio_[1]; ++j) {
                        for (std::int64_t i = 0; i < ratio_[0]; ++i) {
                            sum = sum + fine_comp_(
                                            {fine_base[0] + i,
                                             fine_base[1] + j,
                                             fine_base[2] + k}
                                        );
                        }
                    }
                }
            }

            const auto iv = inv_volume_;
            return sum * iv;
        }
    };

    // -------------------------------------------------------------------------
    // injection restriction kernel
    // samples a single fine cell
    // -------------------------------------------------------------------------
    template <typename FineComp, std::uint64_t Rank>
    struct restrict_injection_t {
        using vec_type = iarray<Rank>;
        using T        = typename FineComp::value_type;

        FineComp fine_comp_;
        vec_type ratio_;

        DUAL restrict_injection_t(FineComp fine, vec_type ratio)
            : fine_comp_(std::move(fine)), ratio_(ratio)
        {
        }

        DUAL T operator()(const vec_type& coarse_coord) const
        {
            vec_type fine_coord;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                fine_coord[d] = coarse_coord[d] * ratio_[d];
            }
            return fine_comp_(fine_coord);
        }
    };

    // -------------------------------------------------------------------------
    // factories / combinators
    // -------------------------------------------------------------------------

    // helper to scale domain down
    template <std::uint64_t Rank>
    constexpr grid::domain_t<Rank>
    scale_domain_down(const grid::domain_t<Rank>& d, const iarray<Rank>& ratio)
    {
        auto start = d.start;
        auto fin   = d.fin;
        for (std::uint64_t i = 0; i < Rank; ++i) {
            // assuming integer alignment (snapped grid)
            // otherwise floor/ceil logic might be needed for bounds
            start[i] /= ratio[i];
            fin[i] /= ratio[i];
        }
        return {start, fin};
    }

    // restrict: creates a computation on the coarse domain
    // usage: coarse_field = restrict<average>(fine_field, ratio)
    template <typename Computation, std::uint64_t Rank>
    auto restrict(const Computation& fine, iarray<Rank> ratio)
    {
        auto coarse_domain = scale_domain_down(fine.domain(), ratio);

        // default to averaging (conservative)
        auto kernel = restrict_average_t<Computation, Rank>(fine, ratio);

        return computation_t<Rank, decltype(kernel)>{
          std::move(kernel),
          coarse_domain
        };
    }

    template <typename Computation, std::uint64_t Rank>
    auto restrict_inject(const Computation& fine, iarray<Rank> ratio)
    {
        auto coarse_domain = scale_domain_down(fine.domain(), ratio);
        auto kernel = restrict_injection_t<Computation, Rank>(fine, ratio);

        return computation_t<Rank, decltype(kernel)>{
          std::move(kernel),
          coarse_domain
        };
    }

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_RESTRICTION_HPP
