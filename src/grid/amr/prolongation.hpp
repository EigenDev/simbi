#ifndef GRID_AMR_PROLONGATION_HPP
#define GRID_AMR_PROLONGATION_HPP

#include "base/concepts.hpp"
#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // constant prolongation kernel (pcm)
    // order 1: sample and hold
    // -------------------------------------------------------------------------
    template <typename CoarseComp, std::uint64_t Rank>
    struct prolong_constant_t {
        using vec_type = iarray<Rank>;
        using T        = typename CoarseComp::value_type;

        CoarseComp coarse_comp_;
        vec_type ratio_;

        DUAL prolong_constant_t(CoarseComp comp, vec_type ratio)
            : coarse_comp_(std::move(comp)), ratio_(ratio)
        {
        }

        DUAL T operator()(const vec_type& fine_coord) const
        {
            vec_type coarse_coord;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                // handle negative coords via floor division
                if (fine_coord[d] >= 0) {
                    coarse_coord[d] = fine_coord[d] / ratio_[d];
                }
                else {
                    coarse_coord[d] =
                        (fine_coord[d] - ratio_[d] + 1) / ratio_[d];
                }
            }

            return coarse_comp_(coarse_coord);
        }
    };

    // -------------------------------------------------------------------------
    // linear prolongation kernel (plm)
    // order 2: reconstruction with slope limiting
    // -------------------------------------------------------------------------
    template <typename CoarseComp, std::uint64_t Rank>
    struct prolong_linear_t {
        using vec_type = iarray<Rank>;
        using T        = std::remove_cvref_t<typename CoarseComp::value_type>;

        CoarseComp coarse_comp_;
        vec_type ratio_;
        vector_t<real, Rank> inv_ratio_;

        DUAL prolong_linear_t(CoarseComp comp, vec_type ratio)
            : coarse_comp_(std::move(comp)), ratio_(ratio)
        {
            for (std::uint64_t d = 0; d < Rank; ++d) {
                inv_ratio_[d] = 1.0 / static_cast<real>(ratio[d]);
            }
        }

        DUAL T operator()(const vec_type& fine_coord) const
        {
            vec_type coarse_coord;
            vector_t<real, Rank> normalized_offset;

            // map fine -> coarse and calculate sub-cell offset
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (fine_coord[d] >= 0) {
                    coarse_coord[d] = fine_coord[d] / ratio_[d];
                }
                else {
                    coarse_coord[d] =
                        (fine_coord[d] - ratio_[d] + 1) / ratio_[d];
                }

                std::int64_t local = fine_coord[d] % ratio_[d];
                if (local < 0) {
                    local += ratio_[d];
                }

                // map 0..ratio-1 to -0.5..0.5 relative to coarse center
                real fine_center   = static_cast<real>(local) + 0.5;
                real coarse_center = static_cast<real>(ratio_[d]) * 0.5;

                normalized_offset[d] =
                    (fine_center - coarse_center) * inv_ratio_[d];
            }

            // get base value
            T val = coarse_comp_(coarse_coord);

            // add gradient corrections
            // central difference slope
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                vec_type left  = coarse_coord;
                vec_type right = coarse_coord;
                left[dd] -= 1;
                right[dd] += 1;

                T v_left  = coarse_comp_(left);
                T v_right = coarse_comp_(right);

                // centered slope (unlimited for now, can inject limiter later)
                T slope = (v_right - v_left) * 0.5;

                if constexpr (is_hydro_primitive_c<T>) {
                    val = val | structs::add_gas(slope * normalized_offset[dd]);
                }
                else {
                    val = val + slope * normalized_offset[dd];
                }
            }

            return val;
        }
    };

    // -------------------------------------------------------------------------
    // factories / combinators
    // -------------------------------------------------------------------------

    // helper to scale domain
    template <std::uint64_t Rank>
    constexpr grid::domain_t<Rank>
    scale_domain_up(const grid::domain_t<Rank>& d, const iarray<Rank>& ratio)
    {
        auto start = d.start;
        auto fin   = d.fin;
        for (std::uint64_t i = 0; i < Rank; ++i) {
            start[i] *= ratio[i];
            fin[i] *= ratio[i];
        }
        return {start, fin};
    }

    // prolong: creates a computation on the fine domain
    // usage: fine_field = prolong<order>(coarse_field, ratio)
    template <std::int64_t Order = 2, typename Computation, std::uint64_t Rank>
    auto prolong(const Computation& coarse, iarray<Rank> ratio)
    {
        auto fine_domain = scale_domain_up(coarse.domain(), ratio);

        if constexpr (Order == 1) {
            auto kernel = prolong_constant_t<Computation, Rank>(coarse, ratio);
            return computation_t<Rank, decltype(kernel)>{
              std::move(kernel),
              fine_domain
            };
        }
        else {
            auto kernel = prolong_linear_t<Computation, Rank>(coarse, ratio);
            return computation_t<Rank, decltype(kernel)>{
              std::move(kernel),
              fine_domain
            };
        }
    }

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_PROLONGATION_HPP
