#ifndef GRID_AMR_PROLONGATION_HPP
#define GRID_AMR_PROLONGATION_HPP

#include "base/concepts.hpp"
#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "utility/helpers.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // van leer slope limiter
    // -------------------------------------------------------------------------
    template <typename T>
    DUAL T limit_slope(const T& slope_left, const T& slope_right)
    {
        using namespace simbi::helpers;

        if constexpr (is_hydro_primitive_c<T> || is_hydro_conserved_c<T>) {
            T limited;
            for (std::uint64_t i = 0; i < T::nmem; ++i) {
                const auto sl = slope_left[i];
                const auto sr = slope_right[i];

                if (sl * sr <= 0) {
                    limited[i] = 0;   // opposite signs
                }
                else {
                    // van leer limiter
                    const auto r =
                        (std::abs(sl) < global::epsilon) ? 1.0 : sr / sl;
                    constexpr real theta = 2.0;
                    limited[i] =
                        sl *
                        my_max(
                            0.0,
                            my_min(theta * r, my_min((1.0 + r) * 0.5, theta))
                        );
                }
            }
            return limited;
        }
        else {
            // scalar type
            const auto sl = slope_left;
            const auto sr = slope_right;

            if (sl * sr <= 0) {
                return T{};
            }

            const auto r = (std::abs(sl) < global::epsilon) ? 1.0 : sr / sl;
            constexpr real theta = 2.0;
            return sl * my_max(
                            0.0,
                            my_min(theta * r, my_min((1.0 + r) * 0.5, theta))
                        );
        }
    }

    // -------------------------------------------------------------------------
    // constant prolongation kernel (pcm)
    // -------------------------------------------------------------------------
    template <typename CoarseComp, std::uint64_t Rank>
    struct prolong_constant_t {
        using value_type = std::remove_cvref_t<typename CoarseComp::value_type>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        CoarseComp coarse_comp_;
        argument_type ratio_;
        domain_t<Rank> coarse_domain_;

        DUAL prolong_constant_t(
            CoarseComp comp,
            argument_type ratio,
            domain_t<Rank> coarse_domain
        )
            : coarse_comp_(std::move(comp)),
              ratio_(ratio),
              coarse_domain_(coarse_domain)
        {
        }

        DUAL value_type operator()(const argument_type& fine_coord) const
        {
            argument_type coarse_coord;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (fine_coord[d] >= 0) {
                    coarse_coord[d] = fine_coord[d] / ratio_[d];
                }
                else {
                    coarse_coord[d] = (fine_coord[d] + 1) / ratio_[d] - 1;
                }
            }

            // clamp to valid coarse domain
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (coarse_coord[d] < coarse_domain_.start[d]) {
                    coarse_coord[d] = coarse_domain_.start[d];
                }
                else if (coarse_coord[d] >= coarse_domain_.fin[d]) {
                    coarse_coord[d] = coarse_domain_.fin[d] - 1;
                }
            }

            return coarse_comp_(coarse_coord);
        }
    };

    // -------------------------------------------------------------------------
    // linear prolongation kernel with slope limiting (plm)
    // -------------------------------------------------------------------------
    template <typename CoarseComp, std::uint64_t Rank>
    struct prolong_linear_t {
        using value_type = std::remove_cvref_t<typename CoarseComp::value_type>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        CoarseComp coarse_comp_;
        argument_type ratio_;
        vector_t<real, Rank> inv_ratio_;
        domain_t<Rank> coarse_domain_;

        DUAL prolong_linear_t(
            CoarseComp comp,
            argument_type ratio,
            domain_t<Rank> coarse_domain
        )
            : coarse_comp_(std::move(comp)),
              ratio_(ratio),
              coarse_domain_(coarse_domain)
        {
            for (std::uint64_t d = 0; d < Rank; ++d) {
                inv_ratio_[d] = 1.0 / static_cast<real>(ratio[d]);
            }
        }

        DUAL value_type operator()(const argument_type& fine_coord) const
        {
            argument_type coarse_coord;
            vector_t<real, Rank> normalized_offset;

            // map fine -> coarse
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (fine_coord[d] >= 0) {
                    coarse_coord[d] = fine_coord[d] / ratio_[d];
                }
                else {
                    coarse_coord[d] = (fine_coord[d] + 1) / ratio_[d] - 1;
                }

                // compute normalized position within coarse cell [-0.5, 0.5]
                std::int64_t local =
                    fine_coord[d] - coarse_coord[d] * ratio_[d];
                real fine_center   = static_cast<real>(local) + 0.5;
                real coarse_center = static_cast<real>(ratio_[d]) * 0.5;
                normalized_offset[d] =
                    (fine_center - coarse_center) * inv_ratio_[d];
            }

            // boundary check and clamp
            bool out_of_bounds = false;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (coarse_coord[d] < coarse_domain_.start[d] ||
                    coarse_coord[d] >= coarse_domain_.fin[d]) {
                    out_of_bounds   = true;
                    coarse_coord[d] = simbi::helpers::my_max(
                        coarse_domain_.start[d],
                        simbi::helpers::my_min(
                            coarse_coord[d],
                            coarse_domain_.fin[d] - 1
                        )
                    );
                }
            }

            value_type v0 = coarse_comp_(coarse_coord);

            if (out_of_bounds) {
                return v0;   // fallback to pcm at boundaries
            }

            // compute limited slopes
            value_type result = v0;
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                argument_type left  = coarse_coord;
                argument_type right = coarse_coord;
                left[dd] -= 1;
                right[dd] += 1;

                bool has_left  = coarse_domain_.contains(left);
                bool has_right = coarse_domain_.contains(right);

                value_type slope;
                if (has_left && has_right) {
                    // centered difference with limiting
                    value_type v_left  = coarse_comp_(left);
                    value_type v_right = coarse_comp_(right);
                    value_type slope_l = v0 - v_left;
                    value_type slope_r = v_right - v0;
                    slope              = limit_slope(slope_l, slope_r);
                }
                else if (has_left) {
                    // backward difference
                    value_type v_left  = coarse_comp_(left);
                    value_type slope_l = v0 - v_left;
                    slope              = limit_slope(slope_l, slope_l);
                }
                else if (has_right) {
                    // forward difference
                    value_type v_right = coarse_comp_(right);
                    value_type slope_r = v_right - v0;
                    slope              = limit_slope(slope_r, slope_r);
                }
                else {
                    // no neighbors
                    slope = value_type{};
                }

                // apply slope
                if constexpr (is_hydro_primitive_c<value_type> ||
                              is_hydro_conserved_c<value_type>) {
                    result = result |
                             structs::add_gas(slope * normalized_offset[dd]);
                }
                else {
                    result = result + slope * normalized_offset[dd];
                }
            }

            return result;
        }
    };

    // -------------------------------------------------------------------------
    // parabolic prolongation kernel (ppm-like)
    // -------------------------------------------------------------------------
    template <typename CoarseComp, std::uint64_t Rank>
    struct prolong_parabolic_t {
        using value_type = std::remove_cvref_t<typename CoarseComp::value_type>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        CoarseComp coarse_comp_;
        argument_type ratio_;
        vector_t<real, Rank> inv_ratio_;
        domain_t<Rank> coarse_domain_;

        DUAL prolong_parabolic_t(
            CoarseComp comp,
            argument_type ratio,
            domain_t<Rank> coarse_domain
        )
            : coarse_comp_(std::move(comp)),
              ratio_(ratio),
              coarse_domain_(coarse_domain)
        {
            for (std::uint64_t d = 0; d < Rank; ++d) {
                inv_ratio_[d] = 1.0 / static_cast<real>(ratio[d]);
            }
        }

        DUAL value_type operator()(const argument_type& fine_coord) const
        {
            argument_type coarse_coord;
            vector_t<real, Rank> x_norm;

            // map fine -> coarse
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (fine_coord[d] >= 0) {
                    coarse_coord[d] = fine_coord[d] / ratio_[d];
                }
                else {
                    coarse_coord[d] = (fine_coord[d] + 1) / ratio_[d] - 1;
                }

                // normalized position within coarse cell [-0.5, 0.5]
                std::int64_t offset =
                    fine_coord[d] - coarse_coord[d] * ratio_[d];
                x_norm[d] =
                    (static_cast<real>(offset) + 0.5) * inv_ratio_[d] - 0.5;
            }

            // boundary check
            bool out_of_bounds = false;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                if (coarse_coord[d] < coarse_domain_.start[d] ||
                    coarse_coord[d] >= coarse_domain_.fin[d]) {
                    out_of_bounds   = true;
                    coarse_coord[d] = simbi::helpers::my_max(
                        coarse_domain_.start[d],
                        simbi::helpers::my_min(
                            coarse_coord[d],
                            coarse_domain_.fin[d] - 1
                        )
                    );
                }
            }

            value_type v0 = coarse_comp_(coarse_coord);

            if (out_of_bounds) {
                return v0;   // fallback to pcm
            }

            value_type result = v0;

            // dimension-by-dimension reconstruction
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                argument_type left  = coarse_coord;
                argument_type right = coarse_coord;
                left[dd] -= 1;
                right[dd] += 1;

                bool has_left  = coarse_domain_.contains(left);
                bool has_right = coarse_domain_.contains(right);

                if (has_left && has_right) {
                    // full 3-point stencil: parabolic reconstruction
                    value_type v_left  = coarse_comp_(left);
                    value_type v_right = coarse_comp_(right);

                    value_type slope_l = v0 - v_left;
                    value_type slope_r = v_right - v0;
                    value_type slope   = limit_slope(slope_l, slope_r);

                    // second derivative (curvature)
                    value_type d2v = v_right - v0 * 2.0 + v_left;

                    // monotonicity check: flatten if curvature opposes slope
                    value_type d2v_limited = d2v;
                    if constexpr (is_hydro_primitive_c<value_type> ||
                                  is_hydro_conserved_c<value_type>) {
                        for (std::uint64_t i = 0; i < value_type::nmem; ++i) {
                            if (d2v[i] * slope[i] < 0.0) {
                                d2v_limited[i] = 0.0;
                            }
                        }
                    }
                    else {
                        if (d2v * slope < 0.0) {
                            d2v_limited = value_type{};
                        }
                    }

                    // add linear term
                    if constexpr (is_hydro_primitive_c<value_type> ||
                                  is_hydro_conserved_c<value_type>) {
                        result = result | structs::add_gas(slope * x_norm[dd]);
                    }
                    else {
                        result = result + slope * x_norm[dd];
                    }

                    // add parabolic term: (d2v/2) * (x^2 - 1/12)
                    real parabolic_factor =
                        x_norm[dd] * x_norm[dd] - (1.0 / 12.0);
                    if constexpr (is_hydro_primitive_c<value_type> ||
                                  is_hydro_conserved_c<value_type>) {
                        result =
                            result | structs::add_gas(
                                         (d2v_limited * 0.5) * parabolic_factor
                                     );
                    }
                    else {
                        result =
                            result + (d2v_limited * 0.5) * parabolic_factor;
                    }
                }
                else {
                    // fallback to linear at boundaries
                    value_type slope;
                    if (has_left) {
                        value_type v_left  = coarse_comp_(left);
                        value_type slope_l = v0 - v_left;
                        slope              = limit_slope(slope_l, slope_l);
                    }
                    else if (has_right) {
                        value_type v_right = coarse_comp_(right);
                        value_type slope_r = v_right - v0;
                        slope              = limit_slope(slope_r, slope_r);
                    }
                    else {
                        slope = value_type{};
                    }

                    if constexpr (is_hydro_primitive_c<value_type> ||
                                  is_hydro_conserved_c<value_type>) {
                        result = result | structs::add_gas(slope * x_norm[dd]);
                    }
                    else {
                        result = result + slope * x_norm[dd];
                    }
                }
            }

            return result;
        }
    };

    // -------------------------------------------------------------------------
    // helper to scale domain up
    // -------------------------------------------------------------------------
    template <std::uint64_t Rank>
    constexpr grid::domain_t<Rank>
    scale_domain_up(const grid::domain_t<Rank>& d, const iarray<Rank>& ratio)
    {
        auto start = d.start;
        auto fin   = d.fin;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            start[ii] *= ratio[ii];
            fin[ii] *= ratio[ii];
        }
        return {start, fin};
    }

    // -------------------------------------------------------------------------
    // factories
    // -------------------------------------------------------------------------
    template <std::int64_t Order = 2, typename Computation, std::uint64_t Rank>
    auto prolong(const Computation& coarse, iarray<Rank> ratio)
    {
        auto fine_domain   = scale_domain_up(coarse.domain(), ratio);
        auto coarse_domain = coarse.domain();

        if constexpr (Order == 1) {
            auto kernel = prolong_constant_t<Computation, Rank>(
                coarse,
                ratio,
                coarse_domain
            );
            return computation_t<Rank, decltype(kernel)>{
              std::move(kernel),
              fine_domain
            };
        }
        else if constexpr (Order == 2) {
            auto kernel = prolong_linear_t<Computation, Rank>(
                coarse,
                ratio,
                coarse_domain
            );
            return computation_t<Rank, decltype(kernel)>{
              std::move(kernel),
              fine_domain
            };
        }
        else if constexpr (Order == 3) {
            auto kernel = prolong_parabolic_t<Computation, Rank>(
                coarse,
                ratio,
                coarse_domain
            );
            return computation_t<Rank, decltype(kernel)>{
              std::move(kernel),
              fine_domain
            };
        }
        else {
            static_assert(Order >= 1 && Order <= 3, "Order must be 1, 2, or 3");
        }
    }

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_PROLONGATION_HPP
