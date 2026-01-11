#ifndef GRID_AMR_RESTRICTION_HPP
#define GRID_AMR_RESTRICTION_HPP

#include "base/concepts.hpp"
#include "compute/computation.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "grid/domain.hpp"

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // conservative restriction kernel
    // averages r^rank fine cells to produce one coarse cell value
    // -------------------------------------------------------------------------
    template <typename FineComp, std::uint64_t Rank>
    struct restrict_average_t
    {
        using value_type = std::remove_cv_t<std::remove_reference_t<typename FineComp::value_type>>;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        FineComp      fine_comp_;
        argument_type ratio_;
        double        inv_volume_;

        DUAL restrict_average_t(FineComp fine, argument_type ratio)
            : fine_comp_(std::move(fine)), ratio_(ratio), inv_volume_(1.0)
        {
            for (std::uint64_t ii = 0; ii < Rank; ++ii) {
                inv_volume_ /= static_cast<double>(ratio[ii]);
            }
        }

        // input: coarse coordinate
        // action: loops over fine children and averages
        DUAL value_type operator()(const argument_type& coarse_coord) const
        {
            using namespace structs;
            argument_type fine_base;
            for (std::uint64_t d = 0; d < Rank; ++d) {
                fine_base[d] = coarse_coord[d] * ratio_[d];
            }

            value_type sum{};

            // manual loop unrolling for typical dimensions
            // we cannot use recursion easily in a device lambda/functor
            // without more template machinery
            if constexpr (Rank == 1) {
                bool first_iter = true;
                for (std::int64_t ii = 0; ii < ratio_[0]; ++ii) {
                    if (first_iter) {
                        sum        = fine_comp_({fine_base[0] + ii});
                        first_iter = false;
                        continue;
                    }
                    if constexpr (is_hydro_conserved_c<value_type>) {
                        sum = sum | add_gas(fine_comp_({fine_base[0] + ii}));
                    }
                    else {
                        sum = sum + fine_comp_({fine_base[0] + ii});
                    }
                }
            }
            else if constexpr (Rank == 2) {
                bool first_iter = true;
                for (std::int64_t jj = 0; jj < ratio_[1]; ++jj) {
                    for (std::int64_t ii = 0; ii < ratio_[0]; ++ii) {
                        if (first_iter) {
                            sum        = fine_comp_({fine_base[0] + ii, fine_base[1] + jj});
                            first_iter = false;
                            continue;
                        }
                        if constexpr (is_hydro_conserved_c<value_type>) {
                            sum = sum | add_gas(fine_comp_({fine_base[0] + ii, fine_base[1] + jj}));
                        }
                        else {
                            sum = sum + fine_comp_({fine_base[0] + ii, fine_base[1] + jj});
                        }
                    }
                }
            }
            else if constexpr (Rank == 3) {
                bool first_iter = true;
                for (std::int64_t kk = 0; kk < ratio_[2]; ++kk) {
                    for (std::int64_t jj = 0; jj < ratio_[1]; ++jj) {
                        for (std::int64_t ii = 0; ii < ratio_[0]; ++ii) {
                            if (first_iter) {
                                sum = fine_comp_(
                                    {fine_base[0] + ii, fine_base[1] + jj, fine_base[2] + kk}
                                );
                                first_iter = false;
                                continue;
                            }
                            if constexpr (is_hydro_conserved_c<value_type>) {
                                sum = sum |
                                      add_gas(fine_comp_(
                                          {fine_base[0] + ii, fine_base[1] + jj, fine_base[2] + kk}
                                      ));
                            }
                            else {
                                sum = sum +
                                      fine_comp_(
                                          {fine_base[0] + ii, fine_base[1] + jj, fine_base[2] + kk}
                                      );
                            }
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
    struct restrict_injection_t
    {
        using value_type                    = typename FineComp::value_type;
        using argument_type                 = iarray<Rank>;
        static constexpr std::uint64_t rank = Rank;

        FineComp      fine_comp_;
        argument_type ratio_;

        DUAL restrict_injection_t(FineComp fine, argument_type ratio)
            : fine_comp_(std::move(fine)), ratio_(ratio)
        {
        }

        DUAL value_type operator()(const argument_type& coarse_coord) const
        {
            argument_type fine_coord;
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
    // requires aligned domains: start and end must be divisible by ratio
    template <std::uint64_t Rank>
    constexpr grid::domain_t<Rank>
    scale_domain_down(const grid::domain_t<Rank>& d, const iarray<Rank>& ratio)
    {
        auto start = d.start;
        auto fin   = d.fin;
        for (std::uint64_t ii = 0; ii < Rank; ++ii) {
            // verify alignment: fine domain must snap to coarse grid
            if (start[ii] % ratio[ii] != 0 || fin[ii] % ratio[ii] != 0) {
#ifndef __CUDA_ARCH__
                throw std::runtime_error(
                    "scale_domain_down: fine domain not aligned to coarse "
                    "grid. "
                    "For refined levels, ghost width must be a multiple of the "
                    "refinement ratio."
                );
#endif
            }
            start[ii] /= ratio[ii];
            fin[ii] /= ratio[ii];
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

        return computation_t<Rank, decltype(kernel)>{std::move(kernel), coarse_domain};
    }

    template <typename Computation, std::uint64_t Rank>
    auto restrict_inject(const Computation& fine, iarray<Rank> ratio)
    {
        auto coarse_domain = scale_domain_down(fine.domain(), ratio);
        auto kernel        = restrict_injection_t<Computation, Rank>(fine, ratio);

        return computation_t<Rank, decltype(kernel)>{std::move(kernel), coarse_domain};
    }

} // namespace simbi::grid::amr

#endif // GRID_AMR_RESTRICTION_HPP
