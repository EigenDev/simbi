#ifndef GRID_AMR_API_HPP
#define GRID_AMR_API_HPP

#include "build_config.hpp"
#include "compute/computation.hpp"
#include "containers/state_ops.hpp"
#include "containers/vector.hpp"
#include "grid/algebra.hpp"
#include "grid/amr/flux_correction.hpp"
#include "grid/amr/prolongation.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // reflux functor
    // applies flux correction to coarse cells
    // -------------------------------------------------------------------------
    template <typename T, std::uint64_t Rank, typename Geometry>
    struct reflux_t
    {
        Geometry geometry;

        DEV T operator()(const iarray<Rank>& coord, const std::pair<T, T>& p) const
        {
            using namespace structs;
            const auto& [u, dfa] = p;
            real inv_vol         = 1.0 / geometry.volume(coord);
            return u | add_gas(dfa * inv_vol);
        }
    };

    // -------------------------------------------------------------------------
    // ghost cell filling (coarse -> fine)
    // -------------------------------------------------------------------------
    // standard version for memory-backed fields
    template <typename T, std::uint64_t Rank, typename Exec>
    void fill_fine_ghosts(
        field_t<T, Rank>&       fine,
        const field_t<T, Rank>& coarse,
        const domain_t<Rank>&   fine_active_domain,
        const iarray<Rank>&     refinement_ratio,
        Exec&                   exec
    )
    {
        using namespace grid::domain_algebra;

        // identify ghost regions
        auto ghost_regions = difference(fine.domain(), fine_active_domain);

        // construct lazy prolongator
        // this creates a computation defined on the fine global index space
        // prolong(coarse) covers the entire fine domain logically
        auto prolong_op = prolong<3>(coarse.view(), refinement_ratio);

        // verify proper nesting: coarse must cover all ghost regions
        for (const auto& ghost_box : ghost_regions) {
            auto coarse_equiv = scale_domain_down(ghost_box, refinement_ratio);
            using namespace grid::domain_algebra;
            auto overlap = intersection(coarse.domain(), coarse_equiv);
            if (overlap != coarse_equiv) {
                throw std::runtime_error(
                    "fill_fine_ghosts: coarse grid does not cover fine ghost "
                    "region (proper nesting violated)"
                );
            }
        }

        // execute fill for each ghost region
        for (const auto& ghost_box : ghost_regions) {
            // assignment triggers the fused kernel:
            // fine[ghost] = prolong(coarse)[ghost]
            // the computation engine handles the coordinate mapping internally
            fine[ghost_box] = prolong_op.with(exec);
        }
    }

    // overload for bound_computation_t (lazy interpolated data)
    template <typename T, std::uint64_t Rank, typename Computation, typename Exec>
    void fill_fine_ghosts(
        field_t<T, Rank>&                                   fine,
        const bound_computation_t<Rank, Computation, Exec>& coarse_bound,
        const domain_t<Rank>&                               fine_active_domain,
        const iarray<Rank>&                                 refinement_ratio,
        Exec&                                               exec
    )
    {
        // materialize the computation into a temporary field
        auto coarse_temp = field_t<T, Rank>(coarse_bound.comp.domain());
        coarse_temp      = coarse_bound;

        // call the standard version
        fill_fine_ghosts(fine, coarse_temp, fine_active_domain, refinement_ratio, exec);
    }

    // -------------------------------------------------------------------------
    // restriction (fine -> coarse)
    // -------------------------------------------------------------------------
    // averages the fine field onto the coarse field.
    // typically applied to the region covered by the fine grid.
    template <typename T, std::uint64_t Rank, typename Exec>
    void restrict_to_coarse(
        field_t<T, Rank>&       coarse,
        const field_t<T, Rank>& fine,
        const iarray<Rank>&     refinement_ratio,
        Exec&                   exec
    )
    {
        // construct lazy restrictor
        // this creates a computation defined on the coarse global index space
        auto restrict_op = restrict(computation(fine.view()), refinement_ratio);

        // determine update region
        // we only update the coarse cells that are overlapped by the fine grid
        // restrict_op.domain() is automatically the coarse equivalent of
        // fine.domain()
        auto update_region = restrict_op.domain();

        // execute
        // coarse[region] = average(fine)
        coarse[update_region] = restrict_op.with(exec);
    }

    // overload for field_view_t
    template <typename T, std::uint64_t Rank, typename Exec>
    void restrict_to_coarse(
        field_t<T, Rank>&            coarse,
        const field_view_t<T, Rank>& fine_view,
        const iarray<Rank>&          refinement_ratio,
        Exec&                        exec
    )
    {
        auto restrict_op      = restrict(computation(fine_view), refinement_ratio);
        auto update_region    = restrict_op.domain();
        coarse[update_region] = restrict_op.with(exec);
    }

    template <typename T, std::uint64_t Rank, typename Geometry, typename Exec>
    void apply_flux_correction(
        field_t<T, Rank>&         coarse,
        flux_register_t<T, Rank>& flux_reg,
        const Geometry&           geometry,
        Exec&                     exec
    )
    {
        reflux_t<T, Rank, Geometry> reflux_op{geometry};

        // iterate over all face directions
        for (std::uint64_t dd = 0; dd < Rank; ++dd) {

            // iterate over left and right registers separately
            for (auto side : {side_t::left, side_t::right}) {
                auto* reg = flux_reg.get_register(dd, side);
                if (!reg) {
                    continue;
                }

                auto region = reg->domain();

                // register now contains (F * dt * area)
                // apply: u += (F * dt * area) / volume
                coarse[region] =
                    coarse[region].zip(*reg, fp::make_pair_func).enum_map(reflux_op).with(exec);
            }
        }
    }

} // namespace simbi::grid::amr

#endif // GRID_AMR_API_HPP
