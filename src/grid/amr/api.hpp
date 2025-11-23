#ifndef GRID_AMR_API_HPP
#define GRID_AMR_API_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "containers/vector.hpp"
#include "grid/algebra.hpp"
#include "grid/amr/flux_correction.hpp"
#include "grid/amr/prolongation.hpp"
#include "grid/connectivity.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"

#include <cstdint>

namespace simbi::grid::amr {

    using namespace simbi::compute;

    // -------------------------------------------------------------------------
    // ghost cell filling (coarse -> fine)
    // -------------------------------------------------------------------------
    // fills the ghost regions of the fine field by interpolating from the
    // coarse field. assumes global indexing alignment.
    template <typename T, std::uint64_t Rank, typename Exec>
    void fill_fine_ghosts(
        field_t<T, Rank>& fine,
        const field_t<T, Rank>& coarse,
        const domain_t<Rank>& fine_active_domain,
        const iarray<Rank>& refinement_ratio,
        Exec& exec
    )
    {
        using namespace grid::domain_algebra;

        // identify ghost regions
        // ghosts = total_domain - active_domain
        // we use the difference set algebra we built earlier
        auto ghost_regions = difference(fine.domain(), fine_active_domain);

        // construct lazy prolongator
        // this creates a computation defined on the fine global index space
        // prolong(coarse) covers the entire fine domain logically
        auto prolong_op = prolong<2>(coarse.view(), refinement_ratio);

        // execute fill for each ghost region
        for (const auto& ghost_box : ghost_regions) {
            // assignment triggers the fused kernel:
            // fine[ghost] = prolong(coarse)[ghost]
            // the computation engine handles the coordinate mapping internally
            fine[ghost_box] = computation(fine.domain(), prolong_op).with(exec);
        }
    }

    // -------------------------------------------------------------------------
    // restriction (fine -> coarse)
    // -------------------------------------------------------------------------
    // averages the fine field onto the coarse field.
    // typically applied to the region covered by the fine grid.
    template <typename T, std::uint64_t Rank, typename Exec>
    void restrict_to_coarse(
        field_t<T, Rank>& coarse,
        const field_t<T, Rank>& fine,
        const iarray<Rank>& refinement_ratio,
        Exec& exec
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

    // -------------------------------------------------------------------------
    // flux correction (reflux)
    // -------------------------------------------------------------------------
    // applies the accumulated flux mismatch in the register to the coarse
    // state. u_coarse += flux_mismatch / cell_volume
    template <typename T, std::uint64_t Rank, typename Geometry, typename Exec>
    void apply_flux_correction(
        field_t<T, Rank>& coarse,
        flux_register_t<T, Rank>& flux_reg,
        const Geometry& geometry,
        Exec& exec
    )
    {
        // iterate over all face directions
        for (std::uint64_t dd = 0; dd < Rank; ++dd) {
            for (auto side : {side_t::left, side_t::right}) {
                auto* reg = flux_reg.get_register(dd, side);
                if (!reg) {
                    continue;
                }

                auto region = reg->domain();

                // reflux kernel: u_new = u_old + flux_mismatch / volume
                // geometry.volume(coord) gives the proper cell volume for
                // curvilinear coordinates (spherical, cylindrical, etc.)
                auto reflux_op = [geometry] DUAL(
                                     const iarray<Rank>& coord,
                                     const T& u,
                                     const T& df
                                 ) {
                    real inv_vol = 1.0 / geometry.volume(coord);
                    return u + df * inv_vol;
                };

                // use enum_map to get coordinate + values
                coarse[region] =
                    coarse[region]
                        .zip(
                            *reg,
                            [](const T& u, const T& df) {
                                return std::make_pair(u, df);
                            }
                        )
                        .enum_map([reflux_op](
                                      const iarray<Rank>& coord,
                                      const std::pair<T, T>& p
                                  ) {
                            return reflux_op(coord, p.first, p.second);
                        })
                        .with(exec);
            }
        }
    }

}   // namespace simbi::grid::amr

#endif   // GRID_AMR_API_HPP
