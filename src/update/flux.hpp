#ifndef FLUX_HPP
#define FLUX_HPP

#include "bcs.hpp"
#include "physics/em/ct_updater.hpp"

#include <cstdint>

namespace simbi::cfd {
    template <typename HydroState, typename CfdOps, typename MeshConfig>
    auto compute_fluxes(
        const HydroState& state,
        const MeshConfig& mesh,
        const CfdOps& ops,
        std::uint64_t dir
    );
}
namespace simbi {
    struct bfield_parameter {
        bool advance_bfields = true;
    };

    template <typename HydroState, typename CfdOps, typename MeshConfig>
    void update_staggered_fields(
        HydroState& state,
        const CfdOps& ops,
        const MeshConfig& mesh,
        bfield_parameter params = {.advance_bfields = true}
    )
    {
        for (std::uint64_t dir = 0; dir < HydroState::dimensions; ++dir) {
            const auto interface_f = cfd::compute_fluxes(state, mesh, ops, dir);
            auto flux              = state.flux[dir][mesh.face_domain[dir]];
            flux                   = flux.map([interface_f](auto coord, auto) {
                return interface_f(coord);
            });
        }

        if constexpr (HydroState::is_mhd) {
            // if we're doing MHD, then we need to make sure
            // that quantities that are staggered (i.e, the fluxes)
            // correctly have the perpendicular boundary conditions
            // applied since we do not save the edge-centered
            // electric fields but rather compute them
            // on-the-fly,
            if (params.advance_bfields) {
                boundary::apply_stagg_bcs(state, mesh);
                em::update_magnetic_fields(state, mesh);
            }
        }
    }

}   // namespace simbi
#endif
