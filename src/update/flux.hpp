#ifndef FLUX_HPP
#define FLUX_HPP

#include "bcs.hpp"
#include "execution/executor.hpp"
#include "execution/future.hpp"
#include "physics/em/perm.hpp"

#include <cstdint>
#include <vector>

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
    template <typename HydroState, typename CfdOps, typename MeshConfig>
    void update_staggered_fields(
        HydroState& state,
        const CfdOps& ops,
        const MeshConfig& mesh
    )
    {
        const auto executor = async::default_executor();
        std::vector<async::future_t<void>> flux_futures;

        for (std::uint64_t dir = 0; dir < HydroState::dimensions; ++dir) {
            const auto interface_f = cfd::compute_fluxes(state, mesh, ops, dir);

            auto future = executor.async([&state, interface_f, dir]() {
                state.flux[dir] =
                    state.flux[dir].map([interface_f](auto coord, auto) {
                        return interface_f(coord);
                    });
            });

            flux_futures.push_back(std::move(future));
        }

        for (auto& future : flux_futures) {
            future.wait();
        }

        if constexpr (HydroState::is_mhd) {
            // if we're doing MHD, then we need to make sure
            // that quantities that are staggered (i.e, the fluxes)
            // correctly have the perpendicular boundary conditions
            // applied since we do not save the edge-centered
            // electric fields but rather compute them
            // on-the-fly,
            boundary::apply_flux_bcs(state, mesh);
            em::update_magnetic_fields(state, mesh);
        }
    }

}   // namespace simbi
#endif
