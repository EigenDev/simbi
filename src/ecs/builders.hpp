#ifndef COMPONENT_BUILDERS_HPP
#define COMPONENT_BUILDERS_HPP

#include "compat.hpp"
#include "components.hpp"
#include "containers//vector.hpp"
#include "math/express_t.hpp"
#include "utility/bimap.hpp"
#include "utility/enums.hpp"
#include "utility/init_conditions.hpp"

#include <cstdint>
#include <vector>

namespace simbi::ecs {

    template <std::uint64_t Dims>
    ecs::simulation_metadata_t<Dims>
    build_metadata_component(const initial_conditions_t& init)
    {
        auto get_shock_smoother = [&init]() {
            return init.fleischmann_limiter
                       ? ShockWaveLimiter::FLEISCHMANN
                       : (init.quirk_smoothing ? ShockWaveLimiter::QUIRK
                                               : ShockWaveLimiter::NONE);
        };

        ecs::simulation_metadata_t<Dims> meta{
          .gamma                = init.gamma,
          .plm_theta            = init.plm_theta,
          .viscosity            = init.viscosity,
          .cfl                  = init.cfl,
          .time                 = init.time,
          .tend                 = init.tend,
          .global_dt            = 0.0,
          .dlogt                = init.dlogt,
          .checkpoint_interval  = init.checkpoint_interval,
          .checkpoint_time      = init.time,
          .prev_checkpoint_time = init.time,
          .ambient_sound_speed  = init.ambient_sound_speed,
          .iteration            = 0,
          .halo_radius          = init.halo_radius,
          .checkpoint_index     = init.checkpoint_index,
          .checkpoint_zones     = init.checkpoint_zones(),
          .regime               = deserialize<Regime>(init.regime),
          .shock_smoother       = get_shock_smoother(),
          .solver               = deserialize<Solver>(init.solver),
          .x1_spacing           = deserialize<Cellspacing>(init.x1_spacing),
          .x2_spacing           = deserialize<Cellspacing>(init.x2_spacing),
          .x3_spacing           = deserialize<Cellspacing>(init.x3_spacing),
          .coord_system         = deserialize<Geometry>(init.coord_system),
          .reconstruction       = deserialize<Reconstruction>(init.reconstruct),
          .timestepping         = deserialize<Timestepping>(init.timestepping),
          .boundary_conditions  = vector_t<BoundaryCondition, 2 * Dims>{},
          .resolution           = {init.nz, init.ny, init.nx},
          .is_mhd               = init.is_mhd,
          .is_relativistic      = init.is_relativistic,
          .data_dir             = init.data_directory,
          .level_dts            = std::vector<real>(init.fmr_max_levels),
          .level_substeps       = init.substeps,
          .subcycling_mode      = init.subcycling_mode
        };

        for (std::uint64_t ii = 0; ii < 2 * Dims; ++ii) {
            auto logical_dim = ii / 2;
            auto side        = ii % 2;
            auto array_dim   = (Dims - 1) - logical_dim;
            auto array_index = array_dim * 2 + side;
            meta.boundary_conditions[array_index] =
                deserialize<BoundaryCondition>(init.boundary_conditions[ii]);
        }

        return meta;
    }

    template <std::uint64_t Dims>
    sources_t<Dims> build_sources_component(const initial_conditions_t& init)
    {
        using exp_t = state::expression_t<Dims>;

        vector_t<exp_t, 2 * Dims> bc_sources;
        bc_sources[0] = exp_t::from_config(init.bx1_inner_expressions);
        bc_sources[1] = exp_t::from_config(init.bx1_outer_expressions);
        if constexpr (Dims >= 2) {
            bc_sources[2] = exp_t::from_config(init.bx2_inner_expressions);
            bc_sources[3] = exp_t::from_config(init.bx2_outer_expressions);
        }
        if constexpr (Dims >= 3) {
            bc_sources[4] = exp_t::from_config(init.bx3_inner_expressions);
            bc_sources[5] = exp_t::from_config(init.bx3_outer_expressions);
        }

        return sources_t<Dims>{
          .hydro_source   = exp_t::from_config(init.hydro_source_expressions),
          .gravity_source = exp_t::from_config(init.gravity_source_expressions),
          .bc_sources     = std::move(bc_sources)
        };
    }

    template <std::uint64_t Dims>
    level_info_t
    build_level_info(std::uint64_t level_id, std::uint64_t ref_ratio)
    {
        return level_info_t{
          .level_id         = level_id,
          .refinement_ratio = ref_ratio
        };
    }
}   // namespace simbi::ecs
#endif
