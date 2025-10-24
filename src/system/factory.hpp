#ifndef ECS_FACTORY_HPP
#define ECS_FACTORY_HPP

#include "base/concepts.hpp"       // for is_mhd_c
#include "compat.hpp"              // for real type
#include "components.hpp"          // for hydro_fields_t, mesh_geometry_t, etc
#include "compute/field.hpp"       // for field_t, from_data_field
#include "containers/vector.hpp"   // for vector_t
#include "functional/fp.hpp"       // for fp::transform, fp::collect, fp::range
#include "mesh/fmr/factory.hpp"    // for fmr::build_hierarchy_from_init
#include "mesh/mesh_config.hpp"    // for mesh_config_t
#include "physics/ib/factory.hpp"
#include "simulation.hpp"                // for simulation_t
#include "utility/bimap.hpp"             // for deserialize
#include "utility/enums.hpp"             // for Geometry, Regime
#include "utility/init_conditions.hpp"   // for initial_conditions_t

#include <bit>
#include <cstdint>
#include <functional>

namespace simbi::ecs {
    using namespace mesh::refinement;
    using namespace body::factory;

    inline ShockWaveLimiter get_shock_smoother(const initial_conditions_t& init)
    {
        return init.fleischmann_limiter
                   ? ShockWaveLimiter::FLEISCHMANN
                   : (init.quirk_smoothing ? ShockWaveLimiter::QUIRK
                                           : ShockWaveLimiter::NONE);
    }

    template <Regime R, std::uint64_t Dims, Geometry G, typename EoS>
    simulation_t<R, Dims, G, EoS> create_simulation(
        const initial_conditions_t& init,
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        using sim_t       = simulation_t<R, Dims, G, EoS>;
        using conserved_t = typename sim_t::conserved_t;
        using primitive_t = typename sim_t::primitive_t;

        sim_t sim;

        // create global metadata entity
        sim.global = sim.registry.create();
        sim.registry.add(
            sim.global,
            simulation_metadata_t<Dims>{
              .gamma                = init.gamma,
              .plm_theta            = init.plm_theta,
              .viscosity            = init.viscosity,
              .cfl                  = init.cfl,
              .time                 = init.time,
              .tend                 = init.tend,
              .dt                   = 0.0,
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
              .shock_smoother       = get_shock_smoother(init),
              .solver               = deserialize<Solver>(init.solver),
              .x1_spacing           = deserialize<Cellspacing>(init.x1_spacing),
              .x2_spacing           = deserialize<Cellspacing>(init.x2_spacing),
              .x3_spacing           = deserialize<Cellspacing>(init.x3_spacing),
              .coord_system         = deserialize<Geometry>(init.coord_system),
              .reconstruction = deserialize<Reconstruction>(init.reconstruct),
              .timestepping   = deserialize<Timestepping>(init.timestepping),
              .boundary_conditions = vector_t<BoundaryCondition, 2 * Dims>{},
              .resolution          = {init.nz, init.ny, init.nx},
              .is_mhd              = init.is_mhd,
              .is_relativistic     = init.is_relativistic,
              .data_dir            = init.data_directory
            }
        );

        // create base level (level 0)
        auto level_0 = sim.registry.create();
        sim.levels.push_back(level_0);

        // populate bfields if MHD
        vector_t<field_t<real, Dims>, Dims> bfield;
        if constexpr (R == Regime::MHD || R == Regime::RMHD) {
            bfield = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                         auto data_ptr = std::bit_cast<real*>(bfield_data[dir]);
                         auto full_shape = init.get_full_shape<Dims>();
                         // create staggered shape
                         iarray<Dims> staggered_shape = full_shape;
                         staggered_shape[dir] += 1;
                         return from_data_field(data_ptr, staggered_shape);
                     }) |
                     fp::collect<vector_t<field_t<real, Dims>, Dims>>;
        }

        // add hydro fields
        auto full_shape = init.get_full_shape<Dims>();
        sim.registry.add(
            level_0,
            hydro_fields_t<conserved_t, primitive_t, Dims>{
              .cons = from_data_field(
                  std::bit_cast<conserved_t*>(cons_data),
                  full_shape
              ),
              .prim = from_data_field(
                  std::bit_cast<primitive_t*>(prim_data),
                  full_shape
              ),
              .flux = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                          // create staggered shape
                          iarray<Dims> staggered_shape = full_shape;
                          staggered_shape[dir] += 1;
                          return field(
                              make_domain(staggered_shape),
                              fp::default_t<conserved_t>{}
                          );
                      }) |
                      fp::collect<vector_t<field_t<conserved_t, Dims>, Dims>>,
              .bfield = std::move(bfield)
            }
        );

        // add mesh geometry
        auto mesh = mesh::mesh_config_t<Dims, G>::from_init_conditions(
            init,
            scale_factor,
            scale_factor_derivative
        );
        sim.registry.add(
            level_0,
            mesh_geometry_t<Dims, G>{.config = std::move(mesh)}
        );

        // add level metadata
        sim.registry.add(
            level_0,
            level_info_t{.level_id = 0, .refinement_ratio = 1}
        );

        auto bodies = create_body_collection_from_init<Dims>(init);

        // add bodies if enabled
        if (bodies) {
            sim.registry.add(
                level_0,
                immersed_bodies_t<Dims>{.bodies = std::move(bodies.value())}
            );
        }

        // add fmr levels if enabled
        if (init.fmr_enabled) {
            auto base_mesh = sim.mesh(0);
            auto hierarchy =
                fmr::build_hierarchy_from_init<Dims>(init, base_mesh);

            for (std::uint64_t lvl = 1; lvl < hierarchy.num_levels; ++lvl) {
                auto level_entity = sim.registry.create();
                sim.levels.push_back(level_entity);

                const auto& level_desc = hierarchy[lvl];

                vector_t<field_t<real, Dims>, Dims> bfield;
                if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                    bfield = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                                 auto full_shape =
                                     level_desc.full_domain.shape();
                                 // create staggered shape
                                 iarray<Dims> staggered_shape = full_shape;
                                 staggered_shape[dir] += 1;
                                 return field(
                                     make_domain(staggered_shape),
                                     fp::default_t<real>{}
                                 );
                             }) |
                             fp::collect<vector_t<field_t<real, Dims>, Dims>>;
                }

                // allocate fields
                sim.registry.add(
                    level_entity,
                    hydro_fields_t<conserved_t, primitive_t, Dims>{
                      .cons = stored_field<conserved_t, Dims>(
                          level_desc.full_domain
                      ),
                      .prim = stored_field<primitive_t, Dims>(
                          level_desc.full_domain
                      ),
                      .flux = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                                  // create staggered shape
                                  iarray<Dims> staggered_shape =
                                      level_desc.full_domain.shape();
                                  staggered_shape[dir] += 1;
                                  return field(
                                      make_domain(staggered_shape),
                                      fp::default_t<conserved_t>{}
                                  );
                              }) |
                              fp::collect<
                                  vector_t<field_t<conserved_t, Dims>, Dims>>,
                      .bfield = std::move(bfield)
                    }
                );

                // create mesh for this level
                auto level_mesh =
                    fmr::create_level_mesh(base_mesh, level_desc, init);
                sim.registry.add(
                    level_entity,
                    mesh_geometry_t<Dims, G>{.config = std::move(level_mesh)}
                );

                sim.registry.add(
                    level_entity,
                    level_info_t{
                      .level_id         = lvl,
                      .refinement_ratio = level_desc.ref_ratio
                    }
                );

                sim.registry.add(
                    level_entity,
                    refinement_child_t{
                      .parent          = sim.levels[lvl - 1],
                      .parent_coverage = level_desc.parent_coverage
                    }
                );
            }
        }

        return sim;
    }

}   // namespace simbi::ecs

#endif
