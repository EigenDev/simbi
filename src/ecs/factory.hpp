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
#include "physics/ib/body.hpp"     // for body_t
#include "physics/ib/body_delta.hpp"     // for body_delta_t
#include "physics/ib/factory.hpp"        // for from_data_body_collection
#include "simulation.hpp"                // for simulation_t
#include "state/express_t.hpp"           // for expression_t
#include "utility/bimap.hpp"             // for deserialize
#include "utility/enums.hpp"             // for Geometry, Regime
#include "utility/init_conditions.hpp"   // for initial_conditions_t

#include <bit>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace simbi::ecs {
    using namespace mesh::fmr;
    using namespace body::factory;
    using namespace state;

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
        using exp_t       = expression_t<Dims>;
        using conserved_t = typename sim_t::conserved_t;
        using primitive_t = typename sim_t::primitive_t;

        sim_t sim;

        // create global metadata entity
        sim.global = sim.registry.create();
        // populate global metadata
        auto meta = simulation_metadata_t<Dims>{
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
          .reconstruction       = deserialize<Reconstruction>(init.reconstruct),
          .timestepping         = deserialize<Timestepping>(init.timestepping),
          .boundary_conditions  = vector_t<BoundaryCondition, 2 * Dims>{},
          .resolution           = {init.nz, init.ny, init.nx},
          .is_mhd               = init.is_mhd,
          .is_relativistic      = init.is_relativistic,
          .data_dir             = init.data_directory
        };
        for (std::uint64_t ii = 0; ii < 2 * Dims; ++ii) {
            auto logical_dim = ii / 2;   // which dimension (x=0, y=1, z=2)
            auto side        = ii % 2;   // which side (inner=0, outer=1)
            // map to array order
            auto array_dim = (Dims - 1) - logical_dim;
            // convert back to flat index
            auto array_index = array_dim * 2 + side;
            meta.boundary_conditions[array_index] =
                deserialize<BoundaryCondition>(init.boundary_conditions[ii]);
        }
        sim.registry.add(sim.global, std::move(meta));

        // create base level (level 0)
        auto level_0 = sim.registry.create();
        sim.levels.push_back(level_0);

        // populate bfields if MHD
        vector_t<field_t<real, Dims>, Dims> bfield;
        if constexpr (R == Regime::MHD || R == Regime::RMHD) {
            const auto mhd_b = 2 * init.is_mhd;
            bfield = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                         auto data_ptr = std::bit_cast<real*>(bfield_data[dir]);
                         auto active_shape = init.get_active_shape<Dims>();
                         // create staggered shape
                         iarray<Dims> staggered_shape = active_shape;
                         staggered_shape[dir] += 1;
                         // add MHD offset to other dimensions
                         for (std::uint64_t d = 0; d < Dims; ++d) {
                             if (d != dir) {
                                 staggered_shape[d] += mhd_b;
                             }
                         }
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
                          auto active_shape = init.get_active_shape<Dims>();
                          const auto mhd_b  = 2 * init.is_mhd;
                          // create staggered shape
                          iarray<Dims> staggered_shape = active_shape;
                          staggered_shape[dir] += 1;
                          // add MHD offset to other dimensions
                          for (std::uint64_t d = 0; d < Dims; ++d) {
                              if (d != dir) {
                                  staggered_shape[d] += mhd_b;
                              }
                          }
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

        auto bodies      = create_body_collection_from_init<Dims>(init);
        auto diagnostics = body::create_diagnostics_accumulator<Dims>();

        // add bodies if enabled
        if (bodies) {
            sim.registry.add(
                sim.global,
                immersed_bodies_t<Dims>{.bodies = std::move(bodies.value())}
            );

            bodies->visit_all([&](const auto& body) {
                using body_type = std::decay_t<decltype(body)>;
                auto delta      = body::body_delta_t<Dims>{
                       .idx          = body.idx,
                       .force_delta  = body.force,
                       .torque_delta = body.torque,
                       .mass_delta   = 0.0
                };
                if constexpr (body::has_accretion_capability_c<body_type>) {
                    delta.prev_mass_delta = body::total_accreted_mass(body);
                }
                diagnostics->accumulate_delta(delta);
            });

            sim.registry.add(
                sim.global,
                body_info_t<Dims>{.diagnostics = std::move(diagnostics)}
            );
        }

        // add sources (if any)
        auto hydro = exp_t::from_config(init.hydro_source_expressions);
        auto grav  = exp_t::from_config(init.gravity_source_expressions);
        vector_t<exp_t, 2 * Dims> bc_sources;
        // set up boundary condition sources
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

        sim.registry.add(
            sim.global,
            sources_t<Dims>{
              .hydro_source   = std::move(hydro),
              .gravity_source = std::move(grav),
              .bc_sources     = std::move(bc_sources)
            }
        );

        // add fmr levels if enabled
        if (init.fmr_enabled) {
            auto base_mesh = sim.mesh(0);
            auto hierarchy = build_hierarchy_from_init<Dims>(init, base_mesh);

            for (std::uint64_t lvl = 1; lvl < hierarchy.num_levels; ++lvl) {
                auto level_entity = sim.registry.create();
                sim.levels.push_back(level_entity);

                const auto& level_desc = hierarchy[lvl];

                vector_t<field_t<real, Dims>, Dims> bfield;
                if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                    bfield = fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                                 const auto mhd_b = 2 * init.is_mhd;
                                 auto stagg_s     = level_desc.domain.shape();
                                 stagg_s[dir] += 1;
                                 // add MHD offset to other dimensions
                                 for (std::uint64_t d = 0; d < Dims; ++d) {
                                     if (d != dir) {
                                         stagg_s[d] += mhd_b;
                                     }
                                 }
                                 return field(
                                     make_domain(stagg_s),
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
                                  const auto mhd_b = 2 * init.is_mhd;
                                  auto stagg_s     = level_desc.domain.shape();
                                  stagg_s[dir] += 1;
                                  // add MHD offset to other dimensions
                                  for (std::uint64_t d = 0; d < Dims; ++d) {
                                      if (d != dir) {
                                          stagg_s[d] += mhd_b;
                                      }
                                  }
                                  return field(
                                      make_domain(stagg_s),
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
                    create_level_mesh(base_mesh, level_desc, init);
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

                prolongate_level_data(sim, lvl);
            }
            sim.registry.add(
                sim.global,
                fmr_hierarchy_t<Dims>{std::move(hierarchy)}
            );
        }

        return sim;
    }

}   // namespace simbi::ecs

#endif
