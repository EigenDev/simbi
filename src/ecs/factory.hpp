#ifndef ECS_FACTORY_HPP
#define ECS_FACTORY_HPP

#include "base/concepts.hpp"   // for is_mhd_c
#include "builders.hpp"
#include "compat.hpp"              // for real type
#include "components.hpp"          // for hydro_fields_t, mesh_geometry_t, etc
#include "compute/field.hpp"       // for field_t, from_data_field
#include "containers/vector.hpp"   // for vector_t
#include "functional/fp.hpp"       // for fp::transform, fp::collect, fp::range
#include "memory/accessor.hpp"     // for accessor_t
#include "mesh/fmr/factory.hpp"    // for fmr::build_hierarchy_from_init
#include "mesh/fmr/prolongation.hpp"   // for fmr::prolongate_conserved
#include "mesh/mesh_config.hpp"        // for mesh_config_t
#include "physics/em/ct_updater.hpp"
#include "physics/hydro/physics.hpp"
#include "physics/ib/body.hpp"           // for body_t
#include "physics/ib/body_delta.hpp"     // for body_delta_t
#include "physics/ib/factory.hpp"        // for from_data_body_collection
#include "simulation.hpp"                // for simulation_t
#include "state/express_t.hpp"           // for expression_t
#include "utility/bimap.hpp"             // for deserialize
#include "utility/enums.hpp"             // for Geometry, Regime
#include "utility/init_conditions.hpp"   // for initial_conditions_t

#include <cstdint>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

namespace simbi::ecs {
    using namespace mesh::fmr;
    using namespace body::factory;
    using namespace state;

    struct cell_centered_tag {
    };
    struct face_centered_tag {
    };

    template <typename State>
    State py_tuple_to_state(py::handle obj)
    {
        State s;
        std::uint64_t chi_position = State::dimensions + 3;

        // check if it's a tuple/sequence
        if (py::isinstance<py::tuple>(obj)) {
            auto tuple = obj.cast<py::tuple>();
            for (std::uint64_t ii = 0; ii < tuple.size(); ++ii) {
                if constexpr (is_hydro_primitive_c<State>) {
                    if (chi_position == ii) {
                        s[ii + State::dimensions] = tuple[ii].cast<real>();
                    }
                    else {
                        s[ii] = tuple[ii].cast<real>();
                    }
                }
                else {
                    s[ii] = tuple[ii].cast<real>();
                }
            }
        }
        else if (py::isinstance<py::sequence>(obj)) {
            auto seq = obj.cast<py::list>();
            for (std::uint64_t ii = 0; ii < seq.size(); ++ii) {
                if constexpr (is_hydro_primitive_c<State>) {
                    if (chi_position == ii) {
                        s[ii + State::dimensions] = seq[ii].cast<real>();
                    }
                    else {
                        s[ii] = seq[ii].cast<real>();
                    }
                }
                else {
                    s[ii] = seq[ii].cast<real>();
                }
            }
        }
        else {
            throw std::runtime_error(
                "Expected tuple or sequence from generator"
            );
        }

        return s;
    }

    template <
        typename element_type,
        std::uint64_t Dims,
        typename Tag = cell_centered_tag>
    field_t<element_type, Dims> from_generator(
        py::iterator gen,
        const iarray<Dims>& full_shape,
        std::uint64_t halo_radius,
        std::uint64_t direction = 0
    )
    {
        auto full_domain   = make_domain(full_shape);
        auto active_domain = [halo_radius, direction, full_domain]() {
            if constexpr (std::is_same_v<Tag, cell_centered_tag>) {
                (void) direction;   // suppress unused warning
                auto width = static_cast<std::int64_t>(halo_radius);
                return domain_algebra::contract(
                    full_domain,
                    ones<Dims, std::int64_t>() * width
                );
            }
            else {
                auto amount = ones<Dims, std::int64_t>();
                amount[direction] -= halo_radius;
                return domain_algebra::contract(full_domain, amount);
            }
        }();

        // allocate full domain (with halos)
        auto accessor = accessor_t<element_type, Dims>{full_domain};
        auto field    = compute_field_t{std::move(accessor), full_domain};
        for (auto coord : active_domain) {
            // get current value
            auto value = *gen;

            if (gen == py::iterator::sentinel()) {
                throw std::runtime_error("Generator exhausted");
            }
            if constexpr (std::is_same_v<element_type, real>) {
                field[coord] = py::cast<real>(value);
            }
            else {
                field[coord] = py_tuple_to_state<element_type>(value);
            }
            // advance iterator for next call
            ++gen;
        }

        // halos remain uninitialized (will be filled by boundary conditions)
        return field;
    }

    template <Regime R, std::uint64_t Dims, Geometry G, typename EoS>
    simulation_t<R, Dims, G, EoS> create_simulation(
        const initial_conditions_t& init,
        py::iterator prim_gen,
        vector_t<py::iterator, 3> bfield_gens,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        {
            using sim_t       = simulation_t<R, Dims, G, EoS>;
            using conserved_t = typename sim_t::conserved_t;
            using primitive_t = typename sim_t::primitive_t;
            sim_t sim;

            // create global metadata entity
            sim.global = sim.registry.create();
            // populate global metadata
            sim.registry.add(sim.global, build_metadata_component<Dims>(init));

            // create base level (level 0)
            auto level_0 = sim.registry.create();
            sim.levels.push_back(level_0);

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

            // add hydro fields
            auto full_shape   = init.get_full_shape<Dims>();
            auto active_shape = init.get_active_shape<Dims>();

            vector_t<field_t<real, Dims>, Dims> bfield;
            if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                const auto mhd_b = 2 * init.is_mhd;
                bfield =
                    fp::range(Dims) | fp::map([&](std::uint64_t dir) {
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
                        auto bn_gen = bfield_gens[dir];
                        return from_generator<real, Dims, face_centered_tag>(
                            bn_gen,
                            staggered_shape,
                            1,   // halo radius for staggered
                            dir
                        );
                    }) |
                    fp::collect<vector_t<field_t<real, Dims>, Dims>>;
            }
            auto prims = from_generator<primitive_t>(
                prim_gen,
                full_shape,
                init.halo_radius
            );
            if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                const auto& base_mesh = sim.mesh(0);
                auto bavg =
                    em::interpolate_face_to_cell_magnetic(bfield, base_mesh);
                auto active_prims = prims[base_mesh.domain];
                active_prims =
                    active_prims.enum_map([bavg](auto coord, auto prim) {
                        prim.mag = bavg(coord);
                        return prim;
                    });
            }
            auto cons = stored_field<conserved_t>(make_domain(full_shape));
            cons      = prims.map([g = init.gamma](auto prim) {
                return hydro::to_conserved(prim, g);
            });

            sim.registry.add(
                level_0,
                hydro_fields_t<conserved_t, primitive_t, Dims>{
                  .cons = std::move(cons),
                  .prim = std::move(prims),
                  .flux =
                      fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                          const auto mhd_b = 2 * init.is_mhd;
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
            // add source terms
            sim.registry.add(sim.global, build_sources_component<Dims>(init));

            // add fmr levels if enabled
            if (init.fmr_enabled) {
                auto base_mesh = sim.mesh(0);
                auto hierarchy =
                    build_hierarchy_from_init<Dims>(init, base_mesh);

                for (std::uint64_t lvl = 1; lvl < hierarchy.num_levels; ++lvl) {
                    auto level_entity = sim.registry.create();
                    sim.levels.push_back(level_entity);

                    const auto& level_desc = hierarchy[lvl];

                    vector_t<field_t<real, Dims>, Dims> bfield;
                    if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                        bfield =
                            fp::range(Dims) | fp::map([&](std::uint64_t dir) {
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
                          .flux =
                              fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                                  return field(
                                      level_desc.face_domains[dir],
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
                        mesh_geometry_t<Dims, G>{
                          .config = std::move(level_mesh)
                        }
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

                    const auto map = create_level_mapping(hierarchy, lvl);
                    prolongate_conservative(
                        sim.hydro(lvl - 1).cons,
                        sim.hydro(lvl).cons,
                        map
                    );
                }
                sim.registry.add(
                    sim.global,
                    fmr_hierarchy_t<Dims>{std::move(hierarchy)}
                );
            }

            return sim;
        }
    }

}   // namespace simbi::ecs

#endif
