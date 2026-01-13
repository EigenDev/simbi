#ifndef ECS_CREATION_FIELD_INITIALIZER_HPP
#define ECS_CREATION_FIELD_INITIALIZER_HPP

// =============================================================================
// field_initializer.hpp
//
// initializes simulation field data from python generators.
// separated from simulation_builder to keep construction and initialization
// as distinct concerns.
//
// usage:
//   auto sim = simulation_builder_t<...>{}.configure(...).build();
//   field_initializer_t<Sim>::initialize(sim, prim_gen, bfield_gens, gamma);
// =============================================================================

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "compute/numerics.hpp"
#include "containers/vector.hpp"
#include "ecs/geometry_visitor.hpp"
#include "ecs/systems.hpp"
#include "functional/fp.hpp"
#include "geometry/volume_scaling.hpp"
#include "grid/amr/prolongation.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace py = pybind11;

namespace simbi::ecs::creation {

    // =============================================================================
    // python generator to field conversion
    // =============================================================================

    template <typename T, std::uint64_t Rank>
    T py_value_to_state(py::handle obj)
    {
        T state;

        if (py::isinstance<py::tuple>(obj)) {
            auto tuple = obj.cast<py::tuple>();
            for (std::uint64_t ii = 0; ii < tuple.size() && ii < T::nmem; ++ii) {
                state[ii] = tuple[ii].cast<real>();
            }
        }
        else if (py::isinstance<py::list>(obj)) {
            auto list = obj.cast<py::list>();
            for (std::uint64_t ii = 0; ii < list.size() && ii < T::nmem; ++ii) {
                state[ii] = list[ii].cast<real>();
            }
        }
        else if constexpr (std::is_same_v<T, real>) {
            state = obj.cast<real>();
        }
        else {
            throw std::runtime_error("expected tuple, list, or scalar from generator");
        }

        return state;
    }

    template <typename T, std::uint64_t Rank>
    grid::field_t<T, Rank> from_generator(
        py::iterator&               gen,
        const grid::domain_t<Rank>& full_domain,
        const grid::domain_t<Rank>& active_domain
    )
    {
        // use unified memory for python generator initialization
        grid::field_t<T, Rank> field(full_domain);

        // iterate over active domain and fill from generator
        std::uint64_t total = active_domain.size();

        for (std::uint64_t linear = 0; linear < total; ++linear) {
            if (gen == py::iterator::sentinel()) {
                throw std::runtime_error("generator exhausted before filling field");
            }

            auto coord = active_domain.linear_to_coord(linear);
            auto value = *gen;

            if constexpr (std::is_same_v<T, real>) {
                field[coord] = py::cast<real>(value);
            }
            else {
                field[coord] = py_value_to_state<T, Rank>(value);
            }

            ++gen;
        }

        return field;
    }

    struct mean_magnetic_t
    {
        template <typename PrimField, std::uint64_t Rank>
        DEV PrimField operator()(PrimField prim, vector_t<real, Rank> bavg) const
        {
            PrimField p = prim;
            p.mag       = bavg;
            return p;
        }
    };

    // =============================================================================
    // field_initializer_t
    // =============================================================================

    template <typename Sim>
    struct field_initializer_t
    {
        using conserved_t                   = typename Sim::conserved_t;
        using primitive_t                   = typename Sim::primitive_t;
        static constexpr std::uint64_t Rank = Sim::rank;
        static constexpr regime_t      R    = Sim::regime;

        // -------------------------------------------------------------------------
        // initialize
        //
        // fills simulation fields from python generators.
        // must be called after simulation is built.
        // -------------------------------------------------------------------------
        static void initialize(
            Sim&                      sim,
            py::iterator              prim_gen,
            vector_t<py::iterator, 3> bfield_gens,
            real                      gamma
        )
        {
            // initialize level 0
            initialize_level(sim, 0, prim_gen, bfield_gens, gamma);

            // initialize refined levels via prolongation
            for (std::uint64_t lvl = 1; lvl < sim.num_levels(); ++lvl) {
                initialize_refined_level(sim, lvl, gamma);
            }
        }

        // -------------------------------------------------------------------------
        // initialize_level
        //
        // fills a single level's fields from generators.
        // -------------------------------------------------------------------------
        static void initialize_level(
            Sim&                      sim,
            std::uint64_t             lvl,
            py::iterator              prim_gen,
            vector_t<py::iterator, 3> bfield_gens,
            real                      gamma
        )
        {
            // for multi-partition, we need to handle each partition
            // for now, assume single partition (generator fills global domain)
            if (sim.num_partitions(lvl) != 1) {
                throw std::runtime_error(
                    "field initialization from generators not yet supported "
                    "for multi-partition simulations"
                );
            }

            auto& part   = sim.partition(lvl, 0);
            auto& fields = sim.partition_hydro(lvl, 0);

            auto full_domain   = part.allocated_domain;
            auto active_domain = part.owned_domain;

            auto        motion = sim.motion_state();
            const auto& mesh   = sim.mesh(lvl);

            // initialize primitives from a host-local generator and then clone
            // to the partition locality. create host-local prims explicitly so
            // initialize primitives from generator (uses unified memory)
            fields.prim = from_generator<primitive_t, Rank>(prim_gen, full_domain, active_domain);

            auto& exec = sim.partition_executor(lvl, 0);

            // initialize magnetic field (mhd only)
            if constexpr (R == regime_t::MHD || R == regime_t::RMHD) {
                vector_t<grid::domain_t<Rank>, Rank> face_domains;
                for (std::uint64_t dir = 0; dir < Rank; ++dir) {
                    auto staggered_domain = active_domain;
                    staggered_domain.fin[dir] += 1;

                    auto staggered_active = active_domain;
                    staggered_active.fin[dir] += 1;

                    fields.bfield[dir] = from_generator<real, Rank>(
                        bfield_gens[dir],
                        staggered_domain,
                        staggered_active
                    );
                    face_domains[dir] = staggered_active;
                }

                // interpolate face-centered B to cell-centered for conserved
                // state
                ecs::with_block_geometry<Sim::coord_system>(
                    mesh,
                    motion,
                    [&exec, face_domains, active_domain, fields](const auto& block_geo) {
                        auto bavg = interpolate_face_to_cell_magnetic(
                            fields.bfield,
                            block_geo,
                            face_domains,
                            active_domain
                        );
                        auto prims = fields.prim[active_domain];
                        prims      = prims.zip(bavg, mean_magnetic_t{}).with(exec);
                    }
                );
            }

            // convert primitives to conserved
            // for moving mesh: store volume-normalized extensive variables to conserve mass

            if (motion.is_moving) {
                // store Q = \rho * scale_factor (mass per comoving volume)
                // physical volume: V_phys(t) = scale_factor(geometry, a) * V_comoving
                // this conserves total mass as mesh expands
                std::cout << "Initializing moving mesh conserved variables..." << std::endl;
                std::cin.get();
                ecs::with_block_geometry<Sim::coord_system>(
                    mesh,
                    motion,
                    [&exec, gamma, active_domain, fields](const auto& block_geo) {
                        auto cons  = fields.cons[active_domain];
                        auto prims = fields.prim[active_domain];
                        cons       = prims.map(numerics::to_conserved_t{gamma})
                                   .enum_map([&block_geo](auto coord, auto u) {
                                       const auto dv = block_geo.volume(coord);
                                       return u * dv;
                                   })
                                   .with(exec);
                    }
                );
            }
            else {
                // static mesh: store intensive conserved variables
                fields.cons[active_domain] =
                    fields.prim[active_domain].map(numerics::to_conserved_t{gamma}).with(exec);
            }
        }

        // -------------------------------------------------------------------------
        // initialize_refined_level
        //
        // initializes a refined level by prolongation from parent.
        // -------------------------------------------------------------------------
        static void initialize_refined_level(Sim& sim, std::uint64_t lvl, real gamma)
        {
            if (sim.num_partitions(lvl) != 1 || sim.num_partitions(lvl - 1) != 1) {
                throw std::runtime_error("prolongation not yet supported for multi-partition");
            }

            // get parent and child fields
            auto& parent_fields = sim.partition_hydro(lvl - 1, 0);
            auto& child_fields  = sim.partition_hydro(lvl, 0);

            // get refinement ratio from level info
            const auto&  level_info = sim.level_info(lvl);
            iarray<Rank> ratio;
            ratio.fill(static_cast<std::int64_t>(level_info.refinement_ratio));

            auto& exec = sim.partition_executor(lvl, 0);

            // prolong primitives from parent to child
            auto parent_prim_comp = parent_fields.prim.as_computation();
            auto prolonged_prim   = grid::amr::prolong<2>(parent_prim_comp, ratio);

            child_fields.prim = prolonged_prim.with(exec);

            // // prolong magnetic fields for mhd
            if constexpr (R == regime_t::MHD || R == regime_t::RMHD) {
                for (std::uint64_t dir = 0; dir < Rank; ++dir) {
                    if (parent_fields.bfield[dir].domain().size() == 0) {
                        continue;
                    }

                    auto parent_b_comp       = parent_fields.bfield[dir].as_computation();
                    auto prolonged_b         = grid::amr::prolong<2>(parent_b_comp, ratio);
                    child_fields.bfield[dir] = prolonged_b.with(exec);
                }

                // interpolate prolonged face B to cell-centered conserved
                // state
                auto& child_part = sim.partition(lvl, 0);
                auto& child_mesh = sim.mesh(lvl);
                auto  motion     = sim.motion_state();
                auto& exec       = sim.partition_executor(lvl, 0);
                ecs::with_block_geometry<Sim::coord_system>(
                    child_mesh,
                    motion,
                    [&](const auto& block_geo) {
                        auto bavg = interpolate_face_to_cell_magnetic(
                            child_fields.bfield,
                            block_geo,
                            child_part.face_domains,
                            child_part.owned_domain
                        );
                        auto prims = child_fields.prim[child_part.owned_domain];
                        prims      = prims.zip(bavg, mean_magnetic_t{}).with(exec);
                    }
                );
            }

            // convert prolonged primitives to conserved
            // for moving mesh: use extensive variables just like in initialize_level
            auto motion = sim.motion_state();

            if (motion.is_moving) {
                // get volume scaling factor for this geometry and dimensionality
                const auto& meta = sim.metadata();
                const real  volume_factor =
                    geometry::get_scaling_factor<Sim::rank>(meta.coord_system, motion.a);

                // store Ũ = ρ * scale_factor (mass per comoving volume)
                child_fields.cons =
                    child_fields.prim.map(numerics::to_conserved_t{gamma})
                        .map([volume_factor](const auto& u) { return u * volume_factor; })
                        .with(exec);
            }
            else {
                // static mesh: store intensive conserved variables
                child_fields.cons =
                    child_fields.prim.map(numerics::to_conserved_t{gamma}).with(exec);
            }
        }
    };

} // namespace simbi::ecs::creation

#endif // ECS_CREATION_FIELD_INITIALIZER_HPP
