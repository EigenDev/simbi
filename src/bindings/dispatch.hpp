#ifndef HYDRO_DISPATCH_HPP
#define HYDRO_DISPATCH_HPP

// =============================================================================
// dispatch.hpp
//
// type dispatch system for hydro simulations.
// converts runtime configuration (strings, enums) to compile-time template
// parameters via nested switch dispatchers.
//
// dispatch chain:
//   regime -> dims -> geometry -> solver -> reconstruction -> eos
//   -> call_visitor_with_state<R, D, G, S, Rec, EoS>(visitor, blueprints, ...)
//
// usage:
//   with_hydro_state(config, prim_gen, bfield_gens, scale_factor, ...,
//       [](auto& sim, auto& ops) {
//           // sim is simulation_t<R, D, G, EoS>
//           // ops is cfd_operations_t<R, D, S, Rec, EoS>
//       });
// =============================================================================

#include "build_config.hpp"
#include "compute/cfd_ops.hpp"
#include "containers/vector.hpp"
#include "ecs/blueprints.hpp"
#include "ecs/components.hpp"
#include "ecs/creation/blueprint_extractor.hpp"
#include "ecs/creation/field_initializer.hpp"
#include "ecs/creation/sim.hpp"
#include "physics/eos/ideal.hpp"
#include "physics/eos/isothermal.hpp"
#include "utility/bimap.hpp"
#include "utility/config_dict.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace py = pybind11;
namespace simbi::dispatch {

    // =============================================================================
    // validity concept - defines which template combinations are supported
    // =============================================================================

    template <regime_t R, std::uint64_t D, geometry_t G, solver_t S, reconstruction_t Rec>
    concept valid_combination =
        // basic constraints
        (D >= 1 && D <= 3) &&

        // reconstruction constraints
        (Rec == reconstruction_t::PCM || Rec == reconstruction_t::PLM) &&

        // geometry constraints
        (G == geometry_t::CARTESIAN || G == geometry_t::CYLINDRICAL ||
         G == geometry_t::AXIS_CYLINDRICAL || G == geometry_t::PLANAR_CYLINDRICAL ||
         G == geometry_t::SPHERICAL) &&

        // geometry-dimension constraints
        ((G != geometry_t::AXIS_CYLINDRICAL && G != geometry_t::PLANAR_CYLINDRICAL) || D == 2) &&
        ((G != geometry_t::CYLINDRICAL) || D == 3) &&

        // regime-specific constraints
        (R != regime_t::RMHD || D == 3) && (R != regime_t::MHD || D == 3) &&

        // solver-regime compatibility
        (S != solver_t::HLLD || (R == regime_t::RMHD || R == regime_t::MHD)) &&
        ((R == regime_t::NEWTONIAN || R == regime_t::SRHD)
             ? (S == solver_t::HLLE || S == solver_t::HLLC)
             : true) &&

        // exclude unimplemented regimes
        (R != regime_t::MHD);

    // =============================================================================
    // error handling
    // =============================================================================

    class configuration_error : public std::runtime_error
    {
      public:
        configuration_error(const std::string& msg) : std::runtime_error(msg) {}
    };

    class unsupported_configuration : public configuration_error
    {
      public:
        unsupported_configuration(const std::string& msg)
            : configuration_error("unsupported hydro configuration: " + msg)
        {
        }
    };

    class invalid_parameter_combination : public configuration_error
    {
      public:
        invalid_parameter_combination(const std::string& msg)
            : configuration_error("invalid parameter combination: " + msg)
        {
        }
    };

    // =============================================================================
    // dispatch implementation
    // =============================================================================

    namespace detail {

        // -------------------------------------------------------------------------
        // terminal dispatch - builds simulation and calls visitor
        // -------------------------------------------------------------------------
        template <
            regime_t         R,
            std::uint64_t    D,
            geometry_t       G,
            solver_t         S,
            reconstruction_t Rec,
            typename EoS,
            typename Visitor>
        auto call_visitor_with_state(
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gens,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const ecs::blueprint_set_t<D>&   blueprints
        ) -> std::enable_if_t<valid_combination<R, D, G, S, Rec>, void>
        {
            using namespace ecs;
            using namespace ecs::builders;
            using ecs::mesh_motion_config_t;

            // build simulation from blueprints
            auto builder = simulation_builder_t<R, D, G, EoS>{}
                               .configure_mesh(blueprints.mesh)
                               .configure_physics(blueprints.physics)
                               .configure_execution(blueprints.execution)
                               .configure_amr(blueprints.amr)
                               .configure_numerics(blueprints.numerics)
                               .configure_expressions(blueprints.expressions)
                               .configure_bodies(blueprints.bodies);

            // configure gravitational system if present
            if (blueprints.gravitational_system.has_value()) {
                builder.configure_gravitational_system(*blueprints.gravitational_system);
            }

            // configure decomposition if present
            if (blueprints.decomposition.has_value()) {
                builder.configure_decomposition(*blueprints.decomposition);
            }

            auto sim = builder.build();

            // configure moving mesh BEFORE field initialization
            // field_initializer needs motion_state() to compute cell volumes
            if (blueprints.mesh.moving_mesh) {
                mesh_motion_config_t motion_config;
                motion_config.scale_factor            = scale_factor;
                motion_config.scale_factor_derivative = scale_factor_derivative;
                motion_config.homologous              = blueprints.mesh.homologous_expansion;
                sim.registry.add(sim.global, std::move(motion_config));
            }

            // initialize fields from generators
            using sim_t = decltype(sim);
            // if restart file is provided, skip initialization
            if (blueprints.execution.restart_file.empty()) {
                creation::field_initializer_t<sim_t>::initialize(
                    sim,
                    prim_gen,
                    bfield_gens,
                    blueprints.physics.gamma
                );
            }

            // create operations bundle
            const auto ops = cfd::cfd_operations_t<R, D, S, Rec, EoS>{};

            // call visitor with typed simulation and ops
            visitor(sim, ops);
        }

        // fallback for invalid combinations
        template <
            regime_t         R,
            std::uint64_t    D,
            geometry_t       G,
            solver_t         S,
            reconstruction_t Rec,
            typename EoS,
            typename Visitor>
        auto call_visitor_with_state(
            Visitor&&,
            py::iterator,
            vector_t<py::iterator, 3>,
            std::function<real(real)> const&,
            std::function<real(real)> const&,
            const ecs::blueprint_set_t<D>&
        ) -> std::enable_if_t<!valid_combination<R, D, G, S, Rec>, void>
        {
            throw unsupported_configuration("invalid combination detected at compile time");
        }

        // -------------------------------------------------------------------------
        // eos dispatch
        // -------------------------------------------------------------------------
        template <
            regime_t         R,
            std::uint64_t    D,
            geometry_t       G,
            solver_t         S,
            reconstruction_t Rec,
            typename Visitor>
        void dispatch_eos(
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const ecs::blueprint_set_t<D>&   blueprints
        )
        {
            if constexpr (R == regime_t::NEWTONIAN) {
                if (blueprints.physics.gamma > 1.0) {
                    call_visitor_with_state<R, D, G, S, Rec, eos::ideal_gas_eos_t<R>>(
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                }
                else {
                    call_visitor_with_state<R, D, G, S, Rec, eos::isothermal_gas_eos_t>(
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                }
            }
            else {
                call_visitor_with_state<R, D, G, S, Rec, eos::ideal_gas_eos_t<R>>(
                    std::forward<Visitor>(visitor),
                    prim_gen,
                    bfield_gen,
                    scale_factor,
                    scale_factor_derivative,
                    blueprints
                );
            }
        }

        // -------------------------------------------------------------------------
        // reconstruction dispatch
        // -------------------------------------------------------------------------
        template <regime_t R, std::uint64_t D, geometry_t G, solver_t S, typename Visitor>
        void dispatch_reconstruction(
            reconstruction_t                 rec,
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const ecs::blueprint_set_t<D>&   blueprints
        )
        {
            switch (rec) {
                case reconstruction_t::PCM:
                    dispatch_eos<R, D, G, S, reconstruction_t::PCM>(
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case reconstruction_t::PLM:
                    dispatch_eos<R, D, G, S, reconstruction_t::PLM>(
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported reconstruction: " + std::to_string(static_cast<int>(rec))
                    );
            }
        }

        // -------------------------------------------------------------------------
        // solver dispatch
        // -------------------------------------------------------------------------
        template <regime_t R, std::uint64_t D, geometry_t G, typename Visitor>
        void dispatch_solver(
            solver_t                         solver,
            reconstruction_t                 rec,
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const ecs::blueprint_set_t<D>&   blueprints
        )
        {
            switch (solver) {
                case solver_t::HLLE:
                    dispatch_reconstruction<R, D, G, solver_t::HLLE>(
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case solver_t::HLLC:
                    dispatch_reconstruction<R, D, G, solver_t::HLLC>(
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case solver_t::HLLD:
                    dispatch_reconstruction<R, D, G, solver_t::HLLD>(
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported solver: " + std::to_string(static_cast<int>(solver))
                    );
            }
        }

        // -------------------------------------------------------------------------
        // geometry dispatch
        // -------------------------------------------------------------------------
        template <regime_t R, std::uint64_t D, typename Visitor>
        void dispatch_geometry(
            geometry_t                       geometry,
            solver_t                         solver,
            reconstruction_t                 rec,
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const ecs::blueprint_set_t<D>&   blueprints
        )
        {
            switch (geometry) {
                case geometry_t::CARTESIAN:
                    dispatch_solver<R, D, geometry_t::CARTESIAN>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case geometry_t::CYLINDRICAL:
                    dispatch_solver<R, D, geometry_t::CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case geometry_t::AXIS_CYLINDRICAL:
                    dispatch_solver<R, D, geometry_t::AXIS_CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case geometry_t::PLANAR_CYLINDRICAL:
                    dispatch_solver<R, D, geometry_t::PLANAR_CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                case geometry_t::SPHERICAL:
                    dispatch_solver<R, D, geometry_t::SPHERICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported geometry: " + std::to_string(static_cast<int>(geometry))
                    );
            }
        }

        // -------------------------------------------------------------------------
        // dimensions dispatch
        //
        // this is where blueprint_set_t<D> is created with the resolved D
        // -------------------------------------------------------------------------
        template <regime_t R, typename Visitor>
        void dispatch_dimensions(
            std::uint64_t                    dims,
            geometry_t                       geometry,
            solver_t                         solver,
            reconstruction_t                 rec,
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const config_dict_t&             config
        )
        {
            switch (dims) {
                case 1: {
                    auto blueprints = ecs::blueprint_set_t<1>::from_config(config);
                    dispatch_geometry<R, 1>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                }
                case 2: {
                    auto blueprints = ecs::blueprint_set_t<2>::from_config(config);
                    dispatch_geometry<R, 2>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                }
                case 3: {
                    auto blueprints = ecs::blueprint_set_t<3>::from_config(config);
                    dispatch_geometry<R, 3>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        blueprints
                    );
                    break;
                }
                default:
                    throw unsupported_configuration(
                        "unsupported dimensions: " + std::to_string(dims)
                    );
            }
        }

        // -------------------------------------------------------------------------
        // regime dispatch (top level)
        // -------------------------------------------------------------------------
        template <typename Visitor>
        void dispatch_regime(
            regime_t                         regime,
            std::uint64_t                    dims,
            geometry_t                       geometry,
            solver_t                         solver,
            reconstruction_t                 rec,
            Visitor&&                        visitor,
            py::iterator                     prim_gen,
            vector_t<py::iterator, 3>        bfield_gen,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const config_dict_t&             config
        )
        {
            switch (regime) {
                case regime_t::NEWTONIAN:
                    dispatch_dimensions<regime_t::NEWTONIAN>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        config
                    );
                    break;
                case regime_t::SRHD:
                    dispatch_dimensions<regime_t::SRHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        config
                    );
                    break;
                case regime_t::RMHD:
                    dispatch_dimensions<regime_t::RMHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        prim_gen,
                        bfield_gen,
                        scale_factor,
                        scale_factor_derivative,
                        config
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported regime: " + std::to_string(static_cast<int>(regime))
                    );
            }
        }

    } // namespace detail

    // =============================================================================
    // main entry point
    // =============================================================================

    /**
     * create hydro state and call visitor with it.
     *
     * visitor receives:
     *   - sim: simulation_t<R, D, G, EoS> (fully typed)
     *   - ops: cfd_operations_t<R, D, S, Rec, EoS>
     *
     * only the specific template combination requested gets compiled.
     */
    template <typename Visitor>
    void with_hydro_state(
        const config_dict_t&             config,
        py::iterator                     prim_gen,
        vector_t<py::iterator, 3>        bfield_gen,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative,
        Visitor&&                        visitor
    )
    {
        using namespace ecs::creation;

        // extract dispatch parameters from config
        auto regime_str   = config.at("regime").get<std::string>();
        auto geometry_str = config.at("coord_system").get<std::string>();
        auto solver_str   = config.at("solver").get<std::string>();
        auto rec_str      = config.at("reconstruction").get<std::string>();
        auto dims         = config.at("dimensionality").get<std::uint64_t>();

        // convert to enums
        auto regime         = deserialize<regime_t>(regime_str);
        auto geometry       = deserialize<geometry_t>(geometry_str);
        auto solver         = deserialize<solver_t>(solver_str);
        auto reconstruction = deserialize<reconstruction_t>(rec_str);

        try {
            detail::dispatch_regime(
                regime,
                dims,
                geometry,
                solver,
                reconstruction,
                std::forward<Visitor>(visitor),
                prim_gen,
                bfield_gen,
                scale_factor,
                scale_factor_derivative,
                config
            );
        }
        catch (const configuration_error&) {
            std::string msg = "regime=" + regime_str + ", dims=" + std::to_string(dims) +
                              ", geometry=" + geometry_str + ", solver=" + solver_str +
                              ", reconstruction=" + rec_str;
            throw unsupported_configuration(msg);
        }
    }

} // namespace simbi::dispatch

#endif // HYDRO_DISPATCH_HPP
