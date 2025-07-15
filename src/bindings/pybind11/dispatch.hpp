#ifndef SIMBI_HYDRO_DISPATCH_HPP
#define SIMBI_HYDRO_DISPATCH_HPP

#include "config.hpp"
#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "data/state/hydro_state.hpp"
#include "physics/eos/ideal.hpp"
#include "physics/eos/isothermal.hpp"
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace simbi::dispatch {

    //==============================================================================
    // VALIDITY CONCEPTS - Define which combinations are supported
    //==============================================================================

    template <
        Regime R,
        std::uint64_t D,
        Geometry G,
        Solver S,
        Reconstruction Rec>
    concept valid_combination =
        // basic constraints
        (D >= 1 && D <= 3) &&

        // reconstruction constraints
        (Rec == Reconstruction::PCM ||
         Rec == Reconstruction::PLM) &&   // only implemented ones

        // geometry constraints
        (G == Geometry::CARTESIAN || G == Geometry::CYLINDRICAL ||
         G == Geometry::AXIS_CYLINDRICAL || G == Geometry::PLANAR_CYLINDRICAL ||
         G == Geometry::SPHERICAL) &&

        // geometry-dimension constraints
        ((G != Geometry::AXIS_CYLINDRICAL &&
          G != Geometry::PLANAR_CYLINDRICAL) ||
         D == 2) &&   // AXIS and PLANAR only in 2D
        ((G != Geometry::CYLINDRICAL) || D == 3) &&   // CYLINDRICAL only in 3D

        // regime-specific constraints
        (R != Regime::RMHD || D == 3) &&   // (r)mhd requires 3D
        (R != Regime::MHD || D == 3) &&

        // solver-regime compatibility
        (
            S != Solver::HLLD || (R == Regime::RMHD || R == Regime::MHD)
        ) &&   // hlld only for mhd regimes
        ((R == Regime::NEWTONIAN || R == Regime::SRHD)
             ? (S == Solver::HLLE || S == Solver::HLLC)
             : true) &&

        // exclude unimplemented regimes
        (R != Regime::MHD);   // classical mhd not implemented yet

    //==============================================================================
    // ERROR HANDLING
    //==============================================================================
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

    //==============================================================================
    // VISITOR PATTERN DISPATCH - Gets you raw template types!
    //==============================================================================

    namespace detail {

        // create specific state and call visitor with it
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec,
            typename EoS,
            typename Visitor>
        auto call_visitor_with_state(
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        ) -> std::enable_if_t<valid_combination<R, D, G, S, Rec>, void>
        {
            auto state = state::hydro_state_t<R, D, G, S, Rec, EoS>::from_init(
                cons_data,
                prim_data,
                bfield_data,
                scale_factor,
                scale_factor_derivative,
                init
            );

            // call visitor
            visitor(state);
        }

        // fallback for invalid combinations
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec,
            typename EoS,
            typename Visitor>
        auto call_visitor_with_state(
            Visitor&&,
            void*,
            void*,
            vector_t<void*, 3>,
            std::function<real(real)> const&,
            std::function<real(real)> const&,
            const InitialConditions&
        ) -> std::enable_if_t<!valid_combination<R, D, G, S, Rec>, void>
        {
            throw unsupported_configuration(
                "invalid combination detected at compile time"
            );
        }

        // EoS dispatch
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec,
            typename Visitor>
        void dispatch_eos(
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            if (init.gamma - 1.0 < 1e-6) {
                call_visitor_with_state<
                    R,
                    D,
                    G,
                    S,
                    Rec,
                    eos::isothermal_gas_eos_t>(
                    std::forward<Visitor>(visitor),
                    cons_data,
                    prim_data,
                    bfield_data,
                    scale_factor,
                    scale_factor_derivative,
                    init
                );
            }
            else {
                call_visitor_with_state<
                    R,
                    D,
                    G,
                    S,
                    Rec,
                    eos::ideal_gas_eos_t<R>>(
                    std::forward<Visitor>(visitor),
                    cons_data,
                    prim_data,
                    bfield_data,
                    scale_factor,
                    scale_factor_derivative,
                    init
                );
            }
        }

        // reconstruction dispatch
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            typename Visitor>
        void dispatch_reconstruction(
            Reconstruction rec,
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            switch (rec) {
                case Reconstruction::PCM:
                    dispatch_eos<R, D, G, S, Reconstruction::PCM>(
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Reconstruction::PLM:
                    dispatch_eos<R, D, G, S, Reconstruction::PLM>(
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported reconstruction: " +
                        std::to_string(static_cast<int>(rec))
                    );
            }
        }

        // solver dispatch
        template <Regime R, std::uint64_t D, Geometry G, typename Visitor>
        void dispatch_solver(
            Solver solver,
            Reconstruction rec,
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            switch (solver) {
                case Solver::HLLE:
                    dispatch_reconstruction<R, D, G, Solver::HLLE>(
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Solver::HLLC:
                    dispatch_reconstruction<R, D, G, Solver::HLLC>(
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Solver::HLLD:
                    dispatch_reconstruction<R, D, G, Solver::HLLD>(
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported solver: " +
                        std::to_string(static_cast<int>(solver))
                    );
            }
        }

        // geometry dispatch
        template <Regime R, std::uint64_t D, typename Visitor>
        void dispatch_geometry(
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            switch (geometry) {
                case Geometry::CARTESIAN:
                    dispatch_solver<R, D, Geometry::CARTESIAN>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Geometry::CYLINDRICAL:
                    dispatch_solver<R, D, Geometry::CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Geometry::AXIS_CYLINDRICAL:
                    dispatch_solver<R, D, Geometry::AXIS_CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Geometry::PLANAR_CYLINDRICAL:
                    dispatch_solver<R, D, Geometry::PLANAR_CYLINDRICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Geometry::SPHERICAL:
                    dispatch_solver<R, D, Geometry::SPHERICAL>(
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported geometry: " +
                        std::to_string(static_cast<int>(geometry))
                    );
            }
        }

        // dimensions dispatch
        template <Regime R, typename Visitor>
        void dispatch_dimensions(
            std::uint64_t dims,
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            switch (dims) {
                case 1:
                    dispatch_geometry<R, 1>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case 2:
                    dispatch_geometry<R, 2>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case 3:
                    dispatch_geometry<R, 3>(
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported dimensions: " + std::to_string(dims)
                    );
            }
        }

        // regime dispatch (top level)
        template <typename Visitor>
        void dispatch_regime(
            Regime regime,
            std::uint64_t dims,
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            Visitor&& visitor,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            const InitialConditions& init
        )
        {
            switch (regime) {
                case Regime::NEWTONIAN:
                    dispatch_dimensions<Regime::NEWTONIAN>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Regime::SRHD:
                    dispatch_dimensions<Regime::SRHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                case Regime::RMHD:
                    dispatch_dimensions<Regime::RMHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        std::forward<Visitor>(visitor),
                        cons_data,
                        prim_data,
                        bfield_data,
                        scale_factor,
                        scale_factor_derivative,
                        init
                    );
                    break;
                default:
                    throw unsupported_configuration(
                        "unsupported regime: " +
                        std::to_string(static_cast<int>(regime))
                    );
            }
        }

    }   // namespace detail

    //==============================================================================
    // MAIN VISITOR DISPATCH FUNCTION
    //==============================================================================

    /**
     * create hydro state and call visitor with it
     *
     * visitor receives the raw hydro_state_t<R,D,G,S,Rec> type!
     * only the specific combination you request gets compiled!
     */
    template <typename Visitor>
    void with_hydro_state(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative,
        const InitialConditions& init,
        Visitor&& visitor
    )
    {
        const auto& regime_str         = init.regime;
        const auto& geometry_str       = init.coord_system;
        const auto& solver_str         = init.solver;
        const auto& reconstruction_str = init.reconstruct;
        const auto dims                = init.dimensionality;
        // convert runtime strings to enum values
        auto regime         = deserialize<Regime>(regime_str);
        auto geometry       = deserialize<Geometry>(geometry_str);
        auto solver         = deserialize<Solver>(solver_str);
        auto reconstruction = deserialize<Reconstruction>(reconstruction_str);

        try {
            // dispatch to visitor with raw template type!
            detail::dispatch_regime(
                regime,
                dims,
                geometry,
                solver,
                reconstruction,
                std::forward<Visitor>(visitor),
                cons_data,
                prim_data,
                bfield_data,
                scale_factor,
                scale_factor_derivative,
                init
            );
        }
        catch (const configuration_error&) {
            // generate helpful error message
            std::string msg =
                "regime=" + regime_str + ", dims=" + std::to_string(dims) +
                ", geometry=" + geometry_str + ", solver=" + solver_str +
                ", reconstruction=" + reconstruction_str;
            throw unsupported_configuration(msg);
        }
    }

}   // namespace simbi::dispatch

#endif   // SIMBI_HYDRO_DISPATCH_HPP
