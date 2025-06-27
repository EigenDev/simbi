#ifndef SIMBI_HYDRO_DISPATCH_HPP
#define SIMBI_HYDRO_DISPATCH_HPP

#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "data/state/hydro_state.hpp"
#include <cstdint>
#include <memory>
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
         G == Geometry::AXIS_CYLINDRICAL ||
         G == Geometry::PLANAR_CYLINDRICAL) &&

        // regime-specific constraints
        (R != Regime::RMHD || D == 3) &&   // rmhd requires 3D
        (R != Regime::MHD ||
         D == 3) &&   // mhd requires at least 3D (when implemented)

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
    // TYPE-ERASED HYDRO STATE - Holds any valid combination
    //==============================================================================
    class hydro_state_base
    {
      public:
        virtual ~hydro_state_base()                   = default;
        virtual std::string get_configuration() const = 0;
    };

    template <
        Regime R,
        std::uint64_t D,
        Geometry G,
        Solver S,
        Reconstruction Rec>
        requires valid_combination<R, D, G, S, Rec>
    class hydro_state_wrapper : public hydro_state_base
    {
      private:
        state::hydro_state_t<R, D, G, S, Rec> state_;

      public:
        template <typename... Args>
        hydro_state_wrapper(Args&&... args)
            : state_(std::forward<Args>(args)...)
        {
        }

        std::string get_configuration() const override
        {
            return "regime=" + std::to_string(static_cast<int>(R)) +
                   ", dims=" + std::to_string(D) +
                   ", geometry=" + std::to_string(static_cast<int>(G)) +
                   ", solver=" + std::to_string(static_cast<int>(S)) +
                   ", reconstruction=" + std::to_string(static_cast<int>(Rec));
        }

        // access to underlying state for advanced operations
        const auto& get_state() const { return state_; }
        auto& get_state() { return state_; }
    };

    using hydro_state_ptr = std::unique_ptr<hydro_state_base>;

    //==============================================================================
    // ERROR HANDLING
    //==============================================================================

    class unsupported_configuration : public std::runtime_error
    {
      public:
        unsupported_configuration(const std::string& msg)
            : std::runtime_error("unsupported hydro configuration: " + msg)
        {
        }
    };

    //==============================================================================
    // LAZY TEMPLATE DISPATCH
    //==============================================================================
    namespace detail {

        // template factory function - only instantiated if valid_combination is
        // true
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec>
        auto create_specific_state(
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        ) -> std::
            enable_if_t<valid_combination<R, D, G, S, Rec>, hydro_state_ptr>
        {
            auto state = state::hydro_state_t<R, D, G, S, Rec>::from_init(
                cons_data,
                prim_data,
                bfield_data,
                init
            );
            return std::make_unique<hydro_state_wrapper<R, D, G, S, Rec>>(
                std::move(state)
            );
        }

        // fallback for invalid combinations - never instantiated due to SFINAE
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec>
        auto create_specific_state(
            void*,
            void*,
            vector_t<void*, 3>,
            const InitialConditions&
        ) -> std::
            enable_if_t<!valid_combination<R, D, G, S, Rec>, hydro_state_ptr>
        {
            throw unsupported_configuration(
                "invalid combination detected at compile time"
            );
        }

        // reconstruction dispatch
        template <Regime R, std::uint64_t D, Geometry G, Solver S>
        hydro_state_ptr dispatch_reconstruction(
            Reconstruction rec,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        )
        {
            switch (rec) {
                case Reconstruction::PCM:
                    return create_specific_state<
                        R,
                        D,
                        G,
                        S,
                        Reconstruction::PCM>(
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Reconstruction::PLM:
                    return create_specific_state<
                        R,
                        D,
                        G,
                        S,
                        Reconstruction::PLM>(
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                // add more as I implement them:
                // case Reconstruction::PPM:
                //     return create_specific_state<R, D, G, S,
                //     Reconstruction::PPM>(cons_data, prim_data, bfield_data,
                //     init);
                default:
                    throw unsupported_configuration(
                        "unsupported reconstruction: " +
                        std::to_string(static_cast<int>(rec))
                    );
            }
        }

        // solver dispatch
        template <Regime R, std::uint64_t D, Geometry G>
        hydro_state_ptr dispatch_solver(
            Solver solver,
            Reconstruction rec,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        )
        {
            switch (solver) {
                case Solver::HLLE:
                    return dispatch_reconstruction<R, D, G, Solver::HLLE>(
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Solver::HLLC:
                    return dispatch_reconstruction<R, D, G, Solver::HLLC>(
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Solver::HLLD:
                    return dispatch_reconstruction<R, D, G, Solver::HLLD>(
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                // add more as you implement them:
                // case Solver::AUSM_PLUS:
                //     return dispatch_reconstruction<R, D, G,
                //     Solver::AUSM_PLUS>(rec, cons_data, prim_data,
                //     bfield_data, init);
                default:
                    throw unsupported_configuration(
                        "unsupported solver: " +
                        std::to_string(static_cast<int>(solver))
                    );
            }
        }

        // geometry dispatch
        template <Regime R, std::uint64_t D>
        hydro_state_ptr dispatch_geometry(
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        )
        {
            switch (geometry) {
                case Geometry::CARTESIAN:
                    return dispatch_solver<R, D, Geometry::CARTESIAN>(
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Geometry::CYLINDRICAL:
                    return dispatch_solver<R, D, Geometry::CYLINDRICAL>(
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Geometry::AXIS_CYLINDRICAL:
                    return dispatch_solver<R, D, Geometry::AXIS_CYLINDRICAL>(
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Geometry::PLANAR_CYLINDRICAL:
                    return dispatch_solver<R, D, Geometry::PLANAR_CYLINDRICAL>(
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                // add more as you implement them:
                // case Geometry::SPHERICAL:
                //     return dispatch_solver<R, D, Geometry::SPHERICAL>(solver,
                //     rec, cons_data, prim_data, bfield_data, init);
                default:
                    throw unsupported_configuration(
                        "unsupported geometry: " +
                        std::to_string(static_cast<int>(geometry))
                    );
            }
        }

        // dimensions dispatch
        template <Regime R>
        hydro_state_ptr dispatch_dimensions(
            std::uint64_t dims,
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        )
        {
            switch (dims) {
                case 1:
                    return dispatch_geometry<R, 1>(
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case 2:
                    return dispatch_geometry<R, 2>(
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case 3:
                    return dispatch_geometry<R, 3>(
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                default:
                    throw unsupported_configuration(
                        "unsupported dimensions: " + std::to_string(dims)
                    );
            }
        }

        // regime dispatch (top level)
        hydro_state_ptr dispatch_regime(
            Regime regime,
            std::uint64_t dims,
            Geometry geometry,
            Solver solver,
            Reconstruction rec,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        );

    }   // namespace detail

    //==============================================================================
    // MAIN DISPATCH FUNCTION - Runtime → Lazy template instantiation
    //==============================================================================

    /**
     * create hydro state with lazy template instantiation
     *
     * only the SPECIFIC combination you request gets compiled!
     * no template bloat, fast compilation, zero runtime overhead after
     * creation.
     */
    hydro_state_ptr create_hydro_state(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        const InitialConditions& init
    );

    //==============================================================================
    // UTILITY FUNCTIONS
    //==============================================================================

    // check if a configuration would be supported (without creating state)
    bool is_configuration_supported(
        const std::string& regime_str,
        std::uint64_t dims,
        const std::string& geometry_str,
        const std::string& solver_str,
        const std::string& reconstruction_str
    );

}   // namespace simbi::dispatch

#endif   // SIMBI_HYDRO_DISPATCH_HPP
