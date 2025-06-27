#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "dispatch.hpp"
#include <cstdint>
#include <exception>
#include <string>

namespace simbi::dispatch {
    namespace detail {
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
        )
        {
            switch (regime) {
                case Regime::NEWTONIAN:
                    return dispatch_dimensions<Regime::NEWTONIAN>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Regime::SRHD:
                    return dispatch_dimensions<Regime::SRHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                case Regime::RMHD:
                    return dispatch_dimensions<Regime::RMHD>(
                        dims,
                        geometry,
                        solver,
                        rec,
                        cons_data,
                        prim_data,
                        bfield_data,
                        init
                    );
                // add more as you implement them:
                // case Regime::MHD:
                //     return dispatch_dimensions<Regime::MHD>(dims, geometry,
                //     solver, rec, cons_data, prim_data, bfield_data, init);
                default:
                    throw unsupported_configuration(
                        "unsupported regime: " +
                        std::to_string(static_cast<int>(regime))
                    );
            }
        }
    }   // namespace detail

    hydro_state_ptr create_hydro_state(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        const InitialConditions& init
    )
    {
        const auto regime_str         = init.regime;
        const auto geometry_str       = init.coord_system;
        const auto solver_str         = init.solver;
        const auto reconstruction_str = init.reconstruct;
        const auto dims               = init.dimensionality;
        // convert runtime strings to enum values
        auto regime         = deserialize<Regime>(regime_str);
        auto geometry       = deserialize<Geometry>(geometry_str);
        auto solver         = deserialize<Solver>(solver_str);
        auto reconstruction = deserialize<Reconstruction>(reconstruction_str);

        try {
            // lazy dispatch
            return detail::dispatch_regime(
                regime,
                dims,
                geometry,
                solver,
                reconstruction,
                cons_data,
                prim_data,
                bfield_data,
                init
            );
        }
        catch (const std::exception&) {
            // generate helpful error message
            std::string msg =
                "regime=" + regime_str + ", dims=" + std::to_string(dims) +
                ", geometry=" + geometry_str + ", solver=" + solver_str +
                ", reconstruction=" + reconstruction_str;
            throw unsupported_configuration(msg);
        }
    }

    // check if a configuration would be supported (without creating state)
    bool is_configuration_supported(
        const std::string& /*regime_str*/,
        std::uint64_t /*dims*/,
        const std::string& /*geometry_str*/,
        const std::string& /*solver_str*/,
        const std::string& /*reconstruction_str*/
    )
    {
        try {
            // auto regime   = deserialize<Regime>(regime_str);
            // auto geometry = deserialize<Geometry>(geometry_str);
            // auto solver   = deserialize<Solver>(solver_str);
            // auto reconstruction =
            //     deserialize<Reconstruction>(reconstruction_str);

            // this is a compile-time check, but we can't easily do it at
            // runtime for now, we'll just try to create and catch exceptions
            // TODO: implement compile-time validation reflection
            return true;   // optimistic approach
        }
        catch (...) {
            return false;
        }
    }

}   // namespace simbi::dispatch
