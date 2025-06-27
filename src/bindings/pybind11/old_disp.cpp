#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "data/state/hydro_state.hpp"
#include "dispatch.hpp"
#include <cstdint>
#include <string>

namespace simbi::dispatch {
    /**
     * main dispatch function - converts runtime parameters to compile-time
     * types
     *
     * creates fully-typed hydro_state_t based on runtime configuration.
     * after this call, python is forgotten and templates take over!
     */
    auto create_hydro_state(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        const InitialConditions& init
    ) -> hydro_state_variant_t
    {
        const auto regime_str         = init.regime;
        const auto dims               = init.dimensionality;
        const auto geometry_str       = init.coord_system;
        const auto solver_str         = init.solver;
        const auto reconstruction_str = init.reconstruct;
        // stage 1: dispatch by regime
        auto regime = deserialize<Regime>(regime_str);

        switch (regime) {
            case Regime::NEWTONIAN:
                return detail::dispatch_newtonian(
                    dims,
                    geometry_str,
                    solver_str,
                    reconstruction_str,
                    cons_data,
                    prim_data,
                    bfield_data,
                    init
                );

            case Regime::SRHD:
                return detail::dispatch_srhd(
                    dims,
                    geometry_str,
                    solver_str,
                    reconstruction_str,
                    cons_data,
                    prim_data,
                    bfield_data,
                    init
                );

            case Regime::RMHD:
                return detail::dispatch_rmhd(
                    dims,
                    geometry_str,
                    solver_str,
                    reconstruction_str,
                    cons_data,
                    prim_data,
                    bfield_data,
                    init
                );

            case Regime::MHD:
                throw unsupported_configuration(
                    "classical mhd not yet implemented"
                );

            default:
                throw unsupported_configuration(
                    "unknown regime: " + regime_str
                );
        }
    }

    namespace detail {

        // stage 2: dispatch newtonian by dimensions and other parameters
        auto dispatch_newtonian(
            std::uint64_t dims,
            const std::string& geometry_str,
            const std::string& solver_str,
            const std::string& reconstruction_str,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        ) -> hydro_state_variant_t
        {
            auto geometry = deserialize<Geometry>(geometry_str);
            auto solver   = deserialize<Solver>(solver_str);
            auto reconstruction =
                deserialize<Reconstruction>(reconstruction_str);

            // validate geometry support
            if (geometry != Geometry::CARTESIAN) {
                throw unsupported_configuration(
                    "newtonian only supports cartesian geometry currently"
                );
            }

            // validate reconstruction support
            if (reconstruction != Reconstruction::PLM) {
                throw unsupported_configuration(
                    "newtonian only supports plm reconstruction currently"
                );
            }

            // dispatch by dimensions and solver
            switch (dims) {
                case 1:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::NEWTONIAN,
                                1,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "1d newtonian only supports hllc solver "
                                "currently"
                            );
                    }

                case 2:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::NEWTONIAN,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::NEWTONIAN,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "2d newtonian supports hllc/hlle solvers only"
                            );
                    }

                case 3:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::NEWTONIAN,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::NEWTONIAN,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "3d newtonian supports hllc/hlle solvers only"
                            );
                    }

                default:
                    throw unsupported_configuration(
                        "newtonian supports 1d, 2d, 3d only"
                    );
            }
        }

        // stage 2: dispatch srhd by dimensions and other parameters
        auto dispatch_srhd(
            std::uint64_t dims,
            const std::string& geometry_str,
            const std::string& solver_str,
            const std::string& reconstruction_str,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        ) -> hydro_state_variant_t
        {
            auto geometry = deserialize<Geometry>(geometry_str);
            auto solver   = deserialize<Solver>(solver_str);
            auto reconstruction =
                deserialize<Reconstruction>(reconstruction_str);

            // validate geometry support
            if (geometry != Geometry::CARTESIAN) {
                throw unsupported_configuration(
                    "srhd only supports cartesian geometry currently"
                );
            }

            // validate reconstruction support
            if (reconstruction != Reconstruction::PLM) {
                throw unsupported_configuration(
                    "srhd only supports plm reconstruction currently"
                );
            }

            // dispatch by dimensions and solver
            switch (dims) {
                case 1:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::SRHD,
                                1,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "1d srhd only supports hllc solver currently"
                            );
                    }

                case 2:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::SRHD,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::SRHD,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "2d srhd supports hllc/hlle solvers only"
                            );
                    }

                case 3:
                    switch (solver) {
                        case Solver::HLLC:
                            return state::hydro_state_t<
                                Regime::SRHD,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLC,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::SRHD,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "3d srhd supports hllc/hlle solvers only"
                            );
                    }

                default:
                    throw unsupported_configuration(
                        "srhd supports 1d, 2d, 3d only"
                    );
            }
        }

        // stage 2: dispatch rmhd by dimensions and other parameters
        auto dispatch_rmhd(
            std::uint64_t dims,
            const std::string& geometry_str,
            const std::string& solver_str,
            const std::string& reconstruction_str,
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        ) -> hydro_state_variant_t
        {
            auto geometry = deserialize<Geometry>(geometry_str);
            auto solver   = deserialize<Solver>(solver_str);
            auto reconstruction =
                deserialize<Reconstruction>(reconstruction_str);

            // validate geometry support
            if (geometry != Geometry::CARTESIAN) {
                throw unsupported_configuration(
                    "rmhd only supports cartesian geometry currently"
                );
            }

            // validate reconstruction support
            if (reconstruction != Reconstruction::PLM) {
                throw unsupported_configuration(
                    "rmhd only supports plm reconstruction currently"
                );
            }

            // rmhd doesn't support 1d
            if (dims == 1) {
                throw unsupported_configuration("rmhd requires at least 2d");
            }

            // dispatch by dimensions and solver
            switch (dims) {
                case 2:
                    switch (solver) {
                        case Solver::HLLD:
                            return state::hydro_state_t<
                                Regime::RMHD,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLD,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::RMHD,
                                2,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "2d rmhd supports hlld/hlle solvers only"
                            );
                    }

                case 3:
                    switch (solver) {
                        case Solver::HLLD:
                            return state::hydro_state_t<
                                Regime::RMHD,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLD,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        case Solver::HLLE:
                            return state::hydro_state_t<
                                Regime::RMHD,
                                3,
                                Geometry::CARTESIAN,
                                Solver::HLLE,
                                Reconstruction::PLM>::
                                from_init(
                                    cons_data,
                                    prim_data,
                                    bfield_data,
                                    init
                                );
                        default:
                            throw unsupported_configuration(
                                "3d rmhd supports hlld/hlle solvers only"
                            );
                    }

                default:
                    throw unsupported_configuration(
                        "rmhd supports 2d, 3d only"
                    );
            }
        }

    }   // namespace detail

}   // namespace simbi::dispatch
