#ifndef SIMBI_STATE_HYDRO_STATE_HPP
#define SIMBI_STATE_HYDRO_STATE_HPP

#include "config.hpp"                        // for real, size_type
#include "core/containers/array.hpp"         // for array_t
#include "core/containers/collapsable.hpp"   // for collapsable
#include "core/containers/ndarray.hpp"       // for ndarray
#include "core/types/alias/alias.hpp"        // for luint
#include "core/utility/bimap.hpp"            // for deserialize, serialize
#include "core/utility/enums.hpp"   // for Regime, Geometry, Cellspacing, Solver, ShockWaveLimiter, Reconstruction, Timestepping, BoundaryCondition
#include "core/utility/init_conditions.hpp"   // for InitialConditions
#include "core/utility/managed.hpp"           // for Managed
#include "core/utility/smart_ptr.hpp"         // for smart_ptr
#include "express_t.hpp"                      // for expression_t
#include "geometry/mesh/mesh.hpp"             // for Mesh
#include "physics/eos/ideal.hpp"              // for ideal_gas_eos_t
// #include "physics/hydro/schemes/ib/delta/collector.hpp"   // for
// GridBodyDeltaCollector #include
// "physics/hydro/schemes/ib/systems/component_body_system.hpp"   // for
// ComponentBodySystem #include
// "physics/hydro/schemes/ib/systems/component_generator.hpp"   // for
// ComponentGenerator
#include <string>    // for string
#include <utility>   // for move, forward

namespace simbi::state {
    using namespace utility;
    using namespace containers;
    using namespace eos;

    template <Regime R, size_type Dims, typename EoS = ideal_gas_eos_t<R>>
    struct hydro_state_t : public Managed<platform::is_gpu> {
        // compile-time constants
        using express_t                       = expression_t<Dims>;
        using eos_t                           = EoS;
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;

        Mesh<Dims> mesh;
        // arrays
        ndarray_t<real> cons;
        ndarray_t<real> prim;
        array_t<ndarray_t<real>, Dims> flux;
        // optional fields
        array_t<ndarray_t<real>, Dims> bstag;
        // util::smart_ptr<ibsystem::ComponentBodySystem<real, Dims>>
        // body_system; util::smart_ptr<ibsystem::GridBodyDeltaCollector<real,
        // Dims>> collector;
        struct meta_data_t {
            // numerics
            real gamma;
            real plm_theta;
            real cfl;
            real time;
            real dt;

            // int
            size_type iteration;
            size_type halo_radius;

            // enums
            Regime regime;
            ShockWaveLimiter shock_smoother;
            Solver solver;
            Cellspacing x1_spacing;
            Cellspacing x2_spacing;
            Cellspacing x3_spacing;
            Geometry coord_system;
            Reconstruction reconstruction;
            Timestepping timestepping;
            array_t<BoundaryCondition, 2 * Dims> boundary_conditions;

            // flags
            bool is_mhd;
            bool is_relativistic;
        } metadata;

        express_t hydro_source;
        express_t gravity_source;
        array_t<express_t, 2 * Dims> bx_inner;

        // track the failure state
        bool in_failure_state{false};

        hydro_state_t() = default;

        static hydro_state_t from_init(
            ndarray_t<real>&& cons,
            ndarray_t<real>&& prim,
            array_t<ndarray_t<real>, Dims>&& bfields,
            InitialConditions&& init
        )
        {

            hydro_state_t state;
            state.mesh = Mesh<Dims>(init);
            state.cons = std::move(cons);
            state.prim = std::move(prim);
            if constexpr (R == Regime::RMHD) {
                state.bstag = std::move(bfields);
            }

            const auto [nia, nja, nka] = init.active_zones();
            const bool is_mhd          = init.is_mhd;
            for (size_type ii = 0; ii < Dims; ++ii) {
                const size_type ni = nia + (ii == 0) + (is_mhd * 2 * (ii != 0));
                const size_type nj = nja + (ii == 1) + (is_mhd * 2 * (ii != 1));
                const size_type nk = nka + (ii == 2) + (is_mhd * 2 * (ii != 2));
                state.flux[ii].resize(ni * nj * nk);
            }

            if (is_mhd) {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    const size_type ni = nia + (ii == 0) + (2 * (ii != 0));
                    const size_type nj = nja + (ii == 1) + (2 * (ii != 1));
                    const size_type nk = nka + (ii == 2) + (2 * (ii != 2));
                    state.bstag[ii].resize(ni * nj * nk);
                }
            }

            // set up expression
            state.initialize_expressions(init);

            // set up metadata
            state.initialize_metadata(init);

            // setup body system if needed
            state.initialize_body_system(init);

            return state;
        }

        void initialize_expressions(const InitialConditions& init)
        {
            // setup source expressions
            hydro_source =
                express_t::from_config(init.hydro_source_expressions);
            gravity_source =
                express_t::from_config(init.gravity_source_expressions);

            // setup boundary expressions
            if constexpr (Dims >= 1) {
                bx_inner[0] =
                    express_t::from_config(init.bx1_inner_expressions);
                bx_inner[1] =
                    express_t::from_config(init.bx1_outer_expressions);
            }

            if constexpr (Dims >= 2) {
                bx_inner[2] =
                    express_t::from_config(init.bx2_inner_expressions);
                bx_inner[3] =
                    express_t::from_config(init.bx2_outer_expressions);
            }

            if constexpr (Dims >= 3) {
                bx_inner[4] =
                    express_t::from_config(init.bx3_inner_expressions);
                bx_inner[5] =
                    express_t::from_config(init.bx3_outer_expressions);
            }
        }

        void initialize_metadata(const InitialConditions& init)
        {
            metadata = {
              .gamma          = init.gamma,
              .plm_theta      = init.plm_theta,
              .cfl            = init.cfl,
              .time           = init.time,
              .dt             = 0.0,
              .iteration      = 0,
              .halo_radius    = init.halo_radius,
              .regime         = deserialize<Regime>(init.regime),
              .shock_smoother = get_shock_smoother(init),
              .solver         = deserialize<Solver>(init.solver),
              .x1_spacing     = deserialize<Cellspacing>(init.x1_spacing),
              .x2_spacing     = deserialize<Cellspacing>(init.x2_spacing),
              .x3_spacing     = deserialize<Cellspacing>(init.x3_spacing),
              .coord_system   = deserialize<Geometry>(init.coord_system),
              .reconstruction = deserialize<Reconstruction>(init.reconstruct),
              .timestepping   = deserialize<Timestepping>(init.timestepping),
              .boundary_conditions = array_t<BoundaryCondition, 2 * Dims>{},
              .is_mhd              = init.is_mhd,
              .is_relativistic     = init.is_relativistic
            };

            for (size_type ii = 0; ii < 2 * Dims; ++ii) {
                metadata.boundary_conditions[ii] =
                    deserialize<BoundaryCondition>(
                        init.boundary_conditions[ii]
                    );
            }
        }

        void initialize_body_system(const InitialConditions& init)
        {
            if (init.immersed_bodies.empty() && !init.contains("body_system")) {
                return;
            }

            // body_system = ibsystem::create_body_system_from_config<real,
            // Dims>(
            //     mesh,
            //     init
            // );

            // if (body_system) {
            //     const auto [nia, nij, nka] = init.active_zones();

            //     collector = util::make_unique<
            //         ibsystem::GridBodyDeltaCollector<real, Dims>>(
            //         collapsable<Dims>{nia, nij, nij},
            //         2   // maximum number of bodies
            //     );
            // }
        }

        static ShockWaveLimiter
        get_shock_smoother(const InitialConditions& init)
        {
            return init.fleischmann_limiter
                       ? ShockWaveLimiter::FLEISCHMANN
                       : (init.quirk_smoothing ? ShockWaveLimiter::QUIRK
                                               : ShockWaveLimiter::NONE);
        }
    };
}   // namespace simbi::state
#endif
