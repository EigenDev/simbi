#ifndef SIMBI_STATE_HYDRO_STATE_HPP
#define SIMBI_STATE_HYDRO_STATE_HPP

#include "compute/math/field.hpp"
#include "config.hpp"
#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "data/state/express_t.hpp"
#include "hydro_state_types.hpp"
#include "physics/eos/ideal.hpp"
#include "physics/hydro/ib/collector.hpp"
#include "physics/hydro/ib/component_body_system.hpp"
#include "physics/hydro/ib/component_generator.hpp"
#include "system/mesh/mesh_config.hpp"
#include "system/mesh/solver.hpp"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace simbi::state {
    using namespace base;
    using namespace mesh;
    using namespace ibsystem;

    /**
     * modern implementation of hydro_state_t using mesh abstractions
     * for cell-face relationships and zero-copy with numpy arrays
     */
    template <
        Regime R,
        std::uint64_t Dims,
        Geometry G,
        Solver S,
        Reconstruction Rec,
        typename EoS = eos::ideal_gas_eos_t<R>>
    struct hydro_state_t {
        // type definitions
        using conserved_t = typename hs_value_traits<R, Dims>::conserved_type;
        using primitive_t = typename hs_value_traits<R, Dims>::primitive_type;
        using geo_t       = geometry_solver_t<Dims, G>;
        using eos_t       = EoS;

        static constexpr std::uint64_t dimensions     = Dims;
        static constexpr Regime regime_t              = R;
        static constexpr Geometry geometry_t          = G;
        static constexpr Solver solver_t              = S;
        static constexpr Reconstruction reconstruct_t = Rec;
        static constexpr bool is_mhd = (R == Regime::MHD || R == Regime::RMHD);
        static constexpr auto nvars  = (is_mhd) ? 9 : Dims + 3;

        // cell-centered data
        field_t<conserved_t, Dims> cons;
        field_t<primitive_t, Dims> prim;

        // face-centered fluxes
        vector_t<field_t<conserved_t, Dims>, Dims> flux;

        // optional magnetic fields for mhd (face-centered)
        vector_t<field_t<real, Dims>, Dims> bstaggs;

        // geometric mesh config
        geo_t geom_solver;

        bool has_sources{false};
        std::unique_ptr<ComponentBodySystem<real, Dims>> body_system;
        std::unique_ptr<GridBodyDeltaCollector<real, Dims>> collector;

        // simulation metadata
        struct meta_data_t {
            // numerics
            real gamma;
            real plm_theta;
            real cfl;
            real time;
            real dt;

            // iteration tracking
            std::uint64_t iteration;
            std::uint64_t halo_radius;

            // simulation configuration
            Regime regime;
            ShockWaveLimiter shock_smoother;
            Solver solver;
            Cellspacing x1_spacing;
            Cellspacing x2_spacing;
            Cellspacing x3_spacing;
            Geometry coord_system;
            Reconstruction reconstruction;
            Timestepping timestepping;
            vector_t<BoundaryCondition, 2 * Dims> boundary_conditions;

            // flags
            bool is_mhd;
            bool is_relativistic;
        } metadata;

        struct sources_t {
            expression_t<Dims> hydro_source;
            expression_t<Dims> gravity_source;
            vector_t<expression_t<Dims>, 2 * Dims> bc_sources;
        } sources;

        // error handling
        bool in_failure_state{false};

        // default constructor
        hydro_state_t() = default;

        /**
         * create hydro_state from init conditions and numpy arrays with
         * zero-copy
         */
        static hydro_state_t from_init(
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const InitialConditions& init
        )
        {
            hydro_state_t state;

            // setup geometric config
            state.geom_solver = {
              .config = mesh_config_t<Dims>::from_init_conditions(init)
            };

            setup_hydro_state(cons_data, prim_data, bfield_data, state, init);

            // set up metadata from init
            setup_metadata(state, init);

            // set up sources from init
            setup_sources(state, init);

            return state;
        }

        /**
         * sync all memory to device for gpu execution
         */
        void to_gpu()
        {
            cons.to_gpu();
            prim.to_gpu();

            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                flux[dir].to_gpu();
            }

            if constexpr (is_mhd) {
                for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                    bstaggs[dir].to_gpu();
                }
            }
        }

        /**
         * sync all memory back to host
         */
        void to_cpu()
        {
            cons.to_cpu();
            prim.to_cpu();

            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                flux[dir].to_cpu();
            }

            if constexpr (is_mhd) {
                for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                    bstaggs[dir].to_cpu();
                }
            }
        }

      private:
        static void setup_hydro_state(
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bstaggs,
            hydro_state_t& state,
            const InitialConditions& init
        )
        {
            state.cons = field_t<conserved_t, Dims>::wrap_external(
                std::bit_cast<conserved_t*>(cons_data),
                {init.nz, init.ny, init.nx},
                false   // don't own memory
            );

            state.prim = field_t<primitive_t, Dims>::wrap_external(
                std::bit_cast<primitive_t*>(prim_data),
                {init.nz, init.ny, init.nx},
                false   // don't own memory
            );

            // staggered fluxes and fields
            const auto [nia, nja, nka] = init.active_zones();
            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                const auto mhd_b = 2 * init.is_mhd;
                const auto nivc  = nia + (dir == 0) + mhd_b * (dir != 0);
                const auto njvc  = nja + (dir == 1) + mhd_b * (dir != 1);
                const auto nkvc  = nka + (dir == 2) + mhd_b * (dir != 2);
                state.flux[dir] =
                    field_t<conserved_t, Dims>::zeros({nkvc, njvc, nivc});
                if constexpr (is_mhd) {
                    state.bstaggs[dir] = field_t<real, Dims>::wrap_external(
                        std::bit_cast<real*>(bstaggs[dir]),
                        {nkvc, njvc, nivc},
                        false   // don't own memory
                    );
                }
            }
        }
        /**
         * set up metadata from init conditions
         */
        static void
        setup_metadata(hydro_state_t& state, const InitialConditions& init)
        {
            state.metadata = {
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
              .boundary_conditions = vector_t<BoundaryCondition, 2 * Dims>{},
              .is_mhd              = init.is_mhd,
              .is_relativistic     = init.is_relativistic
            };

            // set boundary conditions from init
            for (std::uint64_t ii = 0; ii < 2 * Dims; ++ii) {
                state.metadata.boundary_conditions[ii] =
                    deserialize<BoundaryCondition>(
                        init.boundary_conditions[ii]
                    );
            }
        }

        /**
         * set up sources from init conditions
         */
        static void
        setup_sources(hydro_state_t& state, const InitialConditions& init)
        {
            state.sources.hydro_source =
                expression_t<Dims>::from_config(init.hydro_source_expressions);

            state.sources.gravity_source = expression_t<Dims>::from_config(
                init.gravity_source_expressions
            );

            // set up boundary condition sources
            state.sources.bc_sources[0] =
                expression_t<Dims>::from_config(init.bx1_inner_expressions);
            state.sources.bc_sources[1] =
                expression_t<Dims>::from_config(init.bx1_outer_expressions);

            if constexpr (Dims >= 2) {
                state.sources.bc_sources[2] =
                    expression_t<Dims>::from_config(init.bx2_inner_expressions);
                state.sources.bc_sources[3] =
                    expression_t<Dims>::from_config(init.bx2_outer_expressions);
            }

            if constexpr (Dims >= 3) {
                state.sources.bc_sources[4] =
                    expression_t<Dims>::from_config(init.bx3_inner_expressions);
                state.sources.bc_sources[5] =
                    expression_t<Dims>::from_config(init.bx3_outer_expressions);
            }

            // if any hydro source or gravity source is enabled,
            // set has_sources to true
            state.has_sources = state.sources.hydro_source.enabled ||
                                state.sources.gravity_source.enabled;
        }

        /*
         * Setup the body and collector system if provided
         */
        static void
        init_body_system(hydro_state_t& state, const InitialConditions& init)
        {
            state.body_system =
                ibsystem::create_body_system_from_config<real, Dims>(
                    state.geom_solver.config,
                    init
                );

            if (state.body_system) {
                const auto [nax, nay, naz] = init.active_zones();
                state.collector            = util::make_unique<
                               ibsystem::GridBodyDeltaCollector<real, Dims>>(
                    {naz, nay, nax},
                    2   // max bodies
                );
            }
        }

        /**
         * get shock smoother type from init conditions
         */
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

#endif   // SIMBI_STATE_HYDRO_STATE_HPP
