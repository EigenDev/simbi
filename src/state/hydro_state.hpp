#ifndef SIMBI_STATE_HYDRO_STATE_HPP
#define SIMBI_STATE_HYDRO_STATE_HPP

#include "compute/field.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "core/utility/managed.hpp"
#include "hydro_state_types.hpp"
#include "physics/eos/isothermal.hpp"
#include "physics/hydro/ib/collection.hpp"
#include "physics/hydro/ib/factory.hpp"
#include "state/express_t.hpp"
#include "system/io/exceptions.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simbi::state {
    using namespace base;
    using namespace body::factory;
    // using namespace ibsystem;

    /**
     * modern implementation of hydro_state_t using mesh abstractions
     * for cell-face relationships and zero-copy with numpy arrays
     */
    template <
        Regime R,
        std::uint64_t Dims,
        typename EoS = eos::isothermal_gas_eos_t>
    struct hydro_state_t : public managed_t<global::managed_memory> {
        // type definitions
        using conserved_t = typename vtraits<R, Dims, EoS>::conserved_type;
        using primitive_t = typename vtraits<R, Dims, EoS>::primitive_type;
        using eos_t       = EoS;

        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime_t          = R;
        static constexpr bool is_mhd = (R == Regime::MHD || R == Regime::RMHD);
        static constexpr auto nvars  = (is_mhd) ? 9 : Dims + 3;

        // cell-centered data
        field_t<conserved_t, Dims> cons;
        field_t<primitive_t, Dims> prim;

        // face-centered fluxes
        vector_t<field_t<conserved_t, Dims>, Dims> flux;

        // optional magnetic fields for mhd (face-centered)
        vector_t<field_t<real, Dims>, Dims> bstaggs;

        // simulation metadata
        struct meta_data_t {
            // numerics
            real gamma;
            real plm_theta;
            real cfl;
            real time;
            real tend;
            real dt;

            // int tracking
            std::uint64_t iteration;
            std::uint64_t halo_radius;
            std::uint64_t dimensions{Dims};

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
            iarray<3> resolution;

            // flags
            bool is_mhd;
            bool is_relativistic;

            // strings
            std::string data_dir;
        } metadata;

        struct sources_t {
            expression_t<Dims> hydro_source;
            expression_t<Dims> gravity_source;
            vector_t<expression_t<Dims>, 2 * Dims> bc_sources;
        } sources;

        // immersed body stuff
        std::optional<body::body_collection_t<Dims>> bodies;

        // error handling
        bool in_failure_state;

        /**
         * create hydro_state from init conditions and numpy arrays with
         * zero-copy
         */
        static hydro_state_t from_init(
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bfield_data,
            const initial_conditions_t& init
        )
        {

            auto [cons, prims, flux_vec, bstaggs] =
                setup_hydro_state(cons_data, prim_data, bfield_data, init);

            auto bodies = create_body_collection_from_init<Dims>(init);

            return hydro_state_t{
              .cons             = std::move(cons),
              .prim             = std::move(prims),
              .flux             = {std::move(flux_vec)},
              .bstaggs          = {std::move(bstaggs)},
              .metadata         = setup_metadata(init),
              .sources          = setup_sources(init),
              .bodies           = std::move(bodies),
              .in_failure_state = false,
            };
        }

      private:
        static auto setup_hydro_state(
            void* cons_data,
            void* prim_data,
            vector_t<void*, 3> bstaggs,
            const initial_conditions_t& init
        )
        {
            const auto full_shape = init.get_full_shape<dimensions>();
            auto cons             = from_numpy_field(
                std::bit_cast<conserved_t*>(cons_data),
                full_shape
            );

            auto prims = from_numpy_field(
                std::bit_cast<primitive_t*>(prim_data),
                full_shape
            );

            vector_t<field_t<real, Dims>, Dims> bstaggs_vec;
            vector_t<field_t<conserved_t, Dims>, Dims> flux_vec;

            // staggered fluxes and fields
            const auto active_sizes = init.get_active_shape<dimensions>();
            for (std::uint64_t dir = 0; dir < Dims; ++dir) {
                const auto mhd_b = 2 * init.is_mhd;

                // create staggered shape: add 1 to the direction we're
                // staggering in
                iarray<Dims> staggered_shape = active_sizes;
                staggered_shape[dir] += 1;

                // add MHD offset to other dimensions
                for (std::uint64_t d = 0; d < Dims; ++d) {
                    if (d != dir) {
                        staggered_shape[d] += mhd_b;
                    }
                }

                auto flux_field = field(make_domain(staggered_shape), [](auto) {
                    return conserved_t{};
                });
                flux_vec[dir]   = flux_field;

                if constexpr (is_mhd) {
                    bstaggs_vec[dir] = from_numpy_field(
                        std::bit_cast<real*>(bstaggs[dir]),
                        staggered_shape
                    );
                }
            }

            return std::make_tuple(
                std::move(cons),
                std::move(prims),
                std::move(flux_vec),
                std::move(bstaggs_vec)
            );
        }
        /**
         * set up metadata from init conditions
         */
        static auto setup_metadata(const initial_conditions_t& init)
        {
            meta_data_t metadata = {
              .gamma          = init.gamma,
              .plm_theta      = init.plm_theta,
              .cfl            = init.cfl,
              .time           = init.time,
              .tend           = init.tend,
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
              .resolution          = {init.nz, init.ny, init.nx},
              .is_mhd              = init.is_mhd,
              .is_relativistic     = init.is_relativistic,
              .data_dir            = init.data_directory
            };

            // set boundary conditions from init
            for (std::uint64_t ii = 0; ii < 2 * Dims; ++ii) {
                auto logical_dim = ii / 2;   // which dimension (x=0, y=1, z=2)
                auto side        = ii % 2;   // which side (inner=0, outer=1)
                // map to array order
                auto array_dim = (Dims - 1) - logical_dim;
                // convert back to flat index
                auto array_index = array_dim * 2 + side;
                metadata.boundary_conditions[array_index] =
                    deserialize<BoundaryCondition>(
                        init.boundary_conditions[ii]
                    );
            }

            return metadata;
        }

        /**
         * set up sources from init conditions
         */
        static auto setup_sources(const initial_conditions_t& init)
        {
            auto hydro =
                expression_t<Dims>::from_config(init.hydro_source_expressions);

            auto grav = expression_t<Dims>::from_config(
                init.gravity_source_expressions
            );

            vector_t<expression_t<Dims>, 2 * Dims> bc_sources;

            // set up boundary condition sources
            bc_sources[0] =
                expression_t<Dims>::from_config(init.bx1_inner_expressions);
            bc_sources[1] =
                expression_t<Dims>::from_config(init.bx1_outer_expressions);

            if constexpr (Dims >= 2) {
                bc_sources[2] =
                    expression_t<Dims>::from_config(init.bx2_inner_expressions);
                bc_sources[3] =
                    expression_t<Dims>::from_config(init.bx2_outer_expressions);
            }

            if constexpr (Dims >= 3) {
                bc_sources[4] =
                    expression_t<Dims>::from_config(init.bx3_inner_expressions);
                bc_sources[5] =
                    expression_t<Dims>::from_config(init.bx3_outer_expressions);
            }

            return sources_t{
              .hydro_source   = std::move(hydro),
              .gravity_source = std::move(grav),
              .bc_sources     = std::move(bc_sources)
            };
        }

        /**
         * get shock smoother type from init conditions
         */
        static ShockWaveLimiter
        get_shock_smoother(const initial_conditions_t& init)
        {
            return init.fleischmann_limiter
                       ? ShockWaveLimiter::FLEISCHMANN
                       : (init.quirk_smoothing ? ShockWaveLimiter::QUIRK
                                               : ShockWaveLimiter::NONE);
        }
    };
}   // namespace simbi::state

#endif   // SIMBI_STATE_HYDRO_STATE_HPP
