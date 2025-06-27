#ifndef SIMBI_HYDRO_DISPATCH_HPP
#define SIMBI_HYDRO_DISPATCH_HPP

#include "core/utility/bimap.hpp"
#include "core/utility/enums.hpp"
#include "core/utility/init_conditions.hpp"
#include "data/containers/vector.hpp"
#include "data/state/hydro_state.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace simbi::dispatch {

    //==============================================================================
    // SINGLE SOURCE OF TRUTH - All valid combinations defined here!
    //
    // To add new combinations: just add a line below!
    // To remove combinations: just delete/comment the line!
    // Everything else (variant type + dispatch table) auto-generates!
    //==============================================================================

#define VALID_HYDRO_COMBINATIONS(MACRO)                                        \
    /* newtonian combinations */                                               \
    MACRO(NEWTONIAN, 1, CARTESIAN, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 2, CARTESIAN, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 3, CARTESIAN, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 1, CARTESIAN, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 2, CARTESIAN, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 3, CARTESIAN, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 1, CARTESIAN, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 2, CARTESIAN, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 3, CARTESIAN, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 1, CARTESIAN, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 2, CARTESIAN, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 3, CARTESIAN, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 1, SPHERICAL, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 2, SPHERICAL, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 3, SPHERICAL, HLLC, PLM)                                  \
    MACRO(NEWTONIAN, 1, SPHERICAL, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 2, SPHERICAL, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 3, SPHERICAL, HLLE, PLM)                                  \
    MACRO(NEWTONIAN, 1, SPHERICAL, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 2, SPHERICAL, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 3, SPHERICAL, HLLE, PCM)                                  \
    MACRO(NEWTONIAN, 1, SPHERICAL, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 2, SPHERICAL, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 3, SPHERICAL, HLLC, PCM)                                  \
    MACRO(NEWTONIAN, 1, CYLINDRICAL, HLLC, PLM)                                \
    MACRO(NEWTONIAN, 2, CYLINDRICAL, HLLC, PLM)                                \
    MACRO(NEWTONIAN, 3, CYLINDRICAL, HLLC, PLM)                                \
    MACRO(NEWTONIAN, 1, CYLINDRICAL, HLLE, PLM)                                \
    MACRO(NEWTONIAN, 2, CYLINDRICAL, HLLE, PLM)                                \
    MACRO(NEWTONIAN, 3, CYLINDRICAL, HLLE, PLM)                                \
    MACRO(NEWTONIAN, 1, CYLINDRICAL, HLLE, PCM)                                \
    MACRO(NEWTONIAN, 2, CYLINDRICAL, HLLE, PCM)                                \
    MACRO(NEWTONIAN, 3, CYLINDRICAL, HLLE, PCM)                                \
    MACRO(NEWTONIAN, 1, CYLINDRICAL, HLLC, PCM)                                \
    MACRO(NEWTONIAN, 2, CYLINDRICAL, HLLC, PCM)                                \
    MACRO(NEWTONIAN, 3, CYLINDRICAL, HLLC, PCM)                                \
    MACRO(NEWTONIAN, 1, AXIS_CYLINDRICAL, HLLC, PLM)                           \
    MACRO(NEWTONIAN, 2, AXIS_CYLINDRICAL, HLLC, PLM)                           \
    MACRO(NEWTONIAN, 3, AXIS_CYLINDRICAL, HLLC, PLM)                           \
    MACRO(NEWTONIAN, 1, AXIS_CYLINDRICAL, HLLE, PLM)                           \
    MACRO(NEWTONIAN, 2, AXIS_CYLINDRICAL, HLLE, PLM)                           \
    MACRO(NEWTONIAN, 3, AXIS_CYLINDRICAL, HLLE, PLM)                           \
    MACRO(NEWTONIAN, 1, AXIS_CYLINDRICAL, HLLE, PCM)                           \
    MACRO(NEWTONIAN, 2, AXIS_CYLINDRICAL, HLLE, PCM)                           \
    MACRO(NEWTONIAN, 3, AXIS_CYLINDRICAL, HLLE, PCM)                           \
    MACRO(NEWTONIAN, 1, AXIS_CYLINDRICAL, HLLC, PCM)                           \
    MACRO(NEWTONIAN, 2, AXIS_CYLINDRICAL, HLLC, PCM)                           \
    MACRO(NEWTONIAN, 3, AXIS_CYLINDRICAL, HLLC, PCM)                           \
    MACRO(NEWTONIAN, 1, PLANAR_CYLINDRICAL, HLLC, PLM)                         \
    MACRO(NEWTONIAN, 2, PLANAR_CYLINDRICAL, HLLC, PLM)                         \
    MACRO(NEWTONIAN, 3, PLANAR_CYLINDRICAL, HLLC, PLM)                         \
    MACRO(NEWTONIAN, 1, PLANAR_CYLINDRICAL, HLLE, PLM)                         \
    MACRO(NEWTONIAN, 2, PLANAR_CYLINDRICAL, HLLE, PLM)                         \
    MACRO(NEWTONIAN, 3, PLANAR_CYLINDRICAL, HLLE, PLM)                         \
    MACRO(NEWTONIAN, 1, PLANAR_CYLINDRICAL, HLLE, PCM)                         \
    MACRO(NEWTONIAN, 2, PLANAR_CYLINDRICAL, HLLE, PCM)                         \
    MACRO(NEWTONIAN, 3, PLANAR_CYLINDRICAL, HLLE, PCM)                         \
    MACRO(NEWTONIAN, 1, PLANAR_CYLINDRICAL, HLLC, PCM)                         \
    MACRO(NEWTONIAN, 2, PLANAR_CYLINDRICAL, HLLC, PCM)                         \
    MACRO(NEWTONIAN, 3, PLANAR_CYLINDRICAL, HLLC, PCM)                         \
    MACRO(SRHD, 1, CARTESIAN, HLLC, PLM)                                       \
    MACRO(SRHD, 2, CARTESIAN, HLLC, PLM)                                       \
    MACRO(SRHD, 3, CARTESIAN, HLLC, PLM)                                       \
    MACRO(SRHD, 1, CARTESIAN, HLLE, PLM)                                       \
    MACRO(SRHD, 2, CARTESIAN, HLLE, PLM)                                       \
    MACRO(SRHD, 3, CARTESIAN, HLLE, PLM)                                       \
    MACRO(SRHD, 1, CARTESIAN, HLLE, PCM)                                       \
    MACRO(SRHD, 2, CARTESIAN, HLLE, PCM)                                       \
    MACRO(SRHD, 3, CARTESIAN, HLLE, PCM)                                       \
    MACRO(SRHD, 1, CARTESIAN, HLLC, PCM)                                       \
    MACRO(SRHD, 2, CARTESIAN, HLLC, PCM)                                       \
    MACRO(SRHD, 3, CARTESIAN, HLLC, PCM)                                       \
    MACRO(SRHD, 1, SPHERICAL, HLLC, PLM)                                       \
    MACRO(SRHD, 2, SPHERICAL, HLLC, PLM)                                       \
    MACRO(SRHD, 3, SPHERICAL, HLLC, PLM)                                       \
    MACRO(SRHD, 1, SPHERICAL, HLLE, PLM)                                       \
    MACRO(SRHD, 2, SPHERICAL, HLLE, PLM)                                       \
    MACRO(SRHD, 3, SPHERICAL, HLLE, PLM)                                       \
    MACRO(SRHD, 1, SPHERICAL, HLLE, PCM)                                       \
    MACRO(SRHD, 2, SPHERICAL, HLLE, PCM)                                       \
    MACRO(SRHD, 3, SPHERICAL, HLLE, PCM)                                       \
    MACRO(SRHD, 1, SPHERICAL, HLLC, PCM)                                       \
    MACRO(SRHD, 2, SPHERICAL, HLLC, PCM)                                       \
    MACRO(SRHD, 3, SPHERICAL, HLLC, PCM)                                       \
    MACRO(SRHD, 1, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(SRHD, 2, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(SRHD, 3, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(SRHD, 1, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(SRHD, 2, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(SRHD, 3, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(SRHD, 1, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(SRHD, 2, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(SRHD, 3, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(SRHD, 1, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(SRHD, 2, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(SRHD, 3, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(SRHD, 1, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(SRHD, 2, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(SRHD, 3, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(SRHD, 1, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(SRHD, 2, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(SRHD, 3, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(SRHD, 1, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(SRHD, 2, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(SRHD, 3, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(SRHD, 1, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(SRHD, 2, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(SRHD, 3, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(SRHD, 1, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(SRHD, 2, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(SRHD, 3, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(SRHD, 1, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(SRHD, 2, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(SRHD, 3, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(SRHD, 1, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(SRHD, 2, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(SRHD, 3, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(SRHD, 1, PLANAR_CYLINDRICAL, HLLC, PCM)                              \
    MACRO(SRHD, 2, PLANAR_CYLINDRICAL, HLLC, PCM)                              \
    MACRO(SRHD, 3, PLANAR_CYLINDRICAL, HLLC, PCM)                              \
    MACRO(RMHD, 1, CARTESIAN, HLLC, PLM)                                       \
    MACRO(RMHD, 2, CARTESIAN, HLLC, PLM)                                       \
    MACRO(RMHD, 3, CARTESIAN, HLLC, PLM)                                       \
    MACRO(RMHD, 1, CARTESIAN, HLLE, PLM)                                       \
    MACRO(RMHD, 2, CARTESIAN, HLLE, PLM)                                       \
    MACRO(RMHD, 3, CARTESIAN, HLLE, PLM)                                       \
    MACRO(RMHD, 1, CARTESIAN, HLLE, PCM)                                       \
    MACRO(RMHD, 2, CARTESIAN, HLLE, PCM)                                       \
    MACRO(RMHD, 3, CARTESIAN, HLLE, PCM)                                       \
    MACRO(RMHD, 1, CARTESIAN, HLLC, PCM)                                       \
    MACRO(RMHD, 2, CARTESIAN, HLLC, PCM)                                       \
    MACRO(RMHD, 3, CARTESIAN, HLLC, PCM)                                       \
    MACRO(RMHD, 1, SPHERICAL, HLLC, PLM)                                       \
    MACRO(RMHD, 2, SPHERICAL, HLLC, PLM)                                       \
    MACRO(RMHD, 3, SPHERICAL, HLLC, PLM)                                       \
    MACRO(RMHD, 1, SPHERICAL, HLLE, PLM)                                       \
    MACRO(RMHD, 2, SPHERICAL, HLLE, PLM)                                       \
    MACRO(RMHD, 3, SPHERICAL, HLLE, PLM)                                       \
    MACRO(RMHD, 1, SPHERICAL, HLLE, PCM)                                       \
    MACRO(RMHD, 2, SPHERICAL, HLLE, PCM)                                       \
    MACRO(RMHD, 3, SPHERICAL, HLLE, PCM)                                       \
    MACRO(RMHD, 1, SPHERICAL, HLLC, PCM)                                       \
    MACRO(RMHD, 2, SPHERICAL, HLLC, PCM)                                       \
    MACRO(RMHD, 3, SPHERICAL, HLLC, PCM)                                       \
    MACRO(RMHD, 1, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(RMHD, 2, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(RMHD, 3, CYLINDRICAL, HLLC, PLM)                                     \
    MACRO(RMHD, 1, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(RMHD, 2, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(RMHD, 3, CYLINDRICAL, HLLE, PLM)                                     \
    MACRO(RMHD, 1, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(RMHD, 2, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(RMHD, 3, CYLINDRICAL, HLLE, PCM)                                     \
    MACRO(RMHD, 1, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(RMHD, 2, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(RMHD, 3, CYLINDRICAL, HLLC, PCM)                                     \
    MACRO(RMHD, 1, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(RMHD, 2, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(RMHD, 3, AXIS_CYLINDRICAL, HLLC, PLM)                                \
    MACRO(RMHD, 1, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(RMHD, 2, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(RMHD, 3, AXIS_CYLINDRICAL, HLLE, PLM)                                \
    MACRO(RMHD, 1, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(RMHD, 2, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(RMHD, 3, AXIS_CYLINDRICAL, HLLE, PCM)                                \
    MACRO(RMHD, 1, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(RMHD, 2, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(RMHD, 3, AXIS_CYLINDRICAL, HLLC, PCM)                                \
    MACRO(RMHD, 1, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(RMHD, 2, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(RMHD, 3, PLANAR_CYLINDRICAL, HLLC, PLM)                              \
    MACRO(RMHD, 1, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(RMHD, 2, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(RMHD, 3, PLANAR_CYLINDRICAL, HLLE, PLM)                              \
    MACRO(RMHD, 1, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(RMHD, 2, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(RMHD, 3, PLANAR_CYLINDRICAL, HLLE, PCM)                              \
    MACRO(RMHD, 1, PLANAR_CYLINDRICAL, HLLC, PCM)                              \
    MACRO(RMHD, 2, PLANAR_CYLINDRICAL, HLLC, PCM)                              \
    MACRO(RMHD, 3, PLANAR_CYLINDRICAL, HLLC, PCM)

//==============================================================================
// AUTO-GENERATED VARIANT TYPE - Never touch this!
//==============================================================================

// macro to generate variant type entries
#define MAKE_VARIANT_ENTRY(REGIME, DIMS, GEOMETRY, SOLVER, RECONSTRUCTION)     \
    state::hydro_state_t<                                                      \
        Regime::REGIME,                                                        \
        DIMS,                                                                  \
        Geometry::GEOMETRY,                                                    \
        Solver::SOLVER,                                                        \
        Reconstruction::RECONSTRUCTION>,

    // auto-generated variant containing ALL valid combinations
    using hydro_state_variant_t =
        std::variant<VALID_HYDRO_COMBINATIONS(MAKE_VARIANT_ENTRY)
                         std::monostate   // fallback to prevent empty variant
                     >;

#undef MAKE_VARIANT_ENTRY

    //==============================================================================
    // DISPATCH INFRASTRUCTURE
    //==============================================================================

    struct config_key_t {
        Regime regime;
        std::uint64_t dims;
        Geometry geometry;
        Solver solver;
        Reconstruction reconstruction;

        auto operator<=>(const config_key_t&) const = default;
        bool operator==(const config_key_t&) const  = default;
    };

    using factory_fn_t = std::function<hydro_state_variant_t(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        const InitialConditions& init
    )>;

    struct dispatch_entry_t {
        config_key_t key;
        factory_fn_t factory;
    };

    class unsupported_configuration : public std::runtime_error
    {
      public:
        unsupported_configuration(const std::string& msg)
            : std::runtime_error("unsupported hydro configuration: " + msg)
        {
        }
    };

    //==============================================================================
    // AUTO-GENERATED DISPATCH TABLE - Never touch this!
    //==============================================================================

    namespace detail {

        // helper to create factory function for specific template parameters
        template <
            Regime R,
            std::uint64_t D,
            Geometry G,
            Solver S,
            Reconstruction Rec>
        constexpr auto make_factory_function()
        {
            return [](void* cons_data,
                      void* prim_data,
                      vector_t<void*, 3> bfield_data,
                      const InitialConditions& init) -> hydro_state_variant_t {
                auto state = state::hydro_state_t<R, D, G, S, Rec>::from_init(
                    cons_data,
                    prim_data,
                    bfield_data,
                    init
                );
                return hydro_state_variant_t{std::move(state)};
            };
        }

// macro to generate dispatch table entries
#define MAKE_DISPATCH_ENTRY(REGIME, DIMS, GEOMETRY, SOLVER, RECONSTRUCTION)    \
    dispatch_entry_t{                                                          \
      .key =                                                                   \
          {Regime::REGIME,                                                     \
           DIMS,                                                               \
           Geometry::GEOMETRY,                                                 \
           Solver::SOLVER,                                                     \
           Reconstruction::RECONSTRUCTION},                                    \
      .factory = make_factory_function<                                        \
          Regime::REGIME,                                                      \
          DIMS,                                                                \
          Geometry::GEOMETRY,                                                  \
          Solver::SOLVER,                                                      \
          Reconstruction::RECONSTRUCTION>()                                    \
    },

        // compile-time dispatch table (auto-generated from same source as
        // variant!)
        inline const auto& get_dispatch_table()
        {
            static const auto table =
                std::array{VALID_HYDRO_COMBINATIONS(MAKE_DISPATCH_ENTRY)};
            return table;
        }

#undef MAKE_DISPATCH_ENTRY

    }   // namespace detail

    //==============================================================================
    // MAIN DISPATCH FUNCTION - Runtime → Compile-time magic!
    //==============================================================================

    /**
     * convert runtime parameters to fully-typed hydro state
     *
     * after this call, python is forgotten and templates take over with zero
     * overhead!
     */
    inline auto create_hydro_state(
        void* cons_data,
        void* prim_data,
        vector_t<void*, 3> bfield_data,
        const InitialConditions& init
    ) -> hydro_state_variant_t
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

        // create lookup key
        config_key_t key{regime, dims, geometry, solver, reconstruction};

        // find matching factory in auto-generated dispatch table
        const auto& table = detail::get_dispatch_table();
        auto it =
            std::find_if(table.begin(), table.end(), [&key](const auto& entry) {
                return entry.key == key;
            });

        if (it == table.end()) {
            // configuration not found - generate helpful error message
            std::string msg =
                "regime=" + regime_str + ", dims=" + std::to_string(dims) +
                ", geometry=" + geometry_str + ", solver=" + solver_str +
                ", reconstruction=" + reconstruction_str;
            throw unsupported_configuration(msg);
        }

        // call the auto-generated factory function for this specific
        // configuration
        return it->factory(cons_data, prim_data, bfield_data, init);
    }

    //==============================================================================
    // CONVENIENCE UTILITIES
    //==============================================================================

    // get list of all supported configurations (useful for validation/testing)
    inline auto get_supported_configurations()
    {
        const auto& table = detail::get_dispatch_table();
        std::vector<config_key_t> configs;
        configs.reserve(table.size());

        for (const auto& entry : table) {
            configs.push_back(entry.key);
        }

        return configs;
    }

    // check if a configuration is supported without creating state
    inline bool is_configuration_supported(
        const std::string& regime_str,
        std::uint64_t dims,
        const std::string& geometry_str,
        const std::string& solver_str,
        const std::string& reconstruction_str
    )
    {
        try {
            auto regime   = deserialize<Regime>(regime_str);
            auto geometry = deserialize<Geometry>(geometry_str);
            auto solver   = deserialize<Solver>(solver_str);
            auto reconstruction =
                deserialize<Reconstruction>(reconstruction_str);

            config_key_t key{regime, dims, geometry, solver, reconstruction};

            const auto& table = detail::get_dispatch_table();
            return std::find_if(
                       table.begin(),
                       table.end(),
                       [&key](const auto& entry) { return entry.key == key; }
                   ) != table.end();
        }
        catch (...) {
            return false;
        }
    }

    // utility to print all supported combinations (useful for debugging)
    inline void print_supported_configurations()
    {
        const auto configs = get_supported_configurations();
        std::cout << "supported hydro configurations (" << configs.size()
                  << " total):\n";
        for (const auto& config : configs) {
            std::cout << "  regime=" << static_cast<int>(config.regime)
                      << ", dims=" << config.dims
                      << ", geometry=" << static_cast<int>(config.geometry)
                      << ", solver=" << static_cast<int>(config.solver)
                      << ", reconstruction="
                      << static_cast<int>(config.reconstruction) << "\n";
        }
    }

    //==============================================================================
    // INTEGRATION EXAMPLE - How to use this system
    //==============================================================================

    /*
    // python interface usage:
    auto state = create_hydro_state("srhd", 3, "cartesian", "hllc", "plm",
                                    cons_data, prim_data, bfield_data, init);

    // simulation loop with zero overhead after dispatch:
    for (int step = 0; step < nsteps; ++step) {
        apply_boundary_conditions_variant(state);  // templated function call
        compute_fluxes_variant(state);              // templated function call
        update_state_variant(state);                // templated function call
        cons2prim_variant(state);                   // templated function call
        auto dt = compute_timestep_variant(state);  // templated function call
    }

    // to add new combinations: just edit VALID_HYDRO_COMBINATIONS macro above!
    // everything else (variant type + dispatch) auto-regenerates!
    */

}   // namespace simbi::dispatch

#endif   // SIMBI_HYDRO_DISPATCH_HPP
