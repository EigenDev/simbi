// =============================================================================
// blueprints.hpp
//
// configuration containers for simulation construction.
// each blueprint captures a specific aspect of the simulation setup.
// blueprints are extracted from config_dict_t via blueprint_extractor_t,
// then passed to simulation_builder_t to construct the simulation.
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "utility/config_dict.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace simbi::ecs {

    

    // -----------------------------------------------------------------------------
    // mesh_blueprint_t
    //
    // topological and geometric configuration for the computational domain.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct mesh_blueprint_t
    {
        // grid resolution (active cells, excluding ghosts)
        iarray<Rank> active_resolution;

        // physical domain bounds per dimension
        // bounds[d] = {min, max} for dimension d
        // ordered as: [x3, x2, x1]
        std::vector<std::pair<real, real>> bounds;

        // boundary condition types per face
        // ordered as: [x3_left, x3_right, x2_left, x2_right, ...]
        std::vector<std::string> boundary_conditions;

        // coordinate system type
        std::string coord_system;

        // cell spacing type per dimension ("linear", "log")
        // ordered as: [x3, x2, x1]
        std::vector<std::string> spacing;

        // halo width
        std::uint64_t halo_width{0};

        // mesh motion flags
        bool moving_mesh{false};
        bool homologous_expansion{false};
    };

    // -----------------------------------------------------------------------------
    // physics_blueprint_t
    //
    // physics equations and numerical method configuration.
    // -----------------------------------------------------------------------------
    struct physics_blueprint_t
    {
        // physics regime
        regime_t regime;

        // numerical methods (stored as strings for dispatch)
        std::string solver;
        std::string reconstruction;
        std::string timestepping;

        // equation of state parameters
        real gamma;
        real cfl;
        real plm_theta;
        real viscosity;

        // derived flags (set by extractor based on regime)
        bool is_mhd;
        bool is_relativistic;

        // isothermal eos
        bool isothermal;
        real ambient_sound_speed;

        // disk physics
        real shakura_sunyaev_alpha;
    };

    // -----------------------------------------------------------------------------
    // numerics_blueprint_t
    //
    // numerical stability and limiting options.
    // -----------------------------------------------------------------------------
    struct numerics_blueprint_t
    {
        bool use_quirk_smoothing;
        bool use_fleischmann_limiter;
    };

    // -----------------------------------------------------------------------------
    // execution_blueprint_t
    //
    // time integration and output configuration.
    // -----------------------------------------------------------------------------
    struct execution_blueprint_t
    {
        // time bounds
        real start_time;
        real end_time;

        // checkpointing
        real          checkpoint_interval;
        real          dlogt; // logarithmic output spacing (0 = linear)
        std::uint64_t checkpoint_zones;

        // output paths
        std::string   data_directory;
        std::uint64_t start_index;
        std::string   restart_file;
    };

    // -----------------------------------------------------------------------------
    // amr_blueprint_t
    //
    // adaptive/static mesh refinement configuration.
    // -----------------------------------------------------------------------------
    struct amr_blueprint_t
    {
        bool          enabled;
        std::uint64_t max_levels;

        // refinement ratio per level transition
        // refinement_ratios[i] = ratio from level i to level i+1
        std::vector<std::uint64_t> refinement_ratios;

        // static refinement regions (physical coordinates)
        // each inner vector: [x1_min, x1_max, x2_min, x2_max, ...]
        std::vector<std::vector<real>> static_refinement_regions;

        // subcycling configuration
        subcycling_mode_t          subcycling_mode;
        std::vector<std::uint64_t> manual_substeps;
    };

    // -----------------------------------------------------------------------------
    // expressions_blueprint_t
    //
    // user-defined source terms and boundary expressions.
    // stored as raw config_dict_t for deferred compilation.
    // -----------------------------------------------------------------------------
    struct expressions_blueprint_t
    {
        config_dict_t hydro_source;
        config_dict_t gravity_source;

        // boundary injection sources
        // indexed as [2*dim + side] where side: 0=inner, 1=outer
        std::vector<config_dict_t> boundary_sources;
    };

    // -----------------------------------------------------------------------------
    // bodies_blueprint_t
    //
    // immersed boundary object configuration for individual bodies.
    // -----------------------------------------------------------------------------
    struct bodies_blueprint_t
    {
        std::vector<config_dict_t> body_configs;
    };

    // -----------------------------------------------------------------------------
    // binary_system_blueprint_t
    //
    // configuration for binary orbital systems.
    // -----------------------------------------------------------------------------
    struct binary_system_blueprint_t
    {
        // orbital parameters
        real semi_major;
        real eccentricity;
        real mass_ratio;
        real total_mass;
        real orbital_period;

        // dynamics control
        bool        prescribed_motion;
        bool        is_circular_orbit;
        std::string reference_frame;

        // component configurations (exactly 2)
        std::vector<config_dict_t> components;
    };

    // -----------------------------------------------------------------------------
    // gravitational_system_blueprint_t
    //
    // configuration for gravitational systems (binary, triple, n-body).
    // separates system-level dynamics from individual body properties.
    // -----------------------------------------------------------------------------
    struct gravitational_system_blueprint_t
    {
        // system type: "binary", "triple", "nbody"
        std::string system_type;

        // type-specific configuration
        std::optional<binary_system_blueprint_t> binary;
        // future: std::optional<triple_system_blueprint_t> triple;
        // future: std::optional<nbody_system_blueprint_t> nbody;
    };

    // -----------------------------------------------------------------------------
    // decomposition_blueprint_t
    //
    // multi-device domain decomposition configuration.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct decomposition_blueprint_t
    {
        // process topology (e.g., {2, 2, 1} for 4 devices)
        iarray<Rank> topology_dims;

        // mpi rank for distributed runs
        std::int32_t mpi_rank{0};

        // device ids for local partitions
        std::vector<int> device_ids;

        // ghost zone width
        std::int64_t halo_width;
    };

    
    template <std::uint64_t Rank>
    struct blueprint_set_t
    {
        mesh_blueprint_t<Rank>  mesh;
        physics_blueprint_t     physics;
        execution_blueprint_t   execution;
        amr_blueprint_t         amr;
        numerics_blueprint_t    numerics;
        expressions_blueprint_t expressions;
        bodies_blueprint_t      bodies;

        // optional gravitational system (nullopt = individual bodies only)
        std::optional<gravitational_system_blueprint_t> gravitational_system;

        // optional multi-gpu config (nullopt = single device)
        std::optional<decomposition_blueprint_t<Rank>> decomposition;

        // -------------------------------------------------------------------------
        // factory from config_dict_t
        //
        // extracts all blueprints from a raw config dictionary.
        // this is the main entry point from the python binding.
        // -------------------------------------------------------------------------
        static blueprint_set_t from_config(const config_dict_t& config);
    };

} // namespace simbi::ecs
