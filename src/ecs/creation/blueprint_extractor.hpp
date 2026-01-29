// =============================================================================
// blueprint_extractor.hpp
//
// stateless extraction of blueprints from config_dict_t.
// each extractor function:
//   1. reads relevant fields from config
//   2. applies defaults for missing fields
//   3. validates the extracted values
//   4. returns a fully-populated blueprint
//
// usage:
//   auto mesh_bp = blueprint_extractor_t<2>::mesh(config);
//   auto phys_bp = blueprint_extractor_t<2>::physics(config);
//   // ... or use blueprint_set_t::from_config(config) for all at once
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "ecs/blueprints.hpp"
#include "utility/bimap.hpp"
#include "utility/config_dict.hpp"
#include "utility/enums.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <list>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simbi::ecs::creation {

    inline void require(bool condition, const std::string& msg)
    {
        if (!condition) {
            throw std::runtime_error("blueprint validation failed: " + msg);
        }
    }

    template <typename T>
    T require_field(const config_dict_t& config, const std::string& key, const std::string& context)
    {
        auto maybe_val = config::try_read<T>(config, key);
        if (!maybe_val.has_value()) {
            throw std::runtime_error(context + ": required field '" + key + "' not found");
        }
        return *maybe_val;
    }

    template <typename T>
    T read_or_default(const config_dict_t& config, const std::string& key, T default_value)
    {
        return config::try_read<T>(config, key).unwrap_or(default_value);
    }

    template <std::uint64_t Rank>
    struct blueprint_extractor_t
    {

        // -------------------------------------------------------------------------
        // mesh
        // -------------------------------------------------------------------------
        static mesh_blueprint_t<Rank> mesh(const config_dict_t& config)
        {
            mesh_blueprint_t<Rank> bp;

            // resolution (required)
            auto resolution =
                require_field<std::vector<std::int64_t>>(config, "resolution", "mesh");
            // resolution is in nx, ny, nz order
            auto nx = resolution[0];
            auto ny = (Rank >= 2) ? resolution[1] : 1;
            auto nz = (Rank >= 3) ? resolution[2] : 1;

            auto reconstruction = require_field<std::string>(config, "reconstruction", "mesh");
            auto rec            = deserialize<reconstruction_t>(reconstruction);
            if (rec == reconstruction_t::PCM) {
                bp.halo_width = 1;
            }
            else if (rec == reconstruction_t::PLM) {
                bp.halo_width = 2;
            }
            else {
                throw std::runtime_error(
                    "mesh: unsupported reconstruction for halo width: " + reconstruction
                );
            }

            if constexpr (Rank == 1) {
                bp.active_resolution = {nx};
            }
            else if constexpr (Rank == 2) {
                bp.active_resolution = {ny, nx};
            }
            else {
                bp.active_resolution = {nz, ny, nx};
            }

            // bounds (required)
            auto x1_bounds = require_field<std::pair<real, real>>(config, "x1_bounds", "mesh");
            bp.bounds.push_back(x1_bounds);

            if constexpr (Rank >= 2) {
                auto x2_bounds = require_field<std::pair<real, real>>(config, "x2_bounds", "mesh");
                bp.bounds.push_back(x2_bounds);
            }

            if constexpr (Rank >= 3) {
                auto x3_bounds = require_field<std::pair<real, real>>(config, "x3_bounds", "mesh");
                bp.bounds.push_back(x3_bounds);
            }

            std::reverse(bp.bounds.begin(), bp.bounds.end());

            // coordinate system
            bp.coord_system = require_field<std::string>(config, "coord_system", "mesh");

            // spacing
            bp.spacing.push_back(require_field<std::string>(config, "x1_spacing", "mesh"));
            if constexpr (Rank >= 2) {
                bp.spacing.push_back(require_field<std::string>(config, "x2_spacing", "mesh"));
            }
            if constexpr (Rank >= 3) {
                bp.spacing.push_back(require_field<std::string>(config, "x3_spacing", "mesh"));
            }
            std::reverse(bp.spacing.begin(), bp.spacing.end());

            // boundary conditions
            // input is logical order: [x1_left, x1_right, x2_left, x2_right,
            // ...] we need array-index order: reverse dimension pairs, not
            // individual elements
            auto bcs = require_field<std::vector<std::string>>(config, "boundary_conditions", {});
            bp.boundary_conditions.resize(bcs.size());
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                // logical dim dd maps to array-index dim (Rank - 1 - dd)
                std::uint64_t src_offset               = dd * 2;
                std::uint64_t dst_offset               = (Rank - 1 - dd) * 2;
                bp.boundary_conditions[dst_offset]     = bcs[src_offset];     // left
                bp.boundary_conditions[dst_offset + 1] = bcs[src_offset + 1]; // right
            }

            // motion
            bp.moving_mesh          = require_field<bool>(config, "mesh_motion", "mesh");
            bp.homologous_expansion = require_field<bool>(config, "is_homologous", "mesh");

            // validation
            require(bp.active_resolution[Rank - 1] > 0, "x1 resolution must be positive");
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                require(bp.bounds[dd].first < bp.bounds[dd].second, "bounds must have min < max");
            }
            require(
                bp.boundary_conditions.size() == 2 * Rank,
                "must have 2*Rank boundary conditions"
            );

            return bp;
        }

        // -------------------------------------------------------------------------
        // physics
        // -------------------------------------------------------------------------
        static physics_blueprint_t physics(const config_dict_t& config)
        {
            physics_blueprint_t bp;

            // regime
            auto regime_str = require_field<std::string>(config, "regime", "physics");
            bp.regime       = deserialize<regime_t>(regime_str);

            // solver
            bp.solver = require_field<std::string>(config, "solver", "physics");

            // reconstruction
            bp.reconstruction = require_field<std::string>(config, "reconstruction", "physics");

            // timestepping
            bp.timestepping = require_field<std::string>(config, "timestepping", "physics");

            // eos parameters
            bp.gamma     = require_field<real>(config, "adiabatic_index", "physics");
            bp.cfl       = require_field<real>(config, "cfl_number", "physics");
            bp.plm_theta = require_field<real>(config, "plm_theta", "physics");
            bp.viscosity = require_field<real>(config, "viscosity", "physics");

            // flags
            bp.isothermal          = require_field<bool>(config, "isothermal", "physics");
            bp.ambient_sound_speed = require_field<real>(config, "ambient_sound_speed", "physics");
            bp.shakura_sunyaev_alpha =
                require_field<real>(config, "shakura_sunyaev_alpha", "physics");

            // derived flags
            bp.is_mhd          = (bp.regime == regime_t::RMHD || bp.regime == regime_t::MHD);
            bp.is_relativistic = (bp.regime == regime_t::SRHD || bp.regime == regime_t::RMHD);

            // validation
            require(bp.gamma > 0, "gamma must be positive");
            require(bp.cfl > 0 && bp.cfl < 1, "cfl must be in (0, 1)");
            require(bp.plm_theta >= 1 && bp.plm_theta <= 2, "plm_theta must be in [1, 2]");

            return bp;
        }

        // -------------------------------------------------------------------------
        // execution
        // -------------------------------------------------------------------------
        static execution_blueprint_t execution(const config_dict_t& config)
        {
            execution_blueprint_t bp;

            bp.start_time = require_field<real>(config, "start_time", "execution");
            bp.end_time   = require_field<real>(config, "end_time", "execution");

            bp.checkpoint_interval =
                require_field<real>(config, "checkpoint_interval", "execution");
            bp.dlogt = require_field<real>(config, "dlogt", "execution");

            bp.data_directory = require_field<std::string>(config, "data_directory", "execution");
            bp.start_index  = require_field<std::uint64_t>(config, "checkpoint_index", "execution");
            bp.restart_file = read_or_default<std::string>(config, "checkpoint_file", "");
            auto resolution =
                require_field<std::vector<std::int64_t>>(config, "resolution", "execution");
            if (resolution[2] > 1) {
                bp.checkpoint_zones = static_cast<std::uint64_t>(resolution[2]);
            }
            else if (resolution[1] > 1) {
                bp.checkpoint_zones = static_cast<std::uint64_t>(resolution[1]);
            }
            else {
                bp.checkpoint_zones = static_cast<std::uint64_t>(resolution[0]);
            }

            // validation
            require(bp.end_time > bp.start_time, "tend must be > start_time");
            require(bp.checkpoint_interval > 0, "checkpoint_interval must be positive");

            return bp;
        }

        // -------------------------------------------------------------------------
        // amr
        // -------------------------------------------------------------------------
        static amr_blueprint_t amr(const config_dict_t& config)
        {
            amr_blueprint_t bp;

            bp.enabled = read_or_default<bool>(config, "refinement_enabled", false);

            if (!bp.enabled) {
                bp.max_levels = 1;
                return bp;
            }

            bp.max_levels = require_field<std::uint64_t>(config, "refinement_max_levels", "amr");

            // refinement ratios
            auto ratios =
                require_field<std::vector<std::uint64_t>>(config, "refinement_ratios", "amr");
            bp.refinement_ratios = ratios;

            // refinement regions
            auto regions =
                require_field<std::vector<std::vector<real>>>(config, "refinement_regions", "amr");
            bp.static_refinement_regions = regions;

            // subcycling
            auto mode_str = require_field<std::string>(config, "refinement_subcycling_mode", "amr");
            bp.subcycling_mode = deserialize<subcycling_mode_t>(mode_str);

            auto substeps =
                require_field<std::vector<std::uint64_t>>(config, "refinement_substeps", "amr");
            bp.manual_substeps = substeps;

            // validation
            if (bp.enabled && bp.max_levels > 1) {
                require(
                    bp.refinement_ratios.size() >= bp.max_levels - 1,
                    "need refinement_ratios for each refined level"
                );
                require(
                    bp.static_refinement_regions.size() >= bp.max_levels - 1,
                    "need refinement_regions for each refined level"
                );
            }

            return bp;
        }

        // -------------------------------------------------------------------------
        // numerics
        // -------------------------------------------------------------------------
        static numerics_blueprint_t numerics(const config_dict_t& config)
        {
            numerics_blueprint_t bp;

            bp.use_quirk_smoothing = read_or_default<bool>(config, "use_quirk_smoothing", false);
            bp.use_fleischmann_limiter =
                read_or_default<bool>(config, "use_fleischmann_limiter", false);

            return bp;
        }

        // -------------------------------------------------------------------------
        // expressions
        // -------------------------------------------------------------------------
        static expressions_blueprint_t expressions(const config_dict_t& config)
        {
            expressions_blueprint_t bp;

            // hydro source
            auto hydro_src = config::try_read<config_dict_t>(config, "hydro_source_expressions");
            if (hydro_src.has_value()) {
                bp.hydro_source = *hydro_src;
            }

            // gravity source
            auto grav_src = config::try_read<config_dict_t>(config, "gravity_source_expressions");
            if (grav_src.has_value()) {
                bp.gravity_source = *grav_src;
            }

            // boundary sources (x1_inner, x1_outer, x2_inner, ...)
            const std::vector<std::string> bc_keys = {
                "bx1_inner_expressions",
                "bx1_outer_expressions",
                "bx2_inner_expressions",
                "bx2_outer_expressions",
                "bx3_inner_expressions",
                "bx3_outer_expressions"
            };

            for (std::uint64_t ii = 0; ii < 2 * Rank && ii < bc_keys.size(); ++ii) {
                auto bc_expr = config::try_read<config_dict_t>(config, bc_keys[ii]);
                if (bc_expr.has_value()) {
                    bp.boundary_sources.push_back(*bc_expr);
                }
                else {
                    bp.boundary_sources.push_back({});
                }
            }

            return bp;
        }

        // -------------------------------------------------------------------------
        // bodies
        // -------------------------------------------------------------------------
        static bodies_blueprint_t bodies(const config_dict_t& config)
        {
            bodies_blueprint_t bp;

            auto bodies_list =
                config::try_read<std::list<config_dict_t>>(config, "immersed_bodies");
            if (bodies_list.has_value()) {
                for (const auto& body_config : *bodies_list) {
                    bp.body_configs.push_back(body_config);
                }
            }

            return bp;
        }

        // -------------------------------------------------------------------------
        // gravitational_system
        // -------------------------------------------------------------------------
        static std::optional<gravitational_system_blueprint_t>
        gravitational_system(const config_dict_t& config)
        {
            // check if body_system config is present
            auto sys_config = config::try_read<config_dict_t>(config, "body_system");
            if (!sys_config.has_value()) {
                return std::nullopt;
            }

            gravitational_system_blueprint_t bp;

            // read system type
            bp.system_type =
                require_field<std::string>(*sys_config, "system_type", "gravitational_system");

            // dispatch based on system type
            if (bp.system_type == "binary") {
                bp.binary = extract_binary_system(*sys_config);
            }
            else {
                throw std::runtime_error(
                    "unsupported gravitational system type: " + bp.system_type
                );
            }

            return bp;
        }

        // -------------------------------------------------------------------------
        // extract_binary_system (helper)
        // -------------------------------------------------------------------------
        static binary_system_blueprint_t extract_binary_system(const config_dict_t& sys_config)
        {
            binary_system_blueprint_t bp;

            // get binary_config section
            auto binary_cfg =
                require_field<config_dict_t>(sys_config, "binary_config", "binary_system");

            // extract orbital parameters
            bp.semi_major   = require_field<real>(binary_cfg, "semi_major", "binary_system");
            bp.eccentricity = require_field<real>(binary_cfg, "eccentricity", "binary_system");
            bp.mass_ratio   = require_field<real>(binary_cfg, "mass_ratio", "binary_system");
            bp.total_mass   = require_field<real>(binary_cfg, "total_mass", "binary_system");

            // extract dynamics control
            bp.prescribed_motion = read_or_default<bool>(sys_config, "prescribed_motion", true);
            bp.reference_frame =
                require_field<std::string>(sys_config, "reference_frame", "binary_system");

            // extract components
            auto components =
                require_field<std::list<config_dict_t>>(binary_cfg, "components", "binary_system");

            require(components.size() == 2, "binary system must have exactly 2 components");

            for (const auto& comp : components) {
                bp.components.push_back(comp);
            }

            auto orbital_period =
                real{2} * std::numbers::pi_v<real> *
                std::sqrt((bp.semi_major * bp.semi_major * bp.semi_major) / bp.total_mass);
            bp.orbital_period    = orbital_period;
            bp.is_circular_orbit = (bp.eccentricity < real{1e-10});

            // validation
            require(bp.semi_major > 0, "semi_major must be positive");
            require(bp.eccentricity >= 0 && bp.eccentricity < 1, "eccentricity must be in [0, 1)");
            require(bp.mass_ratio > 0, "mass_ratio must be positive");
            require(bp.total_mass > 0, "total_mass must be positive");

            return bp;
        }

        // -------------------------------------------------------------------------
        // decomposition (optional - for multi-gpu)
        // -------------------------------------------------------------------------
        static std::optional<decomposition_blueprint_t<Rank>>
        decomposition(const config_dict_t& config)
        {
            // check if multi-gpu config is present
            auto topo_dims = config::try_read<std::vector<std::int64_t>>(config, "topology_dims");

            if (!topo_dims.has_value()) {
                return std::nullopt; // single-device mode
            }

            decomposition_blueprint_t<Rank> bp;

            // topology dimensions
            for (std::uint64_t ii = 0; ii < Rank && ii < topo_dims->size(); ++ii) {
                bp.topology_dims[ii] = (*topo_dims)[ii];
            }

            // fill remaining with 1
            for (std::uint64_t ii = topo_dims->size(); ii < Rank; ++ii) {
                bp.topology_dims[ii] = 1;
            }

            // mpi rank
            bp.mpi_rank = read_or_default<std::int32_t>(config, "mpi_rank", 0);

            // device ids
            auto dev_ids = config::try_read<std::vector<std::int64_t>>(config, "device_ids");
            if (dev_ids.has_value()) {
                for (auto id : *dev_ids) {
                    bp.device_ids.push_back(static_cast<int>(id));
                }
            }

            // halo width from reconstruction
            auto reconstr = require_field<std::string>(config, "reconstruction", "decomposition");
            if (reconstr == "plm") {
                bp.halo_width = 2;
            }
            else if (reconstr == "ppm") {
                bp.halo_width = 3;
            }
            else { // pcm
                bp.halo_width = 1;
            }

            return bp;
        }
    };

    inline std::uint64_t infer_dimensionality(const config_dict_t& config)
    {
        // auto nx = config::try_read<std::int64_t>(config, "nx").unwrap_or(1);
        auto ny = config::try_read<std::int64_t>(config, "ny").unwrap_or(1);
        auto nz = config::try_read<std::int64_t>(config, "nz").unwrap_or(1);

        auto halo    = config::try_read<std::uint64_t>(config, "halo_radius").unwrap_or(2);
        auto nghosts = 2 * static_cast<std::int64_t>(halo);

        // active cells (subtract ghosts)
        // auto active_nx = nx - nghosts;
        auto active_ny = std::max<std::int64_t>(ny - nghosts, 1);
        auto active_nz = std::max<std::int64_t>(nz - nghosts, 1);

        if (active_nz > 1) {
            return 3;
        }
        if (active_ny > 1) {
            return 2;
        }
        return 1;
    }

} // namespace simbi::ecs::creation

namespace simbi::ecs {
    template <std::uint64_t Rank>
    blueprint_set_t<Rank> blueprint_set_t<Rank>::from_config(const config_dict_t& config)
    {
        using extractor = creation::blueprint_extractor_t<Rank>;

        blueprint_set_t<Rank> set;
        set.mesh                 = extractor::mesh(config);
        set.physics              = extractor::physics(config);
        set.execution            = extractor::execution(config);
        set.amr                  = extractor::amr(config);
        set.numerics             = extractor::numerics(config);
        set.expressions          = extractor::expressions(config);
        set.bodies               = extractor::bodies(config);
        set.gravitational_system = extractor::gravitational_system(config);
        set.decomposition        = extractor::decomposition(config);

        return set;
    }
} // namespace simbi::ecs
