#ifndef COMPONENTS_HPP
#define COMPONENTS_HPP

#include "compat.hpp"                  // for real type
#include "compute/field.hpp"           // for field_t
#include "containers/vector.hpp"       // for vector_t
#include "domain/domain.hpp"           // for domain_t
#include "entity.hpp"                  // for entity_t
#include "mesh/mesh_config.hpp"        // for mesh_config_t
#include "physics/ib/collection.hpp"   // for body_collection_t
#include "state/express_t.hpp"
#include "utility/enums.hpp"   // for Geometry

#include <cmath>     // for std::pow, std::floor
#include <cstdint>   // for std::uint64_t
#include <string>    // for std::string

namespace simbi::ecs {
    /**
     * Here lies the ECS components used in the simulation.
     * Each struct represents a distinct component that can be
     * attached to entities within the ECS framework.
     */

    // hydro fields for one level
    template <typename Conserved, typename Primitive, std::uint64_t Dims>
    struct hydro_fields_t {
        field_t<Conserved, Dims> cons;
        field_t<Primitive, Dims> prim;
        vector_t<field_t<Conserved, Dims>, Dims> flux;
        vector_t<field_t<real, Dims>, Dims> bfield;   // for MHD
    };

    // mesh geometry for one level
    template <std::uint64_t Dims, Geometry G>
    struct mesh_geometry_t {
        mesh::mesh_config_t<Dims, G> config;
    };

    // level metadata
    struct level_info_t {
        std::uint64_t level_id;
        std::uint64_t refinement_ratio;
    };

    // marks refined levels
    template <std::uint64_t Dims>
    struct refinement_child_t {
        entity_t parent;
        domain_t<Dims> parent_coverage;
    };

    // immersed bodies (optional)
    template <std::uint64_t Dims>
    struct immersed_bodies_t {
        body::body_collection_t<Dims> bodies;
    };

    // global simulation state
    template <std::uint64_t Dims>
    struct simulation_metadata_t {
        // numerics
        real gamma;
        real plm_theta;
        real viscosity;
        real cfl;
        real time;
        real tend;
        real dt;
        real dlogt;
        real checkpoint_interval;
        real checkpoint_time;
        real prev_checkpoint_time;
        real ambient_sound_speed;

        // int tracking
        std::uint64_t iteration;
        std::uint64_t halo_radius;
        std::uint64_t checkpoint_index;
        std::uint64_t checkpoint_zones;
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

        // queries
        auto checkpoint_identifier() const
        {
            return dlogt != 0.0 ? checkpoint_index : checkpoint_time;
        }

        void update_checkpoint_time()
        {
            // Set the initial time interval
            // based on the current time, advanced
            // by the checkpoint interval to the nearest
            // place in the log10 scale. If dlogt is 0
            // then the interval is set to the current time
            // shifted towards the nearest checkpoint interval
            // if the checkpoint interval is 0 then the interval
            // is set to the current time
            if (dlogt != 0) {
                checkpoint_time =
                    time * std::pow(10.0, std::floor(std::log10(time) + dlogt));
            }
            else {
                static auto round_place = 1.0 / checkpoint_interval;
                checkpoint_time =
                    checkpoint_interval +
                    std::floor(time * round_place + 0.5) / round_place;
            }
            checkpoint_index += 1;
        }
    };

    // sources
    template <std::uint64_t Dims>
    struct sources_t {
        state::expression_t<Dims> hydro_source;
        state::expression_t<Dims> gravity_source;
        vector_t<state::expression_t<Dims>, 2 * Dims> bc_sources;
    };

}   // namespace simbi::ecs

#endif
