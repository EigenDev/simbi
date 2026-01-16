#ifndef COMPONENTS_HPP
#define COMPONENTS_HPP

// =============================================================================
// components.hpp
//
// ecs components for multi-device domain-decomposed simulations.
//
// architecture overview:
//   - simulation owns levels (amr hierarchy)
//   - each level is decomposed into partitions (one per device)
//   - each partition owns its fields, executor, and halo metadata
//   - halo_graph describes all inter-partition data transfers
//
// key types:
//   - partition_t:           execution context + domain metadata
//   - halo_link_t:           single send/recv pair between partitions
//   - level_decomposition_t: all partitions + halo graph for one level
//   - partition_fields_t:    hydro fields for one partition
//
// migration notes (hesi → xpu):
//   - partition_t.stream → partition_t.executor (executor owns stream)
//   - partition_t.device_id removed (query executor.device_id() if needed)
//   - het::comm::rank_id_t → xpu::comm::rank_id_t
//   - executor stored by value, owns its resources (modern c++ pattern)
// =============================================================================

#include "build_config.hpp"
#include "containers/vector.hpp"
#include "dag/express_t.hpp"
#include "entity.hpp"
#include "geometry/block_geometry.hpp"
#include "grid/amr/flux_correction.hpp"
#include "grid/block_info.hpp"
#include "grid/boundary.hpp"
#include "grid/connectivity.hpp"
#include "grid/decomposition.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "grid/mesh_config.hpp"
#include "grid/patch_id.hpp"
#include "grid/skeleton.hpp"
#include "physics/ib/collection.hpp"
#include "physics/ib/diagnostics.hpp"
#include "platform.hpp"
#include "utility/enums.hpp"
#include "xpu/comm/types.hpp"
#include "xpu/xpu.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace simbi::ecs {

    // =============================================================================
    // partition components
    // these describe how a single device's piece of the domain is configured
    // =============================================================================

    // -----------------------------------------------------------------------------
    // partition_t
    //
    // binds a block's topology to a specific execution context.
    // each partition owns:
    //   - block metadata (geometry, connectivity)
    //   - executor (owns stream + device for async kernel launch)
    //   - domain decomposition (owned cells + ghost padding)
    //
    // the allocated_domain includes ghost cells; owned_domain is the interior
    // that this partition actually computes.
    //
    // note: device_id no longer stored - query executor.device_id() if needed.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct partition_t
    {
        // block metadata from the skeleton (id, geometry, face connectivity)
        grid::block_info_t<Rank> block;

        // the cells this partition computes (no ghosts)
        grid::domain_t<Rank> owned_domain;

        // the cells this partition allocates (owned + ghost padding)
        grid::domain_t<Rank> allocated_domain;

        // face-centered domains for each direction (for mhd)
        // face_domains[d] has one extra cell in dimension d
        vector_t<grid::domain_t<Rank>, Rank> face_domains;

        // edge-centered domains for each direction (for mhd constrained
        // transport) edge_domains[d] has one extra cell in both transverse
        // dimensions
        vector_t<grid::domain_t<Rank>, Rank> edge_domains;

        // execution context for this partition's kernels
        // each partition gets its own executor to enable concurrent execution
        xpu::executor_t<xpu::default_space> executor;

        // mpi rank info (node id + local device id)
        // used by communicator to determine transfer strategy
        xpu::comm::rank_id_t rank_id;
    };

    // -----------------------------------------------------------------------------
    // halo_link_t
    //
    // describes a single halo exchange operation between two partitions.
    // the source partition sends from src_region (its interior boundary),
    // the destination partition receives into dst_region (its ghost zone).
    //
    // for periodic boundaries, src and dst may be the same partition
    // (self-communication with coordinate wrapping).
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct halo_link_t
    {
        // source partition info
        grid::patch_id_t     src_patch;
        xpu::comm::rank_id_t src_rank;
        grid::domain_t<Rank> src_region; // interior boundary to read from

        // destination partition info
        grid::patch_id_t     dst_patch;
        xpu::comm::rank_id_t dst_rank;
        grid::domain_t<Rank> dst_region; // ghost zone to write into

        // which dimension and direction this link corresponds to
        // useful for debugging and for building dimension-ordered exchanges
        std::uint64_t dimension;
        grid::side_t  direction;
    };

    // -----------------------------------------------------------------------------
    // level_decomposition_t
    //
    // complete decomposition state for one amr level.
    // contains all partitions and the halo graph that connects them.
    //
    // the skeleton stores block metadata (patch_id -> block_info).
    // the partitions vector stores runtime state (streams, device ids).
    // the halo_graph stores all required inter-partition transfers.
    // partition_entities maps to ecs entities that hold the actual field data.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct level_decomposition_t
    {
        // block-level metadata (topology, connectivity)
        grid::skeleton_t<Rank> skeleton;

        // runtime partition state (one per device)
        std::vector<partition_t<Rank>> partitions;

        // all halo transfers for this level
        // during exchange, iterate this list and issue transfers
        std::vector<halo_link_t<Rank>> halo_graph;

        // ecs entity handles for partition-specific components
        // partition_entities[i] holds the entity for partitions[i]
        // use this to fetch partition_fields_t, partition_geometry_t, etc.
        std::vector<entity_t> partition_entities;

        // the process topology used to create this decomposition
        // stored for load balancing queries and debugging
        grid::topology_t topology;

        // -----------------------------------------------------------------
        // queries
        // -----------------------------------------------------------------

        std::uint64_t num_partitions() const
        {
            return partitions.size();
        }

        // find partition index by patch id
        // returns -1 if not found (patch may be on another mpi rank)
        std::int64_t find_partition(const grid::patch_id_t& id) const
        {
            for (std::uint64_t ii = 0; ii < partitions.size(); ++ii) {
                if (partitions[ii].block.id == id) {
                    return static_cast<std::int64_t>(ii);
                }
            }
            return -1;
        }

        // check if a patch is local to this mpi rank
        bool is_local(const grid::patch_id_t& id) const
        {
            return find_partition(id) >= 0;
        }
    };

    // =============================================================================
    // field components
    // these hold the actual simulation data for each partition
    // =============================================================================

    // -----------------------------------------------------------------------------
    // partition_fields_t
    //
    // hydro field storage for a single partition.
    // allocated on the partition's device, sized to allocated_domain.
    //
    // this replaces the old hydro_fields_t which was one-per-level.
    // now we have one partition_fields_t per partition per level.
    // -----------------------------------------------------------------------------
    template <typename Conserved, typename Primitive, std::uint64_t Rank>
    struct partition_fields_t
    {
        // conserved variables (density, momentum, energy, ...)
        grid::field_t<Conserved, Rank> cons;

        // primitive variables (density, velocity, pressure, ...)
        grid::field_t<Primitive, Rank> prim;

        // fluxes in each direction
        // flux[d] stores the flux through faces normal to dimension d
        vector_t<grid::field_t<Conserved, Rank>, Rank> flux;

        // face-centered magnetic field components (mhd only)
        // bfield[d] stores B_d at faces normal to axis d
        vector_t<grid::field_t<real, Rank>, Rank> bfield;

        // edge-centered electric field components (mhd only)
        // efield[d] stores E_d at edges parallel to axis d
        // used for constrained transport with amr
        vector_t<grid::field_t<real, Rank>, Rank> efield;
    };

    // -----------------------------------------------------------------------------
    // partition_workspace_t
    //
    // scratch space for runge-kutta time integration on a single partition.
    // stores the initial state and intermediate stages.
    // -----------------------------------------------------------------------------
    template <typename Conserved, typename Primitive, std::uint64_t Rank>
    struct partition_workspace_t
    {
        // conserved state at the beginning of the timestep
        grid::field_t<Conserved, Rank> u_n;

        // primitive state at the beginning of the timestep (for AMR prolongation)
        grid::field_t<Primitive, Rank> prim_n;

        // efield at beginning of the timestep
        vector_t<grid::field_t<real, Rank>, Rank> e_n;

        // intermediate state for multi-stage methods
        grid::field_t<Conserved, Rank> u_star;
    };

    // =============================================================================
    // geometry components
    // these describe the physical coordinate system for each partition
    // =============================================================================

    // -----------------------------------------------------------------------------
    // partition_geometry_t
    //
    // physical geometry for a single partition's domain.
    // stores the mesh configuration (cell positions, widths) for this block.
    //
    // the geometry_service creates these from the global geometry config
    // using the partition's topological coordinates.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank, geometry_t G>
    struct partition_geometry_t
    {
        grid::mesh_config_t<Rank> config;
    };

    // =============================================================================
    // level-wide components
    // these are shared across all partitions of a level
    // =============================================================================

    // -----------------------------------------------------------------------------
    // level_info_t
    //
    // metadata for an amr level.
    // level 0 is the coarsest; higher levels are finer.
    // -----------------------------------------------------------------------------
    struct level_info_t
    {
        // which level in the amr hierarchy (0 = coarsest)
        std::uint64_t level_id;

        // refinement ratio relative to coarser level
        // typically 2 (each coarse cell becomes 2^rank fine cells)
        std::uint64_t refinement_ratio;
    };

    // -----------------------------------------------------------------------------
    // level_mesh_t
    //
    // stores the mesh configuration for a level.
    // includes coordinate maps, boundaries, and cell counts.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct level_mesh_t
    {
        grid::mesh_config_t<Rank> config;
    };

    // -----------------------------------------------------------------------------
    // refinement_child_t
    //
    // marks a level as a refined child of a coarser level.
    // stores the parent entity and which region of the parent is covered.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct refinement_child_t
    {
        // entity handle for the parent level
        entity_t parent;

        // which region of the parent level this child covers
        // used for restriction (fine -> coarse) and prolongation (coarse ->
        // fine)
        grid::domain_t<Rank> parent_coverage;
    };

    // -----------------------------------------------------------------------------
    // flux_register_component_t
    //
    // stores flux registers for coarse-fine boundary flux correction (reflux).
    // one register per partition, tracking flux mismatches at refinement
    // boundaries.
    // -----------------------------------------------------------------------------
    template <typename Conserved, std::uint64_t Rank>
    struct flux_register_component_t
    {
        // one flux register per partition
        // each register tracks the coarse-fine flux mismatch for that partition
        std::vector<grid::amr::flux_register_t<Conserved, Rank>> registers;

        // refinement ratio used for this level pair
        iarray<Rank> ratio;

        // whether registers have been initialized
        bool initialized{false};
    };

    // =============================================================================
    // global simulation components
    // these are attached to the global entity, shared across all levels
    // =============================================================================

    // -----------------------------------------------------------------------------
    // simulation_metadata_t
    //
    // global simulation state: numerics, configuration, timing.
    // single instance attached to the global entity.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct simulation_metadata_t
    {
        // === numeric parameters ===
        real gamma;                // adiabatic index
        real plm_theta;            // plm limiter parameter
        real viscosity;            // artificial viscosity coefficient
        real cfl;                  // courant number
        real time;                 // current simulation time
        real tend;                 // final simulation time
        real global_dt;            // current global timestep
        real dlogt;                // logarithmic output spacing (0 = linear)
        real checkpoint_interval;  // time between checkpoints
        real checkpoint_time;      // next scheduled checkpoint time
        real prev_checkpoint_time; // last checkpoint time
        real ambient_sound_speed;  // for vacuum/floor handling
        real initial_time;         // start time from original initial conditions

        // === integer tracking ===
        std::uint64_t iteration;        // current iteration count
        std::uint64_t halo_radius;      // ghost zone width
        std::uint64_t checkpoint_index; // current checkpoint number
        std::uint64_t checkpoint_zones; // zones per checkpoint file
        std::uint64_t dimensions{Rank}; // spatial dimensions

        // === configuration enums ===
        regime_t                                  regime;         // newtonian, srhd, rmhd
        shockwave_limiter_t                       shock_smoother; // shock detection method
        solver_t                                  solver;         // riemann solver type
        cellspacing_t                             x1_spacing;     // x1 coordinate spacing
        cellspacing_t                             x2_spacing;     // x2 coordinate spacing
        cellspacing_t                             x3_spacing;     // x3 coordinate spacing
        geometry_t                                coord_system; // cartesian, spherical, cylindrical
        reconstruction_t                          reconstruction; // pcm, plm, ppm, weno
        timestepping_t                            timestepping;   // euler, rk2, rk3
        vector_t<grid::boundary_type_t, 2 * Rank> boundary_conditions;

        // global resolution (1, 1, nx) or (1, ny, nx) or (nz, ny, nx)
        iarray<3> resolution;

        // === flags ===
        bool is_mhd;
        bool is_relativistic;

        // === paths ===
        std::string data_dir;

        // === multi-level timestepping ===
        std::vector<real>          level_dts;      // dt for each level
        std::vector<std::uint64_t> level_substeps; // substeps per coarse step
        subcycling_mode_t          subcycling_mode{subcycling_mode_t::STANDARD};

        // -----------------------------------------------------------------
        // methods
        // -----------------------------------------------------------------

        void advance_schedule(auto schedule)
        {
            checkpoint_time  = schedule.checkpoint_time;
            checkpoint_index = schedule.checkpoint_index;
        }

        real checkpoint_identifier() const
        {
            return dlogt != 0.0 ? checkpoint_index : checkpoint_time;
        }
    };

    // -----------------------------------------------------------------------------
    // sources_t
    //
    // user-defined source terms and boundary condition functions.
    // evaluated lazily via expression trees.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct sources_t
    {
        state::expression_t<Rank> hydro_source; // momentum/energy sources
        // gravitational acceleration
        state::expression_t<Rank> gravity_source;

        // per-face boundary condition expressions
        // indexed as [2*dim + side] where side: 0=left, 1=right
        vector_t<state::expression_t<Rank>, 2 * Rank> bc_sources;
    };

    // -----------------------------------------------------------------------------
    // immersed_bodies_t
    //
    // optional component for simulations with immersed boundary objects.
    // stores the collection of solid bodies embedded in the fluid.
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct immersed_bodies_t
    {
        body::body_collection_t<Rank> bodies;
    };

    // -----------------------------------------------------------------------------
    // mesh_motion_config_t
    //
    // host-side storage for moving mesh configuration.
    // stores scale factor callbacks from python and produces motion_state_t
    // snapshots for kernel dispatch.
    //
    // attached to global entity when mesh_motion is enabled.
    // -----------------------------------------------------------------------------
    struct mesh_motion_config_t
    {
        // scale factor a(t) and its derivative da/dt
        std::function<real(real)> scale_factor;
        std::function<real(real)> scale_factor_derivative;

        // expansion mode
        bool homologous{false};

        // produce device-side snapshot at given time
        geometry::motion_state_t snapshot(real time) const
        {
            real a    = scale_factor ? scale_factor(time) : 1.0;
            real adot = scale_factor_derivative ? scale_factor_derivative(time) : 0.0;
            return geometry::motion_state_t{
                .enabled       = true,
                .is_homologous = homologous,
                .a             = a,
                .a_dot         = adot
            };
        }

        // static "no motion" state
        static geometry::motion_state_t static_mesh()
        {
            return geometry::motion_state_t{
                .enabled       = false,
                .is_homologous = false,
                .a             = 1.0,
                .a_dot         = 0.0
            };
        }
    };

    // -----------------------------------------------------------------------------
    // body_info_t
    //
    // diagnostics for immersed bodies (forces, torques, etc).
    // -----------------------------------------------------------------------------
    template <std::uint64_t Rank>
    struct body_info_t
    {
        using diag_t = std::conditional_t<
            platform::is_gpu,
            body::gpu_diagnostics_t<Rank>,
            body::cpu_diagnostics_t<Rank>>;
        std::unique_ptr<diag_t> diagnostics;
    };

    // =============================================================================
    // backward compatibility aliases
    // these allow gradual migration from single-device code
    // =============================================================================

    // the old hydro_fields_t is now partition_fields_t
    template <typename Conserved, typename Primitive, std::uint64_t Rank>
    using hydro_fields_t = partition_fields_t<Conserved, Primitive, Rank>;

    // the old mesh_geometry_t is now partition_geometry_t
    template <std::uint64_t Rank, geometry_t G>
    using mesh_geometry_t = partition_geometry_t<Rank, G>;

} // namespace simbi::ecs

#endif
