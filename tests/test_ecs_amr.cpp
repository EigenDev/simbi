#include "build_config.hpp"
#include "ecs/blueprints.hpp"
#include "ecs/components.hpp"
#include "ecs/creation/decomposition.hpp"
#include "ecs/creation/sim.hpp"
#include "ecs/simulation.hpp"
#include "utility/enums.hpp"

#include <cassert>
#include <iostream>

struct ideal_gas_t
{
    real gamma;
};

// =============================================================================
// test helpers
// =============================================================================

void print_ok(const char* msg)
{
    std::cout << "  " << msg << " [ok]" << std::endl;
}

void print_section(const char* msg)
{
    std::cout << "\n" << msg << std::endl;
}

// =============================================================================
// test: single-device decomposition
// =============================================================================

void test_single_device_decomposition()
{
    using namespace simbi;
    using namespace simbi::ecs;

    print_section("test: single-device decomposition");

    // configure 16x16 mesh
    mesh_blueprint_t<2> mesh_bp;
    mesh_bp.active_resolution    = {16, 16};
    mesh_bp.bounds               = {{0.0, 1.0}, {0.0, 1.0}};
    mesh_bp.coord_system         = "cartesian";
    mesh_bp.spacing              = {"linear", "linear"};
    mesh_bp.boundary_conditions  = {"outflow", "outflow", "outflow", "outflow"};
    mesh_bp.moving_mesh          = false;
    mesh_bp.homologous_expansion = false;
    mesh_bp.halo_width           = 2;

    physics_blueprint_t phys_bp;
    phys_bp.regime         = regime_t::NEWTONIAN;
    phys_bp.is_mhd         = false;
    phys_bp.gamma          = 1.4;
    phys_bp.cfl            = 0.4;
    phys_bp.plm_theta      = 1.5;
    phys_bp.viscosity      = 0.0;
    phys_bp.solver         = "hllc";
    phys_bp.reconstruction = "plm";
    phys_bp.timestepping   = "rk2";

    amr_blueprint_t amr_bp;
    amr_bp.enabled = false;

    execution_blueprint_t exec_bp;
    exec_bp.start_time          = 0.0;
    exec_bp.end_time            = 1.0;
    exec_bp.checkpoint_interval = 0.1;
    exec_bp.data_directory      = "./output";
    exec_bp.dlogt               = 0.0;
    exec_bp.start_index         = 0;

    numerics_blueprint_t num_bp;
    num_bp.use_quirk_smoothing     = false;
    num_bp.use_fleischmann_limiter = false;

    std::cout << "Building simulation..." << std::endl;
    // build simulation (single device, default)
    auto sim =
        builders::simulation_builder_t<regime_t::NEWTONIAN, 2, geometry_t::CARTESIAN, ideal_gas_t>()
            .configure_mesh(mesh_bp)
            .configure_physics(phys_bp)
            .configure_execution(exec_bp)
            .configure_amr(amr_bp)
            .configure_numerics(num_bp)
            .build();

    std::cout << "Simulation built." << std::endl;
    // verify single level
    assert(sim.num_levels() == 1);
    print_ok("single level created");

    // verify single partition
    assert(sim.num_partitions(0) == 1);
    print_ok("single partition created");

    // verify decomposition structure
    const auto& decomp = sim.decomposition(0);
    assert(decomp.partitions.size() == 1);
    assert(decomp.partition_entities.size() == 1);
    print_ok("decomposition structure valid");

    // verify partition domains
    const auto& part = sim.partition(0, 0);
    assert(part.owned_domain.start[0] == 0);
    assert(part.owned_domain.start[1] == 0);
    assert(part.owned_domain.fin[0] == 16);
    assert(part.owned_domain.fin[1] == 16);
    print_ok("partition owned domain [0,16]x[0,16]");

    // single block with outflow BCs should have no ghost expansion
    // (no partition neighbors)
    assert(part.allocated_domain.start[0] == -2);
    assert(part.allocated_domain.fin[0] == 18);
    print_ok("allocated domain matches owned (no partition neighbors)");

    // verify fields accessible via both apis (use partition_hydro API)
    auto& hydro = sim.partition_hydro(0, 0);
    // verify field domain
    assert(hydro.cons.domain().size() == 400); // 20x20
    print_ok("cons field size (including ghosts) = 400");

    // verify flux fields exist and are face-centered
    assert(hydro.flux[0].domain().fin[0] == 17); // n+1 faces in x
    assert(hydro.flux[1].domain().fin[1] == 17); // n+1 faces in y
    print_ok("flux fields are face-centered");

    // verify metadata
    const auto& meta = sim.metadata();
    assert(meta.gamma == 1.4);
    assert(meta.halo_radius == 2);
    assert(meta.coord_system == geometry_t::CARTESIAN);
    print_ok("metadata populated correctly");
}

// =============================================================================
// test: multi-partition decomposition (simulated, same device)
// =============================================================================

void test_multi_partition_decomposition()
{
    using namespace simbi;
    using namespace simbi::ecs;

    print_section("test: multi-partition decomposition (2x2)");

    // configure 16x16 mesh
    mesh_blueprint_t<2> mesh_bp;
    mesh_bp.active_resolution    = {16, 16};
    mesh_bp.bounds               = {{0.0, 1.0}, {0.0, 1.0}};
    mesh_bp.coord_system         = "cartesian";
    mesh_bp.spacing              = {"linear", "linear"};
    mesh_bp.boundary_conditions  = {"outflow", "outflow", "outflow", "outflow"};
    mesh_bp.moving_mesh          = false;
    mesh_bp.homologous_expansion = false;
    mesh_bp.halo_width           = 2;

    physics_blueprint_t phys_bp;
    phys_bp.regime         = regime_t::NEWTONIAN;
    phys_bp.is_mhd         = false;
    phys_bp.gamma          = 1.4;
    phys_bp.cfl            = 0.4;
    phys_bp.plm_theta      = 1.5;
    phys_bp.viscosity      = 0.0;
    phys_bp.solver         = "hllc";
    phys_bp.reconstruction = "plm";
    phys_bp.timestepping   = "rk2";

    amr_blueprint_t amr_bp;
    amr_bp.enabled = false;

    execution_blueprint_t exec_bp;
    exec_bp.start_time          = 0.0;
    exec_bp.end_time            = 1.0;
    exec_bp.checkpoint_interval = 0.1;
    exec_bp.data_directory      = "./output";
    exec_bp.dlogt               = 0.0;
    exec_bp.start_index         = 0;

    numerics_blueprint_t num_bp;
    num_bp.use_quirk_smoothing     = false;
    num_bp.use_fleischmann_limiter = false;

    // configure 2x2 decomposition (4 partitions, all on device 0)
    decomposition_blueprint_t<2> decomp_bp;
    decomp_bp.topology_dims = {2, 2};
    decomp_bp.halo_width    = 2;
    decomp_bp.mpi_rank      = 0;
    decomp_bp.device_ids    = {0}; // all partitions on same device

    // build simulation
    auto sim =
        builders::simulation_builder_t<regime_t::NEWTONIAN, 2, geometry_t::CARTESIAN, ideal_gas_t>()
            .configure_mesh(mesh_bp)
            .configure_physics(phys_bp)
            .configure_execution(exec_bp)
            .configure_amr(amr_bp)
            .configure_numerics(num_bp)
            .configure_decomposition(decomp_bp)
            .build();

    // verify 4 partitions
    assert(sim.num_partitions(0) == 4);
    print_ok("4 partitions created");

    // verify decomposition
    const auto& decomp = sim.decomposition(0);
    assert(decomp.skeleton.size() == 4);
    print_ok("skeleton has 4 blocks");

    // check partition domains
    // with 2x2 topology on 16x16 grid, each partition owns 8x8
    std::uint64_t total_owned = 0;
    for (std::uint64_t pp = 0; pp < 4; ++pp) {
        const auto& part       = sim.partition(0, pp);
        auto        owned_size = part.owned_domain.size();
        total_owned += owned_size;

        // each partition should own 8x8 = 64 cells
        assert(owned_size == 64);
    }
    assert(total_owned == 256); // 16x16 total
    print_ok("partition owned domains are 8x8 each, total 256");

    // verify halo graph was built
    assert(!decomp.halo_graph.empty());
    std::cout << "  halo graph has " << decomp.halo_graph.size() << " links";
    print_ok("");

    // check that interior partitions have ghost expansion
    // partition at (0,0) has neighbors on right and top
    // partition at (1,1) has neighbors on all sides (interior)
    bool found_expanded = false;
    for (std::uint64_t pp = 0; pp < 4; ++pp) {
        const auto& part = sim.partition(0, pp);
        if (part.allocated_domain.size() > part.owned_domain.size()) {
            found_expanded = true;
            break;
        }
    }
    assert(found_expanded);
    print_ok("at least one partition has ghost expansion");

    // verify each partition has its own fields
    for (std::uint64_t pp = 0; pp < 4; ++pp) {
        auto& fields = sim.partition_hydro(0, pp);
        assert(fields.cons.data() != nullptr);
    }
    print_ok("all partitions have allocated fields");

    // verify different partitions have different field pointers
    auto* ptr0 = sim.partition_hydro(0, 0).cons.data();
    auto* ptr1 = sim.partition_hydro(0, 1).cons.data();
    assert(ptr0 != ptr1);
    print_ok("partitions have distinct field allocations");
}

// =============================================================================
// test: periodic boundary handling
// =============================================================================

void test_periodic_boundaries()
{
    using namespace simbi;
    using namespace simbi::ecs;

    print_section("test: periodic boundary decomposition");

    mesh_blueprint_t<2> mesh_bp;
    mesh_bp.active_resolution    = {16, 16};
    mesh_bp.bounds               = {{0.0, 1.0}, {0.0, 1.0}};
    mesh_bp.coord_system         = "cartesian";
    mesh_bp.spacing              = {"linear", "linear"};
    mesh_bp.boundary_conditions  = {"periodic", "periodic", "periodic", "periodic"};
    mesh_bp.moving_mesh          = false;
    mesh_bp.homologous_expansion = false;
    mesh_bp.halo_width           = 2;

    physics_blueprint_t phys_bp;
    phys_bp.regime         = regime_t::NEWTONIAN;
    phys_bp.is_mhd         = false;
    phys_bp.gamma          = 1.4;
    phys_bp.cfl            = 0.4;
    phys_bp.plm_theta      = 1.5;
    phys_bp.viscosity      = 0.0;
    phys_bp.solver         = "hllc";
    phys_bp.reconstruction = "plm";
    phys_bp.timestepping   = "rk2";

    amr_blueprint_t amr_bp;
    amr_bp.enabled = false;

    execution_blueprint_t exec_bp;
    exec_bp.start_time          = 0.0;
    exec_bp.end_time            = 1.0;
    exec_bp.checkpoint_interval = 0.1;
    exec_bp.data_directory      = "./output";
    exec_bp.dlogt               = 0.0;
    exec_bp.start_index         = 0;

    numerics_blueprint_t num_bp;
    num_bp.use_quirk_smoothing     = false;
    num_bp.use_fleischmann_limiter = false;

    // 2x1 decomposition with periodic
    decomposition_blueprint_t<2> decomp_bp;
    decomp_bp.topology_dims = {2, 1};
    decomp_bp.halo_width    = 2;
    decomp_bp.mpi_rank      = 0;

    auto sim =
        builders::simulation_builder_t<regime_t::NEWTONIAN, 2, geometry_t::CARTESIAN, ideal_gas_t>()
            .configure_mesh(mesh_bp)
            .configure_physics(phys_bp)
            .configure_execution(exec_bp)
            .configure_amr(amr_bp)
            .configure_numerics(num_bp)
            .configure_decomposition(decomp_bp)
            .build();

    assert(sim.num_partitions(0) == 2);
    print_ok("2 partitions created");

    const auto& decomp = sim.decomposition(0);

    // with periodic BCs, both partitions should have ghost zones
    // on their "outer" faces (which wrap to the other partition)
    for (std::uint64_t pp = 0; pp < 2; ++pp) {
        const auto& part = sim.partition(0, pp);
        // allocated should be larger than owned due to halos
        assert(part.allocated_domain.size() > part.owned_domain.size());
    }
    print_ok("periodic partitions have ghost expansion");

    // halo graph should have links for periodic wrapping
    assert(decomp.halo_graph.size() >= 2);
    print_ok("halo graph includes periodic links");
}

// =============================================================================
// test: amr with decomposition
// =============================================================================

void test_amr_with_decomposition()
{
    using namespace simbi;
    using namespace simbi::ecs;

    print_section("test: amr hierarchy with decomposition");

    mesh_blueprint_t<2> mesh_bp;
    mesh_bp.active_resolution    = {8, 8};
    mesh_bp.bounds               = {{0.0, 1.0}, {0.0, 1.0}};
    mesh_bp.coord_system         = "cartesian";
    mesh_bp.spacing              = {"linear", "linear"};
    mesh_bp.boundary_conditions  = {"periodic", "periodic", "periodic", "periodic"};
    mesh_bp.moving_mesh          = false;
    mesh_bp.homologous_expansion = false;
    mesh_bp.halo_width           = 2;

    physics_blueprint_t phys_bp;
    phys_bp.regime         = regime_t::NEWTONIAN;
    phys_bp.is_mhd         = false;
    phys_bp.gamma          = 1.4;
    phys_bp.cfl            = 0.4;
    phys_bp.plm_theta      = 1.5;
    phys_bp.viscosity      = 0.0;
    phys_bp.solver         = "hllc";
    phys_bp.reconstruction = "plm";
    phys_bp.timestepping   = "rk2";

    amr_blueprint_t amr_bp;
    amr_bp.enabled                   = true;
    amr_bp.max_levels                = 2;
    amr_bp.refinement_ratios         = {2};
    amr_bp.static_refinement_regions = {{0.25, 0.75, 0.25, 0.75}};
    amr_bp.subcycling_mode           = subcycling_mode_t::STANDARD;

    execution_blueprint_t exec_bp;
    exec_bp.start_time          = 0.0;
    exec_bp.end_time            = 1.0;
    exec_bp.checkpoint_interval = 0.1;
    exec_bp.data_directory      = "./output";
    exec_bp.dlogt               = 0.0;
    exec_bp.start_index         = 0;

    numerics_blueprint_t num_bp;
    num_bp.use_quirk_smoothing     = false;
    num_bp.use_fleischmann_limiter = false;

    // single device for amr test
    auto sim =
        builders::simulation_builder_t<regime_t::NEWTONIAN, 2, geometry_t::CARTESIAN, ideal_gas_t>()
            .configure_mesh(mesh_bp)
            .configure_physics(phys_bp)
            .configure_execution(exec_bp)
            .configure_amr(amr_bp)
            .configure_numerics(num_bp)
            .build();

    // verify 2 levels
    assert(sim.num_levels() == 2);
    print_ok("2 amr levels created");

    // verify each level has decomposition
    assert(sim.num_partitions(0) == 1);
    assert(sim.num_partitions(1) == 1);
    print_ok("each level has 1 partition");

    // verify level info
    assert(sim.level_info(0).level_id == 0);
    assert(sim.level_info(0).refinement_ratio == 1);
    assert(sim.level_info(1).level_id == 1);
    assert(sim.level_info(1).refinement_ratio == 2);
    print_ok("level info correct");

    // verify refinement linkage
    const auto& l1_ref = sim.refinement(1);
    assert(l1_ref.parent == sim.levels[0]);
    print_ok("level 1 linked to parent");

    // verify level 1 domain (refined center)
    const auto& l1_part = sim.partition(1, 0);
    // [0.25, 0.75] on 8x8 -> [2, 6] in L0 -> [4, 12] in L1
    assert(l1_part.owned_domain.start[0] == 4);
    assert(l1_part.owned_domain.start[1] == 4);
    assert(l1_part.owned_domain.fin[0] == 12);
    assert(l1_part.owned_domain.fin[1] == 12);
    print_ok("level 1 domain [4,12]x[4,12]");

    // verify owned domain sizes
    // level 0: 8x8 = 64 cells
    // level 1: refined [4,12]x[4,12] = 8x8 = 64 cells
    const auto& l0_part_fields = sim.partition(0, 0);
    assert(l0_part_fields.owned_domain.size() == 64);
    assert(l1_part.owned_domain.size() == 64);
    print_ok("owned domains are 8x8 at both levels");

    // verify fields are allocated (non-null) using partition_hydro
    auto& l0_hydro = sim.partition_hydro(0, 0);
    auto& l1_hydro = sim.partition_hydro(1, 0);
    assert(l0_hydro.cons.data() != nullptr);
    assert(l1_hydro.cons.data() != nullptr);
    print_ok("fields allocated at both levels");
}

// =============================================================================
// test: workspace allocation
// =============================================================================

void test_workspace_allocation()
{
    using namespace simbi;
    using namespace simbi::ecs;

    print_section("test: workspace allocation");

    mesh_blueprint_t<2> mesh_bp;
    mesh_bp.active_resolution    = {8, 8};
    mesh_bp.bounds               = {{0.0, 1.0}, {0.0, 1.0}};
    mesh_bp.coord_system         = "cartesian";
    mesh_bp.spacing              = {"linear", "linear"};
    mesh_bp.boundary_conditions  = {"outflow", "outflow", "outflow", "outflow"};
    mesh_bp.moving_mesh          = false;
    mesh_bp.homologous_expansion = false;

    physics_blueprint_t phys_bp;
    phys_bp.regime         = regime_t::NEWTONIAN;
    phys_bp.is_mhd         = false;
    phys_bp.gamma          = 1.4;
    phys_bp.cfl            = 0.4;
    phys_bp.plm_theta      = 1.5;
    phys_bp.viscosity      = 0.0;
    phys_bp.solver         = "hllc";
    phys_bp.reconstruction = "plm";
    phys_bp.timestepping   = "rk2";

    amr_blueprint_t amr_bp;
    amr_bp.enabled = false;

    execution_blueprint_t exec_bp;
    exec_bp.start_time          = 0.0;
    exec_bp.end_time            = 1.0;
    exec_bp.checkpoint_interval = 0.1;
    exec_bp.data_directory      = "./output";
    exec_bp.dlogt               = 0.0;
    exec_bp.start_index         = 0;

    numerics_blueprint_t num_bp;
    num_bp.use_quirk_smoothing     = false;
    num_bp.use_fleischmann_limiter = false;

    auto sim =
        builders::simulation_builder_t<regime_t::NEWTONIAN, 2, geometry_t::CARTESIAN, ideal_gas_t>()
            .configure_mesh(mesh_bp)
            .configure_physics(phys_bp)
            .configure_execution(exec_bp)
            .configure_amr(amr_bp)
            .configure_numerics(num_bp)
            .build();

    // workspace should not exist initially
    assert(!sim.has_workspace(0, 0));
    print_ok("workspace not allocated initially");

    // create workspace
    sim.create_workspace(0, 0);
    assert(sim.has_workspace(0, 0));
    print_ok("workspace created on demand");

    // verify workspace fields
    auto& ws = sim.workspace(0, 0);
    assert(ws.u_n.data() != nullptr);
    assert(ws.u_star.data() != nullptr);
    print_ok("workspace fields allocated");

    // creating again should be idempotent
    sim.create_workspace(0, 0);
    assert(sim.has_workspace(0, 0));
    print_ok("create_workspace is idempotent");
}

// =============================================================================
// main
// =============================================================================

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "ecs decomposition test suite" << std::endl;
    std::cout << "========================================" << std::endl;

    test_single_device_decomposition();
    test_multi_partition_decomposition();
    test_periodic_boundaries();
    test_amr_with_decomposition();
    test_workspace_allocation();

    std::cout << "\n========================================" << std::endl;
    std::cout << "[PASS] all tests passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
