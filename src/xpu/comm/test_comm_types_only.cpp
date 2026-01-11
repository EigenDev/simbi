// =============================================================================
// test_comm_types_only.cpp
//
// standalone test for xpu comm types (no cuda dependency)
// validates rank identification and locality queries
// =============================================================================

#include "types.hpp"

#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace simbi::xpu::comm;

void test_rank_id_basic()
{
    std::cout << "testing rank_id_t basic operations...\n";

    rank_id_t rank1{0, 0};
    rank_id_t rank2{0, 1};
    rank_id_t rank3{1, 0};

    assert(rank1.is_local());
    assert(rank2.is_local());
    assert(!rank3.is_local()); // different node

    assert(rank1 == rank1);
    assert(rank1 != rank2);

    std::cout << "  ✓ rank_id_t basic operations work\n";
}

void test_locality_queries()
{
    std::cout << "testing locality queries...\n";

    rank_id_t local1{0, 0};
    rank_id_t local2{0, 1};
    rank_id_t remote{1, 0};

    // same node checks
    assert(same_node(local1, local2));
    assert(!same_node(local1, remote));

    // same device checks
    assert(same_device(local1, local1));
    assert(!same_device(local1, local2));

    // mpi requirement checks
    assert(!requires_mpi(local1, local2));
    assert(requires_mpi(local1, remote));

    // peer copy capability
    assert(can_use_peer_copy(local1, local2));
    assert(!can_use_peer_copy(local1, local1)); // same device
    assert(!can_use_peer_copy(local1, remote)); // different node

    std::cout << "  ✓ locality queries work correctly\n";
}

void test_transfer_strategy()
{
    std::cout << "testing transfer strategy selection...\n";

    rank_id_t rank_same{0, 0};
    rank_id_t rank_local{0, 1};
    rank_id_t rank_remote{1, 0};

    // same device: no transfer needed
    auto strategy1 = get_transfer_strategy(rank_same, rank_same);
    assert(strategy1 == transfer_strategy_t::none);

    // same node, different device: peer copy
    auto strategy2 = get_transfer_strategy(rank_same, rank_local);
    assert(strategy2 == transfer_strategy_t::peer_copy);

    // different node: mpi
    auto strategy3 = get_transfer_strategy(rank_same, rank_remote);
    assert(strategy3 == transfer_strategy_t::mpi_send);

    std::cout << "  ✓ transfer strategy selection works\n";
}

void test_rank_id_hash()
{
    std::cout << "testing rank_id_t hash support...\n";

    std::unordered_map<rank_id_t, int> rank_map;

    rank_id_t rank1{0, 0};
    rank_id_t rank2{0, 1};
    rank_id_t rank3{1, 0};

    rank_map[rank1] = 100;
    rank_map[rank2] = 200;
    rank_map[rank3] = 300;

    assert(rank_map[rank1] == 100);
    assert(rank_map[rank2] == 200);
    assert(rank_map[rank3] == 300);
    assert(rank_map.size() == 3);

    std::cout << "  ✓ rank_id_t hash support works\n";
}

void test_transfer_strategy_coverage()
{
    std::cout << "testing all transfer strategy paths...\n";

    rank_id_t r00{0, 0};
    rank_id_t r01{0, 1};
    rank_id_t r10{1, 0};
    rank_id_t r11{1, 1};

    // none: same device
    assert(get_transfer_strategy(r00, r00) == transfer_strategy_t::none);

    // peer_copy: same node, different device
    assert(get_transfer_strategy(r00, r01) == transfer_strategy_t::peer_copy);
    assert(get_transfer_strategy(r01, r00) == transfer_strategy_t::peer_copy);

    // mpi_send: different node
    assert(get_transfer_strategy(r00, r10) == transfer_strategy_t::mpi_send);
    assert(get_transfer_strategy(r00, r11) == transfer_strategy_t::mpi_send);
    assert(get_transfer_strategy(r10, r00) == transfer_strategy_t::mpi_send);

    std::cout << "  ✓ all transfer strategy paths covered\n";
}

int main()
{
    std::cout << "=============================================================================\n";
    std::cout << "XPU Comm Types Tests (Standalone)\n";
    std::cout << "=============================================================================\n";

    test_rank_id_basic();
    test_locality_queries();
    test_transfer_strategy();
    test_rank_id_hash();
    test_transfer_strategy_coverage();

    std::cout << "\n✓ All comm types tests passed!\n";
    return 0;
}
