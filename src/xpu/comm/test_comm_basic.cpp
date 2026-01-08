// =============================================================================
// test_comm_basic.cpp
//
// basic tests for xpu comm layer types and locality queries
// validates rank identification and transfer strategy selection
// =============================================================================

#include "xpu/xpu.hpp"

#include <cassert>
#include <iostream>

using namespace simbi::xpu;
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

void test_halo_transfer_descriptor()
{
    std::cout << "testing halo_transfer_t descriptor...\n";

    rank_id_t src{0, 0};
    rank_id_t dst{0, 1};

    float data_src[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float data_dst[10] = {};

    halo_transfer_t transfer{src, data_src, dst, data_dst, sizeof(data_src)};

    // descriptor created successfully
    assert(transfer.src_rank == src);
    assert(transfer.dst_rank == dst);
    assert(transfer.bytes == sizeof(data_src));

    std::cout << "  ✓ halo_transfer_t descriptor works\n";
}

int main()
{
    std::cout << "=============================================================================\n";
    std::cout << "XPU Comm Basic Tests\n";
    std::cout << "=============================================================================\n";

    test_rank_id_basic();
    test_locality_queries();
    test_transfer_strategy();
    test_halo_transfer_descriptor();

    std::cout << "\n✓ All comm basic tests passed!\n";
    return 0;
}
