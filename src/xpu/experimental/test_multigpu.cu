// =============================================================================
// test_multigpu.cu
//
// multi-gpu test for xpu framework
// tests device affinity, context switching, and peer-to-peer transfers
//
// requirements:
//   - 2+ nvidia gpus with peer access enabled
//   - cuda 11.0+
//
// usage:
//   cd src/xpu/experimental
//   meson setup build
//   meson compile -C build
//   ./build/test_multigpu
// =============================================================================

#include "xpu/comm/transfer.hpp"
#include "xpu/comm/types.hpp"
#include "xpu/execution/cuda_space.hpp"
#include "xpu/execution/device_guard.hpp"
#include "xpu/execution/executor.hpp"
#include "xpu/mem/block.hpp"
#include "xpu/mem/device_memory.hpp"
#include "xpu/mem/memory_config.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace simbi;

// simple kernel for testing
__global__ void fill_kernel(float* data, std::size_t n, float value)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

// verification kernel
__global__ void verify_kernel(const float* data, std::size_t n, float expected, int* errors)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

void check_cuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << "cuda error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

// test 1: device affinity tracking
bool test_device_affinity()
{
    std::cout << "test 1: device affinity tracking" << std::endl;

    constexpr std::size_t n = 1024;

    // allocate on device 0
    xpu::mem::memory_block_t<xpu::device_memory_t> block0(n * sizeof(float), 0);
    if (block0.device_id() != 0) {
        std::cerr << "  fail: device_id should be 0, got " << block0.device_id() << std::endl;
        return false;
    }

    // allocate on device 1
    xpu::mem::memory_block_t<xpu::device_memory_t> block1(n * sizeof(float), 1);
    if (block1.device_id() != 1) {
        std::cerr << "  fail: device_id should be 1, got " << block1.device_id() << std::endl;
        return false;
    }

    std::cout << "  pass: device affinity tracked correctly" << std::endl;
    return true;
}

// test 2: device context guard
bool test_device_guard()
{
    std::cout << "test 2: device context guard" << std::endl;

    // start on device 0
    check_cuda(cudaSetDevice(0), "set device 0");
    int initial_device;
    check_cuda(cudaGetDevice(&initial_device), "get initial device");

    {
        // guard switches to device 1
        xpu::device_guard_t<xpu::cuda_space> guard(1);

        int current_device;
        check_cuda(cudaGetDevice(&current_device), "get current device");
        if (current_device != 1) {
            std::cerr << "  fail: guard should switch to device 1, got " << current_device
                      << std::endl;
            return false;
        }
    }

    // guard destructor restores device 0
    int restored_device;
    check_cuda(cudaGetDevice(&restored_device), "get restored device");
    if (restored_device != initial_device) {
        std::cerr << "  fail: guard should restore device " << initial_device << ", got "
                  << restored_device << std::endl;
        return false;
    }

    std::cout << "  pass: device guard restores context" << std::endl;
    return true;
}

// test 3: multi-device executors
bool test_multi_device_executors()
{
    std::cout << "test 3: multi-device executors" << std::endl;

    constexpr std::size_t n          = 1024;
    constexpr int         block_size = 256;
    constexpr int         grid_size  = (n + block_size - 1) / block_size;

    // create executors for each device
    xpu::executor_t<xpu::cuda_space> exec0(0);
    xpu::executor_t<xpu::cuda_space> exec1(1);

    // allocate memory on each device
    xpu::mem::memory_block_t<xpu::device_memory_t> block0(n * sizeof(float), 0);
    xpu::mem::memory_block_t<xpu::device_memory_t> block1(n * sizeof(float), 1);

    // fill each device with different values
    {
        xpu::device_guard_t<xpu::cuda_space> guard(0);
        fill_kernel<<<grid_size, block_size, 0, exec0.stream()>>>(
            block0.template as<float>(),
            n,
            42.0f
        );
        check_cuda(cudaGetLastError(), "launch kernel on device 0");
    }

    {
        xpu::device_guard_t<xpu::cuda_space> guard(1);
        fill_kernel<<<grid_size, block_size, 0, exec1.stream()>>>(
            block1.template as<float>(),
            n,
            99.0f
        );
        check_cuda(cudaGetLastError(), "launch kernel on device 1");
    }

    // sync both
    exec0.sync();
    exec1.sync();

    // verify values on each device
    int* errors0;
    int* errors1;
    check_cuda(cudaMalloc(&errors0, sizeof(int)), "allocate errors0");
    check_cuda(cudaMalloc(&errors1, sizeof(int)), "allocate errors1");
    check_cuda(cudaMemset(errors0, 0, sizeof(int)), "zero errors0");
    check_cuda(cudaMemset(errors1, 0, sizeof(int)), "zero errors1");

    {
        xpu::device_guard_t<xpu::cuda_space> guard(0);
        verify_kernel<<<grid_size, block_size, 0, exec0.stream()>>>(
            block0.template as<float>(),
            n,
            42.0f,
            errors0
        );
        check_cuda(cudaGetLastError(), "launch verify on device 0");
    }

    {
        xpu::device_guard_t<xpu::cuda_space> guard(1);
        verify_kernel<<<grid_size, block_size, 0, exec1.stream()>>>(
            block1.template as<float>(),
            n,
            99.0f,
            errors1
        );
        check_cuda(cudaGetLastError(), "launch verify on device 1");
    }

    exec0.sync();
    exec1.sync();

    int h_errors0, h_errors1;
    check_cuda(
        cudaMemcpy(&h_errors0, errors0, sizeof(int), cudaMemcpyDeviceToHost),
        "copy errors0"
    );
    check_cuda(
        cudaMemcpy(&h_errors1, errors1, sizeof(int), cudaMemcpyDeviceToHost),
        "copy errors1"
    );

    cudaFree(errors0);
    cudaFree(errors1);

    if (h_errors0 != 0 || h_errors1 != 0) {
        std::cerr << "  fail: verification errors (dev0=" << h_errors0 << ", dev1=" << h_errors1
                  << ")" << std::endl;
        return false;
    }

    std::cout << "  pass: independent device operations" << std::endl;
    return true;
}

// test 4: peer-to-peer transfer
bool test_peer_transfer()
{
    std::cout << "test 4: peer-to-peer transfer" << std::endl;

    constexpr std::size_t n          = 1024;
    constexpr int         block_size = 256;
    constexpr int         grid_size  = (n + block_size - 1) / block_size;

    // check if peer access is possible
    int can_access = 0;
    check_cuda(cudaDeviceCanAccessPeer(&can_access, 1, 0), "check peer access");
    if (!can_access) {
        std::cout << "  skip: peer access not available between devices 0 and 1" << std::endl;
        return true;
    }

    // enable peer access
    check_cuda(cudaSetDevice(1), "set device 1");
    check_cuda(cudaDeviceEnablePeerAccess(0, 0), "enable peer access");
    check_cuda(cudaSetDevice(0), "set device 0");

    // allocate on device 0, fill with value
    xpu::executor_t<xpu::cuda_space>               exec0(0);
    xpu::mem::memory_block_t<xpu::device_memory_t> src_block(n * sizeof(float), 0);

    {
        xpu::device_guard_t<xpu::cuda_space> guard(0);
        fill_kernel<<<grid_size, block_size, 0, exec0.stream()>>>(
            src_block.template as<float>(),
            n,
            123.0f
        );
        check_cuda(cudaGetLastError(), "fill source");
    }
    exec0.sync();

    // allocate on device 1, zero it
    xpu::executor_t<xpu::cuda_space>               exec1(1);
    xpu::mem::memory_block_t<xpu::device_memory_t> dst_block(n * sizeof(float), 1);

    {
        xpu::device_guard_t<xpu::cuda_space> guard(1);
        check_cuda(cudaMemset(dst_block.data(), 0, n * sizeof(float)), "zero dest");
    }

    // peer copy from device 0 to device 1
    xpu::comm::rank_id_t src_rank{0, 0};
    xpu::comm::rank_id_t dst_rank{0, 1};

    xpu::comm::transfer_sync(
        src_rank,
        src_block.data(),
        dst_rank,
        dst_block.data(),
        n * sizeof(float)
    );

    // verify on device 1
    int* errors;
    check_cuda(cudaMalloc(&errors, sizeof(int)), "allocate errors");
    check_cuda(cudaMemset(errors, 0, sizeof(int)), "zero errors");

    {
        xpu::device_guard_t<xpu::cuda_space> guard(1);
        verify_kernel<<<grid_size, block_size, 0, exec1.stream()>>>(
            dst_block.template as<float>(),
            n,
            123.0f,
            errors
        );
        check_cuda(cudaGetLastError(), "verify transfer");
    }
    exec1.sync();

    int h_errors;
    check_cuda(cudaMemcpy(&h_errors, errors, sizeof(int), cudaMemcpyDeviceToHost), "copy errors");
    cudaFree(errors);

    if (h_errors != 0) {
        std::cerr << "  fail: transfer verification failed (" << h_errors << " errors)"
                  << std::endl;
        return false;
    }

    std::cout << "  pass: peer-to-peer transfer correct" << std::endl;
    return true;
}

int main()
{
    std::cout << "=============================================================================\n";
    std::cout << "multi-gpu test suite\n";
    std::cout << "=============================================================================\n";

    // check device count
    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "get device count");
    std::cout << "detected " << device_count << " cuda devices\n" << std::endl;

    if (device_count < 2) {
        std::cerr << "error: this test requires at least 2 gpus\n";
        return 1;
    }

    // print device properties
    for (int ii = 0; ii < device_count; ++ii) {
        cudaDeviceProp props;
        check_cuda(cudaGetDeviceProperties(&props, ii), "get device properties");
        std::cout << "device " << ii << ": " << props.name << " (" << props.major << "."
                  << props.minor << ")" << std::endl;
    }
    std::cout << std::endl;

    // run tests
    std::vector<bool> results;

    results.push_back(test_device_affinity());
    results.push_back(test_device_guard());
    results.push_back(test_multi_device_executors());
    results.push_back(test_peer_transfer());

    // summary
    std::cout
        << "\n=============================================================================\n";
    int passed = 0;
    int failed = 0;
    for (std::size_t ii = 0; ii < results.size(); ++ii) {
        if (results[ii]) {
            ++passed;
        }
        else {
            ++failed;
        }
    }

    std::cout << "results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "=============================================================================\n";

    return failed > 0 ? 1 : 0;
}
