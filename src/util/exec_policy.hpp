#ifndef EXEC_POLICY_HPP
#define EXEC_POLICY_HPP

#include "build_options.hpp"   // for dim3, luint, global::col_maj, simbiStream_t
#include "device_api.hpp"      // for api::setDevice
#include <exception>           // for exception
#include <iostream>            // for operator<<, char_traits, basic_ostream
#include <vector>              // for vector

struct ExecutionException : public std::exception {
    const char* what() const throw() { return "Invalid constructor args"; }
};

namespace simbi {

    template <typename T = luint, typename U = luint>
    struct ExecutionPolicy {
        T nzones;
        dim3 gridSize;
        dim3 blockSize;
        size_t sharedMemBytes;
        std::vector<simbiStream_t> streams;
        std::vector<int> devices;

        ~ExecutionPolicy() = default;
        ExecutionPolicy()  = default;

        ExecutionPolicy(const T nzones, const U blockSize, int device = 0)
            : blockSize(dim3(blockSize)), sharedMemBytes(0)
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
            set_device(device);
        }

        ExecutionPolicy(
            const T nzones,
            const U blockSize,
            const size_t sharedMemBytes,
            int device = 0
        )
            : blockSize(dim3(blockSize)), sharedMemBytes(sharedMemBytes)
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
            set_device(device);
        }

        ExecutionPolicy(
            const T nzones,
            const U blockSize,
            const size_t sharedMemBytes,
            const std::vector<simbiStream_t>& streams,
            int device = 0
        )
            : blockSize(dim3(blockSize)),
              sharedMemBytes(sharedMemBytes),
              streams(streams)
        {
            const T nBlocks = compute_blocks(nzones, blockSize);
            this->gridSize  = dim3(nBlocks);
            set_device(device);
        }

        ExecutionPolicy(
            const std::vector<T> glist,
            const std::vector<U> blist,
            int device = 0
        )
            : sharedMemBytes(0)
        {
            if (glist.size() != blist.size()) {
                throw ExecutionException();
            }
            build_grid(glist, blist);
            set_device(device);
        }

        ExecutionPolicy(
            const std::vector<T> glist,
            const std::vector<U> blist,
            const size_t sharedMemBytes,
            int device = 0
        )
            : sharedMemBytes(sharedMemBytes)
        {
            if (glist.size() != blist.size()) {
                throw ExecutionException();
            }
            build_grid(glist, blist);
            set_device(device);
        }

        ExecutionPolicy(
            const std::vector<T> glist,
            const std::vector<U> blist,
            const size_t sharedMemBytes,
            const std::vector<simbiStream_t>& streams,
            int device = 0
        )
            : sharedMemBytes(sharedMemBytes), streams(streams)
        {
            if (glist.size() != blist.size()) {
                throw ExecutionException();
            }
            build_grid(glist, blist);
            set_device(device);
        }

        T compute_blocks(const T nzones, const luint nThreads) const
        {
            return (nzones + nThreads - 1) / nThreads;
        }

        constexpr auto get_xextent() const
        {
            if constexpr (global::col_maj) {
                return blockSize.y;
            }
            return blockSize.x;
        }

        constexpr auto get_yextent() const
        {
            if constexpr (global::col_maj) {
                return blockSize.x;
            }
            return blockSize.y;
        }

        constexpr auto get_full_extent() const
        {
            if constexpr (global::on_gpu) {
                return blockSize.z * gridSize.z * blockSize.x * blockSize.y *
                       gridSize.x * gridSize.y;
            }
            else {
                return nzones;
            }
        }

        void build_grid(const std::vector<T> glist, const std::vector<U> blist)
        {
            if (glist.size() == 1) {
                this->gridSize  = dim3((glist[0] + blist[0] - 1) / blist[0]);
                this->blockSize = dim3(blist[0]);
                this->nzones    = glist[0];
            }
            else if (glist.size() == 2) {
                this->nzones   = glist[0] * glist[1];
                luint nxBlocks = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks = (glist[1] + blist[1] - 1) / blist[1];
                if constexpr (global::col_maj) {
                    this->gridSize  = dim3(nyBlocks, nxBlocks);
                    this->blockSize = dim3(blist[1], blist[0]);
                }
                else {
                    this->gridSize  = dim3(nxBlocks, nyBlocks);
                    this->blockSize = dim3(blist[0], blist[1]);
                }
            }
            else if (glist.size() == 3) {
                this->nzones    = glist[0] * glist[1] * glist[2];
                luint nxBlocks  = (glist[0] + blist[0] - 1) / blist[0];
                luint nyBlocks  = (glist[1] + blist[1] - 1) / blist[1];
                luint nzBlocks  = (glist[2] + blist[2] - 1) / blist[2];
                this->gridSize  = dim3(nxBlocks, nyBlocks, nzBlocks);
                this->blockSize = dim3(blist[0], blist[1], blist[2]);
            }
        }

        void set_device(int device)
        {
            this->devices = {device};
            gpu::api::setDevice(device);
        }

        void set_devices(const std::vector<int>& devices)
        {
            this->devices = devices;
        }

        void add_stream(simbiStream_t stream) { streams.push_back(stream); }

        void synchronize() const
        {
            for (const auto& stream : streams) {
                gpu::api::streamSynchronize(stream);
            }
        }
    };

}   // namespace simbi

#endif