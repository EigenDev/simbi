#ifndef HET_COMM_MPI_BACKEND_HPP
#define HET_COMM_MPI_BACKEND_HPP

#include <cstdint>

#ifdef MPI_ENABLED
#include "types.hpp"
#include <mpi.h>
#include <vector>
#endif

namespace simbi::het::comm {

    struct mpi_backend_t {
#ifdef MPI_ENABLED
        MPI_Comm comm_;
        std::int32_t rank_;
        std::int32_t size_;

        mpi_backend_t() : comm_(MPI_COMM_WORLD)
        {
            MPI_Comm_rank(comm_, &rank_);
            MPI_Comm_size(comm_, &size_);
        }

        explicit mpi_backend_t(MPI_Comm comm) : comm_(comm)
        {
            MPI_Comm_rank(comm_, &rank_);
            MPI_Comm_size(comm_, &size_);
        }

        std::int32_t rank() const { return rank_; }
        std::int32_t size() const { return size_; }

        // async send
        MPI_Request isend(
            const void* buf,
            std::size_t count,
            MPI_Datatype dtype,
            std::int32_t dest,
            std::int32_t tag
        )
        {
            MPI_Request req;
            MPI_Isend(buf, count, dtype, dest, tag, comm_, &req);
            return req;
        }

        // async recv
        MPI_Request irecv(
            void* buf,
            std::size_t count,
            MPI_Datatype dtype,
            std::int32_t source,
            std::int32_t tag
        )
        {
            MPI_Request req;
            MPI_Irecv(buf, count, dtype, source, tag, comm_, &req);
            return req;
        }

        // wait for requests
        void wait_all(std::vector<MPI_Request>& requests)
        {
            if (requests.empty()) {
                return;
            }
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
            requests.clear();
        }

        // test if requests complete
        bool test_all(std::vector<MPI_Request>& requests)
        {
            if (requests.empty()) {
                return true;
            }
            int flag;
            MPI_Testall(
                requests.size(),
                requests.data(),
                &flag,
                MPI_STATUSES_IGNORE
            );
            if (flag) {
                requests.clear();
            }
            return flag != 0;
        }

#else
        // stub for non-mpi builds
        std::int32_t rank() const { return 0; }
        std::int32_t size() const { return 1; }
#endif
    };

    // mpi datatype mapper
    template <typename T>
    struct mpi_type_map {
#ifdef MPI_ENABLED
        static MPI_Datatype get()
        {
            if constexpr (std::is_same_v<T, float>) {
                return MPI_FLOAT;
            }
            else if constexpr (std::is_same_v<T, double>) {
                return MPI_DOUBLE;
            }
            else if constexpr (std::is_same_v<T, int>) {
                return MPI_INT;
            }
            else if constexpr (std::is_same_v<T, long>) {
                return MPI_LONG;
            }
            else {
                return MPI_BYTE;   // fallback
            }
        }
#endif
    };

}   // namespace simbi::het::comm

#endif
