#ifndef HET_EXEC_REDUCE_HPP
#define HET_EXEC_REDUCE_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "hesi/backend/reduce.hpp"
#include "hesi/backend/transfer.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/executor.hpp"
#include "hesi/exec/token.hpp"
#include "hesi/mem/block.hpp"
#include "hesi/mem/ops.hpp"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace simbi::het::exec {
    // =========================================================================
    // concept: things that can receive reduction results
    // =========================================================================
    template <typename T>
    concept result_buffer_c = requires(T& buf) {
        { buf.data() } -> std::convertible_to<void*>;
        { buf.locality() } -> std::same_as<locality_t>;
    };

    // =========================================================================
    // form 1: reduce with computation object
    // used when we have lazy expressions (field operations)
    // =========================================================================
    template <
        typename Computation,
        typename T,
        typename BinaryOp,
        typename Result>
        requires result_buffer_c<Result>
    token_t reduce(
        executor_t& exec,
        const Computation& comp,   // has .domain() method and operator()
        Result& result_buffer,     // field_t or block_t
        T init,
        BinaryOp op,
        T identity = T{}
    )
    {
        // empty domain check
        if (comp.domain().empty()) {
            if (exec.backend() == backend_type_t::cpu) {
                *static_cast<T*>(result_buffer.data()) = init;
            }
            else {
                return backend::copy_async(
                    result_buffer.data(),
                    result_buffer.locality(),
                    &init,
                    locality_t::host(),
                    sizeof(T),
                    exec.stream().native(),
                    exec.backend()
                );
            }
            return token_t::immediate(exec.backend());
        }

        // dispatch to backend (no transform, identity transform)
        backend::transform_reduce(
            exec.backend(),
            exec.stream().native(),
            comp,
            static_cast<T*>(result_buffer.data()),
            init,
            [] DEV(const auto& val) { return val; },   // identity transform
            op,
            identity
        );

        // record completion
        auto token = token_t::create(exec.backend());
        token.event_->record(exec.stream());
        return token;
    }

    // =========================================================================
    // form 2: reduce with domain + mapper
    // used when we compute on-the-fly from coordinates
    // =========================================================================
    template <
        std::uint64_t Rank,
        typename T,
        typename Mapper,
        typename BinaryOp,
        typename Result>
        requires result_buffer_c<Result>
    token_t reduce(
        executor_t& exec,
        const grid::domain_t<Rank>& domain,
        Result& result_buffer,
        T init,
        Mapper mapper,
        BinaryOp op,
        T identity = T{}
    )
    {
        // wrapper mapper in computation interface
        struct computation_wrapper_t {
            grid::domain_t<Rank> dom;
            Mapper map;

            // static constexpr std::uint64_t rank = Rank;
            DUAL auto operator()(iarray<Rank> coord) const
            {
                return map(coord);
            }

            const grid::domain_t<Rank>& domain() const { return dom; }
        };

        computation_wrapper_t comp{domain, mapper};

        // delegate to Form 1
        return reduce(exec, comp, result_buffer, init, op, identity);
    }

    // =========================================================================
    // form 3: transform-reduce (general)
    // =========================================================================
    template <
        typename Computation,
        typename T,
        typename TransformOp,
        typename BinaryOp,
        typename Result>
        requires result_buffer_c<Result>
    token_t transform_reduce(
        executor_t& exec,
        const Computation& comp,
        Result& result_buffer,
        T init,
        TransformOp transform,
        BinaryOp op,
        T identity = T{}
    )
    {
        if (comp.domain().empty()) {
            if (exec.backend() == backend_type_t::cpu) {
                *static_cast<T*>(result_buffer.data()) = init;
            }
            else {
                return backend::copy_async(
                    result_buffer.data(),
                    result_buffer.locality(),
                    &init,
                    locality_t::host(),
                    sizeof(T),
                    exec.stream().native(),
                    exec.backend()
                );
            }
            return token_t::immediate(exec.backend());
        }

        backend::transform_reduce(
            exec.backend(),
            exec.stream().native(),
            comp,
            static_cast<T*>(result_buffer.data()),
            init,
            transform,
            op,
            identity
        );

        auto token = token_t::create(exec.backend());
        token.event_->record(exec.stream());
        return token;
    }

    // =========================================================================
    // helper: synchronous reduce (returns value directly)
    // for cases where i just want the answer now
    // =========================================================================
    template <
        std::uint64_t Rank,
        typename T,
        typename Mapper,
        typename BinaryOp>
    T reduce_sync(
        executor_t& exec,
        const grid::domain_t<Rank>& domain,
        T init,
        Mapper mapper,
        BinaryOp op,
        T identity = T{}
    )
    {
        // allocate result buffer
        auto result_loc = (exec.backend() == backend_type_t::cpu)
                              ? locality_t::host()
                              : exec.stream().locality();

        mem::block_t result_block(
            sizeof(T),
            result_loc,
            memory_type_t::host_visible
        );

        // launch reduction
        auto token =
            reduce(exec, domain, result_block, init, mapper, op, identity);
        token.synchronize();

        // copy to host if needed
        if (exec.backend() != backend_type_t::cpu) {
            T host_result;
            mem::copy_raw(
                &host_result,
                locality_t::host(),
                result_block.data(),
                result_block.locality(),
                sizeof(T)
            );
            return host_result;
        }
        else {
            return *result_block.as<T>();
        }
    }

    // =========================================================================
    // raw pointer versions (for boomers)
    // =========================================================================
    template <typename Computation, typename T, typename BinaryOp>
    token_t reduce_raw(
        executor_t& exec,
        const Computation& comp,
        T* result_ptr,
        locality_t result_loc,
        T init,
        BinaryOp op,
        T identity = T{}
    )
    {
        if (comp.domain().empty()) {
            if (exec.backend() == backend_type_t::cpu) {
                *result_ptr = init;
            }
            else {
                return backend::copy_async(
                    result_ptr,
                    result_loc,
                    &init,
                    locality_t::host(),
                    sizeof(T),
                    exec.stream().native(),
                    exec.backend()
                );
            }
            return token_t::immediate(exec.backend());
        }

        backend::transform_reduce(
            exec.backend(),
            exec.stream().native(),
            comp,
            result_ptr,
            init,
            [](const auto& val) { return val; },
            op,
            identity
        );

        auto token = token_t::create(exec.backend());
        token.event_->record(exec.stream());
        return token;
    }

}   // namespace simbi::het::exec

#endif
