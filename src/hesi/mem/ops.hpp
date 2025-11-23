#ifndef HET_MEM_OPS_HPP
#define HET_MEM_OPS_HPP

#include "hesi/backend/transfer.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"
#include "hesi/exec/token.hpp"
#include "hesi/mem/block.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::het::mem {

    // high-level copy (infers locality from blocks)
    inline void copy(block_t& dst, const block_t& src, std::size_t bytes)
    {
        backend::copy(
            dst.data(),
            dst.locality(),
            src.data(),
            src.locality(),
            bytes
        );
    }

    // async copy
    inline exec::token_t copy_async(
        block_t& dst,
        const block_t& src,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        return backend::copy_async(
            dst.data(),
            dst.locality(),
            src.data(),
            src.locality(),
            bytes,
            stream.native(),
            stream.backend()
        );
    }

    // fill block
    inline void fill(block_t& block, std::uint8_t value)
    {
        backend::fill(block.data(), block.locality(), value, block.size());
    }

    // async fill
    inline exec::token_t
    fill_async(block_t& block, std::uint8_t value, exec::stream_t& stream)
    {
        return backend::fill_async(
            block.data(),
            block.locality(),
            value,
            block.size(),
            stream.native(),
            stream.backend()
        );
    }

    // raw pointer versions (for flexibility)
    inline void copy_raw(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes
    )
    {
        backend::copy(dst, dst_loc, src, src_loc, bytes);
    }

    inline exec::token_t copy_raw_async(
        void* dst,
        locality_t dst_loc,
        const void* src,
        locality_t src_loc,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        return backend::copy_async(
            dst,
            dst_loc,
            src,
            src_loc,
            bytes,
            stream.native(),
            stream.backend()
        );
    }

}   // namespace simbi::het::mem

#endif
