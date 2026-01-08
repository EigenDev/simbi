#ifndef HET_MEM_OPS_HPP
#define HET_MEM_OPS_HPP

#include "hesi/backend/transfer.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/stream.hpp"
#include "hesi/exec/token.hpp"
#include "hesi/mem/block.hpp"
#include "hesi/mem/rc.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

    // -------------------------------------------------------------------------
    // handle-oriented helpers (operate on handle_t<block_t>)
    // these keep high-level field logic out of the low-level het:: mem layer.
    // -------------------------------------------------------------------------

    // allocate a ref-counted handle for a block
    inline handle_t<block_t> make_handle(
        std::size_t bytes,
        locality_t loc,
        memory_type_t mem_type = memory_type_t::device_local
    )
    {
        return handle_t<block_t>::make(bytes, loc, mem_type);
    }

    // async copy between handles (honors each handle's locality)
    // semantics:
    //  - schedule device/host copy on provided stream
    //  - store a pending_transfer token in the destination control block
    //    and set device_dirty = true while transfer is in-flight
    //  - do not flip authoritative_loc until transfer completion is observed
    inline exec::token_t copy_handle_async(
        handle_t<block_t> dst,
        const handle_t<block_t>& src,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        if (!dst || !src) {
            throw std::runtime_error("invalid handle for copy_handle_async");
        }

        // schedule the async transfer using block-oriented helper
        auto tok = copy_async(*dst, *src, bytes, stream);

        // record pending transfer in destination control block
        auto* dst_cb = dst.control_block();
        if (dst_cb) {
            // capture native handle/backend before moving ownership
            auto native_h = tok.native();
            auto native_b = tok.backend();
            std::lock_guard<std::mutex> lk(dst_cb->meta_mutex);
            dst_cb->pending_transfer = std::move(tok);
            dst_cb->device_dirty     = true;
            // do not change authoritative_loc here
            // return a non-owning view referencing the same native event
            return exec::token_t(native_h, native_b, false);
        }

        // no control block to own the token; return the created token
        return tok;
    }

    // blocking copy between handles
    // uses async path then synchronizes and finalizes metadata
    inline void copy_handle_sync(
        handle_t<block_t> dst,
        const handle_t<block_t>& src,
        std::size_t bytes
    )
    {
        if (!dst || !src) {
            throw std::runtime_error("invalid handle for copy_handle_sync");
        }

        // perform async copy on a default stream and wait for completion
        exec::stream_t s = exec::make_a_default_stream();
        auto tok         = copy_handle_async(dst, src, bytes, s);
        tok.synchronize();

        // finalize destination metadata: clear pending_transfer and flip
        // authoritative locality to dst
        auto* dst_cb = dst.control_block();
        if (dst_cb) {
            std::lock_guard<std::mutex> lk(dst_cb->meta_mutex);
            dst_cb->pending_transfer  = std::nullopt;
            dst_cb->authoritative_loc = dst->locality();
            dst_cb->host_dirty =
                (dst_cb->authoritative_loc.backend == backend_type_t::cpu);
            dst_cb->device_dirty = !dst_cb->host_dirty;
        }
    }

    // copy from raw host pointer into handle (async)
    inline exec::token_t copy_from_host_async(
        handle_t<block_t> dst,
        const void* host_ptr,
        std::size_t bytes,
        exec::stream_t& stream
    )
    {
        if (!dst) {
            throw std::runtime_error(
                "invalid dst handle for copy_from_host_async"
            );
        }

        // host locality
        locality_t host_loc = locality_t::host();
        return copy_raw_async(
            dst->data(),
            dst->locality(),
            host_ptr,
            host_loc,
            bytes,
            stream
        );
    }

    // ensure host has an up-to-date copy (blocking)
    // semantics:
    //  - if there is a pending_transfer recorded, wait for it to complete
    //  - if device is authoritative or device_dirty is set, perform a
    //  device->host
    //    copy and block until complete, then mark host authoritative
    inline void ensure_host_sync(const handle_t<block_t>& handle)
    {
        if (!handle) {
            return;
        }

        if (handle->locality().backend == backend_type_t::cpu) {
            return;   // already host-local
        }

        auto* cb = handle.control_block();
        exec::token_t pending_tok;
        bool had_pending = false;

        if (cb) {
            // capture and clear pending_transfer under lock so other threads
            // see that the pending transfer is being observed here
            std::unique_lock<std::mutex> lk(cb->meta_mutex);
            if (cb->pending_transfer) {
                pending_tok = std::move(*cb->pending_transfer);
                cb->pending_transfer.reset();
                had_pending = true;
            }
            // if device is not authoritative and no pending transfer, still may
            // need to copy down below
        }

        // wait for any in-flight transfer to complete
        if (had_pending && pending_tok) {
            pending_tok.synchronize();
        }

        // if device is authoritative or device_dirty, copy device -> host
        if (cb) {
            bool need_copy = false;
            {
                std::lock_guard<std::mutex> lk(cb->meta_mutex);
                if (cb->authoritative_loc.backend != backend_type_t::cpu ||
                    cb->device_dirty) {
                    need_copy = true;
                }
            }

            if (need_copy) {
                // allocate temporary host-visible block and copy device -> host
                block_t host_tmp(
                    handle->size(),
                    locality_t::host(),
                    memory_type_t::host_visible
                );
                exec::stream_t s = exec::make_a_default_stream();
                auto tok         = copy_raw_async(
                    host_tmp.data(),
                    host_tmp.locality(),
                    handle->data(),
                    handle->locality(),
                    handle->size(),
                    s
                );
                tok.synchronize();

                // update metadata: host becomes authoritative
                std::lock_guard<std::mutex> lk(cb->meta_mutex);
                cb->authoritative_loc = locality_t::host();
                cb->host_dirty        = false;
                cb->device_dirty      = false;
                cb->pending_transfer.reset();
                // note: a full implementation might swap host_tmp into the
                // handle rather than copying again; this preserves simple
                // semantics.
            }
        }
    }

    // evict device mirror for the specified device locality (non-blocking)
    // behavior:
    //  - wait for any in-flight transfer that touches this handle
    //  - if the device holds the authoritative copy, ensure host copy first
    //  - free the device-local allocation if present and update metadata
    inline void
    evict_device(const handle_t<block_t>& handle, locality_t device_loc)
    {
        if (!handle) {
            return;
        }

        auto* cb = handle.control_block();
        if (!cb) {
            return;
        }

        // if there is an in-flight transfer, wait for it to complete
        {
            std::unique_lock<std::mutex> lk(cb->meta_mutex);
            if (cb->pending_transfer) {
                auto pending = std::move(*cb->pending_transfer);
                // clear entry to indicate this thread is observing/serializing
                cb->pending_transfer.reset();
                lk.unlock();
                if (pending) {
                    pending.synchronize();
                }
                lk.lock();
            }
        }

        // if the device currently holds the authoritative copy, ensure host
        // copy
        if (cb->authoritative_loc.backend == device_loc.backend &&
            cb->authoritative_loc.device_id == device_loc.device_id) {
            // ensure host has latest data before evicting device mirror
            ensure_host_sync(handle);
        }

        // if the underlying block is located on the target device, deallocate
        // it
        {
            block_t& blk = cb->data;
            if (blk.locality().backend == device_loc.backend &&
                blk.locality().device_id == device_loc.device_id &&
                blk.data() != nullptr) {
                backend::deallocate(
                    blk.locality().backend,
                    blk.data(),
                    blk.memory_type()
                );
                // clear block fields to reflect freed storage
                // these members are public in block_t
                blk.ptr_   = nullptr;
                blk.bytes_ = 0;
                blk.loc_   = locality_t::host();
                blk.type_  = memory_type_t::host_visible;
            }
        }

        // update metadata to reflect that host is now authoritative
        {
            std::lock_guard<std::mutex> lk(cb->meta_mutex);
            if (cb->authoritative_loc.backend == device_loc.backend &&
                cb->authoritative_loc.device_id == device_loc.device_id) {
                cb->authoritative_loc = locality_t::host();
                cb->host_dirty        = true;
                cb->device_dirty      = false;
            }
            // ensure no dangling pending token remains
            cb->pending_transfer.reset();
        }
    }

    // evict all device mirrors (blocking)
    // behavior:
    //  - ensure host copy is present (calls ensure_host_sync)
    //  - free any device-local allocation associated with this handle
    //  - clear device-related metadata
    inline void evict_device_all(const handle_t<block_t>& handle)
    {
        if (!handle) {
            return;
        }

        auto* cb = handle.control_block();
        if (!cb) {
            return;
        }

        // wait for and clear any pending transfer
        {
            std::unique_lock<std::mutex> lk(cb->meta_mutex);
            if (cb->pending_transfer) {
                auto pending = std::move(*cb->pending_transfer);
                cb->pending_transfer.reset();
                lk.unlock();
                if (pending) {
                    pending.synchronize();
                }
            }
        }

        // ensure any authoritative device data is copied to host before
        // eviction
        ensure_host_sync(handle);

        // free device-local storage if present
        {
            block_t& blk = cb->data;
            if (blk.locality().backend != backend_type_t::cpu &&
                blk.data() != nullptr) {
                backend::deallocate(
                    blk.locality().backend,
                    blk.data(),
                    blk.memory_type()
                );
                blk.ptr_   = nullptr;
                blk.bytes_ = 0;
                blk.loc_   = locality_t::host();
                blk.type_  = memory_type_t::host_visible;
            }
        }

        // clear metadata
        {
            std::lock_guard<std::mutex> lk(cb->meta_mutex);
            cb->authoritative_loc = locality_t::host();
            cb->host_dirty        = true;
            cb->device_dirty      = false;
            cb->pending_transfer.reset();
        }
    }

}   // namespace simbi::het::mem

#endif
