#ifndef PHYSICAL_LAYOUT_HPP
#define PHYSICAL_LAYOUT_HPP

#include "compute/data_handle.hpp"
#include "domain/domain.hpp"
#include "memory/accessor.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace simbi {
    // one contiguous chunk of memory on one device.
    // this is the concrete "physical" part of a field.
    template <typename T, std::uint64_t Dims>
    struct partition_t {
        domain_t<Dims> domain;
        mem::accessor_t<T, Dims> accessor;
    };

    // a collection of partitions. this describes the
    // complete physical layout of a single distributed field.
    template <typename T, std::uint64_t Dims>
    using partition_list_t = std::vector<partition_t<T, Dims>>;

    // the central "lookup table" for all physical data.
    // it maps a logical handle to its distributed physical layout.
    // the shared_ptr<void> is a type-erased partition_list_t<T, Dims>.
    using data_layout_map_t = std::map<data_handle_t, std::shared_ptr<void>>;
}   // namespace simbi
#endif
