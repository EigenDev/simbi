#ifndef SIMBI_DATA_CTX_HPP
#define SIMBI_DATA_CTX_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "data/containers/vector.hpp"
#include <cstdint>

namespace simbi {
    template <std::uint64_t Dims>
    struct physics_context_t {
        real gamma;
        real dt;
        vector_t<real, Dims> cell_pos;
        real cell_volume;
        real min_cell_width;
        real max_cell_width;
    };
}   // namespace simbi
#endif
