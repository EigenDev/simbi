#ifndef DATA_CTX_HPP
#define DATA_CTX_HPP

#include "base/concepts.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
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
