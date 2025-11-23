#ifndef GRID_PATCH_ID_HPP
#define GRID_PATCH_ID_HPP

#include "containers/vector.hpp"
#include "functional/fp.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

namespace simbi::grid {
    // -------------------------------------------------------------------------
    // patch id: unique key for a block in the amr / fmr hierarchy
    // -------------------------------------------------------------------------
    struct patch_id_t {
        // refinement Level (0 = Coarsest)
        std::int64_t level = 0;

        // we use 3d explicitly because topology is usually managed in 3d
        // integer coordinates (Logical location in the level)
        // space even if the simulation is 2d (z=0) or 1d (z=0, y=0).
        vector_t<std::int64_t, 3> coords;

        // ---------------------------------------------------------------------
        // comparison (delegates to vector_t)
        // ---------------------------------------------------------------------

        bool operator==(const patch_id_t& other) const
        {
            return level == other.level && coords == other.coords;
        }

        // lexicographical ordering for std::map
        bool operator<(const patch_id_t& other) const
        {
            if (level != other.level) {
                return level < other.level;
            }
            for (std::int64_t ii = 0; ii < 3; ++ii) {
                if (coords[ii] != other.coords[ii]) {
                    return coords[ii] < other.coords[ii];
                }
            }
            return false;
        }
    };

    // ostream op for debugging
    inline std::ostream& operator<<(std::ostream& os, const patch_id_t& id)
    {
        os << "L" << id.level << "_(";
        for (std::int64_t ii = 0; ii < 3; ++ii) {
            os << id.coords[ii];
            if (ii < 2) {
                os << ",";
            }
        }
        os << ")";
        return os;
    }

    // -------------------------------------------------------------------------
    // hashing (for std::unordered_map)
    // -------------------------------------------------------------------------
    struct patch_id_hasher {
        std::size_t operator()(const patch_id_t& id) const
        {
            // standard hash combine pattern
            std::size_t h = std::hash<std::int64_t>{}(id.level);

            for (std::int64_t ii = 0; ii < 3; ++ii) {
                h ^= std::hash<std::int64_t>{}(id.coords[ii]) + 0x9e3779b9 +
                     (h << 6) + (h >> 2);
            }

            return h;
        }
    };

}   // namespace simbi::grid

#endif   // SIMBI_GRID_PATCH_ID_HPP
