#ifndef FMR_HIERARCHY_HPP
#define FMR_HIERARCHY_HPP

#include "compat.hpp"              // for real type
#include "compute/field.hpp"       // for field_t
#include "containers/vector.hpp"   // for vector_t
#include "level_descriptor.hpp"    // for level_descriptor_t
#include "mesh/mesh_config.hpp"    // for mesh_config_t
#include "utility/enums.hpp"       // for Geometry

#include <cstdint>   //   for std::uint64_t

namespace simbi::mesh::refinement::fmr {

    template <std::uint64_t Dims>
    struct mesh_hierarchy_t {
        static constexpr std::uint64_t max_levels = 8;

        vector_t<level_descriptor_t<Dims>, max_levels> levels;
        std::uint64_t num_levels{0};

        // accessors
        const level_descriptor_t<Dims>& operator[](std::uint64_t id) const
        {
            return levels[id];
        }

        level_descriptor_t<Dims>& operator[](std::uint64_t id)
        {
            return levels[id];
        }

        // queries
        constexpr bool has_refinement() const { return num_levels > 1; }
        constexpr std::uint64_t finest_level() const { return num_levels - 1; }
    };

    // fmr extensions b/c I need it now (at least I think I do)
    template <
        typename Conserved,
        typename Primitive,
        std::uint64_t Dims,
        Geometry G = Geometry::CARTESIAN>
    struct fmr_extension_t {
        using hierarchy_t = mesh::refinement::fmr::mesh_hierarchy_t<Dims>;

        hierarchy_t hierarchy;

        // per-level data (level 0 is in parent hydro_state)
        struct level_data_t {
            field_t<Conserved, Dims> cons;
            field_t<Primitive, Dims> prim;
            vector_t<field_t<Conserved, Dims>, Dims> flux;
            vector_t<field_t<real, Dims>, Dims> bstaggs;
            vector_t<field_t<real, Dims>, Dims> b_old;

            // each level gets its own mesh_config_t
            mesh::mesh_config_t<Dims, G> mesh;
        };
        // allow up to 7 refinement levels beyond base level
        vector_t<level_data_t, 7> levels;

        std::uint64_t nlevels() const { return hierarchy.num_levels - 1; }
    };

}   // namespace simbi::mesh::refinement::fmr

#endif
