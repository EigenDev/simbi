#ifndef GEOMETRY_BLOCK_GEOMETRY_HPP
#define GEOMETRY_BLOCK_GEOMETRY_HPP

#include "base/concepts.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "geometry/source_terms.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::geometry {

    // -------------------------------------------------------------------------
    // motion state (device side)
    // purely value-based snapshot of the expansion at a specific time
    // -------------------------------------------------------------------------
    struct motion_state_t {
        bool is_moving;
        bool is_homologous;
        real a;       // scale factor
        real a_dot;   // expansion rate

        // helpers for physical vs comoving conversion
        DUAL real physical_len(real comoving_len) const
        {
            return is_moving ? comoving_len * a : comoving_len;
        }

        DUAL real comoving_len(real physical_len) const
        {
            return is_moving ? physical_len / a : physical_len;
        }

        DUAL real grid_velocity(real coord) const
        {
            if (!is_moving) {
                return 0.0;
            }
            if (is_homologous) {
                return (a_dot / a) * coord * a;   // v = H * r_phys
            }
            return a_dot;   // uniform translation
        }
    };

    // -------------------------------------------------------------------------
    // block geometry
    // composes a specific metric implementation with the global motion state
    // -------------------------------------------------------------------------
    template <typename Metric>
    struct block_geometry_t {
        using metric_type = Metric;
        Metric metric;
        motion_state_t motion;

        DUAL block_geometry_t(Metric m, motion_state_t s) : metric(m), motion(s)
        {
        }

        // ---------------------------------------------------------------------
        // metric forwarding (comoving coordinates)
        // ---------------------------------------------------------------------

        template <std::uint64_t Rank>
        DUAL auto volume(const iarray<Rank>& idx) const
        {
            return metric.volume(idx);
        }

        template <std::uint64_t Rank>
        DUAL auto face_area(const iarray<Rank>& idx, std::size_t dim) const
        {
            return metric.face_area(idx, dim);
        }

        template <std::uint64_t Rank>
        DUAL auto centroid(const iarray<Rank>& idx) const
        {
            return metric.centroid(idx);
        }

        // ---------------------------------------------------------------------
        // geometric source term helpers
        // returns the geometric source terms {g_1, g_2, g_3} needed for
        // momentum source update in curvilinear coordinates
        // ---------------------------------------------------------------------
        template <std::uint64_t Rank, is_hydro_primitive_c prim_t>
        DUAL auto geomtric_source_factors(
            const prim_t& prims,
            real gamma,
            const iarray<Rank>& idx
        ) const
        {
            return geometric_source_terms(prims, gamma, idx, metric);
        }

        // ---------------------------------------------------------------------
        // physical scale factors (h_i)
        // used for gradient calc and cfl condition
        // ---------------------------------------------------------------------
        template <std::uint64_t Rank>
        DUAL auto scale_factors(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            auto h = metric.scale_factors(idx);

            // if mesh is expanding homologously, physical length scales with
            // a(t) ds_physical = a(t) * h_comoving * dx
            if (motion.is_moving) {
                for (std::size_t dd = 0; dd < Rank; ++dd) {
                    h[dd] *= motion.a;
                }
            }
            return h;
        }

        // ---------------------------------------------------------------------
        // grid velocity at face center
        // used for flux correction F(u - v_g)
        // ---------------------------------------------------------------------
        template <std::uint64_t Rank>
        DUAL real face_grid_velocity(
            const simbi::vector_t<int64_t, Rank>& idx,
            std::size_t dim
        ) const
        {
            if (!motion.is_moving) {
                return 0.0;
            }

            // get coordinate at the face
            real coord_comoving = 0.0;
            coord_comoving      = metric.face_position(idx, dim);

            // v_grid = v_peculiar + H * r_phys
            // usually just homologous expansion: v = (a_dot / a) * r_phys
            // r_phys = r_comoving * a
            // -> v = a_dot * r_comoving

            return motion.a_dot * coord_comoving;
        }
    };

    // -------------------------------------------------------------------------
    // factory helper
    // -------------------------------------------------------------------------
    template <typename Metric>
    auto block_geometry(Metric m, motion_state_t s)
    {
        return block_geometry_t<Metric>(m, s);
    }

}   // namespace simbi::geometry

#endif   // GRID_GEOMETRY_BLOCK_GEOMETRY_HPP
