#ifndef GEOMETRY_BLOCK_GEOMETRY_HPP
#define GEOMETRY_BLOCK_GEOMETRY_HPP

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "geometry/metrics.hpp"
#include "geometry/source_terms.hpp"

#include <cstddef>
#include <cstdint>

namespace simbi::geometry {

    // =========================================================================
    // moving mesh coordinate system
    // =========================================================================
    //
    // simbi uses a **comoving coordinate system** fgeometry:hor moving meshes:
    //
    // comoving coordinates (r_com):
    //   - stored in mesh config (xmin, xmax, cell centers)
    //   - fixed throughout simulation
    //   - returned by metric.centroid(idx), metric.face_position(idx, dim)
    //
    // physical coordinates (r_phys):
    //   - actual spatial positions at time t
    //   - r_phys(t) = a(t) * r_com
    //   - computed on-the-fly using block_geometry_t methods
    //
    // scale factor a(t):
    //   - a(t=0) typically = 1.0
    //   - updated every timestep via motion_state_t snapshot
    //   - expansion rate: H(t) = a_dot / a
    //
    // grid velocity:
    //   - v_grid = a_dot * r_com (homologous expansion)
    //   - used in flux calculations: F(u, v_grid)
    //   - affects wave speeds and cfl condition
    //
    // usage pattern:
    //   auto motion = sim.mesh_motion();  // {a(t), a_dot(t), ...}
    //   with_block_geometry(mesh_cfg, motion, [&](auto& geo) {
    //       auto r_com  = geo.centroid(idx);           // comoving position
    //       auto r_phys = geo.physical_centroid(idx);  // a(t) * r_com
    //       auto v_grid = geo.face_grid_velocity(idx, dim);  // a_dot * r_com
    //   });
    //
    // =========================================================================

    // -------------------------------------------------------------------------
    // motion state (device side)
    // purely value-based snapshot of the expansion at a specific time
    // -------------------------------------------------------------------------
    struct motion_state_t
    {
        bool is_moving;
        bool is_homologous;
        real a;     // scale factor
        real a_dot; // expansion rate

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
                return (a_dot / a) * coord * a; // v = H * r_phys
            }
            return a_dot; // uniform translation
        }

        static motion_state_t static_mesh()
        {
            return geometry::motion_state_t{
                .is_moving     = false,
                .is_homologous = false,
                .a             = 1.0,
                .a_dot         = 0.0
            };
        }
    };

    // -------------------------------------------------------------------------
    // block geometry
    // composes a specific metric implementation with the global motion state
    // -------------------------------------------------------------------------
    template <typename Metric>
    struct block_geometry_t
    {
        using metric_type = Metric;
        Metric         metric;
        motion_state_t motion;

        DUAL block_geometry_t(Metric m, motion_state_t s) : metric(m), motion(s) {}

        // ---------------------------------------------------------------------
        // metric forwarding (comoving coordinates)
        // ---------------------------------------------------------------------

        template <std::uint64_t Rank>
        DUAL auto volume(const iarray<Rank>& idx) const
        {
            real v_comoving = metric.volume(idx);
            if (!motion.is_moving) {
                return v_comoving;
            }

            // physical volume = v_comoving * a(t)^n
            // where n depends on which coordinates scale with a(t)
            real a_factor = 1.0;
            if constexpr (is_spherical_c<Metric>) {
                // V ~ r^3, only radial scales
                a_factor = motion.a * motion.a * motion.a;
            }
            else if constexpr (is_cylindrical_c<Metric>) {
                // V ~ r^2, only radial scales
                a_factor = motion.a * motion.a;
            }
            else {
                // cartesian: all dimensions scale
                a_factor = motion.a;
                for (std::uint64_t ii = 1; ii < Rank; ++ii) {
                    a_factor *= motion.a;
                }
            }
            return v_comoving * a_factor;
        }

        template <std::uint64_t Rank>
        DUAL auto volume_scaling(const iarray<Rank>& idx) const
        {
            if (!motion.is_moving) {
                return 1.0;
            }
            return volume(idx);
        }

        template <std::uint64_t Rank>
        DUAL auto face_area(const iarray<Rank>& idx, std::size_t dim) const
        {
            real area_comoving = metric.face_area(idx, dim);
            if (!motion.is_moving) {
                return area_comoving;
            }

            // physical area = area_comoving * a(t)^m
            // where m depends on geometry and which face
            real a_factor = 1.0;
            if constexpr (is_spherical_c<Metric>) {
                // all faces scale as r^2
                a_factor = motion.a * motion.a;
            }
            else if constexpr (is_cylindrical_c<Metric>) {
                // index ordering: [z, phi, r] for 3d, [phi, r] for 2d, [r] for 1d
                constexpr std::size_t radial_dim = Rank - 1;
                if (dim == radial_dim) {
                    // r-face: area ~ r
                    a_factor = motion.a;
                }
                else if constexpr (Rank == 3) {
                    if (dim == 0) {
                        // z-face: area ~ r
                        a_factor = motion.a;
                    }
                    // phi-face (dim==1): no scaling
                }
                // for 2d [phi, r]: phi-face (dim==0) doesn't scale
                // for 1d [r]: no orthogonal faces
            }
            else {
                // cartesian: area ~ a^(Rank-1)
                a_factor = motion.a;
                for (std::uint64_t ii = 1; ii < Rank - 1; ++ii) {
                    a_factor *= motion.a;
                }
            }
            return area_comoving * a_factor;
        }

        template <std::uint64_t Rank>
        DUAL auto centroid(const iarray<Rank>& idx) const
        {
            return metric.centroid(idx);
        }

        // physical centroid: comoving coords scaled by a(t)
        template <std::uint64_t Rank>
        DUAL auto physical_centroid(const iarray<Rank>& idx) const
        {
            auto coords = metric.centroid(idx);
            if (motion.is_moving) {
                for (std::size_t dd = 0; dd < Rank; ++dd) {
                    coords[dd] *= motion.a;
                }
            }
            return coords;
        }

        // physical scale: multiply comoving value by a(t)
        DUAL real to_physical(real comoving_value) const
        {
            return motion.is_moving ? comoving_value * motion.a : comoving_value;
        }

        // comoving scale: divide physical value by a(t)
        DUAL real to_comoving(real physical_value) const
        {
            return motion.is_moving ? physical_value / motion.a : physical_value;
        }

        // ---------------------------------------------------------------------
        // geometric source term helpers
        // returns the geometric source terms {g_1, g_2, g_3} needed for
        // momentum source update in curvilinear coordinates
        // ---------------------------------------------------------------------
        template <std::uint64_t Rank, is_hydro_primitive_c prim_t>
        DUAL auto
        geomtric_source_factors(const prim_t& prims, real gamma, const iarray<Rank>& idx) const
        {
            return geometric_source_terms(prims, gamma, idx, metric);
        }

        // ---------------------------------------------------------------------
        // physical scale factors (h_i)
        // used for gradient calc and cfl condition
        // ---------------------------------------------------------------------
        template <std::uint64_t Rank>
        DUAL auto scale_factors(const iarray<Rank>& idx) const
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
        DUAL real face_grid_velocity(const iarray<Rank>& idx, std::size_t dim) const
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

} // namespace simbi::geometry

#endif // GRID_GEOMETRY_BLOCK_GEOMETRY_HPP
