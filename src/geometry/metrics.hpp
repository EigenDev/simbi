#ifndef SIMBI_GEOMETRY_METRIC_HPP
#define SIMBI_GEOMETRY_METRIC_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace simbi::geometry {
    struct dummy_map {
        DUAL constexpr real width(int) const { return 1.0; }
        DUAL constexpr real center(int) const { return 0.0; }
        DUAL constexpr real face(int) const { return 0.0; }

        // call operator returns cell_interval_t-like struct
        struct dummy_interval {
            real width{1.0};
        };
        DUAL constexpr dummy_interval operator()(int) const { return {}; }
    };

    // =========================================================================
    // cartesian metric
    // orthogonal, flat. dv = dx * dy * dz
    // =========================================================================
    template <
        typename Map1,
        typename Map2 = dummy_map,
        typename Map3 = dummy_map>
    struct cartesian_metric_t {
        Map1 x1_map;
        Map2 x2_map;
        Map3 x3_map;

        DUAL constexpr cartesian_metric_t(Map1 m1, Map2 m2, Map3 m3)
            : x1_map(m1), x2_map(m2), x3_map(m3)
        {
        }

        DUAL constexpr cartesian_metric_t(Map1 m1, Map2 m2)
            : x1_map(m1), x2_map(m2), x3_map(dummy_map{})
        {
        }

        DUAL constexpr cartesian_metric_t(Map1 m1)
            : x1_map(m1), x2_map(dummy_map{}), x3_map(dummy_map{})
        {
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto
        scale_factors(const simbi::vector_t<int64_t, Rank>& /*idx*/) const
        {
            vector_t<real, Rank> h;
            h.fill(1.0);
            return h;
        }

        // coordinate-space cell widths (dx, dy, dz)
        template <std::uint64_t Rank>
        DUAL constexpr auto
        cell_widths(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            vector_t<real, Rank> w;
            w[Rank - 1] = x1_map(idx[Rank - 1]).width;
            if constexpr (Rank > 1) {
                w[Rank - 2] = x2_map(idx[Rank - 2]).width;
            }
            if constexpr (Rank > 2) {
                w[Rank - 3] = x3_map(idx[Rank - 3]).width;
            }
            return w;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real face_position(
            const simbi::vector_t<int64_t, Rank>& idx,
            std::size_t dim
        ) const
        {
            constexpr std::uint64_t x1c = Rank - 1;
            constexpr std::uint64_t x2c = Rank - 2;
            constexpr std::uint64_t x3c = Rank - 3;

            // x1-face
            if (dim == x1c) {
                return x1_map.face(idx[x1c]);
            }
            // x2-face
            if constexpr (Rank > 1) {
                if (dim == x2c) {
                    return x2_map.face(idx[x2c]);
                }
            }
            // x3-face
            // (only if Rank > 2)
            if constexpr (Rank > 2) {
                if (dim == x3c) {
                    return x3_map.face(idx[x3c]);
                }
            }
            return 0.0;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real volume(const iarray<Rank>& idx) const
        {
            real v = x1_map(idx[Rank - 1]).width;
            if constexpr (Rank > 1) {
                v *= x2_map(idx[Rank - 2]).width;
            }
            if constexpr (Rank > 2) {
                v *= x3_map(idx[Rank - 3]).width;
            }
            return v;
        }

        // returns area of the LEFT face normal to 'dim'
        template <std::uint64_t Rank>
        DUAL constexpr real
        face_area(const iarray<Rank>& idx, std::size_t dim) const
        {
            real area = 1.0;
            if (dim != 0) {
                area *= x1_map(idx[Rank - 1]).width;
            }
            if constexpr (Rank > 1) {
                if (dim != 1) {
                    area *= x2_map(idx[Rank - 2]).width;
                }
            }
            if constexpr (Rank > 2) {
                if (dim != 2) {
                    area *= x3_map(idx[Rank - 3]).width;
                }
            }
            return area;
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto centroid(const iarray<Rank>& idx) const
        {
            vector_t<real, Rank> c;
            constexpr std::uint64_t x1c = Rank - 1;
            constexpr std::uint64_t x2c = Rank - 2;
            constexpr std::uint64_t x3c = Rank - 3;

            c[x1c] = x1_map.center(idx[x1c]);
            if constexpr (Rank > 1) {
                c[x2c] = x2_map.center(idx[x2c]);
            }
            if constexpr (Rank > 2) {
                c[x3c] = x3_map.center(idx[x3c]);
            }
            return c;
        }

        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        to_cartesian(const vector_t<real, Rank>& vec) const
        {
            // identity
            return vec;
        }
        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        from_cartesian(const vector_t<real, Rank>& vec) const
        {
            // identity
            return vec;
        }
    };

    // =========================================================================
    // spherical metric (r, theta, phi)
    // dv = r^2 sin(theta) dr dtheta dphi
    // =========================================================================
    template <
        typename MapR,
        typename MapTheta = dummy_map,
        typename MapPhi   = dummy_map>
    struct spherical_metric_t {
        MapR r_map;
        MapTheta theta_map;
        MapPhi phi_map;

        DUAL constexpr spherical_metric_t(MapR mr)
            : r_map(mr), theta_map(dummy_map{}), phi_map(dummy_map{})
        {
        }

        DUAL constexpr spherical_metric_t(MapR mr, MapTheta mt)
            : r_map(mr), theta_map(mt), phi_map(dummy_map{})
        {
        }

        DUAL constexpr spherical_metric_t(MapR mr, MapTheta mt, MapPhi mp)
            : r_map(mr), theta_map(mt), phi_map(mp)
        {
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto
        scale_factors(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            vector_t<real, Rank> h;
            h.fill(1.0);

            // radial (h_r = 1)

            // theta (h_theta = r)
            if constexpr (Rank > 1) {
                h[Rank - 2] = r_map.center(idx[Rank - 1]);
            }

            // phi (h_phi = r sin theta)
            if constexpr (Rank > 2) {
                real r      = r_map.center(idx[Rank - 1]);
                real theta  = theta_map.center(idx[Rank - 2]);
                h[Rank - 3] = r * std::sin(theta);
            }
            return h;
        }

        // coordinate-space cell widths (dr, dtheta, dphi)
        template <std::uint64_t Rank>
        DUAL constexpr auto
        cell_widths(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            vector_t<real, Rank> w;
            w[Rank - 1] = r_map(idx[Rank - 1]).width;
            if constexpr (Rank > 1) {
                w[Rank - 2] = theta_map(idx[Rank - 2]).width;
            }
            if constexpr (Rank > 2) {
                w[Rank - 3] = phi_map(idx[Rank - 3]).width;
            }
            return w;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real face_position(
            const simbi::vector_t<int64_t, Rank>& idx,
            std::size_t dim
        ) const
        {
            constexpr std::uint64_t rc = Rank - 1;
            constexpr std::uint64_t tc = Rank - 2;
            constexpr std::uint64_t pc = Rank - 3;

            // r-face
            if (dim == rc) {
                return r_map.face(idx[rc]);
            }

            // theta-face
            if constexpr (Rank > 1) {
                if (dim == tc) {
                    return theta_map.face(idx[tc]);
                }
            }

            // phi-face
            if constexpr (Rank > 2) {
                if (dim == pc) {
                    return phi_map.face(idx[pc]);
                }
            }

            return 0.0;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real volume(const iarray<Rank>& idx) const
        {
            // radial: (r_r^3 - r_l^3) / 3
            auto r_c  = r_map(idx[0]);
            real dv_r = (r_c.end * r_c.end * r_c.end -
                         r_c.start * r_c.start * r_c.start) /
                        3.0;

            // theta: cos(t_l) - cos(t_r)
            real dv_theta = 2.0;   // full sphere default
            if constexpr (Rank > 1) {
                auto t_c = theta_map(idx[1]);
                dv_theta = std::cos(t_c.start) - std::cos(t_c.end);
            }

            // phi: dphi
            real dv_phi = 2.0 * std::numbers::pi;
            if constexpr (Rank > 2) {
                dv_phi = phi_map(idx[2]).width;
            }

            return dv_r * dv_theta * dv_phi;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real
        face_area(const iarray<Rank>& idx, std::size_t dim) const
        {
            constexpr std::uint64_t x1c = Rank - 1;
            constexpr std::uint64_t x2c = Rank - 2;
            constexpr std::uint64_t x3c = Rank - 2;
            // r-face (dim 0)
            if (dim == 0) {
                real r_l = r_map.face(idx[x1c]);

                real d_omega = 4.0 * std::numbers::pi;
                if constexpr (Rank > 1) {
                    auto t_c = theta_map(idx[x2c]);
                    d_omega  = (std::cos(t_c.start) - std::cos(t_c.end)) * 2.0 *
                              std::numbers::pi;
                    if constexpr (Rank > 2) {
                        // scale by phi fraction
                        d_omega *=
                            phi_map(idx[x3c]).width / (2.0 * std::numbers::pi);
                    }
                }
                return r_l * r_l * d_omega;
            }

            // theta-face (dim 1)
            if constexpr (Rank > 1) {
                if (dim == 1) {
                    // area = integral(r dr) * integral(sin(theta_face) dphi)
                    //      = 0.5 * (r_r^2 - r_l^2) * sin(theta_l) * dphi
                    auto r_c = r_map(idx[x1c]);
                    real r_int =
                        0.5 * (r_c.end * r_c.end - r_c.start * r_c.start);

                    real t_l   = theta_map.face(idx[x2c]);
                    real sin_t = std::sin(t_l);

                    real d_phi = 2.0 * std::numbers::pi;
                    if constexpr (Rank > 2) {
                        d_phi = phi_map(idx[x3c]).width;
                    }

                    return r_int * sin_t * d_phi;
                }
            }

            // phi-face (dim 2)
            if constexpr (Rank > 2) {
                if (dim == 2) {
                    // surface normal to phi is the "half-disk" slice.
                    // area = integral(r dr dtheta) = 0.5 * r^2 * dtheta.
                    // correct.

                    auto r_c = r_map(idx[x1c]);
                    real r_int =
                        0.5 * (r_c.end * r_c.end - r_c.start * r_c.start);

                    auto t_c = theta_map(idx[x2c]);
                    // strictly dtheta, not cos/sin differences
                    return r_int * t_c.width;
                }
            }

            return 0.0;
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto centroid(const iarray<Rank>& idx) const
        {
            // note: geometric centroid of spherical shell is not just
            // arithmetic mean of coords but for finite volume, we usually store
            // the 'coordinate center' r_i+1/2 specialized physics might need
            // volume-weighted centroids. for now, returning coordinate centers.
            vector_t<real, Rank> c;
            c[Rank - 1] = r_map.center(idx[Rank - 1]);
            if constexpr (Rank > 1) {
                c[Rank - 2] = theta_map.center(idx[Rank - 2]);
            }
            if constexpr (Rank > 2) {
                c[Rank - 3] = phi_map.center(idx[Rank - 3]);
            }
            return c;
        }

        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        to_cartesian(const vector_t<real, Rank>& vec) const
        {
            return vecops::spherical_to_cartesian(vec);
        }

        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        from_cartesian(const vector_t<real, Rank>& vec) const
        {
            return vecops::cartesian_to_spherical(vec);
        }
    };

    // =========================================================================
    // cylindrical metric (r, phi, z)
    // dv = r dr dphi dz
    // =========================================================================
    struct full_cylindrical_tag {
    };
    struct axis_cylindrical_tag {
    };
    struct planar_cylindrical_tag {
    };
    template <
        typename MapR,
        typename MapPhi  = dummy_map,
        typename MapZ    = dummy_map,
        typename CylType = full_cylindrical_tag>
    struct cylindrical_metric_t {
        MapR r_map;
        MapPhi phi_map;
        MapZ z_map;
        using cyl_type = CylType;

        DUAL constexpr cylindrical_metric_t(MapR mr)
            : r_map(mr), phi_map(dummy_map{}), z_map(dummy_map{})
        {
        }

        DUAL constexpr cylindrical_metric_t(MapR mr, MapPhi mp)
            : r_map(mr), phi_map(mp), z_map(dummy_map{})
        {
        }

        DUAL constexpr cylindrical_metric_t(MapR mr, MapPhi mp, MapZ mz)
            : r_map(mr), phi_map(mp), z_map(mz)
        {
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto
        scale_factors(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            vector_t<real, Rank> h;
            h.fill(1.0);

            // radial (h_r = 1)

            // phi (h_phi = r)
            if constexpr (Rank > 1) {
                if constexpr (!std::is_same_v<CylType, axis_cylindrical_tag>) {
                    h[Rank - 2] = r_map.center(idx[Rank - 1]);
                }
            }

            // z (h_z = 1)
            return h;
        }

        // coordinate-space cell widths (dr, dphi, dz)
        template <std::uint64_t Rank>
        DUAL constexpr auto
        cell_widths(const simbi::vector_t<int64_t, Rank>& idx) const
        {
            vector_t<real, Rank> w;
            w[Rank - 1] = r_map(idx[Rank - 1]).width;
            if constexpr (Rank > 1) {
                if constexpr (std::is_same_v<CylType, axis_cylindrical_tag>) {
                    w[Rank - 2] = z_map(idx[Rank - 2]).width;
                }
                else {
                    w[Rank - 2] = phi_map(idx[Rank - 2]).width;
                }
            }
            if constexpr (Rank > 2) {
                w[Rank - 3] = z_map(idx[Rank - 3]).width;
            }
            return w;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real face_position(
            const simbi::vector_t<int64_t, Rank>& idx,
            std::size_t dim
        ) const
        {
            constexpr std::uint64_t rc = Rank - 1;
            constexpr std::uint64_t pc = Rank - 2;
            constexpr std::uint64_t zc = Rank - 3;

            // r-face
            if (dim == rc) {
                return r_map.face(idx[rc]);
            }

            // phi-face
            if constexpr (Rank > 1) {
                if (dim == pc) {
                    if constexpr (std::is_same_v<
                                      cyl_type,
                                      axis_cylindrical_tag>) {
                        return z_map.face(idx[pc]);
                    }
                    else {
                        return phi_map.face(idx[pc]);
                    }
                }
            }

            // z-face
            if constexpr (Rank > 2) {
                if (dim == zc) {
                    return z_map.face(idx[zc]);
                }
            }

            return 0.0;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real volume(const vector_t<int64_t, Rank>& idx) const
        {
            auto r_c  = r_map(idx[Rank - 1]);
            real dv_r = 0.5 * (r_c.end * r_c.end - r_c.start * r_c.start);

            real dv_phi = 2.0 * std::numbers::pi;
            if constexpr (Rank > 1) {
                if constexpr (!std::is_same_v<CylType, axis_cylindrical_tag>) {
                    dv_phi = phi_map(idx[Rank - 2]).width;
                }
            }

            real dv_z = 1.0;
            if constexpr (Rank > 2) {
                dv_z = z_map(idx[Rank - 3]).width;
            }
            else if constexpr (std::is_same_v<CylType, axis_cylindrical_tag>) {
                dv_z = z_map(idx[Rank - 3]).width;
            }

            return dv_r * dv_phi * dv_z;
        }

        template <std::uint64_t Rank>
        DUAL constexpr real
        face_area(const iarray<Rank>& idx, std::size_t dim) const
        {
            // r-face (0): r * dphi * dz
            if (dim == Rank - 1) {
                real r_l   = r_map.face(idx[Rank - 1]);
                real d_phi = 2.0 * std::numbers::pi;
                if constexpr (Rank > 1) {
                    d_phi = phi_map(idx[Rank - 2]).width;
                }
                real d_z = 1.0;
                if constexpr (Rank > 2) {
                    d_z = z_map(idx[Rank - 3]).width;
                }
                return r_l * d_phi * d_z;
            }

            // phi-face (1): dr * dz
            if constexpr (Rank > 1) {
                if (dim == Rank - 2) {
                    real d_r = r_map(idx[Rank - 1]).width;
                    real d_z = 1.0;
                    if constexpr (Rank > 2) {
                        d_z = z_map(idx[Rank - 3]).width;
                    }
                    return d_r * d_z;
                }
            }

            // z-face (2): 0.5(r^2) * dphi
            if constexpr (Rank > 2) {
                if (dim == Rank - 3) {
                    auto r_c = r_map(idx[Rank - 1]);
                    real area_r =
                        0.5 * (r_c.end * r_c.end - r_c.start * r_c.start);
                    real d_phi = phi_map(idx[Rank - 2]).width;
                    return area_r * d_phi;
                }
            }
            return 0.0;
        }

        template <std::uint64_t Rank>
        DUAL constexpr auto centroid(const iarray<Rank>& idx) const
        {
            // note: geometric centroid of spherical shell is not just
            // arithmetic mean of coords but for finite volume, we usually
            // store the 'coordinate center' r_i+1/2 specialized physics
            // might need volume-weighted centroids. for now, returning
            // coordinate centers.
            vector_t<real, Rank> c;
            c[Rank - 1] = r_map.center(idx[Rank - 1]);
            if constexpr (Rank > 1) {
                if constexpr (std::is_same_v<CylType, axis_cylindrical_tag>) {
                    c[Rank - 2] = z_map.center(idx[Rank - 2]);
                }
                else {
                    c[Rank - 2] = phi_map.center(idx[Rank - 2]);
                }
            }
            if constexpr (Rank > 2) {
                c[Rank - 3] = z_map.center(idx[Rank - 3]);
            }
            return c;
        }

        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        to_cartesian(const vector_t<real, Rank>& vec) const
        {
            return vecops::cylindrical_to_cartesian(vec);
        }

        template <std::uint64_t Rank>
        DUAL vector_t<real, Rank>
        from_cartesian(const vector_t<real, Rank>& vec) const
        {
            return vecops::cartesian_to_cylindrical(vec);
        }
    };

    template <typename T>
    concept is_spherical_c = requires {
        { T::r_map };
        { T::theta_map };
        { T::phi_map };
    };

    template <typename T>
    concept is_cylindrical_c = requires {
        { T::r_map };
        { T::phi_map };
        { T::z_map };
    };

    template <typename T>
    concept is_cartesian_c = requires {
        { T::x1_map };
        { T::x2_map };
        { T::x3_map };
    };

    template <typename M>
    concept is_cylindrical_variant_c = requires {
        { M::cyl_type };
    };

    // CTAD helpers
    template <typename M1>
    cartesian_metric_t(M1) -> cartesian_metric_t<M1>;

    template <typename M1, typename M2>
    cartesian_metric_t(M1, M2) -> cartesian_metric_t<M1, M2>;

    template <typename M1, typename M2, typename M3>
    cartesian_metric_t(M1, M2, M3) -> cartesian_metric_t<M1, M2, M3>;

    template <typename MR>
    spherical_metric_t(MR) -> spherical_metric_t<MR>;

    template <typename MR, typename MTheta>
    spherical_metric_t(MR, MTheta) -> spherical_metric_t<MR, MTheta>;

    template <typename MR, typename MTheta, typename MPhi>
    spherical_metric_t(MR, MTheta, MPhi)
        -> spherical_metric_t<MR, MTheta, MPhi>;

    template <typename MR>
    cylindrical_metric_t(MR) -> cylindrical_metric_t<MR>;

    template <typename MR, typename MPhi>
    cylindrical_metric_t(MR, MPhi) -> cylindrical_metric_t<MR, MPhi>;

    template <typename MR, typename MPhi, typename MZ>
    cylindrical_metric_t(MR, MPhi, MZ) -> cylindrical_metric_t<MR, MPhi, MZ>;

}   // namespace simbi::geometry

#endif   // SIMBI_GEOMETRY_METRIC_HPP
