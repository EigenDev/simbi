#ifndef PHYSICS_EM_API_HPP
#define PHYSICS_EM_API_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "contact.hpp"
#include "containers/vector.hpp"
#include "ct_geom.hpp"
#include "geometry/metrics.hpp"
#include "physics/em/electromagnetism.hpp"
#include "utility/enums.hpp"
#include "zero.hpp"

namespace simbi::em {
    // ex at edge[k,j,i] is at physical {k-1/2, j-1/2, i}
    // needs 4 fluxes and 4 cell centers
    template <typename FluxField, typename PrimField>
    struct ex_stencil_t {
        FluxField fluxes;
        PrimField prims;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            // north/south use fz (flux[0]), east/west use fy (flux[1])
            auto face_ex = vector_t<real, 4>{
              fluxes[1](iarray<3>{kk - 0, jj, ii}).mag[0],   // north fy
              fluxes[1](iarray<3>{kk - 1, jj, ii}).mag[0],   // south fy
              fluxes[0](iarray<3>{kk, jj - 0, ii}).mag[0],   // east fz
              fluxes[0](iarray<3>{kk, jj - 1, ii}).mag[0]    // west fz
            };

            auto density_flux = vector_t<real, 4>{
              fluxes[1](iarray<3>{kk - 0, jj, ii}).den,   // north fy
              fluxes[1](iarray<3>{kk - 1, jj, ii}).den,   // south fy
              fluxes[0](iarray<3>{kk, jj - 0, ii}).den,   // east fz
              fluxes[0](iarray<3>{kk, jj - 1, ii}).den    // west fz
            };

            // cell centers: ne, nw, se, sw
            auto cell_ex = vector_t<real, 4>{
              electric_field(prims(iarray<3>{kk - 0, jj - 0, ii}))[0],   // ne
              electric_field(prims(iarray<3>{kk - 0, jj - 1, ii}))[0],   // nw
              electric_field(prims(iarray<3>{kk - 1, jj - 0, ii}))[0],   // se
              electric_field(prims(iarray<3>{kk - 1, jj - 1, ii}))[0]    // sw
            };

            if constexpr (comp_ct_type == ct_algo_t::ZERO) {
                return ct_zero_formula(face_ex, cell_ex);
            }
            else {
                return ct_contact_formula(face_ex, cell_ex, density_flux);
            }
        }
    };

    // factory
    template <typename FluxField, typename PrimField, typename EdgeDomain>
    inline auto ex_stencil(
        const FluxField& fluxes,
        const PrimField& prims,
        const EdgeDomain& domain
    )
    {
        return compute::computation_t{
          ex_stencil_t<FluxField, PrimField>{fluxes, prims},
          domain
        };
    }

    // ey at edge[k,j,i] is at physical {k-1/2, j, i-1/2}
    template <typename FluxField, typename PrimField>
    struct ey_stencil_t {
        FluxField fluxes;
        PrimField prims;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            auto face_ey = vector_t<real, 4>{
              fluxes[2](iarray<3>{kk - 0, jj, ii}).mag[1],   // north fx
              fluxes[2](iarray<3>{kk - 1, jj, ii}).mag[1],   // south fx
              fluxes[0](iarray<3>{kk, jj, ii - 0}).mag[1],   // east fz
              fluxes[0](iarray<3>{kk, jj, ii - 1}).mag[1]    // west fz
            };

            auto density_flux = vector_t<real, 4>{
              fluxes[2](iarray<3>{kk - 0, jj, ii}).den,   // north fx
              fluxes[2](iarray<3>{kk - 1, jj, ii}).den,   // south fx
              fluxes[0](iarray<3>{kk, jj, ii - 0}).den,   // east fz
              fluxes[0](iarray<3>{kk, jj, ii - 1}).den    // west fz
            };

            auto cell_ey = vector_t<real, 4>{
              electric_field(prims(iarray<3>{kk - 0, jj, ii - 0}))[1],   // ne
              electric_field(prims(iarray<3>{kk - 0, jj, ii - 1}))[1],   // nw
              electric_field(prims(iarray<3>{kk - 1, jj, ii - 0}))[1],   // se
              electric_field(prims(iarray<3>{kk - 1, jj, ii - 1}))[1]    // sw
            };

            if constexpr (comp_ct_type == ct_algo_t::ZERO) {
                return ct_zero_formula(face_ey, cell_ey);
            }
            else {
                return ct_contact_formula(face_ey, cell_ey, density_flux);
            }
        }
    };

    // ey factory
    template <typename FluxField, typename PrimField, typename EdgeDomain>
    auto ey_stencil(
        const FluxField& fluxes,
        const PrimField& prims,
        const EdgeDomain& domain
    )
    {
        return compute::computation_t{
          ey_stencil_t<FluxField, PrimField>{fluxes, prims},
          domain
        };
    }

    // ez at edge[k,j,i] is at physical {k, j-1/2, i-1/2}
    template <typename FluxField, typename PrimField>
    struct ez_stencil_t {
        FluxField fluxes;
        PrimField prims;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            auto face_ez = vector_t<real, 4>{
              fluxes[2](iarray<3>{kk, jj - 0, ii}).mag[2],   // north fx
              fluxes[2](iarray<3>{kk, jj - 1, ii}).mag[2],   // south fx
              fluxes[1](iarray<3>{kk, jj, ii - 0}).mag[2],   // east fy
              fluxes[1](iarray<3>{kk, jj, ii - 1}).mag[2]    // west fy
            };

            auto density_flux = vector_t<real, 4>{
              fluxes[2](iarray<3>{kk, jj - 0, ii}).den,   // north fx
              fluxes[2](iarray<3>{kk, jj - 1, ii}).den,   // south fx
              fluxes[1](iarray<3>{kk, jj, ii - 0}).den,   // east fy
              fluxes[1](iarray<3>{kk, jj, ii - 1}).den    // west fy
            };

            auto cell_ez = vector_t<real, 4>{
              electric_field(prims(iarray<3>{kk, jj - 0, ii - 0}))[2],   // ne
              electric_field(prims(iarray<3>{kk, jj - 0, ii - 1}))[2],   // nw
              electric_field(prims(iarray<3>{kk, jj - 1, ii - 0}))[2],   // se
              electric_field(prims(iarray<3>{kk, jj - 1, ii - 1}))[2]    // sw
            };

            if constexpr (comp_ct_type == ct_algo_t::ZERO) {
                return ct_zero_formula(face_ez, cell_ez);
            }
            else {
                return ct_contact_formula(face_ez, cell_ez, density_flux);
            }
        }
    };

    // ez factory
    template <typename FluxField, typename PrimField, typename EdgeDomain>
    auto ez_stencil(
        const FluxField& fluxes,
        const PrimField& prims,
        const EdgeDomain& domain
    )
    {
        return compute::computation_t{
          ez_stencil_t<FluxField, PrimField>{fluxes, prims},
          domain
        };
    }

    // =========================================================================
    // magnetic field updates from stored e-fields
    // =========================================================================

    // bx face[k,j,i] at physical {k, j, i-1/2}
    // needs ey and ez edges bounding this face
    template <typename EField, typename Geometry>
    struct bx_curl_t {
        EField efield;
        Geometry geometry;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            // ey edges (comp 1): left/right in k direction
            auto ey_vals = vector_t<real, 2>{
              efield[1](iarray<3>{kk + 0, jj, ii}),
              efield[1](iarray<3>{kk + 1, jj, ii})
            };

            // ez edges (comp 0): left/right in j direction
            auto ez_vals = vector_t<real, 2>{
              efield[0](iarray<3>{kk, jj + 0, ii}),
              efield[0](iarray<3>{kk, jj + 1, ii})
            };

            auto edge_emfs = vector_t<vector_t<real, 2>, 2>{ey_vals, ez_vals};
            return discrete_curl<magnetic_comp_t::I>(
                edge_emfs,
                coord,
                geometry
            );
        }
    };

    // factory for bx_curl_t
    template <typename EField, typename Geometry, typename FaceDomain>
    auto bx_curl_op(
        const EField& efield,
        const Geometry& geometry,
        const FaceDomain& domain
    )
    {
        return compute::computation_t{
          bx_curl_t<EField, Geometry>{efield, geometry},
          domain
        };
    }

    // by face[k,j,i] at physical {k, j-1/2, i}
    template <typename EField, typename Geometry>
    struct by_curl_t {
        EField efield;
        Geometry geometry;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            // ez edges (comp 0): left/right in i direction
            auto ez_vals = vector_t<real, 2>{
              efield[0](iarray<3>{kk, jj, ii + 0}),
              efield[0](iarray<3>{kk, jj, ii + 1})
            };

            // ex edges (comp 2): left/right in k direction
            auto ex_vals = vector_t<real, 2>{
              efield[2](iarray<3>{kk + 0, jj, ii}),
              efield[2](iarray<3>{kk + 1, jj, ii})
            };

            auto edge_emfs = vector_t<vector_t<real, 2>, 2>{ez_vals, ex_vals};
            return discrete_curl<magnetic_comp_t::J>(
                edge_emfs,
                coord,
                geometry
            );
        }
    };

    // factory for by_curl_t
    template <typename EField, typename Geometry, typename FaceDomain>
    auto by_curl_op(
        const EField& efield,
        const Geometry& geometry,
        const FaceDomain& domain
    )
    {
        return compute::computation_t{
          by_curl_t<EField, Geometry>{efield, geometry},
          domain
        };
    }

    // bz face[k,j,i] at physical {k-1/2, j, i}
    template <typename EField, typename Geometry>
    struct bz_curl_t {
        EField efield;
        Geometry geometry;

        DEV auto operator()(const iarray<3>& coord) const
        {
            const auto [kk, jj, ii] = coord;

            // ey edges (comp 1): left/right in ii direction
            auto ey_vals = vector_t<real, 2>{
              efield[1](iarray<3>{kk, jj, ii + 0}),
              efield[1](iarray<3>{kk, jj, ii + 1})
            };

            // ex edges (comp 2): left/right in j direction
            auto ex_vals = vector_t<real, 2>{
              efield[2](iarray<3>{kk, jj + 0, ii}),
              efield[2](iarray<3>{kk, jj + 1, ii})
            };

            auto edge_emfs = vector_t<vector_t<real, 2>, 2>{ex_vals, ey_vals};
            return discrete_curl<magnetic_comp_t::K>(
                edge_emfs,
                coord,
                geometry
            );
        }
    };

    // factory for bz_curl_t
    template <typename EField, typename Geometry, typename FaceDomain>
    DEV auto bz_curl_op(
        const EField& efield,
        const Geometry& geometry,
        const FaceDomain& domain
    )
    {
        return compute::computation_t{
          bz_curl_t<EField, Geometry>{efield, geometry},
          domain
        };
    }

    // =========================================================================
    // public interface
    // =========================================================================

    template <
        typename Executor,
        typename HydroState,
        typename EdgeDomains,
        typename FaceDomains,
        typename Domain>
    void compute_edge_efields(
        Executor& exec,
        HydroState& state,
        const EdgeDomains& edge_domains,
        const FaceDomains& face_domains,
        const Domain& cell_domain
    )
    {
        const auto fluxes = vector_t{
          state.flux[0][face_domains[0]],
          state.flux[1][face_domains[1]],
          state.flux[2][face_domains[2]]
        };
        const auto prims = state.prim[cell_domain];

        // compute ex
        auto e1    = state.efield[2];
        auto e1_op = ex_stencil(fluxes, prims, edge_domains[2]);
        e1 = e1.zip(e1_op, [](auto _, auto enp) { return enp; }).with(exec);

        // compute ey
        auto e2    = state.efield[1];
        auto e2_op = ey_stencil(fluxes, prims, edge_domains[1]);
        e2 = e2.zip(e2_op, [](auto _, auto enp) { return enp; }).with(exec);

        // compute ez
        auto e3    = state.efield[0];
        auto e3_op = ez_stencil(fluxes, prims, edge_domains[0]);
        e3 = e3.zip(e3_op, [](auto _, auto enp) { return enp; }).with(exec);
    }

    // =========================================================================
    // interpolate face-centered b to cell centers
    // =========================================================================
    template <typename Bfield, typename Geometry>
    struct interpolate_magnetic_op_t {
        Bfield b1;
        Bfield b2;
        Bfield b3;
        Geometry geometry;

        DEV auto get_face_avg(const auto& bface, auto cminus, int dir) const
        {
            using namespace unit_vectors;
            const auto cplus = cminus + array_offset<3>(dir);

            using metric_t = typename Geometry::metric_type;
            if constexpr (geometry::is_cartesian_c<metric_t>) {
                return 0.5 * (bface(cminus) + bface(cplus));
            }
            else {
                // volume-average for non-Cartesian geometries
                auto al = geometry.face_area(cminus, dir);
                auto ar = geometry.face_area(cplus, dir);
                return (bface(cminus) * al + bface(cplus) * ar) / (al + ar);
            }
        }

        DEV auto operator()(auto coord) const
        {
            return vector_t<real, 3>{
              get_face_avg(b1, coord, 2),
              get_face_avg(b2, coord, 1),
              get_face_avg(b3, coord, 0)
            };
        }
    };

    template <
        typename BField,
        typename Geometry,
        typename Domain,
        typename FaceDomains>
    auto interpolate_face_to_cell_magnetic(
        const BField& bfield,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const Domain& cell_domain
    )
    {
        return compute::computation_t{
          interpolate_magnetic_op_t{
            bfield[2][face_domains[2]],
            bfield[1][face_domains[1]],
            bfield[0][face_domains[0]],
            geometry
          },
          cell_domain
        };
    }

    template <
        typename ConsField,
        typename BField,
        typename Geometry,
        typename Domain,
        typename FaceDomains,
        typename Executor>
    void update_energy_density(
        Executor& exec,
        ConsField& cons,
        const BField& bfields,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const Domain& cell_domain
    )
    {
        auto bavg = interpolate_face_to_cell_magnetic(
            bfields,
            geometry,
            face_domains,
            cell_domain
        );
        auto u_p = cons[cell_domain];

        u_p = u_p.zip(
                     bavg,
                     [](auto u, auto bv) {
                         const auto bmean = u.mag;
                         const auto e_old = 0.5 * vecops::dot(bmean, bmean);
                         const auto e_new = 0.5 * vecops::dot(bv, bv);
                         u.nrg += (e_new - e_old);
                         return u;
                     }
        ).with(exec);
    }

    template <
        typename Executor,
        typename HydroState,
        typename Geometry,
        typename Domain,
        typename FaceDomains>
    void interpolate_magnetic_fields(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const Domain& cell_domain
    )
    {
        auto bavg = interpolate_face_to_cell_magnetic(
            state.bfield,
            geometry,
            face_domains,
            cell_domain
        );
        auto u_p = state.cons[cell_domain];

        u_p = u_p.zip(
                     bavg,
                     [](auto u, auto b_new) {
                         // since my Godunov does not affect the cell-centered
                         // magnetic fields, we do not need to update the energy
                         // here.
                         // const auto b_old = u.mag;
                         u.mag = b_new;
                         // const auto e_old = 0.5 * vecops::dot(b_old, b_old);
                         // const auto e_new = 0.5 * vecops::dot(b_new, b_new);
                         // u.nrg += (e_new - e_old);
                         return u;
                     }
        ).with(exec);
    }

    template <
        typename Executor,
        typename HydroState,
        typename Geometry,
        typename FaceDomains,
        typename Domain>
    void update_magnetic_fields(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const Domain& cell_domain,
        real dt
    )
    {
        // update bx (comp 2)
        auto b1    = state.bfield[2][face_domains[2]];
        auto b1_op = bx_curl_op(state.efield, geometry, face_domains[2]);
        b1         = b1.zip(b1_op, [dt](auto b, auto curl_e) {
                   return b - dt * curl_e;
               }).with(exec);

        auto b2    = state.bfield[1][face_domains[1]];
        auto b2_op = by_curl_op(state.efield, geometry, face_domains[1]);
        b2         = b2.zip(b2_op, [dt](auto b, auto curl_e) {
                   return b - dt * curl_e;
               }).with(exec);

        auto b3    = state.bfield[0][face_domains[0]];
        auto b3_op = bz_curl_op(state.efield, geometry, face_domains[0]);
        b3         = b3.zip(b3_op, [dt](auto b, auto curl_e) {
                   return b - dt * curl_e;
               }).with(exec);

        // interpolate to cell centers and update energy
        em::interpolate_magnetic_fields(
            exec,
            state,
            geometry,
            face_domains,
            cell_domain
        );

        // std::cin.get();
    }

}   // namespace simbi::em

#endif   // PHYSICS_EM_NEW_API_HPP
