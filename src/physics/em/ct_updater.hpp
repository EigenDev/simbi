#ifndef MHD_NLOGIC_HPP
#define MHD_NLOGIC_HPP

#include "compat.hpp"
#include "compute/computation.hpp"
#include "contact.hpp"
#include "containers/vector.hpp"
#include "ct_geom.hpp"
#include "functional/fp.hpp"
#include "geometry/metrics.hpp"
#include "grid/domain.hpp"
#include "physics/em/electromagnetism.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <type_traits>

namespace simbi::em {
    using namespace simbi::unit_vectors;

    // ========================================================================
    // CORE PERMUTATION TYPE
    // ========================================================================

    template <std::uint8_t L, std::uint8_t M, std::uint8_t N>
    struct coord_permutation_t {
        static constexpr std::uint8_t horizontal_axis = L;
        static constexpr std::uint8_t vertical_axis   = M;
        static constexpr std::uint8_t normal_axis     = N;

        static constexpr auto flux_indices()
        {
            return vector_t<std::uint64_t, 2>{L, M};
        }

        static constexpr auto vary_index(magnetic_comp_t mag_comp)
        {
            if (mag_comp == magnetic_comp_t::I) {
                return M;
            }
            else if (mag_comp == magnetic_comp_t::J) {
                return (L == 2) ? L : M;
            }
            else {
                return L;
            }
        }

        static constexpr auto e_field_component() { return N; }
    };

    using IJ_permutation = coord_permutation_t<2, 1, 0>;
    using JK_permutation = coord_permutation_t<1, 0, 2>;
    using IK_permutation = coord_permutation_t<2, 0, 1>;

    template <typename... Perms>
    struct permutation_list_t {
    };

    template <magnetic_comp_t MagComp>
    constexpr auto permutation_list()
    {
        if constexpr (MagComp == magnetic_comp_t::K) {
            return permutation_list_t<JK_permutation, IK_permutation>{};
        }
        else if constexpr (MagComp == magnetic_comp_t::J) {
            return permutation_list_t<IJ_permutation, JK_permutation>{};
        }
        else {
            return permutation_list_t<IK_permutation, IJ_permutation>{};
        }
    }

    template <typename... Perms, typename Func>
    constexpr auto map_permutations(permutation_list_t<Perms...>, Func&& func)
    {
        return vector_t{func(Perms{})...};
    }

    // ========================================================================
    // COORDINATE UTILITIES
    // ========================================================================

    constexpr std::int64_t to_array_index(int doubled_coord)
    {
        return doubled_coord / 2;
    }

    constexpr auto to_doubled_coord(const iarray<3>& coord)
    {
        return iarray<3>{2 * coord[0], 2 * coord[1], 2 * coord[2]};
    }

    constexpr auto to_array_index_coord(const iarray<3>& doubled_coord)
    {
        return iarray<3>{
          to_array_index(doubled_coord[0]),
          to_array_index(doubled_coord[1]),
          to_array_index(doubled_coord[2])
        };
    }

    // ========================================================================
    // COORDINATE GENERATION
    // ========================================================================

    template <magnetic_comp_t MagComp, typename Permutation>
    constexpr auto gen_edge_coords(const iarray<3>& face_coord)
    {
        constexpr auto vary_index = Permutation::vary_index(MagComp);
        auto face_doubled         = to_doubled_coord(face_coord);

        constexpr auto face_idx = static_cast<uint8_t>(MagComp);
        constexpr auto half     = 1;
        face_doubled[face_idx] -= half;

        auto make_edge = [&](int offset) {
            auto edge = face_doubled;
            edge[vary_index] += offset;
            return edge;
        };

        return vector_t<iarray<3>, 2>{make_edge(-1), make_edge(+1)};
    }

    template <typename Permutation>
    constexpr auto flux_stencil(const iarray<3>& edge_doubled_coord)
    {
        auto make_flux_coord = [&](std::int64_t h_offset,
                                   std::int64_t v_offset) {
            auto coord = edge_doubled_coord;
            coord[Permutation::horizontal_axis] += h_offset;
            coord[Permutation::vertical_axis] += v_offset;
            return to_array_index_coord(coord);
        };
        constexpr auto half = 1;
        return vector_t{
          make_flux_coord(half, +1),
          make_flux_coord(half, -1),
          make_flux_coord(+1, half),
          make_flux_coord(-1, half)
        };
    }

    template <typename Permutation>
    constexpr auto prim_stencil(const iarray<3>& edge_doubled_coord)
    {
        auto make_prim_coord = [&](std::int64_t h_offset,
                                   std::int64_t v_offset) {
            auto coord = edge_doubled_coord;
            coord[Permutation::horizontal_axis] += h_offset;
            coord[Permutation::vertical_axis] += v_offset;
            return to_array_index_coord(coord);
        };

        return vector_t{
          make_prim_coord(+1, +1),
          make_prim_coord(-1, +1),
          make_prim_coord(+1, -1),
          make_prim_coord(-1, -1)
        };
    }

    // ========================================================================
    // FIELD ACCESS
    // ========================================================================

    template <typename Permutation, typename FluxField>
    auto
    face_efields(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();
        constexpr std::uint64_t rank  = FluxField::rank;
        constexpr auto nhat = ehat<rank>(Permutation::e_field_component());

        return vector_t{
          flux[h_flux_idx](coords[0]).mag[index(nhat)],
          flux[h_flux_idx](coords[1]).mag[index(nhat)],
          flux[v_flux_idx](coords[2]).mag[index(nhat)],
          flux[v_flux_idx](coords[3]).mag[index(nhat)]
        };
    }

    template <typename Permutation, typename FluxField>
    auto den_fluxes(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();
        return vector_t<real, 4>{
          flux[h_flux_idx](coords[0]).den,
          flux[h_flux_idx](coords[1]).den,
          flux[v_flux_idx](coords[2]).den,
          flux[v_flux_idx](coords[3]).den
        };
    }

    template <typename Permutation, typename PrimField>
    auto
    center_efields(const PrimField& prim, const vector_t<iarray<3>, 4>& coords)
    {
        constexpr std::uint64_t rank = PrimField::rank;
        constexpr auto nhat = ehat<rank>(Permutation::e_field_component());

        return vector_t{
          em::electric_field(prim(coords[0]))[index(nhat)],
          em::electric_field(prim(coords[1]))[index(nhat)],
          em::electric_field(prim(coords[2]))[index(nhat)],
          em::electric_field(prim(coords[3]))[index(nhat)]
        };
    }

    // ========================================================================
    // CT MAGNETIC UPDATE (geometry-based)
    // ========================================================================
    template <
        magnetic_comp_t MagComp,
        typename FluxField,
        typename PrimField,
        typename Geometry>
    struct ct_magnetic_update_op_t {
        FluxField fluxes;
        PrimField prims;
        real dt;
        Geometry geometry;

        DEV auto operator()(auto face_coord) const
        {
            constexpr auto perm_list = permutation_list<MagComp>();

            const auto emf_computer = [&]<typename Perm>(Perm) {
                return [&](const iarray<3>& edge_coord) {
                    auto flux_coords = flux_stencil<Perm>(edge_coord);
                    auto prim_coords = prim_stencil<Perm>(edge_coord);

                    auto ef    = face_efields<Perm>(fluxes, flux_coords);
                    auto ec    = center_efields<Perm>(prims, prim_coords);
                    auto densf = den_fluxes<Perm>(fluxes, flux_coords);

                    return ct_contact_formula(ef, ec, densf);
                };
            };

            const auto edge_generator = [&]<typename Permutation>(Permutation) {
                return gen_edge_coords<MagComp, Permutation>(face_coord);
            };

            auto emfs = map_permutations(perm_list, [&]<typename Perm>(Perm p) {
                return edge_generator(p) | fp::map(emf_computer(p)) |
                       fp::collect<vector_t<real, 2>>;
            });

            real curl = discrete_curl<MagComp>(emfs, face_coord, geometry);
            return -dt * curl;   // Faraday's law
        }
    };

    template <magnetic_comp_t MagComp>
    auto make_ct_magnetic_update_op(
        auto&& fluxes,
        auto&& prims,
        real dt,
        auto&& geometry
    )
    {
        return ct_magnetic_update_op_t<
            MagComp,
            std::decay_t<decltype(fluxes)>,
            std::decay_t<decltype(prims)>,
            std::decay_t<decltype(geometry)>>{
          std::forward<decltype(fluxes)>(fluxes),
          std::forward<decltype(prims)>(prims),
          dt,
          std::forward<decltype(geometry)>(geometry)
        };
    }

    template <
        magnetic_comp_t MagComp,
        typename HydroState,
        typename Geometry,
        typename Domain>
    auto ct_magnetic_update(
        const HydroState& state,
        const Geometry& geometry,
        const Domain& face_domain,
        real dt
    )
    {
        constexpr auto comp = static_cast<std::uint64_t>(MagComp);
        const auto fluxes   = vector_t{
          state.flux[0][face_domain[0]],
          state.flux[1][face_domain[1]],
          state.flux[2][face_domain[2]]
        };
        const auto prim_domain = state.prim.domain();

        return compute::computation_t{
          make_ct_magnetic_update_op<MagComp>(
              fluxes,
              state.prim[prim_domain],
              dt,
              geometry
          ),
          grid::extents(face_domain[comp].shape())
        };
    }

    // ========================================================================
    // INTERPOLATION FIELDS (geometry-based)
    // ========================================================================
    template <typename Bfield, typename Geometry>
    struct interpolate_magnetic_op_t {
        Bfield b1;
        Bfield b2;
        Bfield b3;
        Geometry geometry;

        DEV auto get_face_avg(const auto& bface, auto cminus, int dir) const
        {
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
          interpolate_magnetic_op_t<
              std::decay_t<decltype(bfield[2][face_domains[2]])>,
              Geometry>{
            bfield[2][face_domains[2]],
            bfield[1][face_domains[1]],
            bfield[0][face_domains[0]],
            geometry
          },
          grid::extents(cell_domain.shape())
        };
    }

    // ========================================================================
    // HIGH-LEVEL INTERFACE (geometry-based)
    // ========================================================================
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

        u_p = u_p.enum_map(
                     [bavg](auto coord, auto u) {
                         const auto b_interp = bavg(coord);
                         const auto bmean    = u.mag;
                         const auto old_emag = 0.5 * vecops::dot(bmean, bmean);
                         const auto new_emag =
                             0.5 * vecops::dot(b_interp, b_interp);
                         u.nrg += (new_emag - old_emag);
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

        u_p = u_p.enum_map(
                     [bavg](auto coord, auto u) {
                         u.mag = bavg(coord);
                         return u;
                     }
        ).with(exec);
    }

    template <
        magnetic_comp_t MagComp,
        typename HydroState,
        typename Geometry,
        typename Domain,
        typename Executor>
    void update_magnetic_component(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const Domain& face_domain,
        real dt
    )
    {
        constexpr auto comp = static_cast<std::uint64_t>(MagComp);
        auto db = ct_magnetic_update<MagComp>(state, geometry, face_domain, dt);
        auto bfield = state.bfield[comp][face_domain[comp]];

        bfield = bfield
                     .enum_map([db](auto coord, auto b_old) {
                         return b_old + db(coord);
                     })
                     .with(exec);
    }

    template <
        typename HydroState,
        typename Geometry,
        typename Domain,
        typename FaceDomains,
        typename Executor>
    void update_magnetic_fields(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const Domain& cell_domain,
        real dt
    )
    {
        update_magnetic_component<magnetic_comp_t::I>(
            exec,
            state,
            geometry,
            face_domains,
            dt
        );
        update_magnetic_component<magnetic_comp_t::J>(
            exec,
            state,
            geometry,
            face_domains,
            dt
        );
        update_magnetic_component<magnetic_comp_t::K>(
            exec,
            state,
            geometry,
            face_domains,
            dt
        );
        interpolate_magnetic_fields(
            exec,
            state,
            geometry,
            face_domains,
            cell_domain
        );
    }

    // ========================================================================
    // EDGE E-FIELD COMPUTATION
    // computes and stores edge-centered electric fields from face fluxes
    // ========================================================================

    // helper to get the correct permutation for computing E component
    // IJ_permutation: e_field_component = 0 (E_z)
    // IK_permutation: e_field_component = 1 (E_y)
    // JK_permutation: e_field_component = 2 (E_x)
    template <std::uint64_t EComp>
    constexpr auto efield_permutation()
    {
        if constexpr (EComp == 0) {
            return IJ_permutation{};
        }
        else if constexpr (EComp == 1) {
            return IK_permutation{};
        }
        else {
            return JK_permutation{};
        }
    }

    // operator that computes E at edges parallel to a given axis
    // edge_comp: which E component (0=Ez, 1=Ey, 2=Ex in array-index order)
    template <std::uint64_t EdgeComp, typename FluxField, typename PrimField>
    struct compute_edge_efield_op_t {
        FluxField fluxes;
        PrimField prims;

        DEV auto operator()(auto edge_coord) const
        {
            // get the permutation that extracts this E component
            constexpr auto perm = efield_permutation<EdgeComp>();

            // convert to doubled coordinates for stencil computation
            auto edge_doubled = to_doubled_coord(edge_coord);

            // gather face E-fields, cell-center E-fields, and density fluxes
            auto flux_coords = flux_stencil<decltype(perm)>(edge_doubled);
            auto prim_coords = prim_stencil<decltype(perm)>(edge_doubled);

            auto ef    = face_efields<decltype(perm)>(fluxes, flux_coords);
            auto ec    = center_efields<decltype(perm)>(prims, prim_coords);
            auto densf = den_fluxes<decltype(perm)>(fluxes, flux_coords);

            return ct_contact_formula(ef, ec, densf);
        }
    };

    // compute edge E-field for a single component
    template <
        std::uint64_t EdgeComp,
        typename Executor,
        typename HydroState,
        typename EdgeDomain,
        typename Domain>
    void compute_edge_efield_component(
        Executor& exec,
        HydroState& state,
        const EdgeDomain& edge_domain,
        const Domain& cell_domain
    )
    {
        const auto fluxes = vector_t{
          state.flux[0][state.flux[0].domain()],
          state.flux[1][state.flux[1].domain()],
          state.flux[2][state.flux[2].domain()]
        };
        const auto prims = state.prim[cell_domain];

        using op_t = compute_edge_efield_op_t<
            EdgeComp,
            std::decay_t<decltype(fluxes)>,
            std::decay_t<decltype(prims)>>;

        auto efield_view = state.efield[EdgeComp][edge_domain];

        efield_view = efield_view
                          .space_map([op = op_t{fluxes, prims}](auto coord) {
                              return op(coord);
                          })
                          .with(exec);
    }

    // compute all edge E-fields from face fluxes and primitives
    template <
        typename Executor,
        typename HydroState,
        typename EdgeDomains,
        typename Domain>
    void compute_edge_efields(
        Executor& exec,
        HydroState& state,
        const EdgeDomains& edge_domains,
        const Domain& cell_domain
    )
    {
        compute_edge_efield_component<0>(
            exec,
            state,
            edge_domains[0],
            cell_domain
        );
        compute_edge_efield_component<1>(
            exec,
            state,
            edge_domains[1],
            cell_domain
        );
        compute_edge_efield_component<2>(
            exec,
            state,
            edge_domains[2],
            cell_domain
        );
    }

    // ========================================================================
    // CT UPDATE FROM STORED E-FIELDS
    // reads pre-computed edge E-fields instead of computing on-the-fly
    // ========================================================================

    template <magnetic_comp_t MagComp, typename EField, typename Geometry>
    struct ct_update_from_efield_op_t {
        EField efield;
        real dt;
        Geometry geometry;

        DEV auto operator()(auto face_coord) const
        {
            // gather edge E values for curl computation
            // for B_d update, we need E from the two transverse directions
            constexpr auto comp = static_cast<std::uint64_t>(MagComp);

            // the two transverse axes
            constexpr std::uint64_t t1 = (comp + 1) % 3;
            constexpr std::uint64_t t2 = (comp + 2) % 3;

            // edge E values at the 4 edges of this face
            // E_t1 at edges parallel to t1 (varies in t2)
            // E_t2 at edges parallel to t2 (varies in t1)
            auto c00 = face_coord;
            auto c01 = face_coord;
            c01[t2] += 1;
            auto c10 = face_coord;
            c10[t1] += 1;
            auto c11 = face_coord;
            c11[t1] += 1;
            c11[t2] += 1;

            // for curl: dE_t2/dt1 - dE_t1/dt2
            real e_t1_lo = efield[t1](c00);
            real e_t1_hi = efield[t1](c10);
            real e_t2_lo = efield[t2](c00);
            real e_t2_hi = efield[t2](c01);

            // get scale factors for proper curl
            const auto h = geometry.scale_factors(face_coord);

            real curl =
                (e_t2_hi - e_t2_lo) / h[t1] - (e_t1_hi - e_t1_lo) / h[t2];

            return -dt * curl;
        }
    };

    template <
        magnetic_comp_t MagComp,
        typename Executor,
        typename HydroState,
        typename Geometry,
        typename FaceDomain,
        typename EdgeDomains>
    void update_magnetic_from_efield(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const FaceDomain& face_domain,
        const EdgeDomains& edge_domains,
        real dt
    )
    {
        constexpr auto comp = static_cast<std::uint64_t>(MagComp);

        const auto efield = vector_t{
          state.efield[0][edge_domains[0]],
          state.efield[1][edge_domains[1]],
          state.efield[2][edge_domains[2]]
        };

        using op_t = ct_update_from_efield_op_t<
            MagComp,
            std::decay_t<decltype(efield)>,
            Geometry>;

        auto bfield = state.bfield[comp][face_domain];

        bfield =
            bfield
                .enum_map(
                    [op = op_t{efield, dt, geometry}](auto coord, auto b_old) {
                        return b_old + op(coord);
                    }
                )
                .with(exec);
    }

    // high-level interface using stored E-fields
    template <
        typename Executor,
        typename HydroState,
        typename Geometry,
        typename FaceDomains,
        typename EdgeDomains,
        typename CellDomain>
    void update_magnetic_fields_from_efield(
        Executor& exec,
        HydroState& state,
        const Geometry& geometry,
        const FaceDomains& face_domains,
        const EdgeDomains& edge_domains,
        const CellDomain& cell_domain,
        real dt
    )
    {
        // magnetic_comp_t::I = 2 (Bx), J = 1 (By), K = 0 (Bz)
        // face_domains[d] corresponds to faces normal to axis d
        // so face_domains[comp] is the correct domain for B_comp
        update_magnetic_from_efield<magnetic_comp_t::K>(
            exec,
            state,
            geometry,
            face_domains[0],
            edge_domains,
            dt
        );
        update_magnetic_from_efield<magnetic_comp_t::J>(
            exec,
            state,
            geometry,
            face_domains[1],
            edge_domains,
            dt
        );
        update_magnetic_from_efield<magnetic_comp_t::I>(
            exec,
            state,
            geometry,
            face_domains[2],
            edge_domains,
            dt
        );
        interpolate_magnetic_fields(
            exec,
            state,
            geometry,
            face_domains,
            cell_domain
        );
    }

}   // namespace simbi::em

#endif   // MHD_NLOGIC_HPP
