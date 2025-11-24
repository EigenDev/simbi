#ifndef PHYSICS_EM_MHD_NLOGIC_HPP
#define PHYSICS_EM_MHD_NLOGIC_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"
#include "geometry/metrics.hpp"
#include "physics/em/contact.hpp"
#include "physics/em/ct_geom.hpp"
#include "physics/em/electromagnetism.hpp"
#include "utility/enums.hpp"

#include <cstdint>

namespace simbi::em {
    using namespace simbi::unit_vectors;

    // ========================================================================
    // CORE PERMUTATION TYPE
    // Maps logic indices (Normal, Transverse1, Transverse2) to storage indices
    // (I, J, K) for generic kernel implementation.
    // ========================================================================

    template <std::uint8_t L, std::uint8_t M, std::uint8_t N>
    struct coord_permutation_t {
        static constexpr std::uint8_t horizontal_axis = L;   // First transverse
        static constexpr std::uint8_t vertical_axis = M;   // Second transverse
        static constexpr std::uint8_t normal_axis   = N;   // Field component

        // Indices of the fluxes needed to compute E_N
        static constexpr auto flux_indices()
        {
            return vector_t<std::uint64_t, 2>{L, M};
        }

        // For a given B component, which direction does E_N vary?
        // Used for reconstructing E at corners.
        static constexpr auto vary_index(magnetic_comp_t mag_comp)
        {
            if (mag_comp == magnetic_comp_t::I) {
                return M;   // Bx update needs dEz/dy
            }
            if (mag_comp == magnetic_comp_t::J) {
                return (L == 2) ? L : M;
            }
            return L;
        }

        static constexpr auto e_field_component() { return N; }
    };

    // Standard permutations for 3D MHD
    // Corresponds to E_z (from Fx, Fy), E_y (from Fx, Fz), E_x (from Fy, Fz)
    using IJ_permutation =
        coord_permutation_t<2, 1, 0>;   // E_z (fluxes x,y) - Wait, index 0 is
                                        // z?
    // Clarification on Indexing:
    // Simbi standard: 0=Z (k), 1=Y (j), 2=X (i)
    // E_z is component 0. Needs Flux X (2) and Flux Y (1). -> <2, 1, 0>
    // E_y is component 1. Needs Flux X (2) and Flux Z (0). -> <2, 0, 1>
    // E_x is component 2. Needs Flux Y (1) and Flux Z (0). -> <1, 0, 2>

    using Ez_perm = coord_permutation_t<2, 1, 0>;
    using Ey_perm = coord_permutation_t<2, 0, 1>;
    using Ex_perm = coord_permutation_t<1, 0, 2>;

    // ========================================================================
    // STENCIL UTILITIES
    // ========================================================================

    // Convert face/cell index to "doubled" index (2x resolution)
    // This allows representing corners and edges integer coordinates
    constexpr auto to_doubled_coord(const iarray<3>& coord)
    {
        return iarray<3>{2 * coord[0], 2 * coord[1], 2 * coord[2]};
    }

    // Convert doubled index back to array storage index (floor division)
    constexpr auto to_array_index_coord(const iarray<3>& doubled_coord)
    {
        return iarray<3>{
          doubled_coord[0] / 2,
          doubled_coord[1] / 2,
          doubled_coord[2] / 2
        };
    }

    // Generate stencil coordinates for fluxes around an edge
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
        constexpr auto half = 1;   // 1 unit in doubled space = 0.5 cell
        return vector_t{
          make_flux_coord(half, +1),   // F_h(i+1/2, j+1)
          make_flux_coord(half, -1),   // F_h(i+1/2, j)
          make_flux_coord(+1, half),   // F_v(i+1, j+1/2)
          make_flux_coord(-1, half)    // F_v(i, j+1/2)
        };
    }

    // Generate stencil coordinates for primitives (cell centers)
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
        // Offsets from edge to cell centers (+/- 0.5 cell = +/- 1 doubled)
        return vector_t{
          make_prim_coord(+1, +1),   // NE
          make_prim_coord(-1, +1),   // NW
          make_prim_coord(+1, -1),   // SE
          make_prim_coord(-1, -1)    // SW
        };
    }

    // ========================================================================
    // DATA EXTRACTION HELPERS
    // ========================================================================

    template <typename Permutation, typename FluxField>
    DEV inline auto
    face_efields(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        // Map logical Horizontal/Vertical to storage indices
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();
        constexpr std::uint64_t rank  = FluxField::rank;

        // The Electric field at a face is (-v x B) part of the flux
        // In Simbi, flux.mag stores the Electric field term directly if
        // shift_electric_field is used
        constexpr auto nhat = ehat<rank>(Permutation::e_field_component());

        return vector_t{
          flux[h_flux_idx](coords[0]).mag[index(nhat)],
          flux[h_flux_idx](coords[1]).mag[index(nhat)],
          flux[v_flux_idx](coords[2]).mag[index(nhat)],
          flux[v_flux_idx](coords[3]).mag[index(nhat)]
        };
    }

    template <typename Permutation, typename FluxField>
    DEV inline auto
    den_fluxes(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
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
    DEV inline auto
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
    // PHASE 1: COMPUTE EDGE E-FIELDS
    // ========================================================================

    template <std::uint64_t EComp>
    constexpr auto efield_permutation()
    {
        if constexpr (EComp == 0) {
            return Ez_perm{};   // Ez (0)
        }
        else if constexpr (EComp == 1) {
            return Ey_perm{};   // Ey (1)
        }
        else {
            return Ex_perm{};   // Ex (2)
        }
    }

    template <std::uint64_t EdgeComp, typename FluxField, typename PrimField>
    struct compute_edge_efield_op_t {
        FluxField fluxes;
        PrimField prims;

        DEV auto operator()(iarray<3> edge_coord) const
        {
            constexpr auto perm = efield_permutation<EdgeComp>();

            // Coordinates in the "doubled" lattice (2x resolution)
            // Edge centers on doubled grid have one odd coordinate (the edge
            // axis) But here we are mapping from the 'edge_domain' which uses
            // standard indices. E_z(i,j,k) lives at (i+1/2, j+1/2, k). doubled:
            // (2i+1, 2j+1, 2k).

            // The 'edge_coord' passed here is the storage index.
            // We need to shift to the physical location for stencil gathering.
            auto doubled = to_doubled_coord(edge_coord);

            // Shift to edge location based on component
            // Ex (2): lives at (i, j+1/2, k+1/2) -> shift Y and Z
            if constexpr (EdgeComp == 2) {
                doubled[1] += 1;
                doubled[0] += 1;
            }   // X is 2
            if constexpr (EdgeComp == 1) {
                doubled[2] += 1;
                doubled[0] += 1;
            }   // Y is 1
            if constexpr (EdgeComp == 0) {
                doubled[2] += 1;
                doubled[1] += 1;
            }   // Z is 0

            auto flux_coords = flux_stencil<decltype(perm)>(doubled);
            auto prim_coords = prim_stencil<decltype(perm)>(doubled);

            auto ef = face_efields<decltype(perm)>(fluxes, flux_coords);
            auto ec = center_efields<decltype(perm)>(prims, prim_coords);
            auto df = den_fluxes<decltype(perm)>(fluxes, flux_coords);

            return ct_contact_formula(ef, ec, df);
        }
    };

    template <
        typename Executor,
        typename HydroState,   // Contains .flux[] and .prim
        typename EdgeDomains,
        typename FaceDomain,
        typename Domain>
    void compute_edge_efields(
        Executor& exec,
        HydroState& state,
        const EdgeDomains& edge_domains,
        const FaceDomain& face_domains,
        const Domain& cell_domain
    )
    {
        // Create lightweight view of fluxes
        const auto fluxes = vector_t{
          state.flux[0][face_domains[0]],
          state.flux[1][face_domains[1]],
          state.flux[2][face_domains[2]]
        };
        const auto prims = state.prim[cell_domain];   // must allow ghost access

        // Launch computation for each component
        // E_z (0)
        state.efield[0][edge_domains[0]] =
            state.efield[0][edge_domains[0]]
                .space_map(
                    compute_edge_efield_op_t<
                        0,
                        decltype(fluxes),
                        decltype(prims)>{fluxes, prims}
                )
                .with(exec);

        // E_y (1)
        state.efield[1][edge_domains[1]] =
            state.efield[1][edge_domains[1]]
                .space_map(
                    compute_edge_efield_op_t<
                        1,
                        decltype(fluxes),
                        decltype(prims)>{fluxes, prims}
                )
                .with(exec);

        // E_x (2)
        state.efield[2][edge_domains[2]] =
            state.efield[2][edge_domains[2]]
                .space_map(
                    compute_edge_efield_op_t<
                        2,
                        decltype(fluxes),
                        decltype(prims)>{fluxes, prims}
                )
                .with(exec);
    }

    // ========================================================================
    // PHASE 2: UPDATE MAGNETIC FIELDS
    // ========================================================================

    template <typename BField, typename Geometry>
    struct interpolate_magnetic_op_t {
        BField b1, b2, b3;   // Bx, By, Bz views
        Geometry geometry;

        DEV auto operator()(iarray<3> coord) const
        {
            using metric_t = typename Geometry::metric_type;

            auto get_avg = [&](const auto& field, int dim) {
                auto c_plus = coord;
                c_plus[dim] += 1;   // face index n+1

                if constexpr (geometry::is_cartesian_c<metric_t>) {
                    return 0.5 * (field(coord) + field(c_plus));
                }
                else {
                    // Volume weighted average
                    // V_cell * B_center = (V_L*B_L + V_R*B_R) ? No
                    // Standard: Arithmetic is usually fine for visualization,
                    // but consistency requires: B_c = 0.5*(B_l + B_r)
                    // For rigorous volume average in curvilinear,
                    // implementation is complex.
                    return 0.5 * (field(coord) + field(c_plus));
                }
            };

            return vector_t<real, 3>{
              get_avg(b3, 2),   // Bx (dim 2)
              get_avg(b2, 1),   // By (dim 1)
              get_avg(b1, 0)    // Bz (dim 0)
            };
        }
    };

    template <magnetic_comp_t MagComp, typename EField, typename Geometry>
    struct ct_update_from_efield_op_t {
        EField efield;
        real dt;
        Geometry geometry;

        DEV auto operator()(iarray<3> face_coord) const
        {
            constexpr auto comp = static_cast<std::uint64_t>(MagComp);

            // Indices of transverse components
            constexpr std::uint64_t t1 = (comp + 1) % 3;
            constexpr std::uint64_t t2 = (comp + 2) % 3;

            // We need to feed the discrete_curl function with:
            // edge_emfs[0] -> along direction t1
            // edge_emfs[1] -> along direction t2

            // E_t1 edges are at: (face_coord) and (face_coord + 1_in_t2)
            auto c_t1_lo = face_coord;
            auto c_t1_hi = face_coord;
            c_t1_hi[t2] += 1;

            // E_t2 edges are at: (face_coord) and (face_coord + 1_in_t1)
            auto c_t2_lo = face_coord;
            auto c_t2_hi = face_coord;
            c_t2_hi[t1] += 1;

            // Gather into the format expected by ct_geom
            vector_t<vector_t<real, 2>, 2> edges;

            // discrete_curl expects:
            // [0] = edge along first transverse (t1)
            // [1] = edge along second transverse (t2)

            edges[0][0] = efield[t1](c_t1_lo);
            edges[0][1] = efield[t1](c_t1_hi);

            edges[1][0] = efield[t2](c_t2_lo);
            edges[1][1] = efield[t2](c_t2_hi);

            // Compute curl and apply -dt
            real curl = discrete_curl<MagComp>(edges, face_coord, geometry);
            return -dt * curl;
        }
    };

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
        // Lightweight view vector of E-fields
        const auto efields = vector_t{
          state.efield[0][edge_domains[0]],
          state.efield[1][edge_domains[1]],
          state.efield[2][edge_domains[2]]
        };

        // Update Bz (0)
        state.bfield[0][face_domains[0]] =
            state.bfield[0][face_domains[0]]
                .enum_map(
                    [op = ct_update_from_efield_op_t<
                         magnetic_comp_t::K,
                         decltype(efields),
                         Geometry>{efields, dt, geometry}](auto coord, auto b) {
                        return b + op(coord);
                    }
                )
                .with(exec);

        // Update By (1)
        state.bfield[1][face_domains[1]] =
            state.bfield[1][face_domains[1]]
                .enum_map(
                    [op = ct_update_from_efield_op_t<
                         magnetic_comp_t::J,
                         decltype(efields),
                         Geometry>{efields, dt, geometry}](auto coord, auto b) {
                        return b + op(coord);
                    }
                )
                .with(exec);

        // Update Bx (2)
        state.bfield[2][face_domains[2]] =
            state.bfield[2][face_domains[2]]
                .enum_map(
                    [op = ct_update_from_efield_op_t<
                         magnetic_comp_t::I,
                         decltype(efields),
                         Geometry>{efields, dt, geometry}](auto coord, auto b) {
                        return b + op(coord);
                    }
                )
                .with(exec);

        // Interpolate to cell center and update energy
        auto b_interp_op =
            interpolate_magnetic_op_t<decltype(state.bfield[0]), Geometry>{
              state.bfield[2],
              state.bfield[1],
              state.bfield[0],
              geometry
            };

        state.cons[cell_domain] =
            state.cons[cell_domain]
                .enum_map([b_interp_op](auto coord, auto u) {
                    auto b_new = b_interp_op(coord);
                    auto b_old = u.mag;
                    // Energy correction: u.E += 0.5*B_new^2 - 0.5*B_old^2
                    u.nrg += 0.5 * (vecops::dot(b_new, b_new) -
                                    vecops::dot(b_old, b_old));
                    u.mag = b_new;
                    return u;
                })
                .with(exec);
    }

}   // namespace simbi::em

#endif   // PHYSICS_EM_MHD_NLOGIC_HPP
