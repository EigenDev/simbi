#ifndef SIMBI_CT_PERMUTATION_HPP
#define SIMBI_CT_PERMUTATION_HPP

#include "domain/domain.hpp"

#include "compute/functional/fp.hpp"
#include "config.hpp"
#include "contact.hpp"
#include "containers/vector.hpp"
#include "core/utility/enums.hpp"
#include "ct_geom.hpp"
#include "physics/em/electromagnetism.hpp"
#include "physics/em/interp.hpp"
#include <cstdint>
#include <utility>

namespace simbi::em {
    using namespace simbi::unit_vectors;

    // ========================================================================
    // CORE PERMUTATION TYPE
    // ========================================================================

    template <std::uint8_t L, std::uint8_t M, std::uint8_t N>
    struct coord_permutation_t {
        // L = horizontal axis index in array coordinates [K,J,I]
        // M = vertical axis index in array coordinates
        // N = normal axis index in array coordinates

        static constexpr auto horizontal_axis = L;
        static constexpr auto vertical_axis   = M;
        static constexpr auto normal_axis     = N;

        // get flux component indices for this permutation
        static constexpr auto flux_indices()
        {
            return vector_t<std::uint64_t, 2>{
              L,
              M
            };   // horizontal and vertical flux components
        }

        static constexpr auto vary_index(magnetic_comp_t mag_comp)
        {
            // return the varying index based on magnetic component
            if (mag_comp == magnetic_comp_t::I) {
                // if permutation is IK, vary k,
                // if permutation is IJ, vary j
                return M;
            }
            else if (mag_comp == magnetic_comp_t::J) {
                if constexpr (L == 2) {
                    return L;   // I varies in horizontal direction
                }
                else {
                    return M;   // K varies in vertical direction
                }
            }
            else {
                return L;   // I,J vary in horizontal direction
            }
        }

        // get E-field component this permutation computes
        static constexpr auto e_field_component()
        {
            return N;   // normal direction determines E-field component
        }

        // apply coordinate transformation through this permutation
        template <typename Func>
        static constexpr auto
        apply_transform(Func&& func, const iarray<3>& base_coord)
        {
            return std::forward<Func>(func)(base_coord, L, M, N);
        }
    };

    // ========================================================================
    // THE THREE FUNDAMENTAL PERMUTATIONS (Array indexing: [K,J,I])
    // ========================================================================

    // I=horiz, J=vert, K=normal
    using IJ_permutation = coord_permutation_t<2, 1, 0>;
    // J=horiz, K=vert, I=normal
    using JK_permutation = coord_permutation_t<1, 0, 2>;
    // I=horiz, K=vert, J=normal
    using IK_permutation = coord_permutation_t<2, 0, 1>;

    // type-level permutation list b/c I'm feeling fancy
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

    // compile-time fold over permutation types
    template <typename PermList, typename Func>
    constexpr auto fold_permutations(Func&& func);

    template <typename... Perms, typename Func>
    constexpr auto fold_permutations(permutation_list_t<Perms...>, Func&& func)
    {
        return vector_t{func(Perms{})...};
    }

    template <typename PermList, typename Func>
    constexpr auto map_permutations(Func&& func);

    template <typename... Perms, typename Func>
    constexpr auto map_permutations(permutation_list_t<Perms...>, Func&& func)
    {
        return vector_t{func(Perms{})...};
    }

    // ========================================================================
    // COORDINATE UTILITIES
    // ========================================================================

    // convert doubled coordinate to array index
    constexpr std::int64_t to_array_index(int doubled_coord)
    {
        return doubled_coord / 2;
    }

    // convert array indices to doubled coordinate for face centers
    constexpr auto to_doubled_coord(const iarray<3>& coord)
    {
        return iarray<3>{2 * coord[0], 2 * coord[1], 2 * coord[2]};
    }

    // convert doubled coords to array indices
    constexpr auto to_array_index_coord(const iarray<3>& doubled_coord)
    {
        return iarray<3>{
          to_array_index(doubled_coord[0]),
          to_array_index(doubled_coord[1]),
          to_array_index(doubled_coord[2])
        };
    }

    // ========================================================================
    // GENERIC COORDINATE GENERATION USING PERMUTATIONS
    // ========================================================================
    template <magnetic_comp_t MagComp, typename Permutation>
    constexpr auto gen_edge_coords(const iarray<3>& face_coord)
    {
        // determine which index to VARY based on permutation
        // (from the 2 non-fixed indices, pick one based on permutation)
        constexpr auto vary_index = Permutation::vary_index(MagComp);
        auto face_doubled         = to_doubled_coord(face_coord);

        // take the face-centered index and adjust it to the semantic
        // face coordinate in the doubled system
        constexpr auto face_idx = static_cast<uint8_t>(MagComp);
        constexpr auto half = 1;   // conceptual half-step for normal direction
        face_doubled[face_idx] -= half;

        auto make_edge = [&](int offset) {
            auto edge = face_doubled;
            edge[vary_index] += offset;   // vary only in this direction
            return edge;
        };

        return vector_t<iarray<3>, 2>{
          make_edge(-1),   // negative direction
          make_edge(+1)    // positive direction
        };
    }

    template <typename Permutation>
    constexpr auto flux_stencil(const iarray<3>& edge_doubled_coord)
    {
        // for any permutation, flux stencil is \pm 1 in the tangent directions
        auto make_flux_coord = [&](std::int64_t h_offset,
                                   std::int64_t v_offset) {
            auto coord = edge_doubled_coord;
            coord[Permutation::horizontal_axis] += h_offset;
            coord[Permutation::vertical_axis] += v_offset;
            return to_array_index_coord(coord);
        };
        constexpr auto half = 1;   // conceptual half-step for fluxes
        // standard N/S/E/W pattern in permuted coordinates
        return vector_t{
          make_flux_coord(half, +1),   // North (positive vertical)
          make_flux_coord(half, -1),   // South (negative vertical)
          make_flux_coord(+1, half),   // East (positive horizontal)
          make_flux_coord(-1, half)    // West (negative horizontal)
        };
    }

    template <typename Permutation>
    constexpr auto prim_stencil(const iarray<3>& edge_doubled_coord)
    {
        // primitive stencil is \pm 1 in both directions (cell diagonals)

        auto make_prim_coord = [&](std::int64_t h_offset,
                                   std::int64_t v_offset) {
            auto coord = edge_doubled_coord;
            coord[Permutation::horizontal_axis] += h_offset;
            coord[Permutation::vertical_axis] += v_offset;
            return to_array_index_coord(coord);
        };

        return vector_t{
          make_prim_coord(+1, +1),   // NE corner
          make_prim_coord(-1, +1),   // NW corner
          make_prim_coord(+1, -1),   // SE corner
          make_prim_coord(-1, -1)    // SW corner
        };
    }

    // ========================================================================
    // PERMUTATION-AWARE FIELD ACCESS
    // ========================================================================

    template <typename Permutation, typename FluxField>
    auto
    face_efields(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        // when computing the Riemann solver, we store the electric field
        // in the magnetic field portion of the flux field for convenience.
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();
        constexpr auto dims           = FluxField::dimensions;
        constexpr auto nhat = ehat<dims>(Permutation::e_field_component());

        return vector_t{
          // North - horizontal flux
          flux[h_flux_idx][coords[0]].mag[index(nhat)],
          // South - horizontal flux
          flux[h_flux_idx][coords[1]].mag[index(nhat)],
          // East - vertical flux
          flux[v_flux_idx][coords[2]].mag[index(nhat)],
          // West - vertical flux
          flux[v_flux_idx][coords[3]].mag[index(nhat)]
        };
    }

    template <typename Permutation, typename FluxField>
    auto den_fluxes(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        // useful for computing CT contact EMF b/c of upwinding
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();

        return vector_t<real, 4>{
          flux[h_flux_idx][coords[0]].den,   // North density flux
          flux[h_flux_idx][coords[1]].den,   // South density flux
          flux[v_flux_idx][coords[2]].den,   // East density flux
          flux[v_flux_idx][coords[3]].den    // West density flux
        };
    }

    template <typename Permutation, typename PrimField>
    auto
    center_efields(const PrimField& prim, const vector_t<iarray<3>, 4>& coords)
    {
        constexpr auto dims = PrimField::dimensions;
        constexpr auto nhat = ehat<dims>(Permutation::e_field_component());

        return vector_t{
          em::electric_field(prim[coords[0]])[index(nhat)],   // NE
          em::electric_field(prim[coords[1]])[index(nhat)],   // NW
          em::electric_field(prim[coords[2]])[index(nhat)],   // SE
          em::electric_field(prim[coords[3]])[index(nhat)]    // SW
        };
    }

    // ========================================================================
    // MAGNETIC COMPONENT TO PERMUTATION MAPPING
    // ========================================================================
    template <magnetic_comp_t MagComp, typename Func>
    constexpr auto apply_to_permutations(Func&& func)
    {
        if constexpr (MagComp == magnetic_comp_t::K) {
            return vector_t{func(JK_permutation{}), func(IK_permutation{})};
        }
        else if constexpr (MagComp == magnetic_comp_t::J) {
            return vector_t{func(JK_permutation{}), func(IJ_permutation{})};
        }
        else {
            return vector_t{func(IJ_permutation{}), func(IK_permutation{})};
        }
    }

    // ========================================================================
    // UNIFIED EDGE EMF COMPUTATION
    // ========================================================================

    template <typename Permutation, typename HydroState>
    auto compute_edge_emf(
        const iarray<3>& edge_doubled_coord,
        const HydroState& state
    )
    {
        const auto fluxes = vector_t{
          state.flux[0].contract(iarray<3>{0, 1, 1}),
          state.flux[1].contract(iarray<3>{1, 0, 1}),
          state.flux[2].contract(iarray<3>{1, 1, 0})
        };
        const auto p = state.prim[state.mesh.domain];
        // generate all stencil coordinates using permutation
        auto flux_coords = flux_stencil<Permutation>(edge_doubled_coord);
        auto prim_coords = prim_stencil<Permutation>(edge_doubled_coord);

        // extract field values using permutation-aware accessors
        auto flux_e_fields  = face_efields<Permutation>(fluxes, flux_coords);
        auto cell_e_fields  = center_efields<Permutation>(p, prim_coords);
        auto density_fluxes = den_fluxes<Permutation>(fluxes, flux_coords);

        // compute EMF using CT Contact algorithm
        return ct_contact_formula(flux_e_fields, cell_e_fields, density_fluxes);
    }

    //========================================================================
    // MAGNETIC FIELD UPDATE EXPRESSION
    // ========================================================================
    template <magnetic_comp_t MagComp, typename HydroState>
    struct ct_magnetic_update_t
        : expr::expression_t<ct_magnetic_update_t<MagComp, HydroState>> {
        HydroState& state;

        auto domain() const
        {
            if constexpr (MagComp == magnetic_comp_t::I) {
                return state.flux[2].domain();
            }
            else if constexpr (MagComp == magnetic_comp_t::J) {
                return state.flux[1].domain();
            }
            else {
                return state.flux[0].domain();
            }
        }

        template <typename Coord>
        real eval(Coord face_coord) const
        {
            constexpr auto perm_list = permutation_list<MagComp>();

            const auto emf_computer = [&]<typename Permutation>(Permutation) {
                return [&](const iarray<3>& edge_coord) {
                    return compute_edge_emf<Permutation>(edge_coord, state);
                };
            };

            const auto edge_generator = [&]<typename Permutation>(Permutation) {
                return gen_edge_coords<MagComp, Permutation>(face_coord);
            };
            auto emfs = map_permutations(perm_list, [&]<typename Perm>(Perm p) {
                return edge_generator(p) | fp::map(emf_computer(p)) |
                       fp::collect<vector_t<real, 2>>;
            });

            // apply discrete curl
            real curl = discrete_curl<MagComp>(emfs, face_coord, state.mesh);

            // Faraday's law: ∂B/∂t = -∇ × E
            return -state.metadata.dt * curl;
        }
    };

    template <magnetic_comp_t MagComp, typename HydroState>
    auto field_update(HydroState& state)
    {
        return ct_magnetic_update_t<MagComp, HydroState>{.state = state};
    }

    // ========================================================================
    // HIGH-LEVEL INTERFACE
    // ========================================================================
    template <typename HydroState, std::uint64_t Dims = HydroState::dimensions>
    void update_energy_density(
        HydroState& state,
        const domain_t<Dims>& active_domain
    )
    {
        auto u    = state.cons[active_domain];
        auto bavg = interpolate_face_to_cell_magnetic(state, active_domain);

        u = u.map([bavg](const auto& coord, auto u) {
            // update energy to maintain consistency as discussed in
            // Mignone and Bodo 2006,  (Eqn. 76)
            // and Balsara & Spicer 1999,
            // and Toth 2000,
            auto b_interp = bavg(coord);
            auto bmean    = u.mag;
            auto old_emag = 0.5 * vecops::dot(bmean, bmean);
            auto new_emag = 0.5 * vecops::dot(b_interp, b_interp);
            u.nrg += (new_emag - old_emag);
            return u;
        });
    }

    template <typename HydroState, std::uint64_t Dims = HydroState::dimensions>
    void interpolate_magnetic_fields(
        HydroState& state,
        const domain_t<Dims>& active_domain
    )
    {
        active_domain |
            fp::transform_domain(interpolate_face_to_cell_magnetic(state)) |
            fp::for_each([&state](const auto& pair) {
                const auto [coord, bavg] = pair;
                // update conversed magnetic field
                state[active_domain][coord].mag = bavg;
            });
    }

    template <magnetic_comp_t MagComp, typename HydroState, typename MeshConfig>
    void update_magnetic_component(HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto comp    = static_cast<std::uint64_t>(MagComp);
        constexpr auto ct_algo = field_update<MagComp, HydroState>;
        const auto face_domain = mesh.face_domain[comp];
        state.bstaggs[comp][face_domain] += ct_algo(state);
    }

    template <typename HydroState, typename MeshConfig>
    void update_magnetic_fields(HydroState& state, const MeshConfig& mesh)
    {
        update_magnetic_component<magnetic_comp_t::I>(state, mesh);
        update_magnetic_component<magnetic_comp_t::J>(state, mesh);
        update_magnetic_component<magnetic_comp_t::K>(state, mesh);
        interpolate_magnetic_fields(state, mesh.domain);
    }

}   // namespace simbi::em

#endif   // SIMBI_CT_PERMUTATION_HPP
