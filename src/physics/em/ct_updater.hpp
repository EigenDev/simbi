#ifndef MHD_LOGIC_HPP
#define MHD_LOGIC_HPP

#include "compute/field.hpp"
#include "compat.hpp"
#include "contact.hpp"
#include "containers/vector.hpp"
#include "ct_geom.hpp"
#include "domain/domain.hpp"
#include "functional/fp.hpp"
#include "physics/em/electromagnetism.hpp"
#include "utility/enums.hpp"

#include <cstdint>

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

    // the three fundamental permutations
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
        auto make_prim_coord = [&](std::int64_t h_offset,
                                   std::int64_t v_offset) {
            auto coord = edge_doubled_coord;
            coord[Permutation::horizontal_axis] += h_offset;
            coord[Permutation::vertical_axis] += v_offset;
            return to_array_index_coord(coord);
        };

        return vector_t{
          make_prim_coord(+1, +1),   // ne
          make_prim_coord(-1, +1),   // nw
          make_prim_coord(+1, -1),   // se
          make_prim_coord(-1, -1)    // sw
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
        constexpr auto dims           = FluxField::dimensions;
        constexpr auto nhat = ehat<dims>(Permutation::e_field_component());

        return vector_t{
          flux[h_flux_idx][coords[0]].mag[index(nhat)],   // north
          flux[h_flux_idx][coords[1]].mag[index(nhat)],   // south
          flux[v_flux_idx][coords[2]].mag[index(nhat)],   // east
          flux[v_flux_idx][coords[3]].mag[index(nhat)]    // west
        };
    }

    template <typename Permutation, typename FluxField>
    auto den_fluxes(const FluxField& flux, const vector_t<iarray<3>, 4>& coords)
    {
        auto [h_flux_idx, v_flux_idx] = Permutation::flux_indices();
        return vector_t<real, 4>{
          flux[h_flux_idx][coords[0]].den,   // north
          flux[h_flux_idx][coords[1]].den,   // south
          flux[v_flux_idx][coords[2]].den,   // east
          flux[v_flux_idx][coords[3]].den    // west
        };
    }

    template <typename Permutation, typename PrimField>
    auto
    center_efields(const PrimField& prim, const vector_t<iarray<3>, 4>& coords)
    {
        constexpr auto dims = PrimField::dimensions;
        constexpr auto nhat = ehat<dims>(Permutation::e_field_component());

        return vector_t{
          em::electric_field(prim[coords[0]])[index(nhat)],   // ne
          em::electric_field(prim[coords[1]])[index(nhat)],   // nw
          em::electric_field(prim[coords[2]])[index(nhat)],   // se
          em::electric_field(prim[coords[3]])[index(nhat)]    // sw
        };
    }

    // ========================================================================
    // CT MAGNETIC UPDATE
    // ========================================================================
    template <
        magnetic_comp_t MagComp,
        typename FluxField,
        typename PrimField,
        typename MeshConfig>
    struct ct_magnetic_update_op_t {
        FluxField fluxes;
        PrimField prims;
        real dt;
        MeshConfig mesh;

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

            real curl = discrete_curl<MagComp>(emfs, face_coord, mesh);
            return -dt * curl;   // Faraday's law
        }
    };

    template <magnetic_comp_t MagComp>
    auto make_ct_magnetic_update_op(
        auto&& fluxes,
        auto&& prims,
        real dt,
        auto&& mesh
    )
    {
        return ct_magnetic_update_op_t<
            MagComp,
            std::decay_t<decltype(fluxes)>,
            std::decay_t<decltype(prims)>,
            std::decay_t<decltype(mesh)>>{
          std::forward<decltype(fluxes)>(fluxes),
          std::forward<decltype(prims)>(prims),
          dt,
          std::forward<decltype(mesh)>(mesh)
        };
    }

    template <magnetic_comp_t MagComp, typename HydroState, typename MeshConfig>
    auto ct_magnetic_update(const HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto comp = static_cast<std::uint64_t>(MagComp);
        const auto fluxes   = vector_t{
          state.flux[0][mesh.face_domain[0]],
          state.flux[1][mesh.face_domain[1]],
          state.flux[2][mesh.face_domain[2]]
        };

        return compute_field_t{
          make_ct_magnetic_update_op<MagComp>(
              fluxes,
              state.prim[mesh.domain],
              state.metadata.dt,
              mesh
          ),
          make_domain(mesh.face_domain[comp].shape())
        };
    }

    // ========================================================================
    // INTERPOLATION FIELDS
    // ========================================================================
    template <typename Bfield, typename MeshConfig>
    struct interpolate_magnetic_op_t {
        Bfield b1;
        Bfield b2;
        Bfield b3;
        const MeshConfig& mesh;

        DEV auto get_face_avg(const auto& bface, auto cminus, int dir) const
        {
            const auto cplus = cminus + array_offset<3>(dir);
            if constexpr (MeshConfig::geometry == Geometry::CARTESIAN) {
                return 0.5 * (bface[cminus] + bface[cplus]);
            }
            else {
                // volume-average for non-Cartesian geometries
                auto al = mesh::face_area(cminus, dir, Dir::W, mesh);
                auto ar = mesh::face_area(cplus, dir, Dir::E, mesh);
                return (bface[cminus] * al + bface[cplus] * ar) / (al + ar);
            }
        }

        DEV auto operator()(auto coord) const
        {
            return vector_t<real, 3>{
              get_face_avg(b1, coord, 2),   // x1-component
              get_face_avg(b2, coord, 1),   // x2-component
              get_face_avg(b3, coord, 0)    // x3-component
            };
        }
    };

    template <typename HydroState, typename MeshConfig>
    auto interpolate_face_to_cell_magnetic(
        const HydroState& state,
        const MeshConfig& mesh
    )
    {
        return compute_field_t{
          interpolate_magnetic_op_t{
            state.bstaggs[2][mesh.face_domain[2]],
            state.bstaggs[1][mesh.face_domain[1]],
            state.bstaggs[0][mesh.face_domain[0]],
            mesh
          },
          make_domain(mesh.domain.shape())
        };
    }

    // ========================================================================
    // HIGH-LEVEL INTERFACE
    // ========================================================================

    template <typename HydroState, typename MeshConfig>
    void update_energy_density(HydroState& state, const MeshConfig& mesh)
    {
        auto bavg = interpolate_face_to_cell_magnetic(state, mesh);
        auto u_p  = state.cons[mesh.domain];

        u_p = u_p.enum_map([bavg](auto coord, auto u) {
            const auto b_interp = bavg(coord);
            const auto bmean    = u.mag;
            const auto old_emag = 0.5 * vecops::dot(bmean, bmean);
            const auto new_emag = 0.5 * vecops::dot(b_interp, b_interp);
            u.nrg += (new_emag - old_emag);
            return u;
        });
    }

    template <typename HydroState, typename MeshConfig>
    void interpolate_magnetic_fields(HydroState& state, const MeshConfig& mesh)
    {
        auto bavg = interpolate_face_to_cell_magnetic(state, mesh);
        auto u_p  = state.cons[mesh.domain];

        u_p = u_p.enum_map([bavg](auto coord, auto u) {
            // update magnetic field in the conservative state
            u.mag = bavg(coord);
            return u;
        });
    }

    template <magnetic_comp_t MagComp, typename HydroState, typename MeshConfig>
    void update_magnetic_component(HydroState& state, const MeshConfig& mesh)
    {
        constexpr auto comp = static_cast<std::uint64_t>(MagComp);
        auto db             = ct_magnetic_update<MagComp>(state, mesh);
        auto bfield         = state.bstaggs[comp][mesh.face_domain[comp]];
        bfield              = bfield.enum_map([db](auto coord, auto b_old) {
            return b_old + db(coord);
        });
    }

    template <typename HydroState, typename MeshConfig>
    void update_magnetic_fields(HydroState& state, const MeshConfig& mesh)
    {
        update_magnetic_component<magnetic_comp_t::I>(state, mesh);
        update_magnetic_component<magnetic_comp_t::J>(state, mesh);
        update_magnetic_component<magnetic_comp_t::K>(state, mesh);
        interpolate_magnetic_fields(state, mesh);
    }

}   // namespace simbi::em

#endif   // MHD_LOGIC_HPP
