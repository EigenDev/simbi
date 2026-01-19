// =============================================================================
// exchange_pattern.hpp
//
// builds the halo exchange pattern for a decomposed grid.
// defines `exchange_builder_t`, a service that constructs a vector of
// `transfer_op_t` objects. each object describes a required data transfer
// (halo exchange) between neighboring grid blocks based on the grid's
// `skeleton_t`.
//
// usage:
//   auto pattern = exchange_builder_t::create(skeleton, halo_width, geo_calc);
// =============================================================================
#pragma once

#include "connectivity.hpp"
#include "domain.hpp"
#include "grid/amr/geometry.hpp"
#include "patch_id.hpp"
#include "skeleton.hpp"

#include <cstdint>
#include <vector>

namespace simbi::grid {

    template <std::uint64_t Rank>
    struct transfer_op_t
    {
        patch_id_t     src_id;
        patch_id_t     dst_id;
        std::int64_t   dim;
        side_t         side;
        domain_t<Rank> send_box;
        domain_t<Rank> recv_box;
    };

    struct exchange_builder_t
    {

        template <std::uint64_t Rank>
        static domain_t<Rank> compute_halo_slice(
            const domain_t<Rank>& geom,
            std::int64_t          dim,
            side_t                side,
            std::int64_t          halo_width
        )
        {
            // ... (same as before) ...
            domain_t<Rank> slice = geom;
            if (side == side_t::right) {
                slice.start[dim] = geom.fin[dim] - halo_width;
                slice.fin[dim]   = geom.fin[dim];
            }
            else {
                slice.fin[dim]   = geom.start[dim] + halo_width;
                slice.start[dim] = geom.start[dim];
            }
            return slice;
        }

        template <std::uint64_t Rank>
        static std::vector<transfer_op_t<Rank>> create(
            const skeleton_t<Rank>&           skeleton,
            std::int64_t                      halo_width,
            const amr::geometry_calculator_t& geo_calc // NEW dependency
        )
        {
            std::vector<transfer_op_t<Rank>> ops;

            for (const auto& [my_id, info] : skeleton) {
                for (std::uint64_t d = 0; d < Rank; ++d) {
                    // ... (left/right iteration logic) ...

                    // generic processor for a face
                    auto process_face = [&](side_t s) {
                        const auto& conn = info.get_face(d, s);
                        if (!conn.is_connected()) {
                            return;
                        }

                        // my full face slice (in my level coords)
                        auto my_slice = compute_halo_slice(info.geometry, d, s, halo_width);

                        for (const auto& neighbor_id : conn.neighbors) {
                            transfer_op_t<Rank> op;
                            op.src_id = my_id;
                            op.dst_id = neighbor_id;
                            op.dim    = d;
                            op.side   = s;

                            // calculate neighbor's geometric bounds
                            auto neighbor_geom_native = geo_calc.get_domain<Rank>(neighbor_id);

                            // map neighbor to my level
                            // this allows us to intersect in a common
                            // coordinate system
                            auto neighbor_geom_projected = geo_calc.map_domain(
                                neighbor_geom_native,
                                neighbor_id.level,
                                my_id.level
                            );

                            // intersect to find the shared surface
                            // "what part of my face does this neighbor actually
                            // cover?" note: we intersect the slice, not the
                            // whole block. we must expand the neighbor geom
                            // slightly to catch the ghost region overlap? no,
                            // 'my_slice' is interior. the neighbor is exterior.
                            // they strictly do not overlap in space. they
                            // touch.

                            // wait! logic check.
                            // 'my_slice' is indices [98, 100].
                            // neighbor is [100, 200].
                            // intersection is empty.

                            // the "projection" logic needs to act on the face
                            // plane. we ignore the dimension 'd' for the
                            // intersection check.

                            domain_t<Rank> valid_send = my_slice;
                            for (std::uint64_t i = 0; i < Rank; ++i) {
                                if (i == d) {
                                    continue; // skip normal direction
                                }

                                // clip transverse dimensions
                                valid_send.start[i] =
                                    std::max(valid_send.start[i], neighbor_geom_projected.start[i]);
                                valid_send.fin[i] =
                                    std::min(valid_send.fin[i], neighbor_geom_projected.fin[i]);
                            }

                            op.send_box = valid_send;

                            // note: recv_box calculation requires mapping back
                            // to dest level for now, we leave it to the
                            // communicator or calc it here:
                            op.recv_box =
                                geo_calc.map_domain(valid_send, my_id.level, neighbor_id.level);

                            // shift recv_box to ghost position?
                            // handled by the logic of "who receives".
                            // usually we specify "where the data goes".
                            // that logic is complex (offsetting to ghost
                            // region).

                            if (!valid_send.empty()) {
                                ops.push_back(op);
                            }
                        }
                    };

                    process_face(side_t::left);
                    process_face(side_t::right);
                }
            }
            return ops;
        }
    };

} // namespace simbi::grid
