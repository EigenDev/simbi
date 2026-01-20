// =============================================================================
// diagnostics.hpp
//
// runtime diagnostic utilities for debugging simulation failures.
// provides detailed error reporting when physics solvers fail.
//
// usage:
//   catch (SimulationFailureException& e) {
//       diagnose_cons2prim_failure(sim, progress_table);
//       throw;
//   }
// =============================================================================
#pragma once

#include "ecs/geometry_visitor.hpp"
#include "io/exceptions.hpp"
#include "physics/hydro/conversion.hpp"

#include <cstdint>
#include <sstream>

namespace simbi::diagnostics {

    // helper to iterate domain and find first cons2prim failure
    template <typename Sim, typename Table>
    void diagnose_cons2prim_failure(Sim& sim, Table& table)
    {
        const auto gamma  = sim.metadata().gamma;
        auto       motion = sim.motion_state();

        // scan all levels and partitions to find the failing zone
        for (std::uint64_t lvl = 0; lvl < sim.num_levels(); ++lvl) {
            for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
                auto&       fields = sim.partition_hydro(lvl, pp);
                auto&       part   = sim.partition(lvl, pp);
                const auto& mesh   = sim.mesh(lvl);
                const auto  domain = part.owned_domain; // only scan active cells

                // use geometry visitor to get physical coordinates
                ecs::with_block_geometry<Sim::coord_system>(
                    mesh,
                    motion,
                    [&](const auto& block_geo) {
                        // serial scan through domain using range-based for
                        for (const auto& coord : domain) {
                            auto cons       = fields.cons(coord);
                            auto maybe_prim = hydro::to_primitive(cons, gamma);

                            if (!maybe_prim.has_value()) {
                                // found the failure - extract all context
                                auto pos = block_geo.labframe_centroid(coord);

                                // post each line separately to avoid breaking the message board
                                std::ostringstream header;
                                header << "cons2prim failure at level " << lvl << ", partition "
                                       << pp << ", index " << format_coord(coord);
                                table.post_error(header.str());

                                std::ostringstream detail;
                                detail << "  position: " << format_position(pos) << ", error: "
                                       << helpers::error_code_to_string(maybe_prim.error_code());
                                table.post_error(detail.str());

                                table.post_error("  " + format_conserved(cons));

                                return;
                            }
                        }
                    }
                );
            }
        }

        // if we get here, no failure found (shouldn't happen)
        // table.post_error("cons2prim failure reported but no failing zone found");
    }

} // namespace simbi::diagnostics
