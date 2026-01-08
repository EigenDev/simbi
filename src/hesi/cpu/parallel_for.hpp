#ifndef HET_BACKEND_CPU_PARALLEL_FOR_HPP
#define HET_BACKEND_CPU_PARALLEL_FOR_HPP

#include "containers/vector.hpp"
#include "grid/domain.hpp"
#include "hesi/core/types.hpp"
#include "hesi/exec/policy.hpp"

#include <algorithm>
#include <cstdint>

namespace simbi::het::backend::cpu {

    // serial iteration
    template <std::uint64_t Rank, typename Functor>
    void parallel_for_serial(const grid::domain_t<Rank>& domain, Functor&& f)
    {
        iarray<Rank> coord = domain.start;

        if constexpr (Rank == 1) {
            for (coord[0] = domain.start[0]; coord[0] < domain.fin[0];
                 ++coord[0]) {
                f(coord);
            }
        }
        else if constexpr (Rank == 2) {
            for (coord[0] = domain.start[0]; coord[0] < domain.fin[0];
                 ++coord[0]) {
                for (coord[1] = domain.start[1]; coord[1] < domain.fin[1];
                     ++coord[1]) {
                    f(coord);
                }
            }
        }
        else if constexpr (Rank == 3) {
            for (coord[0] = domain.start[0]; coord[0] < domain.fin[0];
                 ++coord[0]) {
                for (coord[1] = domain.start[1]; coord[1] < domain.fin[1];
                     ++coord[1]) {
                    for (coord[2] = domain.start[2]; coord[2] < domain.fin[2];
                         ++coord[2]) {
                        f(coord);
                    }
                }
            }
        }
    }

    // openmp tiled iteration
    template <std::uint64_t Rank, typename Functor>
    void parallel_for_omp(
        const grid::domain_t<Rank>& domain,
        const exec::launch_policy_t& policy,
        Functor&& f
    )
    {
        auto tile_size = policy.block;

        // decompose domain into tiles
        auto shape = domain.shape();

        std::uint64_t tx = std::max<std::uint32_t>(1, tile_size.x);
        std::uint64_t ty = std::max<std::uint32_t>(1, tile_size.y);
        std::uint64_t tz = std::max<std::uint32_t>(1, tile_size.z);

        std::uint32_t sx = (Rank >= 1) ? shape[Rank - 1] : 1;
        std::uint32_t sy = (Rank >= 2) ? shape[Rank - 2] : 1;
        std::uint32_t sz = (Rank == 3) ? shape[0] : 1;

        if constexpr (Rank == 1) {
            sx = shape[0];
            sy = 1;
            sz = 1;
        }

        dim3_t num_tiles{
          static_cast<std::uint32_t>((sx + tx - 1) / tx),
          static_cast<std::uint32_t>((sy + ty - 1) / ty),
          static_cast<std::uint32_t>((sz + tz - 1) / tz)
        };

#pragma omp parallel for collapse(3)
        for (std::uint64_t tile_z = 0; tile_z < num_tiles.z; ++tile_z) {
            for (std::uint64_t tile_y = 0; tile_y < num_tiles.y; ++tile_y) {
                for (std::uint64_t tile_x = 0; tile_x < num_tiles.x; ++tile_x) {
                    // calculate tile bounds
                    grid::domain_t<Rank> tile = domain;

                    if constexpr (Rank == 1) {
                        tile.start[0] = domain.start[0] + tile_x * tx;
                        tile.fin[0]   = std::min<std::int64_t>(
                            tile.start[0] + tx,
                            domain.fin[0]
                        );
                    }
                    else if constexpr (Rank == 2) {
                        tile.start[0] = domain.start[0] + tile_y * ty;
                        tile.start[1] = domain.start[1] + tile_x * tx;

                        tile.fin[0] = std::min<std::int64_t>(
                            tile.start[0] + ty,
                            domain.fin[0]
                        );
                        tile.fin[1] = std::min<std::int64_t>(
                            tile.start[1] + tx,
                            domain.fin[1]
                        );
                    }
                    else if constexpr (Rank == 3) {
                        tile.start[0] = domain.start[0] + tile_z * tz;
                        tile.start[1] = domain.start[1] + tile_y * ty;
                        tile.start[2] = domain.start[2] + tile_x * tx;

                        tile.fin[0] = std::min<std::int64_t>(
                            tile.start[0] + tz,
                            domain.fin[0]
                        );
                        tile.fin[1] = std::min<std::int64_t>(
                            tile.start[1] + ty,
                            domain.fin[1]
                        );
                        tile.fin[2] = std::min<std::int64_t>(
                            tile.start[2] + tx,
                            domain.fin[2]
                        );
                    }

                    // iterate tile serially
                    parallel_for_serial(tile, f);
                }
            }
        }
    }

    // dispatcher
    template <std::uint64_t Rank, typename Functor>
    void parallel_for(
        const exec::launch_policy_t& policy,
        const grid::domain_t<Rank>& domain,
        Functor&& f,
        bool use_openmp
    )
    {
        if (use_openmp) {
            parallel_for_omp(domain, policy, std::forward<Functor>(f));
        }
        else {
            parallel_for_serial(domain, std::forward<Functor>(f));
        }
    }

}   // namespace simbi::het::backend::cpu

#endif
