#ifndef SIMIB_MESH_MESH_CONFIG_HPP
#define SIMIB_MESH_MESH_CONFIG_HPP

#include "compute/functional/fp.hpp"
#include "compute/math/domain.hpp"
#include "config.hpp"   // for real, DUAL, DEV, global::using_four_velocity
#include "core/utility/bimap.hpp"             // for deserialize
#include "core/utility/enums.hpp"             // for Cellspacing enum
#include "core/utility/init_conditions.hpp"   // for InitialConditions
#include "data/containers/vector.hpp"         // for vector_t
#include <cstddef>                            // for std::size_t
#include <cstdint>                            // for std::int64_t
#include <functional>                         // for std::function

namespace simbi::mesh {
    template <std::uint64_t Dims, Geometry G>
    struct mesh_config_t {
        static constexpr auto geometry = G;
        iarray<Dims> shape;         // nk, nj, ni
        iarray<Dims> full_shape;    // nk+2*halo, nj+2*halo, ni+2*halo
        std::int64_t halo_radius;   // halo radius

        domain_t<Dims> full_domain;   // full domain with halo
        domain_t<Dims> domain;        // active computational domain

        vector_t<real, Dims> bounds_min;   // x1min, x2min, x3min
        vector_t<real, Dims> bounds_max;   // x1max, x2max, x3max
        // linear vs log per direction
        vector_t<Cellspacing, Dims> spacing_types;

        // time-dependent state
        bool homologous{false};
        bool mesh_motion{false};
        // current expansion (updated each timestep)
        real expansion_factor{1.0};
        real expansion_rate{0.0};
        std::function<real(real)> sf{nullptr};
        std::function<real(real)> sf_derivative{nullptr};

        DUAL constexpr std::uint64_t full_size() const
        {
            return fp::product(full_shape);
        }

        DUAL constexpr std::uint64_t size() const { return fp::product(shape); }

        // time-dependent bounds (handles homologous expansion)
        DUAL constexpr vector_t<real, Dims> current_bounds_min() const
        {
            if (!mesh_motion) {
                return bounds_min;
            }

            vector_t<real, Dims> result = bounds_min;
            // [TODO]: handle other dimensions if needed
            if (homologous) {
                // homologous expansion scales the bounds
                result[Dims - 1] *= expansion_factor;
            }
            else {
                // uniform expansion translation
                result[Dims - 1] += expansion_factor;
            }
            return result;
        }

        DUAL constexpr vector_t<real, Dims> current_bounds_max() const
        {
            if (!mesh_motion) {
                return bounds_max;
            }

            vector_t<real, Dims> result = bounds_max;
            if (homologous) {
                result[Dims - 1] *= expansion_factor;
            }
            else {
                result[Dims - 1] += expansion_factor;
            }
            return result;
        }

        static mesh_config_t from_init_conditions(
            const InitialConditions& init,
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
        )
        {
            mesh_config_t config;
            config.shape       = init.get_active_shape<Dims>();
            config.full_shape  = init.get_full_shape<Dims>();
            config.halo_radius = static_cast<std::int64_t>(init.halo_radius);

            config.domain      = make_domain(config.shape);
            config.full_domain = make_domain(config.full_shape);

            // geometry setup
            if constexpr (Dims == 1) {
                config.bounds_min = vector_t<real, 1>{init.x1bounds.first};
                config.bounds_max = vector_t<real, 1>{init.x1bounds.second};
            }
            else if constexpr (Dims == 2) {
                config.bounds_min =
                    vector_t<real, 2>{init.x2bounds.first, init.x1bounds.first};
                config.bounds_max = vector_t<real, 2>{
                  init.x2bounds.second,
                  init.x1bounds.second
                };
            }
            else if constexpr (Dims == 3) {
                config.bounds_min = vector_t<real, 3>{
                  init.x3bounds.first,
                  init.x2bounds.first,
                  init.x1bounds.first
                };
                config.bounds_max = vector_t<real, 3>{
                  init.x3bounds.second,
                  init.x2bounds.second,
                  init.x1bounds.second
                };
            }

            // spacing types
            if constexpr (Dims == 1) {
                config.spacing_types = vector_t<Cellspacing, 1>{
                  deserialize<Cellspacing>(init.x1_spacing)
                };
            }
            else if constexpr (Dims == 2) {
                config.spacing_types = vector_t<Cellspacing, 2>{
                  deserialize<Cellspacing>(init.x2_spacing),
                  deserialize<Cellspacing>(init.x1_spacing)
                };
            }
            else if constexpr (Dims == 3) {
                config.spacing_types = vector_t<Cellspacing, 3>{
                  deserialize<Cellspacing>(init.x3_spacing),
                  deserialize<Cellspacing>(init.x2_spacing),
                  deserialize<Cellspacing>(init.x1_spacing)
                };
            }

            // time-dependent properties
            config.homologous       = init.homologous;
            config.mesh_motion      = init.mesh_motion;
            config.expansion_factor = 1.0;
            config.sf               = init.mesh_motion ? a : nullptr;
            config.sf_derivative    = init.mesh_motion ? adot : nullptr;

            return config;
        }
    };

    // update expansion state (called once per timestep)
    // here, we return a new mesh_config_t with updated expansion
    // this allows us to chain updates
    template <typename mesh_t>
    mesh_t update_mesh(const mesh_t& current_mesh, real time, real dt)
    {
        mesh_t new_c = current_mesh;
        if (!current_mesh.mesh_motion) {
            return new_c;
        }

        new_c.expansion_rate = new_c.sf_derivative(time) / new_c.sf(time);
        new_c.expansion_factor += dt * current_mesh.expansion_rate;
        new_c.bounds_min = new_c.current_bounds_min();
        new_c.bounds_max = new_c.current_bounds_max();
        return new_c;
    }
}   // namespace simbi::mesh

#endif   // SIMBI_MESH_MESH_CONFIG_HPP
