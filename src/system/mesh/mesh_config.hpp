#ifndef SIMIB_MESH_MESH_CONFIG_HPP
#define SIMIB_MESH_MESH_CONFIG_HPP

#include "compute/functional/fp.hpp"
#include "config.hpp"   // for real, DUAL, DEV, global::using_four_velocity
#include "core/utility/bimap.hpp"             // for deserialize
#include "core/utility/enums.hpp"             // for Cellspacing enum
#include "core/utility/init_conditions.hpp"   // for InitialConditions
#include "data/containers/vector.hpp"         // for vector_t
#include <cstddef>                            // for std::size_t
#include <cstdint>                            // for std::int64_t
#include <functional>                         // for std::function

namespace simbi::mesh {
    using namespace base;

    template <std::uint64_t Dims>
    struct mesh_config_t {
        uarray<Dims> shape;        // nk, nj, ni
        std::size_t halo_radius;   // halo radius

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

        DUAL constexpr uarray<Dims> full_shape() const
        {
            uarray<Dims> result;
            for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                result[ii] = shape[ii] + 2 * halo_radius;
            }
            return result;
        }

        DUAL constexpr std::uint64_t full_size() const
        {
            return fp::product(full_shape());
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
                result[0] *= expansion_factor;
            }
            else {
                // uniform expansion translation
                result[0] += expansion_factor;
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
                result[0] *= expansion_factor;
            }
            else {
                result[0] += expansion_factor;
            }
            return result;
        }

        // update expansion state (called once per timestep)
        // here, we return a new mesh_config_t with updated expansion
        // this allows us to chain updates
        mesh_config_t update_expansion(real time, real dt)
        {
            mesh_config_t new_c = *this;
            if (!mesh_motion) {
                return new_c;
            }

            new_c.expansion_rate = new_c.sf_derivative(time) / new_c.sf(time);
            new_c.expansion_factor += dt * expansion_rate;
            new_c.bounds_min = new_c.current_bounds_min();
            new_c.bounds_max = new_c.current_bounds_max();
            return new_c;
        }

        static mesh_config_t from_init_conditions(
            const InitialConditions& init,
            std::function<real(real)> const& a,
            std::function<real(real)> const& adot
        )
        {
            mesh_config_t config;
            const auto [nia, nja, nka] = init.active_zones();

            // grid setup
            if constexpr (Dims == 1) {
                config.shape = uarray<Dims>{nia};
            }
            else if constexpr (Dims == 2) {
                config.shape = uarray<Dims>{nia, nja};
            }
            else if constexpr (Dims == 3) {
                config.shape = uarray<Dims>{nia, nja, nka};
            }
            config.halo_radius = init.halo_radius;

            // geometry setup
            if constexpr (Dims == 1) {
                config.bounds_min = vector_t<real, 1>{init.x1bounds.first};
                config.bounds_max = vector_t<real, 1>{init.x1bounds.second};
            }
            else if constexpr (Dims == 2) {
                config.bounds_min =
                    vector_t<real, 2>{init.x1bounds.first, init.x2bounds.first};
                config.bounds_max = vector_t<real, 2>{
                  init.x1bounds.second,
                  init.x2bounds.second
                };
            }
            else if constexpr (Dims == 3) {
                config.bounds_min = vector_t<real, 3>{
                  init.x1bounds.first,
                  init.x2bounds.first,
                  init.x3bounds.first
                };
                config.bounds_max = vector_t<real, 3>{
                  init.x1bounds.second,
                  init.x2bounds.second,
                  init.x3bounds.second
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
                  deserialize<Cellspacing>(init.x1_spacing),
                  deserialize<Cellspacing>(init.x2_spacing)
                };
            }
            else if constexpr (Dims == 3) {
                config.spacing_types = vector_t<Cellspacing, 3>{
                  deserialize<Cellspacing>(init.x1_spacing),
                  deserialize<Cellspacing>(init.x2_spacing),
                  deserialize<Cellspacing>(init.x3_spacing)
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
}   // namespace simbi::mesh

#endif   // SIMBI_MESH_MESH_CONFIG_HPP
