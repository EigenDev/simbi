#include "factory.hpp"
#include "config.hpp"
#include "core/utility/config_dict.hpp"
#include <stdexcept>
#include <string>

namespace simbi::body::factory {
    // ========================================================================
    // capability detection from config
    // ========================================================================

    namespace detail {

        // check if config has gravitational properties
        bool has_gravitational_config(const config_dict_t& props)
        {
            return config::try_read<real>(props, "softening_length")
                .has_value();
        }

        // check if config has accretion properties
        bool has_accretion_config(const config_dict_t& props)
        {
            return config::try_read<real>(props, "accretion_efficiency")
                       .has_value() ||
                   config::try_read<bool>(props, "is_an_accretor")
                       .unwrap_or(false);
        }

        // check if config has rigid properties
        bool has_rigid_config(const config_dict_t& props)
        {
            return config::try_read<real>(props, "inertia").has_value();
        }

        // determine body type from config
        std::string determine_body_type(const config_dict_t& props)
        {
            bool grav  = has_gravitational_config(props);
            bool accr  = has_accretion_config(props);
            bool rigid = has_rigid_config(props);

            if (grav && accr && !rigid) {
                return "black_hole";
            }
            if (grav && !accr && rigid) {
                return "planet";
            }
            if (grav && !accr && !rigid) {
                return "gravitational";
            }
            if (!grav && !accr && rigid) {
                return "rigid_sphere";
            }
            if (!grav && !accr && !rigid) {
                return "basic";
            }

            throw std::runtime_error("unsupported body capability combination");
        }
    }   // namespace detail
}   // namespace simbi::body::factory
