#ifndef SYSTEM_CONFIG_HPP
#define SYSTEM_CONFIG_HPP

#include <string>

// houses pure data structures for a system
namespace simbi::ibsystem::config {
    // basic gravitational config
    template <typename T>
    struct GravitationalConfig {
        bool prescribed_motion      = false;
        std::string reference_frame = "center_of_mass";
    };

    template <typename T>
    struct BinaryConfig {
        T semi_major                  = 1.0;
        T eccentricity                = 0.0;
        T mass_ratio                  = 1.0;    // q = m2/m1
        bool circular                 = true;   // shorthand for e = 0
        bool equal_mass               = true;   // shorthand for q = 1
        T total_mass                  = 1.0;
        T inclination                 = 0.0;
        T longitude_of_ascending_node = 0.0;
        T argument_of_periapsis       = 0.0;
        T true_anomaly                = 0.0;
    };

    template <typename T>
    struct PlanetaryConfig {
        T central_mass = T(1.0);
        std::vector<T> planet_masses;
        std::vector<T> planet_semi_majors;
        std::vector<T> planet_eccentricities;
    };
}   // namespace simbi::ibsystem::config
#endif   // SYSTEM_CONFIG_HPP
