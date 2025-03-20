#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include "build_options.hpp"   // for real
namespace simbi {
    struct HydroContext {
        // The context class is a container for all the objects that are needed
        // to run a simulation. It is a way to pass all the necessary objects
        // to the various functions that need them without having to pass them
        // all individually.
        real gamma;
        bool is_isothermal;
        real ambient_sound_speed;
    };
}   // namespace simbi
#endif   // CONTEXT_HPP
