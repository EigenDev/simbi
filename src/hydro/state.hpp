#ifndef STATE_HPP
#define STATE_HPP

#include "base.hpp"

namespace simbi
{
    namespace hydrostate
    {   
        template<typename T>
        struct Caster {
            using objtype = decltype(T);

            static auto& instance(){
                static Caster<T> me;
                return me;
            }
        };

        std::unique_ptr<HydroBase> create(
            std::vector<std::vector<real>> &state,
            const InitialConditions &init_cond,
            const std::string regime,
            const int dim
        );
    } // namespace hydrostate    
} // namespace simbi

#endif