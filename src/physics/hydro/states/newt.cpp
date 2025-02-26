#include "physics/hydro/states/newt.hpp"
#include "core/cython/state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of Newtonian class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<Newtonian<1>>(state, init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };

        template <>
        void simulate<2, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<Newtonian<2>>(state, init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };

        template <>
        void simulate<3, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<Newtonian<3>>(state, init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };
    }   // namespace hydrostate

}   // namespace simbi
