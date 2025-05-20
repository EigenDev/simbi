#include "physics/hydro/states/rmhd.hpp"
#include "core/cython/state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of RMHD class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<3, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<RMHD<3>>(state, init_cond);
            self->run(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate

}   // namespace simbi
