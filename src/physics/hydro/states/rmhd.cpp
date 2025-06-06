#include "physics/hydro/states/rmhd.hpp"
#include "core/pybind11/state.hpp"
#include "core/types/containers/ndarray.hpp"
#include "physics/hydro/types/generic_structs.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of RMHD class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<3, Regime::RMHD>(
            ndarray<anyConserved<3, Regime::RMHD>, 3>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<RMHD<3>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate

}   // namespace simbi
