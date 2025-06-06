#include "physics/hydro/states/newt.hpp"
#include "core/pybind11/state.hpp"
#include "core/types/utility/enums.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of Newtonian class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, Regime::NEWTONIAN>(
            ndarray<anyConserved<1, Regime::NEWTONIAN>>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self =
                std::make_unique<Newtonian<1>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };

        template <>
        void simulate<2, Regime::NEWTONIAN>(
            ndarray<anyConserved<2, Regime::NEWTONIAN>, 2>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self =
                std::make_unique<Newtonian<2>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };

        template <>
        void simulate<3, Regime::NEWTONIAN>(
            ndarray<anyConserved<3, Regime::NEWTONIAN>, 3>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self =
                std::make_unique<Newtonian<3>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        };
    }   // namespace hydrostate

}   // namespace simbi
