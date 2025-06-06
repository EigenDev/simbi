#include "physics/hydro/states/srhd.hpp"
#include "core/pybind11/state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of SRHD class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, Regime::SRHD>(
            ndarray<anyConserved<1, Regime::SRHD>, 1>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<1>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<2, Regime::SRHD>(
            ndarray<anyConserved<2, Regime::SRHD>, 2>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<2>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<3, Regime::SRHD>(
            ndarray<anyConserved<3, Regime::SRHD>, 3>&& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<3>>(std::move(state), init_cond);
            self->run(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate
}   // namespace simbi
