#include "rmhd.hpp"
#include "state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of Mesh class
template struct Mesh<
    simbi::RMHD<1>,
    1,
    anyConserved<1, simbi::Regime::RMHD>,
    anyPrimitive<1, simbi::Regime::RMHD>>;

template struct Mesh<
    simbi::RMHD<2>,
    2,
    anyConserved<2, simbi::Regime::RMHD>,
    anyPrimitive<2, simbi::Regime::RMHD>>;

template struct Mesh<
    simbi::RMHD<3>,
    3,
    anyConserved<3, simbi::Regime::RMHD>,
    anyPrimitive<3, simbi::Regime::RMHD>>;

// Explicit instantiation of RMHD class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<RMHD<1>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<2, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<RMHD<2>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<3, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<RMHD<3>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate

}   // namespace simbi
