#include "srhd.hpp"
#include "state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of SRHD class
namespace simbi {
    // Explicit instantiation of Mesh class
    template struct Mesh<
        simbi::SRHD<1>,
        1,
        anyConserved<1, simbi::Regime::SRHD>,
        anyPrimitive<1, simbi::Regime::SRHD>>;

    template struct Mesh<
        simbi::SRHD<2>,
        2,
        anyConserved<2, simbi::Regime::SRHD>,
        anyPrimitive<2, simbi::Regime::SRHD>>;

    template struct Mesh<
        simbi::SRHD<3>,
        3,
        anyConserved<3, simbi::Regime::SRHD>,
        anyPrimitive<3, simbi::Regime::SRHD>>;

    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<1>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<2, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<2>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }

        template <>
        void simulate<3, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto self = std::make_unique<SRHD<3>>(state, init_cond);
            self->simulate(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate
}   // namespace simbi
