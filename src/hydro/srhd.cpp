#include "srhd.hpp"
#include "state.hpp"
#include <memory>   // for make_unique, unique_ptr

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

// Explicit instantiation of SRHD class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        )
        {
            auto self = std::make_unique<SRHD<1>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        }

        template <>
        void simulate<2, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        )
        {
            auto self = std::make_unique<SRHD<2>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        }

        template <>
        void simulate<3, HydroRegime::SRHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        )
        {
            auto self = std::make_unique<SRHD<3>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        }
    }   // namespace hydrostate
}   // namespace simbi
