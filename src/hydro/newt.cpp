#include "newt.hpp"
#include "state.hpp"
#include <memory>   // for make_unique, unique_ptr

// Explicit instantiation of Mesh class
template struct Mesh<
    simbi::Newtonian<1>,
    1,
    anyConserved<1, simbi::Regime::NEWTONIAN>,
    anyPrimitive<1, simbi::Regime::NEWTONIAN>>;
template struct Mesh<
    simbi::Newtonian<2>,
    2,
    anyConserved<2, simbi::Regime::NEWTONIAN>,
    anyPrimitive<2, simbi::Regime::NEWTONIAN>>;
template struct Mesh<
    simbi::Newtonian<3>,
    3,
    anyConserved<3, simbi::Regime::NEWTONIAN>,
    anyPrimitive<3, simbi::Regime::NEWTONIAN>>;

// Explicit instantiation of Newtonian class
namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        )
        {
            auto self = std::make_unique<Newtonian<1>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        };

        template <>
        void simulate<2, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        )
        {
            auto self = std::make_unique<Newtonian<2>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        };

        template <>
        void simulate<3, HydroRegime::Newtonian>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        )
        {
            auto self = std::make_unique<Newtonian<3>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        };
    }   // namespace hydrostate

}   // namespace simbi
