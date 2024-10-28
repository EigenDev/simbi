#include "rmhd.hpp"
#include "state.hpp"
#include <memory>   // for make_unique, unique_ptr

namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<1>> const& bsources,
            std::vector<fopt<1>> const& hsources,
            std::vector<fopt<1>> const& gsources
        )
        {
            auto self = std::make_unique<RMHD<1>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        }

        template <>
        void simulate<2, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<2>> const& bsources,
            std::vector<fopt<2>> const& hsources,
            std::vector<fopt<2>> const& gsources
        )
        {
            auto self = std::make_unique<RMHD<2>>(state, init_cond);
            self->simulate(
                scale_factor,
                scale_factor_derivative,
                bsources,
                hsources,
                gsources
            );
        }

        template <>
        void simulate<3, HydroRegime::RMHD>(
            std::vector<std::vector<real>>& state,
            const InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative,
            std::vector<fopt<3>> const& bsources,
            std::vector<fopt<3>> const& hsources,
            std::vector<fopt<3>> const& gsources
        )
        {
            auto self = std::make_unique<RMHD<3>>(state, init_cond);
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
