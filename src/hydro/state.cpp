/**
 * the hydro headers can only be included here since
 * they are templated and cython is unaware of the gpu-specific
 * code present throughout the simulation calls since it
 * is forced to use the host compiler there
 */
#include "state.hpp"
#include <memory>    // for make_unique, unique_ptr
#include <variant>   // for visit, variant
#include <vector>

// Forward declarations
namespace simbi {
    template <int D>
    class Newtonian;

    template <int D>
    class SRHD;

    template <int D>
    class RMHD;
}   // namespace simbi

namespace simbi {
    namespace hydrostate {
        template <>
        void simulate<1>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            using sr_rm_or_nt = std::variant<
                std::unique_ptr<Newtonian<1>>,
                std::unique_ptr<SRHD<1>>,
                std::unique_ptr<RMHD<1>>>;
            auto self = [&]() -> sr_rm_or_nt {
                if (regime == "srhd") {
                    return std::make_unique<SRHD<1>>(state, init_cond);
                }
                else if (regime == "srmhd") {
                    return std::make_unique<RMHD<1>>(state, init_cond);
                }
                else {
                    return std::make_unique<Newtonian<1>>(state, init_cond);
                }
            }();

            std::visit(
                [=](auto&& arg) {
                    arg->simulate(scale_factor, scale_factor_derivative);
                },
                self
            );
        };

        template <>
        void simulate<2>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            using sr_rm_or_nt = std::variant<
                std::unique_ptr<Newtonian<2>>,
                std::unique_ptr<SRHD<2>>,
                std::unique_ptr<RMHD<2>>>;
            auto self = [&]() -> sr_rm_or_nt {
                if (regime == "srhd") {
                    return std::make_unique<SRHD<2>>(state, init_cond);
                }
                else if (regime == "srmhd") {
                    return std::make_unique<RMHD<2>>(state, init_cond);
                }
                else {
                    return std::make_unique<Newtonian<2>>(state, init_cond);
                }
            }();
            std::visit(
                [=](auto&& arg) {
                    arg->simulate(scale_factor, scale_factor_derivative);
                },
                self
            );
        };

        template <>
        void simulate<3>(
            std::vector<std::vector<real>>& state,
            InitialConditions& init_cond,
            const std::string& regime,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            using sr_rm_or_nt = std::variant<
                std::unique_ptr<Newtonian<3>>,
                std::unique_ptr<SRHD<3>>,
                std::unique_ptr<RMHD<3>>>;
            auto self = [&]() -> sr_rm_or_nt {
                if (regime == "srhd") {
                    return std::make_unique<SRHD<3>>(state, init_cond);
                }
                else if (regime == "srmhd") {
                    return std::make_unique<RMHD<3>>(state, init_cond);
                }
                else {
                    return std::make_unique<Newtonian<3>>(state, init_cond);
                }
            }();

            std::visit(
                [=](auto&& arg) {
                    arg->simulate(scale_factor, scale_factor_derivative);
                },
                self
            );
        };
    }   // namespace hydrostate

}   // namespace simbi
