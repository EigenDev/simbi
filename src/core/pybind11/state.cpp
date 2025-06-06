#include "state.hpp"
#include "physics/hydro/states/newt.hpp"
#include "physics/hydro/states/rmhd.hpp"
#include "physics/hydro/states/srhd.hpp"
#include <memory>

// Forward declarations
// namespace simbi {
//     template <int D>
//     class Newtonian;

//     template <int D>
//     class SRHD;

//     template <int D>
//     class RMHD;
// }   // namespace simbi

namespace simbi {
    namespace hydrostate {
        // Define specializations for simulate with ndarray

        // 1D Newtonian
        template <>
        void simulate<1, Regime::NEWTONIAN>(
            ndarray<anyConserved<1, Regime::NEWTONIAN>, 1>&& cons,
            ndarray<Maybe<anyPrimitive<1, Regime::NEWTONIAN>>, 1>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<Newtonian<1>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 1D SRHD
        template <>
        void simulate<1, Regime::SRHD>(
            ndarray<anyConserved<1, Regime::SRHD>, 1>&& cons,
            ndarray<Maybe<anyPrimitive<1, Regime::SRHD>>, 1>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<SRHD<1>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 2D Newtonian
        template <>
        void simulate<2, Regime::NEWTONIAN>(
            ndarray<anyConserved<2, Regime::NEWTONIAN>, 2>&& cons,
            ndarray<Maybe<anyPrimitive<2, Regime::NEWTONIAN>>, 2>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<Newtonian<2>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 2D SRHD
        template <>
        void simulate<2, Regime::SRHD>(
            ndarray<anyConserved<2, Regime::SRHD>, 2>&& cons,
            ndarray<Maybe<anyPrimitive<2, Regime::SRHD>>, 2>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<SRHD<2>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 3D Newtonian
        template <>
        void simulate<3, Regime::NEWTONIAN>(
            ndarray<anyConserved<3, Regime::NEWTONIAN>, 3>&& cons,
            ndarray<Maybe<anyPrimitive<3, Regime::NEWTONIAN>>, 3>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<Newtonian<3>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 3D SRHD
        template <>
        void simulate<3, Regime::SRHD>(
            ndarray<anyConserved<3, Regime::SRHD>, 3>&& cons,
            ndarray<Maybe<anyPrimitive<3, Regime::SRHD>>, 3>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<SRHD<3>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }

        // 3D RMHD
        template <>
        void simulate<3, Regime::RMHD>(
            ndarray<anyConserved<3, Regime::RMHD>, 3>&& cons,
            ndarray<Maybe<anyPrimitive<3, Regime::RMHD>>, 3>&& prim,
            InitialConditions& init_cond,
            std::function<real(real)> const& scale_factor,
            std::function<real(real)> const& scale_factor_derivative
        )
        {
            auto simulator = std::make_unique<RMHD<3>>(
                std::move(cons),
                std::move(prim),
                init_cond
            );
            simulator->simulate(scale_factor, scale_factor_derivative);
        }
    }   // namespace hydrostate
}   // namespace simbi
