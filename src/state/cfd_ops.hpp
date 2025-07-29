#ifndef SIMBI_CFD_OPERATIONS_HPP
#define SIMBI_CFD_OPERATIONS_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "core/base/stencil_view.hpp"
#include "core/utility/enums.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/solvers/hlld.hpp"
#include "physics/hydro/solvers/hlle.hpp"
#include "state/hydro_state_types.hpp"

#include <cstdint>
#include <utility>

namespace simbi::cfd {
    using namespace simbi::state;
    using namespace simbi::base::stencils;

    template <
        Regime R,
        std::uint64_t Dims,
        Solver S,
        Reconstruction Rec,
        typename EoS>
    struct cfd_operations_t {
        using primitive_t   = typename vtraits<R, Dims, EoS>::primitive_type;
        using conserved_t   = typename vtraits<R, Dims, EoS>::conserved_type;
        using unit_vector_t = simbi::unit_vector_t<Dims>;
        static constexpr auto rec_t = Rec;

        static constexpr auto compute_flux = []() {
            if constexpr (S == Solver::HLLE) {
                return hydro::hlle_flux<primitive_t>;
            }
            else if constexpr (S == Solver::HLLC) {
                if constexpr (R == Regime::NEWTONIAN) {
                    return hydro::newtonian::hllc_flux<primitive_t>;
                }
                else if constexpr (R == Regime::SRHD) {
                    return hydro::srhd::hllc_flux<primitive_t>;
                }
                else if constexpr (R == Regime::RMHD) {
                    return hydro::rmhd::hllc_flux<primitive_t>;
                }
            }
            else if constexpr (S == Solver::HLLD && R == Regime::RMHD) {
                return hydro::rmhd::hlld_flux<primitive_t>;
            }
            else {
                static_assert(false, "Invalid solver/regime combination");
            }
        }();

        DEV conserved_t flux(
            const primitive_t& primL,
            const primitive_t& primR,
            const unit_vector_t& nhat,
            real vface,
            real gamma,
            ShockWaveLimiter limiter = ShockWaveLimiter::NONE
        ) const
        {
            return compute_flux(primL, primR, nhat, vface, gamma, limiter);
        }

        template <typename field_type>
        DEV std::pair<primitive_t, primitive_t> reconstruct(
            const stencil_view_t<Rec, field_type, Dims>& stencil,
            real theta
        ) const
        {
            auto [left_vals, right_vals] = stencil.neighbor_values();
            return {
              reconstruct_left<Rec>(left_vals, theta),
              reconstruct_right<Rec>(right_vals, theta)
            };
        }
    };

}   // namespace simbi::cfd

#endif   // CFD_OPERATIONS_HPP
