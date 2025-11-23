#ifndef CFD_OPERATIONS_HPP
#define CFD_OPERATIONS_HPP

#include "base/stencil_view.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/hydro_state_types.hpp"
#include "physics/hydro/solvers/hllc.hpp"
#include "physics/hydro/solvers/hlld.hpp"
#include "physics/hydro/solvers/hlle.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <utility>

namespace simbi::cfd {
    using namespace simbi::ecs;
    using namespace simbi::base::stencils;

    template <
        regime_t R,
        std::uint64_t Rank,
        solver_t S,
        reconstruction_t Rec,
        typename EoS>
    struct cfd_operations_t {
        using primitive_t   = typename vtraits<R, Rank, EoS>::primitive_type;
        using conserved_t   = typename vtraits<R, Rank, EoS>::conserved_type;
        using unit_vector_t = simbi::unit_vector_t<Rank>;
        static constexpr auto rec_t = Rec;

        // need template function b/c nvcc complains about if constexpr in
        // lambda sigh...
        DEV conserved_t flux(
            const primitive_t& primL,
            const primitive_t& primR,
            const unit_vector_t& nhat,
            real vface,
            real gamma,
            shockwave_limiter_t limiter = shockwave_limiter_t::NONE
        ) const
        {
            if constexpr (S == solver_t::HLLE) {
                return hydro::hlle_flux<primitive_t>(
                    primL,
                    primR,
                    nhat,
                    vface,
                    gamma,
                    limiter
                );
            }
            else if constexpr (S == solver_t::HLLC) {
                if constexpr (R == regime_t::NEWTONIAN) {
                    return hydro::newtonian::hllc_flux<primitive_t>(
                        primL,
                        primR,
                        nhat,
                        vface,
                        gamma,
                        limiter
                    );
                }
                else if constexpr (R == regime_t::SRHD) {
                    return hydro::srhd::hllc_flux<primitive_t>(
                        primL,
                        primR,
                        nhat,
                        vface,
                        gamma,
                        limiter
                    );
                }
                else if constexpr (R == regime_t::RMHD) {
                    return hydro::rmhd::hllc_flux<primitive_t>(
                        primL,
                        primR,
                        nhat,
                        vface,
                        gamma,
                        limiter
                    );
                }
            }
            else if constexpr (S == solver_t::HLLD && R == regime_t::RMHD) {
                return hydro::rmhd::hlld_flux<primitive_t>(
                    primL,
                    primR,
                    nhat,
                    vface,
                    gamma,
                    limiter
                );
            }
            else {
                []<bool flag = false>() {
                    static_assert(flag, "Invalid solver/regime combination");
                }();
            }
        }

        template <typename field_type>
        DEV std::pair<primitive_t, primitive_t> reconstruct(
            const stencil_view_t<Rec, field_type, Rank>& stencil,
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
