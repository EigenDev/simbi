#ifndef SIMBI_PHYSICS_CONTACT_PROPERTIES
#define SIMBI_PHYSICS_CONTACT_PROPERTIES

#include "base/concepts.hpp"   // for is_hydro_primitive_c, is_hydro_conserved_c, is_srhd_c, is_relativistic_c
#include "config.hpp"              // for real, DEV
#include "containers/vector.hpp"   // for vector_t
#include "utility/helpers.hpp"     // for sgn

#include <cstdint>       // for std::uint64_t
#include <tuple>         // for std::tuple_size, std::tuple_element
#include <type_traits>   // for std::integral_constant, std::is_same_v
#include <utility>       // for std::tuple_size, std::tuple_element
#include <utility>

namespace simbi::hydro {
    struct contact_properties_t;
}

// structured bindings for contact_properties_t
namespace std {
    template <>
    struct tuple_size<simbi::hydro::contact_properties_t>
        : std::integral_constant<std::uint64_t, 2> {
    };

    template <>
    struct tuple_element<0, simbi::hydro::contact_properties_t> {
        using type = simbi::real;
    };

    template <>
    struct tuple_element<1, simbi::hydro::contact_properties_t> {
        using type = simbi::real;
    };
}   // namespace std

namespace simbi::hydro {
    using namespace simbi::concepts;
    using namespace simbi::helpers;
    struct contact_properties_t {
        real speed, pressure;
        // structured bindings support
        template <std::uint64_t Index>
        DEV std::tuple_element_t<Index, contact_properties_t>& get()
        {
            if constexpr (Index == 0) {
                return speed;
            }
            if constexpr (Index == 1) {
                return pressure;
            }
        }

        template <std::uint64_t Index>
        DEV const std::tuple_element_t<Index, contact_properties_t>& get() const
        {
            if constexpr (Index == 0) {
                return speed;
            }
            if constexpr (Index == 1) {
                return pressure;
            }
        }
    };

    template <
        is_hydro_primitive_c primitive_t,
        is_hydro_conserved_c conserved_t>
    DEV conserved_t star_state(
        const primitive_t& prim,
        const conserved_t& cons,
        const real a,
        const real a_star,
        const real p_star,
        const unit_vector_t<primitive_t::dimensions>& nhat
    )
    {
        const auto& mom = cons.mom;
        const auto vn   = vecops::dot(prim.vel, nhat);
        const auto& pre = prim.pre;
        const auto& den = cons.den;
        const auto& chi = cons.chi;
        const auto e    = cons.nrg + den;
        const auto fac  = 1.0 / (a - a_star);

        const auto ds   = fac * (a - vn) * den;
        const auto chis = fac * (a - vn) * chi;
        const auto ms   = (mom * (a - vn) + nhat * (-pre + p_star)) * fac;
        const auto es   = fac * (e * (a - vn) + p_star * a_star - pre * vn);

        if constexpr (is_relativistic_c<primitive_t>) {
            return conserved_t{ds, ms, es - ds, chis};
        }
        else {
            return conserved_t{ds, ms, es, chis, chis};
        }
    }

    template <is_hydro_conserved_c conserved_t>
    DEV auto contact_props(
        const conserved_t& uL,
        const conserved_t& uR,
        const conserved_t& fL,
        const conserved_t& fR,
        const unit_vector_t<conserved_t::dimensions>& nhat,
        real aL,
        real aR
    ) -> contact_properties_t
    {
        // this is the SRHD formulation from Mignone and Bodo (2005)
        // only implemented for SRHD, not Newtonian hydro, so we use a
        // static assert to prevent me from accidentally using it
        static_assert(
            is_srhd_c<conserved_t>,
            "Contact properties are only implemented for SRHD."
        );

        //-------------------Calculate the HLL Intermediate State
        const auto hll_state = (uR * aR - uL * aL - fR + fL) / (aR - aL);

        //------------------Calculate the RHLLE Flux---------------
        const auto hll_flux =
            (fL * aR - fR * aL + (uR - uL) * aR * aL) / (aR - aL);

        const auto& uhlld   = hll_state.den;
        const auto& uhlls   = hll_state.mom;
        const auto& uhlltau = hll_state.nrg;
        const auto& fhlld   = hll_flux.den;
        const auto& fhlls   = hll_flux.mom;
        const auto& fhlltau = hll_flux.nrg;
        const auto e        = uhlltau + uhlld;
        const auto snorm    = vecops::dot(uhlls, nhat);
        const auto fe       = fhlltau + fhlld;
        const auto fsnorm   = vecops::dot(fhlls, nhat);

        //------Calculate the contact wave velocity and pressure
        const auto a    = fe;
        const auto b    = -(e + fsnorm);
        const auto c    = snorm;
        const auto quad = -0.5 * (b + sgn(b) * std::sqrt(b * b - 4.0 * a * c));
        const auto a_star = c * (1.0 / quad);
        const auto p_star = -a_star * fe + fsnorm;
        return {a_star, p_star};
    }

}   // namespace simbi::hydro
#endif
