#ifndef HYDRO_STATE_TYPES_HPP
#define HYDRO_STATE_TYPES_HPP

#include "containers/state_struct.hpp"
#include "utility/enums.hpp"
#include <cstdint>

namespace simbi::ecs {
    // type traits to select the correct primitive/conserved type based on
    // regime
    template <regime_t R, std::uint64_t Rank, typename EoS>
    struct vtraits;

    // specialization for NEWTONIAN regime
    template <std::uint64_t Rank, typename EoS>
    struct vtraits<regime_t::NEWTONIAN, Rank, EoS> {
        using conserved_type =
            typename structs::conserved_t<regime_t::NEWTONIAN, Rank, EoS>;
        using primitive_type =
            typename structs::primitive_t<regime_t::NEWTONIAN, Rank, EoS>;
    };

    // specialization for SRHD regime
    template <std::uint64_t Rank, typename EoS>
    struct vtraits<regime_t::SRHD, Rank, EoS> {
        using conserved_type =
            typename structs::conserved_t<regime_t::SRHD, Rank, EoS>;
        using primitive_type =
            typename structs::primitive_t<regime_t::SRHD, Rank, EoS>;
    };

    // specialization for RMHD regime
    template <std::uint64_t Rank, typename EoS>
    struct vtraits<regime_t::RMHD, Rank, EoS> {
        using conserved_type =
            typename structs::mhd_conserved_t<regime_t::RMHD, Rank, EoS>;
        using primitive_type =
            typename structs::mhd_primitive_t<regime_t::RMHD, Rank, EoS>;
    };
}   // namespace simbi::ecs

#endif   // STATE_HYDRO_STATE_TYPES_HPP
