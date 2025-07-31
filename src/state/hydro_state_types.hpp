#ifndef STATE_HYDRO_STATE_TYPES_HPP
#define STATE_HYDRO_STATE_TYPES_HPP

#include "containers/state_struct.hpp"
#include "utility/enums.hpp"
#include <cstdint>

namespace simbi::state {
    // Type traits to select the correct primitive/conserved type based on
    // regime
    template <Regime R, std::uint64_t Dims, typename EoS>
    struct vtraits;

    // Specialization for NEWTONIAN regime
    template <std::uint64_t Dims, typename EoS>
    struct vtraits<Regime::NEWTONIAN, Dims, EoS> {
        using conserved_type =
            typename structs::conserved_t<Regime::NEWTONIAN, Dims, EoS>;
        using primitive_type =
            typename structs::primitive_t<Regime::NEWTONIAN, Dims, EoS>;
    };

    // Specialization for SRHD regime
    template <std::uint64_t Dims, typename EoS>
    struct vtraits<Regime::SRHD, Dims, EoS> {
        using conserved_type =
            typename structs::conserved_t<Regime::SRHD, Dims, EoS>;
        using primitive_type =
            typename structs::primitive_t<Regime::SRHD, Dims, EoS>;
    };

    // Specialization for RMHD regime
    template <std::uint64_t Dims, typename EoS>
    struct vtraits<Regime::RMHD, Dims, EoS> {
        using conserved_type =
            typename structs::mhd_conserved_t<Regime::RMHD, Dims, EoS>;
        using primitive_type =
            typename structs::mhd_primitive_t<Regime::RMHD, Dims, EoS>;
    };
}   // namespace simbi::state

#endif   // STATE_HYDRO_STATE_TYPES_HPP
