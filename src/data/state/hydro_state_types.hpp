#ifndef SIMBI_STATE_HYDRO_STATE_TYPES_HPP
#define SIMBI_STATE_HYDRO_STATE_TYPES_HPP

#include "core/utility/enums.hpp"
#include "data/containers/state_struct.hpp"
#include <cstdint>

namespace simbi::state {
    // Type traits to select the correct primitive/conserved type based on
    // regime
    template <Regime R, std::uint64_t Dims>
    struct hs_value_traits;

    // Specialization for NEWTONIAN regime
    template <std::uint64_t Dims>
    struct hs_value_traits<Regime::NEWTONIAN, Dims> {
        using conserved_type =
            typename structs::conserved_t<Regime::NEWTONIAN, Dims>;
        using primitive_type =
            typename structs::primitive_t<Regime::NEWTONIAN, Dims>;
    };

    // Specialization for SRHD regime
    template <std::uint64_t Dims>
    struct hs_value_traits<Regime::SRHD, Dims> {
        using conserved_type =
            typename structs::conserved_t<Regime::SRHD, Dims>;
        using primitive_type =
            typename structs::primitive_t<Regime::SRHD, Dims>;
    };

    // Specialization for RMHD regime
    template <std::uint64_t Dims>
    struct hs_value_traits<Regime::RMHD, Dims> {
        using conserved_type =
            typename structs::mhd_conserved_t<Regime::RMHD, Dims>;
        using primitive_type =
            typename structs::mhd_primitive_t<Regime::RMHD, Dims>;
    };
}   // namespace simbi::state

#endif   // SIMBI_STATE_HYDRO_STATE_TYPES_HPP
