// =============================================================================
// electromagnetism.hpp
//
// core helper functions for electromagnetic quantities.
// provides utility functions for mhd simulations, such as `electric_field`
// which computes the e-field from primitive variables, and
// `shift_electric_field` for transforming flux quantities.
//
// usage:
//   auto E = electric_field(prim_state);
// =============================================================================
#pragma once

#include "base/concepts.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"

#include <cstdint>

namespace simbi::em {
    using namespace simbi::concepts;
    template <is_hydro_conserved_c conserved_t, std::uint64_t Rank = conserved_t::rank>
    DEV constexpr conserved_t
    shift_electric_field(const conserved_t& flux, const unit_vector_t<Rank>& nhat)
        requires(Rank == 3)
    {
        auto       new_flux = flux;
        const auto efield   = -vecops::cross(nhat, flux.mag);
        new_flux.mag        = efield;
        return new_flux;
    }

    template <is_mhd_primitive_c prim_t>
    DEV constexpr auto electric_field(const prim_t& prim)
    {
        return -vecops::cross(prim.vel, prim.mag);
    }

} // namespace simbi::em
