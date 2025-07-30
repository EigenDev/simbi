#ifndef SIMBI_PHYSICS_EM_HPP
#define SIMBI_PHYSICS_EM_HPP

#include "base/concepts.hpp"
#include "config.hpp"

#include "containers/vector.hpp"
#include <cstdint>

namespace simbi::em {
    using namespace simbi::concepts;
    template <
        is_hydro_conserved_c conserved_t,
        std::uint64_t Dims = conserved_t::dimensions>
    DEV conserved_t shift_electric_field(
        const conserved_t& flux,
        const unit_vector_t<Dims>& nhat
    )
        requires(Dims == 3)
    {
        auto new_flux     = flux;
        const auto efield = -vecops::cross(nhat, flux.mag);
        new_flux.mag      = efield;
        return new_flux;
    }

    template <
        is_mhd_primitive_c prim_t,
        std::uint64_t Dims = prim_t::dimensions>
    DEV auto electric_field(const prim_t& prim)
    {
        return -vecops::cross(prim.vel, prim.mag);
    }

}   // namespace simbi::em
#endif
