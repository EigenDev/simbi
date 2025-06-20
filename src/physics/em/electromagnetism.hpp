#ifndef SIMBI_PHYSICS_EM_HPP
#define SIMBI_PHYSICS_EM_HPP

#include "config.hpp"
#include "core/containers/vector.hpp"
#include "core/memory/values/value_concepts.hpp"

namespace simbi::em {
    using namespace simbi::concepts;
    template <is_hydro_conserved_c conserved_t>
    DEV conserved_t shift_electric_field(
        conserved_t&& flux,
        const unit_vector_t<conserved_t::dimensions>& nhat
    )
    {
        const auto efield = -vecops::cross(nhat, flux.mag);
        flux.mag          = efield;
        return std::move(flux);
    }

}   // namespace simbi::em
#endif
