#ifndef SIMBI_PHYSICS_EM_HPP
#define SIMBI_PHYSICS_EM_HPP

#include "config.hpp"
#include "core/base/concepts.hpp"
#include "data/containers/vector.hpp"
#include <cstdint>

namespace simbi::em {
    using namespace simbi::concepts;
    template <
        is_hydro_conserved_c conserved_t,
        std::uint64_t Dims = conserved_t::dimensions>
    DEV conserved_t
    shift_electric_field(conserved_t&& flux, const unit_vector_t<Dims>& nhat)
        requires(Dims == 3)
    {
        const auto efield = -vecops::cross(nhat, flux.mag);
        flux.mag          = efield;
        return flux;
    }

}   // namespace simbi::em
#endif
