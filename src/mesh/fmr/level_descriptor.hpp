#ifndef FMR_LEVEL_DESCRIPTOR_HPP
#define FMR_LEVEL_DESCRIPTOR_HPP

#include "compat.hpp"              // for real type
#include "containers/vector.hpp"   // for vector_t
#include "domain/domain.hpp"       // for domain_t

#include <cstdint>   // for std::uint64_t

namespace simbi::mesh::fmr {

    template <std::uint64_t Dims>
    struct level_descriptor_t {
        std::uint64_t level_id;
        domain_t<Dims> domain;        // active cells at this level
        domain_t<Dims> full_domain;   // including ghost zones
        std::uint64_t ref_ratio;      // cumulative ratio from level 0
        vector_t<real, Dims> dx;      // cell spacing

        // parent relationship
        std::uint64_t parent_level_id;
        domain_t<Dims> parent_coverage;   // which region of parent we refine
    };

}   // namespace simbi::mesh::fmr

#endif
