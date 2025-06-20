#ifndef STATE_VALUE_HPP
#define STATE_VALUE_HPP

#include "config.hpp"
#include "core/containers/vector.hpp"
#include "core/types/alias/alias.hpp"
#include "core/utility/enums.hpp"
#include "state_ops.hpp"

namespace simbi::values {
    // forward declarations
    // these are used to define the counterpart_t type
    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct primitive_value_t;

    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct conserved_value_t;

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_primitive_value_t;

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_conserved_value_t;

    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct primitive_value_t {
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        using counterpart_t                   = conserved_value_t<R, Dims>;
        real rho;
        spatial_vector_t<real, Dims> vel;
        real pre;
        real chi;
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct conserved_value_t {
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        using counterpart_t                   = primitive_value_t<R, Dims>;
        real den;
        spatial_vector_t<real, Dims> mom;
        real nrg;
        real chi;

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == Regime::NEWTONIAN) {
                return nrg;
            }
            else {
                return nrg + den;
            };
        }
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_primitive_value_t {
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        using counterpart_t                   = mhd_conserved_value_t<R, Dims>;
        real rho;
        spatial_vector_t<real, Dims> vel;
        real pre;
        magnetic_vector_t<real, Dims> mag;
        real chi;
    };

    template <Regime R, size_type Dims>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_conserved_value_t {
        static constexpr size_type dimensions = Dims;
        static constexpr Regime regime        = R;
        using counterpart_t                   = mhd_primitive_value_t<R, Dims>;
        real den;
        spatial_vector_t<real, Dims> mom;
        real nrg;
        magnetic_vector_t<real, Dims> mag;
        real chi;

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == Regime::NEWTONIAN) {
                return nrg;
            }
            else {
                return nrg + den;
            };
        }
    };
}   // namespace simbi::values

#endif   // STATE_VALUE_HPP
