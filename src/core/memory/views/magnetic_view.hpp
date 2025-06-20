#include "config.hpp"
#include "core/containers/array.hpp"
#include "core/utility/enums.hpp"

namespace simbi::views {
    // magnetic field view for different regimes
    template <Regime R>
    struct magnetic_view_t {
        // default implementation for regimes without magnetic fields
        DUAL real operator[](size_type) const { return 0.0; }
        DUAL real operator[](size_type) { return 0.0; }
        static constexpr bool has_magnetic_field() { return false; }
        static constexpr size_type size() { return 0; }
    };

    // with magnetic field (MHD, RMHD) always 3D
    template <>
    struct magnetic_view_t<Regime::MHD> {
        array_t<real*, 3> bvec;   // points to b1, b2, b3 arrays
        DUAL auto& operator[](size_type ii) const { return bvec[ii]; }
        DUAL auto& operator[](size_type ii) { return bvec[ii]; }
        static constexpr bool has_magnetic_field() { return true; }
        static constexpr size_type size() { return 3; }   // always 3D
    };

    // same for RMHD
    template <>
    struct magnetic_view_t<Regime::RMHD> : public magnetic_view_t<Regime::MHD> {
    };
}   // namespace simbi::views
