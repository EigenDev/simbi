// =============================================================================
// state_struct.hpp
//
// defines the core data structures for physical quantities.
// this file contains the definitions for the fundamental pod-like structs that
// hold simulation state variables, such as `primitive_t`, `conserved_t`,
// `mhd_primitive_t`, and `mhd_conserved_t`. they are templated on physics
// regime, rank, and equation of state.
//
// usage:
//   using prim_t = structs::primitive_t<srhd, 1, ideal_eos>;
//   prim_t q;
//   q.rho = 1.0;
//   q.vel[0] = 0.5;
//   q.pre = 2.5;
// =============================================================================
#pragma once

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "utility/enums.hpp"

#include <cstdint>
#include <ostream>

namespace simbi::structs {
    // forward declarations
    // these are used to define the counterpart_t type
    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    struct primitive_t;

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    struct conserved_t;

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    struct mhd_primitive_t;

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    struct mhd_conserved_t;

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    struct primitive_t
    {
        static constexpr std::uint64_t rank   = Rank;
        static constexpr regime_t      regime = R;
        static constexpr std::uint64_t nmem   = Rank + 3; // rho, vel, pre, chi
        using counterpart_t                   = conserved_t<R, Rank, EoS>;
        using eos_t                           = EoS;
        real                 rho{0.0};
        vector_t<real, Rank> vel{0.0};
        real                 pre{0.0};
        real                 chi{0.0};

        DEV constexpr real* data() noexcept
        {
            return &rho;
        }
        DEV constexpr const real* data() const noexcept
        {
            return &rho;
        }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Rank + 1) {
                return vel[idx - 1];
            }
            else if (idx == Rank + 1) {
                return pre;
            }
            else {
                return chi;
            }
        }

        DEV constexpr const real& operator[](std::uint64_t idx) const noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Rank + 1) {
                return vel[idx - 1];
            }
            else if (idx == Rank + 1) {
                return pre;
            }
            else {
                return chi;
            }
        }
    };

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    struct conserved_t
    {
        static constexpr std::uint64_t rank   = Rank;
        static constexpr regime_t      regime = R;
        static constexpr std::uint64_t nmem   = Rank + 3; // den, mom, nrg, chi
        using counterpart_t                   = primitive_t<R, Rank, EoS>;
        using eos_t                           = EoS;
        real                 den{0.0};
        vector_t<real, Rank> mom{0.0};
        real                 nrg{0.0};
        real                 chi{0.0};

        DEV constexpr real* data() noexcept
        {
            return &den;
        }
        DEV constexpr const real* data() const noexcept
        {
            return &den;
        }

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == regime_t::NEWTONIAN) {
                return nrg;
            }
            else {
                return nrg + den;
            };
        }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return den;
            }
            else if (idx < Rank + 1) {
                return mom[idx - 1];
            }
            else if (idx == Rank + 1) {
                return nrg;
            }
            else {
                return chi;
            }
        }

        DEV constexpr const real& operator[](std::uint64_t idx) const noexcept
        {
            if (idx == 0) {
                return den;
            }
            else if (idx < Rank + 1) {
                return mom[idx - 1];
            }
            else if (idx == Rank + 1) {
                return nrg;
            }
            else {
                return chi;
            }
        }
    };

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    struct mhd_primitive_t
    {
        static constexpr std::uint64_t rank   = Rank;
        static constexpr regime_t      regime = R;
        // rho, vel, pre, mag, chi
        static constexpr std::uint64_t nmem = 2 * Rank + 3;
        using counterpart_t                 = mhd_conserved_t<R, Rank, EoS>;
        using eos_t                         = EoS;
        real                 rho{0.0};
        vector_t<real, Rank> vel{0.0};
        real                 pre{0.0};
        vector_t<real, Rank> mag{0.0};
        real                 chi{0.0};

        DEV constexpr real* data() noexcept
        {
            return &rho;
        }
        DEV constexpr const real* data() const noexcept
        {
            return &rho;
        }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Rank + 1) {
                return vel[idx - 1];
            }
            else if (idx == Rank + 1) {
                return pre;
            }
            else if (idx < 2 * Rank + 2) {
                return mag[idx - Rank - 2];
            }
            else {
                return chi;
            }
        }

        DEV constexpr const real& operator[](std::uint64_t idx) const noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Rank + 1) {
                return vel[idx - 1];
            }
            else if (idx == Rank + 1) {
                return pre;
            }
            else if (idx < 2 * Rank + 2) {
                return mag[idx - Rank - 2];
            }
            else {
                return chi;
            }
        }

        // dummy accesor for the Alfven speed
        DEV constexpr real& alfven() noexcept
        {
            return chi;
        }
        DEV constexpr const real& alfven() const noexcept
        {
            return chi;
        }
    };

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    struct mhd_conserved_t
    {
        static constexpr std::uint64_t rank   = Rank;
        static constexpr regime_t      regime = R;
        // den, mom, nrg, mag, chi
        static constexpr std::uint64_t nmem = 2 * Rank + 3;
        using counterpart_t                 = mhd_primitive_t<R, Rank, EoS>;
        using eos_t                         = EoS;

        real                 den{0.0};
        vector_t<real, Rank> mom{0.0};
        real                 nrg{0.0};
        vector_t<real, Rank> mag{0.0};
        real                 chi{0.0};

        DEV constexpr real* data() noexcept
        {
            return &den;
        }
        DEV constexpr const real* data() const noexcept
        {
            return &den;
        }

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == regime_t::NEWTONIAN) {
                return nrg;
            }
            else {
                return nrg + den;
            };
        }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return den;
            }
            else if (idx < Rank + 1) {
                return mom[idx - 1];
            }
            else if (idx == Rank + 1) {
                return nrg;
            }
            else if (idx < 2 * Rank + 2) {
                return mag[idx - Rank - 2];
            }
            else {
                return chi;
            }
        }

        DEV constexpr const real& operator[](std::uint64_t idx) const noexcept
        {
            if (idx == 0) {
                return den;
            }
            else if (idx < Rank + 1) {
                return mom[idx - 1];
            }
            else if (idx == Rank + 1) {
                return nrg;
            }
            else if (idx < 2 * Rank + 2) {
                return mag[idx - Rank - 2];
            }
            else {
                return chi;
            }
        }
    };

    // ostream operator overloads for primitive and conserved states
    // for future debugging and logging
    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    std::ostream& operator<<(std::ostream& os, const primitive_t<R, Rank, EoS>& p)
    {
        // os << "Primitive State (Regime: " << serialize(R) << ", Rank: " <<
        // Rank
        //    << "):";
        os << "( " << p.rho << ", ";
        os << p.vel << ", " << p.pre << ", " << p.chi << " )";

        return os;
    }

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::NEWTONIAN || R == regime_t::SRHD)
    std::ostream& operator<<(std::ostream& os, const conserved_t<R, Rank, EoS>& c)
    {
        // os << "Conserved State (Regime: " << serialize(R) << ", Rank: " <<
        // Rank
        //    << "):";
        os << "( " << c.den << ", ";
        os << c.mom << ", " << c.nrg << ", " << c.chi << " )";
        return os;
    }

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    std::ostream& operator<<(std::ostream& os, const mhd_primitive_t<R, Rank, EoS>& p)
    {
        // os << "MHD Primitive State (Regime: " << serialize(R)
        //    << ", Rank: " << Rank << "):";
        os << "( " << p.rho << ", ";
        os << p.vel << ", " << p.pre << ", " << p.mag << ", " << p.chi << " )";
        return os;
    }

    template <regime_t R, std::uint64_t Rank, typename EoS>
        requires(R == regime_t::MHD || R == regime_t::RMHD)
    std::ostream& operator<<(std::ostream& os, const mhd_conserved_t<R, Rank, EoS>& c)
    {
        // os << "MHD Primitive State (Regime: " << serialize(R)
        //    << ", Rank: " << Rank << "):";
        os << "( " << c.den << ", ";
        os << c.mom << ", " << c.nrg << ", " << c.mag << ", " << c.chi << " )";
        return os;
    }

    template <is_hydro_primitive_c T, is_hydro_primitive_c U>
    DEV constexpr bool operator==(const T& a, const U& b) noexcept
    {
        if constexpr (T::rank != U::rank) {
            return false;
        }
        for (std::uint64_t ii = 0; ii < T::nmem; ++ii) {
            if (a[ii] != b[ii]) {
                return false;
            }
        }
        return true;
    }

    template <is_hydro_conserved_c T, is_hydro_conserved_c U>
    DEV constexpr bool operator==(const T& a, const U& b) noexcept
    {
        if constexpr (T::rank != U::rank) {
            return false;
        }
        for (std::uint64_t ii = 0; ii < T::nmem; ++ii) {
            if (a[ii] != b[ii]) {
                return false;
            }
        }
        return true;
    }

    template <is_hydro_primitive_c T, is_hydro_primitive_c U>
    DEV constexpr bool operator!=(const T& a, const U& b) noexcept
    {
        return !(a == b);
    }

    template <is_hydro_conserved_c T, is_hydro_conserved_c U>
    DEV constexpr bool operator!=(const T& a, const U& b) noexcept
    {
        return !(a == b);
    }

    template <is_mhd_primitive_c T, is_mhd_primitive_c U>
    DEV constexpr bool operator==(const T& a, const U& b) noexcept
    {
        if constexpr (T::rank != U::rank) {
            return false;
        }
        for (std::uint64_t ii = 0; ii < T::nmem; ++ii) {
            if (a[ii] != b[ii]) {
                return false;
            }
        }
        return true;
    }

    template <is_mhd_conserved_c T, is_mhd_conserved_c U>
    DEV constexpr bool operator==(const T& a, const U& b) noexcept
    {
        if constexpr (T::rank != U::rank) {
            return false;
        }
        for (std::uint64_t ii = 0; ii < T::nmem; ++ii) {
            if (a[ii] != b[ii]) {
                return false;
            }
        }
        return true;
    }

    template <is_mhd_primitive_c T, is_mhd_primitive_c U>
    DEV constexpr bool operator!=(const T& a, const U& b) noexcept
    {
        return !(a == b);
    }

    template <is_mhd_conserved_c T, is_mhd_conserved_c U>
    DEV constexpr bool operator!=(const T& a, const U& b) noexcept
    {
        return !(a == b);
    }

} // namespace simbi::structs
