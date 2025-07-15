#ifndef STATE_STRUCTS_HPP
#define STATE_STRUCTS_HPP

#include "config.hpp"
#include "core/utility/enums.hpp"
#include "data/containers/vector.hpp"
#include "state_ops.hpp"
#include <cstdint>
#include <ostream>

namespace simbi::structs {
    // forward declarations
    // these are used to define the counterpart_t type
    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct primitive_t;

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct conserved_t;

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_primitive_t;

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_conserved_t;

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct primitive_t {
        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime            = R;
        static constexpr std::uint64_t nmem = Dims + 3;   // rho, vel, pre, chi
        using counterpart_t                 = conserved_t<R, Dims, EoS>;
        using eos_t                         = EoS;
        real rho{0.0};
        vector_t<real, Dims> vel{0.0};
        real pre{0.0};
        real chi{0.0};

        DEV constexpr real* data() noexcept { return &rho; }
        DEV constexpr const real* data() const noexcept { return &rho; }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Dims + 1) {
                return vel[idx - 1];
            }
            else if (idx == Dims + 1) {
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
            else if (idx < Dims + 1) {
                return vel[idx - 1];
            }
            else if (idx == Dims + 1) {
                return pre;
            }
            else {
                return chi;
            }
        }
    };

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    struct conserved_t {
        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime            = R;
        static constexpr std::uint64_t nmem = Dims + 3;   // den, mom, nrg, chi
        using counterpart_t                 = primitive_t<R, Dims, EoS>;
        using eos_t                         = EoS;
        real den{0.0};
        vector_t<real, Dims> mom{0.0};
        real nrg{0.0};
        real chi{0.0};

        DEV constexpr real* data() noexcept { return &den; }
        DEV constexpr const real* data() const noexcept { return &den; }

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == Regime::NEWTONIAN) {
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
            else if (idx < Dims + 1) {
                return mom[idx - 1];
            }
            else if (idx == Dims + 1) {
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
            else if (idx < Dims + 1) {
                return mom[idx - 1];
            }
            else if (idx == Dims + 1) {
                return nrg;
            }
            else {
                return chi;
            }
        }
    };

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_primitive_t {
        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime            = R;
        // rho, vel, pre, mag, chi
        static constexpr std::uint64_t nmem = 2 * Dims + 3;
        using counterpart_t                 = mhd_conserved_t<R, Dims, EoS>;
        using eos_t                         = EoS;
        real rho{0.0};
        vector_t<real, Dims> vel{0.0};
        real pre{0.0};
        vector_t<real, Dims> mag{0.0};
        real chi{0.0};

        DEV constexpr real* data() noexcept { return &rho; }
        DEV constexpr const real* data() const noexcept { return &rho; }

        DEV constexpr real& operator[](std::uint64_t idx) noexcept
        {
            if (idx == 0) {
                return rho;
            }
            else if (idx < Dims + 1) {
                return vel[idx - 1];
            }
            else if (idx == Dims + 1) {
                return pre;
            }
            else if (idx < 2 * Dims + 2) {
                return mag[idx - Dims - 2];
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
            else if (idx < Dims + 1) {
                return vel[idx - 1];
            }
            else if (idx == Dims + 1) {
                return pre;
            }
            else if (idx < 2 * Dims + 2) {
                return mag[idx - Dims - 2];
            }
            else {
                return chi;
            }
        }

        // dummy accesor for the Alfven speed
        DEV constexpr real& alfven() noexcept { return chi; }
        DEV constexpr const real& alfven() const noexcept { return chi; }
    };

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    struct mhd_conserved_t {
        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime            = R;
        // den, mom, nrg, mag, chi
        static constexpr std::uint64_t nmem = 2 * Dims + 3;
        using counterpart_t                 = mhd_primitive_t<R, Dims, EoS>;
        using eos_t                         = EoS;

        real den{0.0};
        vector_t<real, Dims> mom{0.0};
        real nrg{0.0};
        vector_t<real, Dims> mag{0.0};
        real chi{0.0};

        DEV constexpr real* data() noexcept { return &den; }
        DEV constexpr const real* data() const noexcept { return &den; }

        DEV constexpr auto total_energy() const noexcept -> real
        {
            if constexpr (R == Regime::NEWTONIAN) {
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
            else if (idx < Dims + 1) {
                return mom[idx - 1];
            }
            else if (idx == Dims + 1) {
                return nrg;
            }
            else if (idx < 2 * Dims + 2) {
                return mag[idx - Dims - 2];
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
            else if (idx < Dims + 1) {
                return mom[idx - 1];
            }
            else if (idx == Dims + 1) {
                return nrg;
            }
            else if (idx < 2 * Dims + 2) {
                return mag[idx - Dims - 2];
            }
            else {
                return chi;
            }
        }
    };

    // ostream operator overloads for primitive and conserved states
    // for future debugging and logging
    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    std::ostream&
    operator<<(std::ostream& os, const primitive_t<R, Dims, EoS>& p)
    {
        // os << "Primitive State (Regime: " << serialize(R) << ", Dims: " <<
        // Dims
        //    << "):\n";
        os << "( " << p.rho << ", ";
        os << p.vel << ", " << p.pre << ", " << p.chi << " )\n";

        return os;
    }

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::NEWTONIAN || R == Regime::SRHD)
    std::ostream&
    operator<<(std::ostream& os, const conserved_t<R, Dims, EoS>& c)
    {
        // os << "Conserved State (Regime: " << serialize(R) << ", Dims: " <<
        // Dims
        //    << "):\n";
        os << "( " << c.den << ", ";
        os << c.mom << ", " << c.nrg << ", " << c.chi << " )\n";
        return os;
    }

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    std::ostream&
    operator<<(std::ostream& os, const mhd_primitive_t<R, Dims, EoS>& p)
    {
        // os << "MHD Primitive State (Regime: " << serialize(R)
        //    << ", Dims: " << Dims << "):\n";
        os << "( " << p.rho << ", ";
        os << p.vel << ", " << p.pre << ", " << p.mag << ", " << p.chi
           << " )\n";
        return os;
    }

    template <Regime R, std::uint64_t Dims, typename EoS>
        requires(R == Regime::MHD || R == Regime::RMHD)
    std::ostream&
    operator<<(std::ostream& os, const mhd_conserved_t<R, Dims, EoS>& c)
    {
        // os << "MHD Primitive State (Regime: " << serialize(R)
        //    << ", Dims: " << Dims << "):\n";
        os << "( " << c.den << ", ";
        os << c.mom << ", " << c.nrg << ", " << c.mag << ", " << c.chi
           << " )\n";
        return os;
    }
}   // namespace simbi::structs

#endif   // STATE_VALUE_HPP
