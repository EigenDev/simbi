// =============================================================================
// soa_ref.hpp
//
// proxy reference type for SoA field access. holds pointers into each
// component array plus a linear index. provides implicit conversion to
// the value type (read) and assignment from value type (write).
//
// usage:
//   soa_ref_t<primitive_t> ref = field(i, j, k);
//   primitive_t prim = ref;        // read: assembles struct
//   ref = new_prim;                // write: scatters to arrays
// =============================================================================
#pragma once

#include "base/concepts.hpp"
#include "build_config.hpp"
#include "decorators.hpp"

#include <array>
#include <cstdint>

namespace simbi::soa {

    // primary template
    template <typename StateT>
    struct soa_ref_t;

    // hydro primitive proxy
    template <typename StateT>
        requires concepts::is_hydro_primitive_c<StateT> && (!concepts::is_mhd_primitive_c<StateT>)
    struct soa_ref_t<StateT>
    {
        static constexpr std::uint64_t rank = StateT::rank;

        real*                   rho_ptr;
        std::array<real*, rank> vel_ptr;
        real*                   pre_ptr;
        real*                   chi_ptr;
        std::uint64_t           idx;

        // read: assemble struct from SoA arrays
        DEV operator StateT() const
        {
            StateT s;
            s.rho = rho_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.vel[dd] = vel_ptr[dd][idx];
            }
            s.pre = pre_ptr[idx];
            s.chi = chi_ptr[idx];
            return s;
        }

        // write: scatter struct to SoA arrays
        DEV soa_ref_t& operator=(const StateT& s)
        {
            rho_ptr[idx] = s.rho;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                vel_ptr[dd][idx] = s.vel[dd];
            }
            pre_ptr[idx] = s.pre;
            chi_ptr[idx] = s.chi;
            return *this;
        }
    };

    // hydro conserved proxy
    template <typename StateT>
        requires concepts::is_hydro_conserved_c<StateT> && (!concepts::is_mhd_conserved_c<StateT>)
    struct soa_ref_t<StateT>
    {
        static constexpr std::uint64_t rank = StateT::rank;

        real*                   den_ptr;
        std::array<real*, rank> mom_ptr;
        real*                   nrg_ptr;
        real*                   chi_ptr;
        std::uint64_t           idx;

        DEV operator StateT() const
        {
            StateT s;
            s.den = den_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.mom[dd] = mom_ptr[dd][idx];
            }
            s.nrg = nrg_ptr[idx];
            s.chi = chi_ptr[idx];
            return s;
        }

        DEV soa_ref_t& operator=(const StateT& s)
        {
            den_ptr[idx] = s.den;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                mom_ptr[dd][idx] = s.mom[dd];
            }
            nrg_ptr[idx] = s.nrg;
            chi_ptr[idx] = s.chi;
            return *this;
        }
    };

    // mhd primitive proxy
    template <typename StateT>
        requires concepts::is_mhd_primitive_c<StateT>
    struct soa_ref_t<StateT>
    {
        static constexpr std::uint64_t rank = StateT::rank;

        real*                   rho_ptr;
        std::array<real*, rank> vel_ptr;
        real*                   pre_ptr;
        std::array<real*, rank> mag_ptr;
        real*                   chi_ptr;
        std::uint64_t           idx;

        DEV operator StateT() const
        {
            StateT s;
            s.rho = rho_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.vel[dd] = vel_ptr[dd][idx];
            }
            s.pre = pre_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.mag[dd] = mag_ptr[dd][idx];
            }
            s.chi = chi_ptr[idx];
            return s;
        }

        DEV soa_ref_t& operator=(const StateT& s)
        {
            rho_ptr[idx] = s.rho;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                vel_ptr[dd][idx] = s.vel[dd];
            }
            pre_ptr[idx] = s.pre;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                mag_ptr[dd][idx] = s.mag[dd];
            }
            chi_ptr[idx] = s.chi;
            return *this;
        }
    };

    // mhd conserved proxy
    template <typename StateT>
        requires concepts::is_mhd_conserved_c<StateT>
    struct soa_ref_t<StateT>
    {
        static constexpr std::uint64_t rank = StateT::rank;

        real*                   den_ptr;
        std::array<real*, rank> mom_ptr;
        real*                   nrg_ptr;
        std::array<real*, rank> mag_ptr;
        real*                   chi_ptr;
        std::uint64_t           idx;

        DEV operator StateT() const
        {
            StateT s;
            s.den = den_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.mom[dd] = mom_ptr[dd][idx];
            }
            s.nrg = nrg_ptr[idx];
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                s.mag[dd] = mag_ptr[dd][idx];
            }
            s.chi = chi_ptr[idx];
            return s;
        }

        DEV soa_ref_t& operator=(const StateT& s)
        {
            den_ptr[idx] = s.den;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                mom_ptr[dd][idx] = s.mom[dd];
            }
            nrg_ptr[idx] = s.nrg;
            for (std::uint64_t dd = 0; dd < rank; ++dd) {
                mag_ptr[dd][idx] = s.mag[dd];
            }
            chi_ptr[idx] = s.chi;
            return *this;
        }
    };

    // const version for read-only access
    template <typename StateT>
    struct soa_cref_t
    {
        const soa_ref_t<StateT> inner;

        DEV operator StateT() const
        {
            return static_cast<StateT>(inner);
        }
    };

} // namespace simbi::soa
