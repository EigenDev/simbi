#ifndef SIMBI_PHYSICS_HPP
#define SIMBI_PHYSICS_HPP

#include "config.hpp"               // for global::using_four_velocity
#include "core/base/concepts.hpp"   // for is_hydro_primitive_c, is_mhd_primitive_c, is_rmhd_c, is_srhd_c, is_hydro_conserved_c
#include "data/containers/vector.hpp"   // for vector_t
#include "physics/em/electromagnetism.hpp"
#include "physics/eos/isothermal.hpp"   // for isothermal_gas_eos_t
#include <concepts>                     // for std::same_as
#include <cstddef>
#include <iostream>

namespace simbi::hydro {
    using namespace simbi::concepts;
    using namespace simbi::eos;

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto lorentz_factor(const primitive_t& prim)
    {
        if constexpr (is_relativistic_c<primitive_t>) {
            if constexpr (global::using_four_velocity) {
                return std::sqrt(1.0 + vecops::dot(prim.vel, prim.vel));
            }
            return 1.0 / std::sqrt(1.0 - vecops::dot(prim.vel, prim.vel));
        }
        else {
            return 1.0;   // non-relativistic case
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto lorentz_factor_squared(const primitive_t& prim)
    {
        if constexpr (is_relativistic_c<primitive_t>) {
            if constexpr (global::using_four_velocity) {
                return 1.0 + vecops::dot(prim.vel, prim.vel);
            }
            return 1.0 / (1.0 - vecops::dot(prim.vel, prim.vel));
        }
        else {
            return 1.0;   // non-relativistic case
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto sound_speed(const primitive_t& prim, real gamma)
    {
        using eos_t    = typename primitive_t::eos_t;
        const auto eos = eos_t{gamma};
        return eos.sound_speed(prim.rho, prim.pre);
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto sound_speed_squared(const primitive_t& prim, real gamma)
    {
        using eos_t    = typename primitive_t::eos_t;
        const auto eos = eos_t{gamma};
        const auto cs  = eos.sound_speed(prim.rho, prim.pre);
        return cs * cs;
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto enthalpy(const primitive_t& prim, real gamma)
    {
        using eos_t    = typename primitive_t::eos_t;
        const auto eos = eos_t{gamma};
        return eos.enthalpy(prim.rho, prim.pre);
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto magnetic_pressure(const primitive_t& prim)
    {
        if constexpr (is_mhd_primitive_c<primitive_t>) {
            if constexpr (!is_relativistic_c<primitive_t>) {
                return 0.5 * vecops::dot(prim.mag, prim.mag);
            }
            else {
                const auto bsq = vecops::dot(prim.mag, prim.mag);
                const auto wsq = lorentz_factor_squared(prim);
                const auto vdb = vecops::dot(prim.vel, prim.mag);
                return 0.5 * (bsq / wsq + vdb * vdb);
            }
        }
        else {
            return 0.0;   // non-MHD case
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr magnetic_four_vector_t<real>
    magnetic_four_vector(const primitive_t& prim)
    {
        if constexpr (!is_mhd_primitive_c<primitive_t>) {
            return magnetic_four_vector_t<real>{0.0, 0.0, 0.0, 0.0};
        }
        else {
            const auto& vel = prim.vel;
            const auto& mag = prim.mag;
            const auto vdb =
                (is_rmhd_c<primitive_t>) ? vecops::dot(vel, mag) : 0.0;
            const auto w = lorentz_factor(prim);
            return magnetic_four_vector_t<real>{
              w * vdb,
              mag[0] / w + w * vel[0] * vdb,
              mag[1] / w + w * vel[1] * vdb,
              mag[2] / w + w * vel[2] * vdb
            };
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto enthalpy_density(const primitive_t& prim, real gamma)
    {
        using eos_t    = typename primitive_t::eos_t;
        const auto eos = eos_t{gamma};
        if constexpr (is_newtonian_c<primitive_t>) {
            return prim.rho;
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            return prim.rho * eos.enthalpy(prim.rho, prim.pre);
        }
        else {   // RMHD case
            return prim.rho * eos.enthalpy(prim.rho, prim.pre) +
                   2.0 * magnetic_pressure(prim);
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto energy_density(const primitive_t& prim, real gamma)
    {
        using eos_t = typename primitive_t::eos_t;
        if constexpr (std::same_as<eos_t, isothermal_gas_eos_t>) {
            if constexpr (!is_mhd_primitive_c<primitive_t>) {
                return 0.0;
            }
            else {
                return 0.5 * vecops::dot(prim.mag, prim.mag);
            }
        }
        const auto eos = eos_t{gamma};
        if constexpr (is_newtonian_c<primitive_t>) {
            const auto gas_part =
                prim.pre / (gamma - 1.0) +
                0.5 * prim.rho * vecops::dot(prim.vel, prim.vel);
            if constexpr (is_mhd_primitive_c<primitive_t>) {
                return gas_part + 0.5 * vecops::dot(prim.mag, prim.mag);
            }
            return gas_part;
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            const auto h   = eos.enthalpy(prim.rho, prim.pre);
            const auto wsq = lorentz_factor_squared(prim);
            const auto rho = prim.rho;
            return rho * h * wsq - prim.pre - rho * std::sqrt(wsq);
        }
        else {   // RMHD case
            const auto h             = eos.enthalpy(prim.rho, prim.pre);
            const auto wsq           = lorentz_factor_squared(prim);
            const auto rho           = prim.rho;
            const auto bsq           = vecops::dot(prim.mag, prim.mag);
            const auto vsq           = vecops::dot(prim.vel, prim.vel);
            const auto vdb           = vecops::dot(prim.vel, prim.mag);
            const auto magnetic_part = 0.5 * (bsq + bsq * vsq - vdb * vdb);
            return rho * h * wsq - prim.pre - rho * std::sqrt(wsq) +
                   magnetic_part;
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto linear_momentum(const primitive_t& prim, real gamma)
    {
        using eos_t = typename primitive_t::eos_t;
        if constexpr (is_newtonian_c<primitive_t>) {
            return prim.rho * prim.vel;
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            const auto eos = eos_t{gamma};
            const auto h   = eos.enthalpy(prim.rho, prim.pre);
            const auto wsq = lorentz_factor_squared(prim);
            return prim.rho * h * wsq * prim.vel;
        }
        else {   // RMHD case
            const auto eos = eos_t{gamma};
            const auto h   = eos.enthalpy(prim.rho, prim.pre);
            const auto wsq = lorentz_factor_squared(prim);
            const auto bsq = vecops::dot(prim.mag, prim.mag);
            const auto vdb = vecops::dot(prim.vel, prim.mag);
            const auto ed  = prim.rho * h * wsq;
            return (ed + bsq) * prim.vel - vdb * prim.mag;
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto labframe_density(const primitive_t& prim)
    {
        return prim.rho * lorentz_factor(prim);
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto
    proper_velocity(const primitive_t& prim, std::size_t comp)
    {
        if (comp == 2) {
            if constexpr (primitive_t::dimensions < 2) {
                return 0.0;   // no y-velocity in 1D
            }
        }
        if (comp == 3) {
            if constexpr (primitive_t::dimensions < 3) {
                return 0.0;   // no z-velocity in 2D or 1D
            }
        }
        if constexpr (global::using_four_velocity) {
            return prim.vel[comp - 1] / lorentz_factor(prim);
        }
        else {
            return prim.vel[comp - 1];
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto total_pressure(const primitive_t& prim)
    {
        if constexpr (is_mhd_primitive_c<primitive_t>) {
            return prim.pre + magnetic_pressure(prim);
        }
        else {
            return prim.pre;   // non-MHD case
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto to_conserved(const primitive_t& prim, real gamma)
    {
        const auto den    = labframe_density(prim);
        using conserved_t = typename primitive_t::counterpart_t;
        if constexpr (is_mhd_primitive_c<primitive_t>) {
            return conserved_t{
              .den = den,
              .mom = linear_momentum(prim, gamma),
              .nrg = energy_density(prim, gamma),
              .mag = prim.mag,
              .chi = den * prim.chi
            };
        }
        else {
            return conserved_t{
              .den = den,
              .mom = linear_momentum(prim, gamma),
              .nrg = energy_density(prim, gamma),
              .chi = den * prim.chi
            };
        }
    }

    template <is_hydro_primitive_c primitive_t>
    DEV constexpr auto to_flux(
        const primitive_t& prim,
        const unit_vector_t<primitive_t::dimensions>& nhat,
        real gamma
    )
    {
        using conserved_t = typename primitive_t::counterpart_t;
        const auto den    = labframe_density(prim);
        const auto pre    = prim.pre;
        const auto mom    = linear_momentum(prim, gamma);
        const auto mn     = vecops::dot(mom, nhat);
        const auto ed     = energy_density(prim, gamma);
        const auto vn     = vecops::dot(prim.vel, nhat);
        if constexpr (is_newtonian_c<primitive_t>) {
            return conserved_t{
              .den = den * vn,
              .mom = mom * vn + pre * nhat,
              .nrg = (ed + pre) * vn,
              .chi = den * vn * prim.chi
            };
        }
        else if constexpr (is_srhd_c<primitive_t>) {
            return conserved_t{
              .den = den * vn,
              .mom = mom * vn + pre * nhat,
              .nrg = mn - den * vn,
              .chi = den * vn * prim.chi,
            };
        }
        else if constexpr (is_rmhd_c<primitive_t>) {
            const auto bn        = vecops::dot(prim.mag, nhat);
            const auto vdb       = vecops::dot(prim.vel, prim.mag);
            const auto w         = lorentz_factor(prim);
            const auto bmu       = prim.mag / w + prim.vel * w * vdb;
            const auto efield    = em::electric_field(prim);
            const auto induction = vecops::cross(nhat, efield);
            return conserved_t{
              .den = den * vn,
              .mom = mom * vn + total_pressure(prim) * nhat - bmu * bn / w,
              .nrg = mn - den * vn,
              .mag = induction,
              .chi = den * vn * prim.chi
            };
        }
        else if constexpr (is_mhd_primitive_c<primitive_t>) {   // MHD
            const auto bn        = vecops::dot(prim.mag, nhat);
            const auto vdb       = vecops::dot(prim.vel, prim.mag);
            const auto bvec      = prim.mag;
            const auto efield    = em::electric_field(prim);
            const auto induction = vecops::cross(nhat, efield);
            return conserved_t{
              .den = den * vn,
              .mom = mom * vn + total_pressure(prim) * nhat - bvec * bn,
              .nrg = (ed + total_pressure(prim)) * vn,
              .mag = induction,
              .chi = den * vn * prim.chi
            };
        }
        else {   // non-MHD (should not happen)
            std::cout << "Warning: Non-MHD primitive to flux conversion.\n";
            return conserved_t{
              .den = den * vn,
              .mom = mom * vn + pre * nhat,
              .nrg = mn,
              .chi = den * vn * prim.chi
            };
        }
    }

    // a few conserved operations
    template <is_hydro_conserved_c conserved_t>
        requires(is_newtonian_c<conserved_t>)
    DEV constexpr auto
    pressure_from_conserved(const conserved_t& cons, real gamma)
    {
        using eos_t = conserved_t::eos_t;
        if constexpr (std::same_as<eos_t, isothermal_gas_eos_t>) {
            // I store the sound speed squared in the energy density
            // for isothermal runs since we don't use the energy density
            // anyway
            return cons.nrg * cons.den;
        }
        const auto vel = cons.mom / cons.den;
        return (gamma - 1.0) *
               (cons.nrg - 0.5 * cons.den * vecops::dot(vel, vel));
    }
}   // namespace simbi::hydro
#endif   // PHYSICS_HPP
