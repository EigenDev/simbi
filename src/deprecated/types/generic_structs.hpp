/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            generic_structs.hpp
 *  * @brief           Generic Structs for any Hydrodynamic Regime
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef GENERIC_STRUCTS_HPP
#define GENERIC_STRUCTS_HPP

#include "config.hpp"
#include "containers/vector.hpp"
#include "core/functional/monad/maybe.hpp"
#include "core/traits.hpp"
#include "core/utility/enums.hpp"
#include "util/tools/algorithms.hpp"
#include "util/tools/helpers.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace simbi::helpers;

namespace simbi {
    namespace sim_type {
        template <Regime R>
        concept MHD = R == Regime::RMHD;

        template <Regime R>
        concept Relativistic = R == Regime::SRHD || R == Regime::RMHD;

        template <Regime R>
        concept Newtonian = R == Regime::NEWTONIAN;
    }   // namespace sim_type

    struct MignoneDelZannaVariables {
        real lamL, lamR, aL, aR, dL, dR, vjL, vjR, vkL, vkR, vnorm{0.0};
    };

    template <Regime R>
    DUAL constexpr bool has_mdz_vars()
    {
        if constexpr ((comp_ct_type == CTAlgo::MdZ) && sim_type::MHD<R>) {
            return true;
        }
        else {
            return false;
        }
    }

    // Base empty storage that takes up no space when not needed
    struct EmptyMdZStorage {
    };

    // Storage for actual MdZ variables when needed
    struct MdZVariablesStorage {
        MignoneDelZannaVariables vars;

        DUAL constexpr MdZVariablesStorage() : vars{} {}
        DUAL constexpr MdZVariablesStorage(
            const MignoneDelZannaVariables& other
        )
            : vars(other)
        {
        }

        DUAL constexpr const MignoneDelZannaVariables& get() const
        {
            return vars;
        }
        DUAL constexpr MignoneDelZannaVariables& get() { return vars; }
        DUAL constexpr void set(const MignoneDelZannaVariables& other)
        {
            vars = other;
        }
    };

    // Decorator for anyHydro that conditionally adds MdZ storage
    template <bool UseMdZ, typename Base>
    class MdZStorageDecorator : public Base
    {
      public:
        using Base::Base;   // Inherit constructors

        // Forward the returned value and keep the same API
        DUAL constexpr auto mdz_vars() const
        {
            if constexpr (UseMdZ) {
                // Return a Just wrapper with the actual variables
                return Maybe(
                    static_cast<const MdZVariablesStorage&>(*this).get()
                );
            }
            else {
                // Return Nothing when MdZ variables are not enabled
                return Nothing;
            }
        }

        DUAL constexpr auto mdz_vars()
        {
            if constexpr (UseMdZ) {
                return Maybe(static_cast<MdZVariablesStorage&>(*this).get());
            }
            else {
                return Nothing;
            }
        }

        DUAL constexpr void set_mdz_vars(const MignoneDelZannaVariables& vars)
        {
            if constexpr (UseMdZ) {
                static_cast<MdZVariablesStorage&>(*this).set(vars);
            }
        }
    };

    namespace generic_hydro {
        template <size_type Dims, typename Derived, Regime R>
        struct anyHydro {
            static constexpr int nmem = (R == Regime::RMHD) ? 9 : 3 + Dims;

            struct Offsets {
                static constexpr size_type density  = 0;
                static constexpr size_type velocity = 1;
                static constexpr size_type energy   = Dims + 1;
                static constexpr size_type bfield   = Dims + 2;
                static constexpr size_type chi      = nmem - 1;

                // vector components
                static constexpr size_type v1 = 1;
                static constexpr size_type v2 = 2;
                static constexpr size_type v3 = 3;
                static constexpr size_type b1 = Dims + 2;
                static constexpr size_type b2 = Dims + 3;
                static constexpr size_type b3 = Dims + 4;
            };

            ~anyHydro() = default;

            // Constructors
            // zero-argument constructor
            DUAL constexpr anyHydro() : vals_{} {}

            // Base from base
            DUAL constexpr anyHydro(const anyHydro& other) : vals_{}
            {
                algorithms::copy(other.vals_, other.vals_ + nmem, vals_);
            }

            // Move constructor
            DUAL constexpr anyHydro(anyHydro&& other) noexcept : vals_{}
            {
                algorithms::move(other.vals_, other.vals_ + nmem, vals_);
                algorithms::fill_n(other.vals_, nmem, real{0});
            }

            // Copy-assignment operator
            DUAL anyHydro& operator=(const anyHydro& other)
            {
                if (this != &other) {
                    algorithms::copy(other.vals_, other.vals_ + nmem, vals_);
                }
                return *this;
            }

            // bool operator
            DUAL bool operator==(const anyHydro& other) const
            {
                for (luint i = 0; i < nmem; i++) {
                    if (vals_[i] != other.vals_[i]) {
                        return false;
                    }
                }
                return true;
            }

            // bool operator
            DUAL bool operator!=(const anyHydro& other) const
            {
                return !(*this == other);
            }

            // Vector constructor with initialization list
            template <
                typename VType,
                typename BType = magnetic_vector_view_t<real, Dims>>
            DUAL constexpr anyHydro(
                const real density,
                const VType& velocity,
                const real edensity,
                const real chi      = 0.0,
                const BType& bfield = BType{}
            )
                requires VectorLike<VType> && VectorLike<BType>
                : vals_{}
            {
                vals_[Offsets::density] = density;
                vals_[Offsets::energy]  = edensity;
                vals_[Offsets::chi]     = chi;

                // Copy vectors
                algorithms::copy_n(
                    velocity.data(),
                    Dims,
                    vals_ + Offsets::velocity
                );
                if constexpr (sim_type::MHD<R>) {
                    algorithms::copy_n(
                        bfield.data(),
                        3,
                        vals_ + Offsets::bfield
                    );
                }
            }

            template <typename... Args>
            DUAL constexpr anyHydro(Args... args)
                requires(
                    (std::is_same_v<std::remove_cvref_t<Args>, real> && ...) &&
                    sizeof...(Args) <= nmem
                )
                : vals_{}
            {
                size_type i = 0;
                ((vals_[i++] = args), ...);
            }

            // Assignment operators
            DUAL Derived& operator=(Derived&& other) noexcept
            {
                if (this != &other) {
                    algorithms::move(other.vals_, other.vals_ + nmem, vals_);
                }
                return *derived();
            }

            DUAL Derived& operator=(const Derived& other)
            {
                if (this != &other) {
                    algorithms::copy(other.vals_, other.vals_ + nmem, vals_);
                }
                return *derived();
            }

            // + operator
            DUAL Derived operator+(const Derived& other) const
            {
                Derived result;
                for (luint i = 0; i < nmem; i++) {
                    result.vals_[i] = vals_[i] + other.vals_[i];
                }
                return result;
            }

            // - operator
            DUAL Derived operator-(const Derived& other) const
            {
                Derived result;
                for (luint i = 0; i < nmem; i++) {
                    result.vals_[i] = vals_[i] - other.vals_[i];
                }
                return result;
            }

            // Scalar division
            DUAL Derived operator/(const real c) const
            {
                Derived result;
                for (luint i = 0; i < nmem; i++) {
                    result.vals_[i] = vals_[i] / c;
                }
                return result;
            }

            // Scalar multiplication
            DUAL Derived operator*(const real c) const
            {
                Derived result;
                for (luint i = 0; i < nmem; i++) {
                    result.vals_[i] = vals_[i] * c;
                }
                return result;
            }

            DUAL Derived& operator-=(const Derived& prim)
            {
                for (luint i = 0; i < nmem; i++) {
                    vals_[i] -= prim.vals_[i];
                }
                return *derived();
            }

            DUAL Derived& operator+=(const Derived& prim)
            {
                for (luint i = 0; i < nmem; i++) {
                    vals_[i] += prim.vals_[i];
                }
                return *derived();
            }

            // generic accessors
            // Magnetic field for MHD runs
            DEV auto bfield() const
            {
                if constexpr (sim_type::MHD<R>) {
                    return const_magnetic_vector_view_t<real, 3>(
                        vals_ + Offsets::bfield
                    );
                }
                else {
                    if constexpr (!platform::is_gpu) {
                        static const ZeroMagneticVectorView zero_view;
                        return zero_view;
                    }
                    else {
                        const ZeroMagneticVectorView zero_view;
                        return zero_view;
                    }
                }
            }

            DUAL auto bfield()
                requires sim_type::MHD<R>
            {
                return magnetic_vector_view_t<real, 3>(vals_ + Offsets::bfield);
            }

            // Individual magnetic field component
            DUAL real& b1()
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b1];
            }

            DUAL real& b2()
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b2];
            }

            DUAL real& b3()
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b3];
            }

            DUAL const real& b1() const
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b1];
            }

            DUAL const real& b2() const
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b2];
            }

            DUAL const real& b3() const
                requires sim_type::MHD<R>
            {
                return vals_[Offsets::b3];
            }

            DUAL const real& bcomponent(const luint nhat) const
                requires sim_type::MHD<R>
            {
                return (nhat == 1) ? b1() : (nhat == 2) ? b2() : b3();
            }

            DUAL real& bcomponent(const luint nhat)
                requires sim_type::MHD<R>
            {
                return (nhat == 1) ? b1() : (nhat == 2) ? b2() : b3();
            }

            // passive scalar
            DUAL real& chi() { return vals_[Offsets::chi]; }

            DUAL const real& chi() const { return vals_[Offsets::chi]; }
            // output the hydro state
            DUAL void print() const
            {
                for (luint i = 0; i < nmem; i++) {
                    std::cout << vals_[i] << " ";
                }
                std::cout << std::endl;
            }

            DUAL bool is_valid() const
            {
                for (luint i = 0; i < nmem; i++) {
                    if (std::isnan(vals_[i])) {
                        return false;
                    }
                }
                return true;
            }

            // overload the ostream operator
            DUAL friend std::ostream&
            operator<<(std::ostream& os, const anyHydro& hydro)
            {
                for (luint i = 0; i < nmem; i++) {
                    os << hydro.vals_[i] << " ";
                }
                return os;
            }

          protected:
            real vals_[nmem]{};

          public:
            // accessors
            DUAL real& operator[](size_t i) { return vals_[i]; }

            DUAL const real& operator[](size_t i) const { return vals_[i]; }

            DUAL real* data() { return vals_; }
            DUAL const real* data() const { return vals_; }

          private:
            DUAL constexpr Derived* derived()
            {
                return static_cast<Derived*>(this);
            }
        };

        // allow for scalar multiplication from right-hand side
        template <size_type Dims, typename Derived, Regime R>
        DUAL anyHydro<Dims, Derived, R>
        operator*(const real c, const anyHydro<Dims, Derived, R>& hydro)
        {
            anyHydro<Dims, Derived, R> result;
            for (luint ii = 0; ii < anyHydro<Dims, Derived, R>::nmem; ii++) {
                result[ii] = c * hydro[ii];
            }
            return result;
        }

    }   // namespace generic_hydro

    //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-
    // HYDRO STRUCTURES
    //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    template <size_type Dims, Regime R>
    struct anyConserved
        : public generic_hydro::anyHydro<Dims, anyConserved<Dims, R>, R>,
          public std::conditional_t<
              (comp_ct_type == CTAlgo::MdZ) && sim_type::MHD<R>,
              MdZVariablesStorage,
              EmptyMdZStorage> {
        using generic_hydro::anyHydro<Dims, anyConserved<Dims, R>, R>::anyHydro;
        using Base = generic_hydro::anyHydro<Dims, anyConserved<Dims, R>, R>;
        using Base::bfield;
        using Base::chi;
        using Base::vals_;
        using typename Base::Offsets;

        // Vector-based accessors
        DUAL real& dens() { return vals_[Offsets::density]; }

        DUAL const real& dens() const { return vals_[Offsets::density]; }

        // Momentum as vector
        DUAL auto momentum() const
        {
            return const_spatial_vector_view_t<real, Dims>{
              this->vals_ + Offsets::velocity
            };
        }

        DUAL auto momentum()
        {
            return spatial_vector_view_t<real, Dims>{
              this->vals_ + Offsets::velocity
            };
        }

        // Individual momentum components for compatibility
        DUAL real& m1() { return this->vals_[Offsets::v1]; }

        DUAL real& m2()
            requires(Dims > 1)
        {
            return vals_[Offsets::v2];
        }

        DUAL real& m3()
            requires(Dims > 2)
        {
            return vals_[Offsets::v3];
        }

        DUAL const real& m1() const { return vals_[Offsets::v1]; }

        DUAL const real& m2() const
            requires(Dims > 1)
        {
            return vals_[Offsets::v2];
        }

        DUAL const real& m3() const
            requires(Dims > 2)
        {
            return vals_[Offsets::v3];
        }

        // Energy
        DUAL real& nrg() { return vals_[Offsets::energy]; }

        DUAL const real& nrg() const { return vals_[Offsets::energy]; }

        DUAL real& mcomponent(const luint nhat)
        {
            if (nhat == 1) {
                return m1();
            }
            if constexpr (Dims > 1) {
                if (nhat == 2) {
                    return m2();
                }
                if constexpr (Dims > 2) {
                    if (nhat == 3) {
                        return m3();
                    }
                }
            }
            if constexpr (platform::is_gpu) {
                asm("trap;");
                return vals_[Offsets::v1];
            }
            else {
                throw std::out_of_range("Invalid momentum component");
            }
        }

        DUAL real mcomponent(const luint nhat) const
        {
            if (nhat == 1) {
                return m1();
            }
            if constexpr (Dims > 1) {
                if (nhat == 2) {
                    return m2();
                }
                if constexpr (Dims > 2) {
                    if (nhat == 3) {
                        return m3();
                    }
                }
            }
            return static_cast<real>(0.0);
        }

        // physical calculations
        DUAL real momentum_magnitude() const
        {
            return std::sqrt(vecops::dot(momentum(), momentum()));
        }

        DUAL real magnetic_energy() const
            requires sim_type::MHD<R>
        {
            return 0.5 * vecops::dot(bfield(), bfield());
        }

        DUAL real magnetic_norm_squared() const
            requires sim_type::MHD<R>
        {
            return vecops::dot(bfield(), bfield());
        }

        DUAL real total_energy() const
        {
            if constexpr (sim_type::Relativistic<R>) {
                return nrg() + dens();
            }
            else {
                return nrg();
            }
        }

        DUAL anyConserved<Dims, R> increment_gas_terms(
            const anyConserved<Dims, R>& other,
            bool isothermal = false
        )
        {
            dens() += other.dens();
            momentum() += other.momentum();
            if (!isothermal) {
                nrg() += other.nrg();
            }
            chi() += other.chi();
            return *this;
        }

        DUAL anyConserved<Dims, R>
        decrement_gas_terms(const anyConserved<Dims, R>& other)
            requires sim_type::MHD<R>
        {
            dens() -= other.dens();
            momentum() -= other.momentum();
            nrg() -= other.nrg();
            chi() -= other.chi();
            return *this;
        }

        // in MHD runs, the magnetic components are actually the EMF fluxes
        // so we add a method to get the electric field which is simply - nhat x
        // F
        DUAL auto calc_electric_field(const unit_vector_t<Dims>& nhat)
            requires sim_type::MHD<R>
        {
            // since the flux vector magnetic field components are actually
            // the electric field components, but in a different order, we
            // we compute the cross product with the unit vector direction
            // and store it back into the magnetic field for later electric
            // field retrieval
            if constexpr (Dims == 3) {
                const auto efield = -vecops::cross(nhat, bfield());
                // if (ehat == 1) {
                //     std::cout << "Bfield: " << bfield() << std::endl;
                //     std::cout << "Efield: " << efield << std::endl;
                //     std::cout << "========================" << std::endl;
                // }

                // std::cin.get();
                this->b1() = efield[0];
                this->b2() = efield[1];
                this->b3() = efield[2];
            }
        }

        DUAL auto ecomponent(const luint nhat) const
            requires sim_type::MHD<R>
        {
            // when calling this function in the conserved_t
            // struct, we are actually just grabbing the electric
            // field components computed from the Riemann problem
            // so we just return the magnetic field components
            return this->bcomponent(nhat);
        }

        DUAL auto pressure(const real gamma, bool isothermal) const
            requires sim_type::Newtonian<R>
        {
            if (isothermal) {
                // for isothermal runs, we store
                // the square of the sound speed
                // in the energy density
                return dens() * nrg();
            }
            const auto vel = momentum() / dens();
            return (gamma - 1.0) *
                   (nrg() - 0.5 * dens() * vecops::dot(vel, vel));
        }

        // MdZ accessors
        DUAL constexpr auto mdz_vars() const
        {
            if constexpr ((comp_ct_type == CTAlgo::MdZ) && sim_type::MHD<R>) {
                return Maybe(
                    static_cast<const MdZVariablesStorage*>(this)->get()
                );
            }
            else {
                return Nothing;
            }
        }

        DUAL constexpr auto mdz_vars()
        {
            if constexpr ((comp_ct_type == CTAlgo::MdZ) && sim_type::MHD<R>) {
                return Maybe(static_cast<MdZVariablesStorage*>(this)->get());
            }
            else {
                return Nothing;
            }
        }

        DUAL constexpr void set_mdz_vars(const MignoneDelZannaVariables& vars)
        {
            if constexpr ((comp_ct_type == CTAlgo::MdZ) && sim_type::MHD<R>) {
                static_cast<MdZVariablesStorage*>(this)->set(vars);
            }
        }

        DUAL constexpr real vLtrans(const luint perm) const
        {
            if constexpr (has_mdz_vars<R>()) {
                const auto& mdz = mdz_vars().value();
                return perm == 1 ? mdz.vjL : mdz.vkL;
            }
            else {
                return 0.0;
            }
        }

        DUAL constexpr real vRtrans(const luint perm) const
        {
            if constexpr (has_mdz_vars<R>()) {
                const auto& mdz = mdz_vars().value();
                return perm == 1 ? mdz.vjR : mdz.vkR;
            }
            else {
                return 0.0;
            }
        }
    };

    template <size_type Dims, Regime R>
    struct anyPrimitive
        : public generic_hydro::anyHydro<Dims, anyPrimitive<Dims, R>, R> {
        using counterpart_t = anyConserved<Dims, R>;
        using generic_hydro::anyHydro<Dims, anyPrimitive<Dims, R>, R>::anyHydro;
        using Base = generic_hydro::anyHydro<Dims, anyPrimitive<Dims, R>, R>;
        using Base::bfield;
        using Base::chi;
        using Base::vals_;
        using typename Base::Offsets;

        // Vector-based accessors
        DUAL real& rho() { return vals_[Offsets::density]; }

        DUAL const real& rho() const { return vals_[Offsets::density]; }

        // vector views
        DUAL auto velocity() const
        {
            return const_spatial_vector_view_t<real, Dims>{
              this->vals_ + Offsets::velocity
            };
        }

        DUAL auto velocity()
        {
            return spatial_vector_view_t<real, Dims>{
              this->vals_ + Offsets::velocity
            };
        }

        // Individual velocity components for compatibility
        DUAL real& v1() { return this->vals_[Offsets::v1]; }

        DUAL real& v2()
            requires(Dims > 1)
        {
            return vals_[Offsets::v2];
        }

        DUAL real& v3()
            requires(Dims > 2)
        {
            return vals_[Offsets::v3];
        }

        DUAL const real& v1() const { return vals_[Offsets::v1]; }

        DUAL const real& v2() const
            requires(Dims > 1)
        {
            return vals_[Offsets::v2];
        }

        DUAL const real& v3() const
            requires(Dims > 2)
        {
            return vals_[Offsets::v3];
        }

        // Pressure
        DUAL real& press() { return vals_[Offsets::energy]; }

        DUAL const real& press() const { return vals_[Offsets::energy]; }

        // need a dummy accessor for sotring Alfven
        // velocity in HLLD calculations
        DUAL real& alfven() { return vals_[Offsets::chi]; }

        DUAL const real& alfven() const { return vals_[Offsets::chi]; }

        DUAL real vcomponent(const luint nhat) const
        {
            if (nhat == 1) {
                return v1();
            }
            if constexpr (Dims > 1) {
                if (nhat == 2) {
                    return v2();
                }
                if constexpr (Dims > 2) {
                    if (nhat == 3) {
                        return v3();
                    }
                }
            }
            return static_cast<real>(0.0);
        }

        DUAL real& vcomponent(const luint nhat)
        {
            if (nhat == 1) {
                return v1();
            }
            if constexpr (Dims > 1) {
                if (nhat == 2) {
                    return v2();
                }
                if constexpr (Dims > 2) {
                    if (nhat == 3) {
                        return v3();
                    }
                }
            }
            if constexpr (platform::is_gpu) {
                asm("trap;");
                return vals_[Offsets::v1];
            }
            else {
                throw std::out_of_range("Invalid velocity component");
            }
        }

        DUAL real proper_velocity(const luint ehat) const
        {
            if constexpr (global::using_four_velocity) {
                return vcomponent(ehat) / lorentz_factor();
            }
            else {
                return vcomponent(ehat);
            }
        }

        // Physics
        DUAL auto sound_speed(real gamma) const
        {
            return std::sqrt(gamma * press() / (rho() * enthalpy(gamma)));
        }

        DUAL auto sound_speed_squared(real gamma) const
        {
            return gamma * press() / (rho() * enthalpy(gamma));
        }
        DUAL auto spatial_momentum(const real gamma) const
        {
            if constexpr (R == Regime::SRHD) {
                return (
                    velocity() * enthalpy_density(gamma) *
                    lorentz_factor_squared()
                );
            }
            else if constexpr (R == Regime::RMHD) {
                return (
                    velocity() *
                        (enthalpy(gamma) * rho() * lorentz_factor_squared() +
                         bsquared()) -
                    bfield() * vecops::dot(velocity(), bfield())
                );
            }
            else {
                return velocity() * rho();
            }
        }

        DUAL auto specific_energy(real gamma) const
        {
            return energy(gamma) / rho();
        }

        DEV constexpr auto calc_magnetic_four_vector() const
        {
            if constexpr (R != Regime::RMHD) {
                return magnetic_four_vector_t<real>();
            }
            else {
                return vecops::as_fourvec(
                    bfield(),
                    velocity(),
                    lorentz_factor()
                );
            }
        }

        DUAL constexpr real lorentz_factor() const
        {
            if constexpr (sim_type::Relativistic<R>) {
                if constexpr (global::using_four_velocity) {
                    return std::sqrt(1.0 + vsquared());
                }
                return 1.0 / std::sqrt(1.0 - vsquared());
            }
            else {
                return 1.0;
            }
        }

        DUAL constexpr real lorentz_factor_squared() const
        {
            if constexpr (sim_type::Relativistic<R>) {
                if constexpr (global::using_four_velocity) {
                    return 1.0 + vsquared();
                }
                return 1.0 / (1.0 - vsquared());
            }
            else {
                return 1.0;
            }
        }

        DUAL constexpr real enthalpy(const real gamma) const
        {
            if constexpr (sim_type::Relativistic<R>) {
                return 1.0 + gamma * press() / (rho() * (gamma - 1.0));
            }
            else {
                return 1.0;
            }
        }

        DUAL constexpr real vsquared() const
        {
            return vecops::dot(velocity(), velocity());
        }

        DUAL constexpr real bsquared() const
        {
            return vecops::dot(bfield(), bfield());
        }

        DUAL constexpr real vdotb() const
        {
            if constexpr (sim_type::MHD<R>) {
                return vecops::dot(velocity(), bfield());
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL constexpr real total_pressure() const
        {
            if constexpr (sim_type::MHD<R>) {
                return press() + bpressure();
            }
            else {
                return press();
            }
        }

        DUAL constexpr real bpressure() const
        {
            if constexpr (R == Regime::RMHD) {
                const auto bmu = calc_magnetic_four_vector();
                return 0.5 * bmu.inner_product(bmu);
            }
            else if constexpr (sim_type::MHD<R>) {
                return 0.5 * bsquared();
            }
            else {
                return static_cast<real>(0.0);
            }
        }

        DUAL constexpr real enthalpy_density(const real gamma) const
        {
            if constexpr (sim_type::MHD<R>) {
                return rho() * enthalpy(gamma) + 2.0 * bpressure();
            }
            else {
                return rho() * enthalpy(gamma);
            }
        }

        DUAL constexpr real labframe_density() const
        {
            return rho() * lorentz_factor();
        }

        DUAL constexpr real energy(const real gamma) const
        {
            if constexpr (R == Regime::NEWTONIAN) {
                if (helpers::goes_to_zero(gamma - 1.0)) {
                    return 0.0;
                }
                return press() / (gamma - 1.0) + 0.5 * rho() * vsquared();
            }
            else if constexpr (R == Regime::SRHD) {
                return enthalpy_density(gamma) * lorentz_factor_squared() -
                       press() - rho() * lorentz_factor();
            }
            else {
                return rho() * lorentz_factor_squared() * enthalpy(gamma) -
                       press() - rho() * lorentz_factor() +
                       0.5 * (bsquared() + vsquared() * bsquared() -
                              vdotb() * vdotb());
            }
        }

        DUAL constexpr auto pkdelta(const unit_vector_t<Dims>& nhat) const
        {
            if constexpr (sim_type::MHD<R>) {
                return nhat * total_pressure();
            }
            else {
                return nhat * press();
            }
        }

        DUAL constexpr auto electric_field() const
            requires sim_type::MHD<R>
        {
            return -vecops::cross(velocity(), bfield());
        }

        DUAL constexpr auto electric_field(luint ehat) const
        {
            return -vecops::cross_component(velocity(), bfield(), ehat);
        }

        DUAL constexpr real ecomponent(const luint nhat) const
            requires sim_type::MHD<R>
        {
            if constexpr (Dims == 3) {
                return electric_field(nhat);
            }
            else {
                return 0.0;
            }
        }

        DUAL anyConserved<Dims, R> to_conserved(const real gamma) const
        {
            return {
              labframe_density(),
              spatial_momentum(gamma),
              energy(gamma),
              chi() * labframe_density(),
              bfield()
            };
        }

        // conversion to flux
        DUAL anyConserved<Dims, R>
        to_flux(const real gamma, const unit_vector_t<Dims>& nhat) const
        {
            const auto vnorm = vecops::dot(velocity(), nhat);
            const auto mom   = spatial_momentum(gamma);
            const auto mnorm = vecops::dot(mom, nhat);
            if constexpr (R == Regime::NEWTONIAN) {
                return {
                  mnorm,
                  mom * vnorm + pkdelta(nhat),
                  (energy(gamma) + press()) * vnorm,
                  chi()
                };
            }
            else if constexpr (R == Regime::SRHD) {
                const auto d = labframe_density();
                return {
                  d * vnorm,
                  mom * vnorm + pkdelta(nhat),
                  mnorm - d * vnorm,
                  chi() * d * vnorm
                };
            }
            else {
                if constexpr (Dims == 3) {
                    const auto bnorm = vecops::dot(bfield(), nhat);
                    const auto induction =
                        vecops::cross(nhat, electric_field());
                    const auto bmu_spatial =
                        bfield() / lorentz_factor() +
                        velocity() * lorentz_factor() * vdotb();
                    const auto d = labframe_density();
                    return {
                      d * vnorm,
                      mom * vnorm + pkdelta(nhat) -
                          bmu_spatial * bnorm / lorentz_factor(),
                      mnorm - vnorm * d,
                      chi() * d * vnorm,
                      induction.as_magnetic(),
                    };
                }
                else {
                    return {};
                }
            }
        }

        // output the hydro state
        void error_at(
            const luint ii,
            const luint jj,
            const luint kk,
            const real x1,
            const real x2,
            const real x3,
            const ErrorCode error_code,
            auto& table
        ) const
        {
            std::ostringstream oss;
            oss << "Primitives in non-physical state.\n";
            if (error_code != ErrorCode::NONE) {
                oss << "reason: " << error_code_to_string(error_code) << "\n";
            }
            if constexpr (Dims == 1) {
                oss << "location: (" << x1 << "): \n";
            }
            else if constexpr (Dims == 2) {
                if (x2 ==
                    std::numeric_limits<real>::infinity()) {   // an effective
                                                               // 1D run
                    oss << "location: (" << x1 << "): \n";
                    oss << "index: [" << ii << "]\n";
                }
                else {
                    oss << "location: (" << x1 << ", " << x2 << "): \n";
                    oss << "indices: [" << ii << ", " << jj << "]\n";
                }
            }
            else {
                if (x2 ==
                    std::numeric_limits<real>::infinity()) {   // an effective
                                                               // 1D run
                    oss << "location: (" << x1 << "): \n";
                    oss << "indicies: [" << ii << "]\n";
                }
                else if (x3 == std::numeric_limits<real>::infinity()) {
                    oss << "location: (" << x1 << ", " << x2 << "): \n";
                    oss << "indices: [" << ii << ", " << jj << "]\n";
                }
                else {
                    oss << "location: (" << x1 << ", " << x2 << ", " << x3
                        << "): \n";
                    oss << "indices: [" << ii << ", " << jj << ", " << kk
                        << "]\n";
                }
            }
            table.post_error(oss.str());
        }
    };

    struct wave_speeds_t {
        static constexpr size_t size = 6;
        real data[size];

        DUAL constexpr wave_speeds_t() : data{} {}

        DUAL constexpr wave_speeds_t(
            real v1p,
            real v1m,
            real v2p,
            real v2m,
            real v3p,
            real v3m
        )
            : data{v1p, v1m, v2p, v2m, v3p, v3m}
        {
        }

        // Array access
        DUAL constexpr real& operator[](size_t ii) { return data[ii]; }

        DUAL constexpr const real& operator[](size_t ii) const
        {
            return data[ii];
        }
    };

    template <size_type Dims, Regime R>
    struct Eigenvals {
        constexpr static int nvals = 4;
        real vals_[nvals];

        Eigenvals()  = default;
        ~Eigenvals() = default;

        // Generic Constructor
        template <typename... Args>
        DUAL Eigenvals(Args... args) : vals_{static_cast<real>(args)...}
        {
        }

        // Define accessors for the wave speeds
        DUAL constexpr real aL() const { return vals_[0]; }

        DUAL constexpr real aR() const { return vals_[1]; }

        DUAL constexpr real afL() const { return vals_[0]; }

        DUAL constexpr real afR() const { return vals_[1]; }

        DUAL constexpr real asL() const { return vals_[2]; }

        DUAL constexpr real asR() const { return vals_[3]; }

        DUAL constexpr real aStar() const { return vals_[2]; }

        DUAL constexpr real pStar() const { return vals_[3]; }
    };

    struct EntropyDetector {
        DEV static real
        compute_local_mach(const auto& prL, const auto& prR, real gamma)
        {
            const auto velL          = prL.velocity();
            const auto velR          = prR.velocity();
            const auto sound_speed_L = prL.sound_speed(gamma);
            const auto sound_speed_R = prR.sound_speed(gamma);

            // use the maximum of the Mach numbers from both sides
            // as suggested by Fleischmann et al. (2020)
            const real mach_L = vecops::norm(velL) / sound_speed_L;
            const real mach_R = vecops::norm(velR) / sound_speed_R;

            return my_max(mach_L, mach_R);
        }

        DEV static real
        detect_interface_correction(const auto& prL, const auto& prR)
        {
            // detect material interfaces (RT-style contact discontinuities)
            const real rho_jump = std::abs(prL.rho() - prR.rho()) /
                                  (0.5 * (prL.rho() + prR.rho()));
            const real pressure_jump = std::abs(prL.press() - prR.press()) /
                                       (0.5 * (prL.press() + prR.press()));

            // interface = large density jump, small pressure jump (nearly
            // isentropic)
            const bool is_contact = (rho_jump > 0.1) && (pressure_jump < 0.05);

            if (is_contact) {
                // need moderate dissipation to prevent interface oscillations
                return 0.4;   // fixed moderate phi for interfaces
            }
            else {
                return 0.0;   // no correction
            }
        }

        DEV static real detect_shock_correction(
            const auto& prL,
            const auto& prR,
            real gamma,
            const auto& nhat
        )
        {
            // entropy production (shocks increase entropy)
            const real sL = std::log(prL.press()) - gamma * std::log(prL.rho());
            const real sR = std::log(prR.press()) - gamma * std::log(prR.rho());
            const real entropy_production = sR - sL;

            // velocity convergence (shocks compress flow)
            const real vL_normal = vecops::dot(prL.velocity(), nhat);
            const real vR_normal = vecops::dot(prR.velocity(), nhat);
            const real velocity_convergence =
                vL_normal - vR_normal;   // > 0 for compression

            // combined shock indicator
            const bool is_shock =
                (entropy_production > 0.01) && (velocity_convergence > 0.0);

            if (is_shock) {
                return 1.0;   // force standard HLLC for shocks
            }
            else {
                return 0.0;   // no correction
            }
        }

        DEV static real detect_stagnation_correction(
            const auto& prL,
            const auto& prR,
            real gamma
        )
        {
            // detect very low velocity regions (want maximum LM treatment)
            const real cL     = prL.sound_speed(gamma);
            const real cR     = prR.sound_speed(gamma);
            const real vL_mag = vecops::norm(prL.velocity());
            const real vR_mag = vecops::norm(prR.velocity());

            const real mach_L   = vL_mag / cL;
            const real mach_R   = vR_mag / cR;
            const real max_mach = my_max(mach_L, mach_R);

            if (max_mach < 0.01) {
                // nearly stagnant → force low dissipation
                return -0.5;   // negative correction to reduce phi
            }
            else {
                return 0.0;   // no correction
            }
        }

        DEV static real detect_alignment_correction(
            const auto& prL,
            const auto& prR,
            real gamma,
            const auto& nhat
        )
        {
            // check if flow is aligned with interface (carbuncle risk)
            const real vL_normal = vecops::dot(prL.velocity(), nhat);
            const real vR_normal = vecops::dot(prR.velocity(), nhat);
            const real vL_mag    = vecops::norm(prL.velocity());
            const real vR_mag    = vecops::norm(prR.velocity());

            if (vL_mag > 1e-10 && vR_mag > 1e-10) {
                const real alignment_L   = std::abs(vL_normal) / vL_mag;
                const real alignment_R   = std::abs(vR_normal) / vR_mag;
                const real max_alignment = my_max(alignment_L, alignment_R);

                // high speed + high alignment = carbuncle risk
                const real avg_mach = 0.5 * (vL_mag / prL.sound_speed(gamma) +
                                             vR_mag / prR.sound_speed(gamma));

                if ((max_alignment > 0.8) && (avg_mach > 0.5)) {
                    return 1.0;   // force standard HLLC to prevent carbuncle
                }
            }

            return 0.0;   // no correction
        }

        DEV static real compute_adaptive_phi(
            const auto& prL,
            const auto& prR,
            const auto& nhat,
            real gamma,
            bool use_fleischmann
        )
        {
            if (!use_fleischmann) {
                return 1.0;   // no adaptive phi, use standard HLLC
            }

            // base Mach number criterion
            // This number is found in Fleischamnn et al. (2020)
            // A shock-stable modification of the HLLC Riemann solver with
            // reduced numerical dissipation
            constexpr real mach_lim = 0.1;
            const real ma_local     = compute_local_mach(prL, prR, gamma);
            real phi = std::sin(my_min(1.0, ma_local / mach_lim) * M_PI * 0.5);

            // physics-based corrections
            const real correction_factors[] = {
              detect_interface_correction(
                  prL,
                  prR
              ),   // boost phi for interfaces
              detect_shock_correction(
                  prL,
                  prR,
                  gamma,
                  nhat
              ),   // force phi=1 for shocks
              detect_stagnation_correction(
                  prL,
                  prR,
                  gamma
              ),   // reduce phi for stagnant regions
              detect_alignment_correction(
                  prL,
                  prR,
                  gamma,
                  nhat
              )   // boost phi for aligned flows
            };

            // apply the strongest correction
            for (auto factor : correction_factors) {
                phi = my_max(phi, factor);
            }

            return my_min(1.0, phi);
        }
    };

}   // namespace simbi

//=======================================================
// TYPE TRAITS
//=======================================================
// Partial specialization for anyPrimitive
template <size_type Dims, simbi::Regime R>
struct is_primitive<simbi::anyPrimitive<Dims, R>> {
    static const bool value = true;
};

// Partial specialization for anyConserved
template <size_type Dims, simbi::Regime R>
struct is_conserved<simbi::anyConserved<Dims, R>> {
    static const bool value = true;
};

// Partial specialization for MHD
template <size_type Dims, simbi::Regime R>
struct is_mhd<simbi::anyConserved<Dims, R>> {
    static const bool value = (R == simbi::Regime::RMHD);
};

// Partial specialization for relativistic
template <size_type Dims, simbi::Regime R>
struct is_relativistic<simbi::anyConserved<Dims, R>> {
    static const bool value =
        (R == simbi::Regime::SRHD) || (R == simbi::Regime::RMHD);
};
#endif
