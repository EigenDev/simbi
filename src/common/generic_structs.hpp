/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       hydro_structs.hpp
 * @brief      the data structs for states, primitives, and sim configuration
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 */
#ifndef HYDRO_STRUCTS_HPP
#define HYDRO_STRUCTS_HPP

#include "build_options.hpp"
#include "enums.hpp"
#include "traits.hpp"
#include "util/tabulate.hpp"
#include "util/vector.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace simbi;

/**
 * @brief kronecker delta
 *
 * @param i
 * @param j
 * @return 1 for identity, 0 otherwise
 */
STATIC
constexpr unsigned int kdelta(luint i, luint j) { return (i == j); }

//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------
struct InitialConditions {
    real tstart, chkpt_interval, dlogt;
    real plm_theta, engine_duration, gamma, cfl, tend;
    luint nx, ny, nz, chkpt_idx;
    bool quirk_smoothing, constant_sources;
    std::vector<std::vector<real>> sources, gsources, osources, bfield;
    std::vector<bool> object_cells;
    std::string data_directory, coord_system, solver;
    std::string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing, regime;
    std::string hydro_source_lib, gravity_source_lib, boundary_source_lib;
    std::string spatial_order, time_order;
    std::vector<std::string> boundary_conditions;
    std::vector<real> x1, x2, x3;
};

namespace generic_hydro {

    template <int dim, typename Derived, Regime R>
    struct anyHydro {
        static constexpr int nmem = (R == Regime::RMHD) ? 9 : 3 + dim;

        // base storage for iteration/performance
        alignas(32) real vals_[nmem];

        // Wave speed components
        real aL_, aR_, dL_, dR_, vjL_, vjR_, vkL_, vkR_, lamL_, lamR_;

        // Constructors
        anyHydro()  = default;
        ~anyHydro() = default;

        // generic constructor
        template <typename... Args>
        anyHydro(Args... args) : vals_{static_cast<real>(args)...}
        {
            if constexpr (R == Regime::RMHD) {
                // Initialize MHD components
                static_assert(
                    sizeof...(args) == 9,
                    "RMHD requires 9 components"
                );
            }
            else {
                static_assert(
                    sizeof...(args) == dim + 3,
                    "Non-MHD requires dim + 3 components"
                );
            }
            init_refs();
        }

        // move constructor
        anyHydro(const anyHydro& other) = default;

        // copy constructor
        anyHydro(const Derived& other) = default;

        anyHydro(const Derived& other) : vals_{other.vals_} { init_refs(); }

        // move assignment
        anyHydro& operator=(const anyHydro& other) = default;

        // copy assignment
        anyHydro& operator=(const Derived& other) = default;

        // math operations
        DUAL Derived& operator-=(const Derived& other)
        {
            for (size_t i = 0; i < nmem; ++i) {
                vals_[i] -= other.vals_[i];
            }
            return *derived();
        }

        DUAL Derived& operator+=(const Derived& other)
        {
            for (size_t i = 0; i < nmem; ++i) {
                vals_[i] += other.vals_[i];
            }
            return *derived();
        }

        DUAL Derived& operator*=(const real c)
        {
            for (size_t i = 0; i < nmem; ++i) {
                vals_[i] *= c;
            }
            return *derived();
        }

        DUAL Derived& operator/=(const real c)
        {
            for (size_t i = 0; i < nmem; ++i) {
                vals_[i] /= c;
            }
            return *derived();
        }

        DUAL Derived& operator=(const Derived& other)
        {
            for (size_t i = 0; i < nmem; ++i) {
                vals_[i] = other.vals_[i];
            }
            return *derived();
        }

        DUAL Derived operator*(const real scalar)
        {
            Derived result;
            for (size_t i = 0; i < nmem; ++i) {
                result.vals_[i] = vals_[i] * scalar;
            }
            return result;
        }

        DUAL Derived operator/(const real scalar)
        {
            Derived result;
            for (size_t i = 0; i < nmem; ++i) {
                result.vals_[i] = vals_[i] / scalar;
            }
            return result;
        }

        Dual Derived operator+(const real scalar)
        {
            Derived result;
            for (size_t i = 0; i < nmem; ++i) {
                result.vals_[i] = vals_[i] + scalar;
            }
            return result;
        }

      protected:
        // Vector views through unions for zero-overhead access
        union {
            // Raw array view
            real raw_[nmem];

            // Primitive view
            struct {
                real& rho_;   // Rest-mass density
                Vector<real, 3, VectorType::SPATIAL>& vel_;
                real& press_;
                std::conditional_t<
                    R == Regime::RMHD,
                    Vector<real, 3, VectorType::MAGNETIC>&,
                    std::monostate>
                    bfield_;
                real& chi_;
            };

            // Conserved view
            struct {
                real& dens_;   // Lab-frame density
                Vector<real, 3, VectorType::SPATIAL>& mom_;
                real& nrg_;
                std::conditional_t<
                    R == Regime::RMHD,
                    Vector<real, 3, VectorType::MAGNETIC>&,
                    std::monostate>
                    cons_bfield_;
                real& cons_chi_;
            };
        };

      public:
        // accessors
        DUAL real& operator[](size_t i) { return vals_[i]; }

        DUAL const real& operator[](size_t i) const { return vals_[i]; }

        // implicit conversion to underlying array
        DUAL operator real*() { return vals_; }

      private:
        // Initialize union references
        DUAL void init_refs()
        {
            // Point primitive view to array sections
            rho_ = vals_[0];
            new (&vel_) Vector<real, 3, VectorType::SPATIAL>(
                vals_ + 1,
                vals_ + std::min(4, dim + 1)
            );
            press_ = vals_[dim + 1];

            if constexpr (R == Regime::RMHD) {
                new (&bfield_) Vector<real, 3, VectorType::MAGNETIC>(
                    vals_ + dim + 2,
                    vals_ + dim + 5
                );
            }
            chi_ = vals_[nmem - 1];

            // Point conserved view to same memory
            dens_ = vals_[0];
            new (&mom_) Vector<real, 3, VectorType::SPATIAL>(
                vals_ + 1,
                vals_ + std::min(4, dim + 1)
            );
            nrg_ = vals_[dim + 1];

            if constexpr (R == Regime::RMHD) {
                new (&cons_bfield_) Vector<real, 3, VectorType::MAGNETIC>(
                    vals_ + dim + 2,
                    vals_ + dim + 5
                );
            }
            cons_chi_ = vals_[nmem - 1];
        }

        DUAL auto* derived() { return static_cast<Derived&>(*this); }
    };
}   // namespace generic_hydro

//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==--=-=-=-=-=-=-=-=-
// HYDRO STRUCTURES
//=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
template <size_type Dims, Regime R>
anyConserved;

template <size_type Dims, Regime R>
anyPrimitive;

template <size_type Dims, Regime R>
template <size_type Dims, Regime R>
struct anyConserved : generic_hydro::anyHydro<Dims, anyConserved<Dims, R>, R> {
    using Base = generic_hydro::anyHydro<Dims, anyConserved<Dims, R>, R>;
    using Base::vals_;

    // Vector-based accessors
    DUAL real& dens() { return this->dens_; }

    DUAL const real& dens() const { return this->dens_; }

    // Momentum as vector
    DUAL const Vector<real, 3, VectorType::SPATIAL>& momentum() const
    {
        return this->mom_;
    }

    DUAL Vector<real, 3, VectorType::SPATIAL>& momentum() { return this->mom_; }

    // Individual momentum components for compatibility
    DUAL real& m1() { return this->mom_[0]; }

    DUAL real& m2()
        requires(Dims > 1)
    {
        return this->mom_[1];
    }

    DUAL real& m3()
        requires(Dims > 2)
    {
        return this->mom_[2];
    }

    DUAL const real& m1() const { return this->mom_[0]; }

    DUAL const real& m2() const
        requires(Dims > 1)
    {
        return this->mom_[1];
    }

    DUAL const real& m3() const
        requires(Dims > 2)
    {
        return this->mom_[2];
    }

    // Energy
    DUAL real& nrg() { return this->nrg_; }

    DUAL const real& nrg() const { return this->nrg_; }

    // Magnetic field for MHD runs
    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL const Vector<real, 3, VectorType::MAGNETIC>& bfield() const
    {
        return this->cons_bfield_;
    }

    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL Vector<real, 3, VectorType::MAGNETIC>& bfield()
    {
        return this->cons_bfield_;
    }

    // Individual magnetic field component
    DUAL real& b1()
        requires(R == Regime::RMHD)
    {
        return this->cons_bfield_[0];
    }

    DUAL real& b2()
        requires(R == Regime::RMHD)
    {
        return this->cons_bfield_[1];
    }

    DUAL real& b3()
        requires(R == Regime::RMHD)
    {
        return this->cons_bfield_[2];
    }

    DUAL real& chi() { return this->cons_chi_; }

    DUAL const real& chi() const { return this->cons_chi_; }

    // physical calculations
    DUAL real momentum_magnitude() const
    {
        return std::sqrt(momentum().dot(momentum()));
    }

    DUAL spatial_vector_t spatial_momentum(const real gamma) const
    {
        if constexpr (R == Regime::SRHD) {
            return (
                velocity() * enthalpy_density(gamma) * lorentz_factor_squared()
            );
        }
        else if constexpr (R == Rgime::RMHD) {
            return (
                velocity() *
                    (enthalpy(gamma) * rho() * lorentz_factor_squared() +
                     bsquared()) -
                bfield() * (velocity().dot(bfield()))
            );
        }
        return velocity() * rho();
    }

    DUAL mag_four calc_magnetic_four_vector() const
        requires(R == Regime::RMHD)
    {
        return bfield().compose_magnetic_fourvector(
            velocity(),
            lorentz_factor()
        );
    }

    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL real magnetic_energy() const
    {
        return 0.5 * bfield().dot(bfield());
    }

    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL real magnetic_magnitude_squared() const
    {
        return bfield().dot(bfield());
    }

    DUAL real total_energy() const
    {
        if constexpr (R == Regime::RMHD || R == Regime::SRHD) {
            return nrg() + dens();
        }
        else {
            return nrg();
        }
    }

    // EMF calculations for MHD runs
    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL Vector<real, 3, VectorType::MAGNETIC>
    calc_emf(const anyPrimitive<Dims, R>& prim) const
    {
        return prim.velocity().cross(bfield());
    }

    // override the + and - operators to ignore
    // the magnetic field componeents if detecting an MHD
    // run
    DUAL anyConserved<Dims, R> operator-(const anyConserved<Dims, R>& other
    ) const
    {
        if constexpr (R == Regime::RMHD) {
            return anyConserved<Dims, R>{
              dens() - other.dens(),
              momentum() - other.momentum(),
              nrg() - other.nrg(),
              bfield(),
              chi() - other.chi()
            };
        }
        else {
            return anyConserved<Dims, R>{
              dens() - other.dens(),
              momentum() - other.momentum(),
              nrg() - other.nrg(),
              chi() - other.chi()
            };
        }
    }

    DUAL anyConserved<Dims, R> operator+(const anyConserved<Dims, R>& other
    ) const
    {
        if constexpr (R == Regime::RMHD) {
            return anyConserved<Dims, R>{
              dens() + other.dens(),
              momentum() + other.momentum(),
              nrg() + other.nrg(),
              bfield(),
              chi() + other.chi()
            };
        }
        else {
            return anyConserved<Dims, R>{
              dens() + other.dens(),
              momentum() + other.momentum(),
              nrg() + other.nrg(),
              chi() + other.chi()
            };
        }
    }
};

tempalte<size_type Dims, Regime R> anyPrimitive
    : generic_hydro < anyPrimitive<Dims, anyPrimitive<Dims, R>, R>
{
    using Base = generic_hydro::anyHydro<Dims, anyPrimitive<Dims, R>, R>;
    using Base::vals_;

    // Vector-based accessors
    DUAL real& rho() { return this->rho_; }

    DUAL const real& rho() const { return this->rho_; }

    // Velocity as vector
    DUAL const Vector<real, 3, VectorType::SPATIAL>& velocity() const
    {
        return this->vel_;
    }

    // vector views
    DUAL const Vector<real, 3, VectorType::SPATIAL>& velocity() const
    {
        return this->vel_;
    }

    DUAL Vector<real, 3, VectorType::SPATIAL>& velocity() { return this->vel_; }

    // Individual velocity components for compatibility
    DUAL real& v1() { return this->vel_[0]; }

    DUAL real& v2()
        requires(Dims > 1)
    {
        return this->vel_[1];
    }

    DUAL real& v3()
        requires(Dims > 2)
    {
        return this->vel_[2];
    }

    DUAL const real& v1() const { return this->vel_[0]; }

    DUAL const real& v2() const
        requires(Dims > 1)
    {
        return this->vel_[1];
    }

    DUAL const real& v3() const
        requires(Dims > 2)
    {
        return this->vel_[2];
    }

    // Pressure
    DUAL real& press() { return this->press_; }

    DUAL const real& press() const { return this->press_; }

    // Magnetic field for MHD runs
    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL const Vector<real, 3, VectorType::MAGNETIC>& bfield() const
    {
        return this->bfield_;
    }

    template <typename = std::enable_if_t<R == Regime::RMHD>>
    DUAL Vector<real, 3, VectorType::MAGNETIC>& bfield()
    {
        return this->bfield_;
    }

    // Individual magnetic field component
    DUAL real& b1()
        requires(R == Regime::RMHD)
    {
        return this->bfield_[0];
    }

    DUAL real& b2()
        requires(R == Regime::RMHD)
    {
        return this->bfield_[1];
    }

    DUAL real& b3()
        requires(R == Regime::RMHD)
    {
        return this->bfield_[2];
    }

    // passive scalar
    DUAL real& chi() { return this->chi_; }
    DUAL real& chi() const { return this->chi; }

    DUAL constexpr real lorentz_factor() const
    {
        if constexpr (R == Regime::RMHD || R == Regime::SRHD) {
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
        if constexpr (R == Regime::RMHD || R == Regime::SRHD) {
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
        if constexpr (R == Regime::RMHD || R == Regime::SRHD) {
            return 1.0 + gamma * press() / (rho() * (gamma - 1.0));
        }
        else {
            return 1.0;
        }
    }

    DUAL constexpr real vsquared() const { return velocity().dot(velocity()); }
    DUAL constexpr real bsquared() const { return bfield().dot(bfield()); }

    DUAl constexpr real vdotb() const
    {
        if constexpr (R == Regime::RMHD) {
            return velocity().dot(bfield());
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real total_pressure() const
    {
        if constexpr (R == Regime::RMHD) {
            return press() + 0.5 * bsquared();
        }
        else {
            return press();
        }
    }

    DUAL constexpr real enthalpy_density(const real gamma) const
    {
        if constexpr (R == Regime::RMHD) {
            return rho() * enthalpy(gamma) + bpressure();
        }
        else {
            return rho() * enthalpy(gamma);
        }
    }

    DUAL constexpr real labframe_density() const
    {
        return rho() * lorentz_factor();
    }

    DUAL constexpr real total_energy(const real gamma) const
    {
        if constexpr (R == Regime::NEWTONIAN) {
            return p() / (gamma - 1.0) + 0.5 * rho() * vsquared();
        }
        else if constexpr (R == Regime::SRHD) {
            return rho() * lorentz_factor_squared() * enthalpy(gamma) - press();
        }
        else {
            return rho() * lorentz_factor_squared() * enthalpy(gamma) -
                   press() +
                   0.5 * (bsquared() + bsquared() + vsquared() * bsquared() -
                          vdotb() * vdotb());
        }
    }

    DUAL constexpr real pkdelta(const unit_vector_t& nhat) const
    {
        if constexpr (R == Regime::RMHD) {
            return nhat * total_pressure();
        }
        else {
            return nhat * press();
        }
    }

    // conversion to conserved
    DUAL anyConserved<Dims, R> to_conserved(const real gamma) const
    {
        return {
          labframe_density(),
          spatial_momentum(gamma),
          total_energy(gamma),
          chi() * labframe_density()
        };
    }

    // conversion to flux
    DUAL anyConserved<Dims, R> to_flux(
        const real gamma,
        const unit_vector_t& nhat
    ) const
    {
        auto vnorm = velocity().dot(nhat);
        auto mom   = spatial_momentum(gamma);
        auto mnorm = mom.dot(nhat);
        if constexpr (R == Regime::NEWTONIAN) {
            return {
              mnorm,
              mom * vorm + pkdelta(),
              (total_energy(gamma) + press()) * vnorm,
              chi()
            };
        }
        else if constexpr (R == Regime::SRHD) {
            auto d = labframe_density();
            return {
              d * vnorm,
              mom * vnorm + pkdelta(),
              mnorm - d * vnorm,
              chi() * d * vnorm
            };
        }
        else {
            auto bnorm     = bfield().dot(nhat);
            auto induction = -nhat.cross(velocity().cross(bfield()));
            auto bmu       = bfield.as_fourvec(velocity(), lorentz_factor());
            auto bmu_contr = bmu.spatial_part() * bnorm / lorentz_factor();
            auto d         = labframe_density();
            return {
              d * vnorm,
              mom * vnorm + pkdelta() - bmu_contr,
              mnorm - vnorm * d,
              induction,
              chi() * d * vnorm
            };
        }
    }
};

struct WaveSpeeds {
    real v1p, v1m, v2p, v2m, v3p, v3m;
};

template <int dim, Regime R>
struct Eigenvals {
    constexpr static int nvals = 4 + 2 * (R == Regime::NEWTONIAN && dim > 1);
    real vals[nvals];

    Eigenvals()  = default;
    ~Eigenvals() = default;

    // Generic Constructor
    template <typename... Args>
    DUAL Eigenvals(Args... args) : vals{static_cast<real>(args)...}
    {
    }

    // Define accessors for the wave speeds
    DUAL constexpr real aL() const { return this->vals[0]; }

    DUAL constexpr real aR() const { return this->vals[1]; }

    DUAL constexpr real afL() const { return this->vals[0]; }

    DUAL constexpr real afR() const { return this->vals[1]; }

    DUAL constexpr real asL() const { return this->vals[2]; }

    DUAL constexpr real asR() const { return this->vals[3]; }

    DUAL constexpr real csL() const { return this->vals[2]; }

    DUAL constexpr real csR() const { return this->vals[3]; }

    DUAL constexpr real aStar() const
    {
        if constexpr (nvals > 4) {
            return this->vals[4];
        }
        else if constexpr (R == Regime::NEWTONIAN && dim == 1) {
            return this->vals[2];
        }
        return static_cast<real>(0.0);
    }

    DUAL constexpr real pStar() const
    {
        if constexpr (nvals > 4) {
            return this->vals[5];
        }
        else if constexpr (R == Regime::NEWTONIAN && dim == 1) {
            return this->vals[3];
        }
        return static_cast<real>(0.0);
    }
};

//=======================================================
// TYPE TRAITS
//=======================================================
template <>
struct is_1D_mhd_primitive<anyPrimitive<1, Regime::RMHD>> : std::true_type {
};

template <>
struct is_2D_mhd_primitive<anyPrimitive<2, Regime::RMHD>> : std::true_type {
};

template <>
struct is_3D_mhd_primitive<anyPrimitive<3, Regime::RMHD>> : std::true_type {
};

template <int dim>
struct is_relativistic_mhd<anyPrimitive<dim, Regime::RMHD>> : std::true_type {
};

template <int dim>
struct is_relativistic_mhd<anyConserved<dim, Regime::RMHD>> : std::true_type {
};

template <int dim>
struct is_relativistic<anyPrimitive<dim, Regime::SRHD>> : std::true_type {
};

template <int dim>
struct is_relativistic<anyConserved<dim, Regime::RMHD>> : std::true_type {
};

template <>
struct is_1D_primitive<anyPrimitive<1, Regime::NEWTONIAN>> : std::true_type {
};

template <>
struct is_1D_primitive<anyPrimitive<1, Regime::SRHD>> : std::true_type {
};

template <>
struct is_2D_primitive<anyPrimitive<2, Regime::NEWTONIAN>> : std::true_type {
};

template <>
struct is_2D_primitive<anyPrimitive<2, Regime::SRHD>> : std::true_type {
};

template <>
struct is_3D_primitive<anyPrimitive<3, Regime::NEWTONIAN>> : std::true_type {
};

template <>
struct is_3D_primitive<anyPrimitive<3, Regime::SRHD>> : std::true_type {
};
#endif