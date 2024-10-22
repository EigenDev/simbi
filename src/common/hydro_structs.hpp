/**
 * ***********************(C) COPYRIGHT 2024 Marcus DuPont**********************
 * @file       hydro_structs.hpp
 * @brief      the data structs for states, primitives, and sim configuration
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont                   md4469@nyu.edu
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
#include <cmath>
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
struct PrimData {
    std::vector<real> rho, v1, v2, v3, p, b1, b2, b3, chi;
};

struct DataWriteMembers {
    int nx, ny, nz;
    int xactive_zones, yactive_zones, zactive_zones;
    int chkpt_idx, dimensions;
    bool using_fourvelocity, mesh_motion;
    real t, ad_gamma;
    real x1min, x1max, x2min, x2max, x3min, x3max, dt;
    std::string coord_system, regime;
    std::string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing;
    std::string spatial_order, time_order;
    std::vector<real> x1, x2, x3;
    std::vector<std::string> boundary_conditions;

    DataWriteMembers()
        : nx(1),
          ny(1),
          nz(1),
          x1min(0.0),
          x1max(0.0),
          x2min(0.0),
          x2max(0.0),
          x3min(0.0),
          x3max(0.0)
    {
    }
};

struct InitialConditions {
    real tstart, chkpt_interval, dlogt;
    real plm_theta, engine_duration, gamma, cfl, tend;
    luint nx, ny, nz, chkpt_idx;
    bool quirk_smoothing, constant_sources;
    std::vector<std::vector<real>> sources, gsources, bsources, bfield;
    std::vector<bool> object_cells;
    std::string data_directory, coord_system, solver;
    std::string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing, regime;
    std::string spatial_order, time_order;
    std::vector<std::string> boundary_conditions;
    std::vector<std::vector<real>> boundary_sources;
    std::vector<real> x1, x2, x3;
};

namespace generic_hydro {
    // implementing curiously recurring template pattern (CRTP)
    template <int dim, typename Derived, Regime R>
    struct anyHydro {
        constexpr static int nmem = []() {
            if constexpr (R == Regime::RMHD) {
                return 9;
            }
            return 3 + dim;
        }();
        real vals[nmem];
        // Default Destructor
        ~anyHydro() = default;

        // Default Constructor
        anyHydro() = default;

        // Generic Constructor
        template <typename... Args>
        anyHydro(Args... args) : vals{static_cast<real>(args)...}
        {
            // if chi not defined, set to zero
            if constexpr (sizeof...(args) == nmem - 1) {
                vals[nmem - 1] = 0.0;
            }
            else {
                static_assert(
                    sizeof...(args) == nmem,
                    "Number of arguments must match nmem"
                );
            }
        }

        // access operator for the values
        DUAL real& operator[](const luint i) { return vals[i]; }

        DUAL real operator[](const luint i) const { return vals[i]; }

        // Copy Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            for (luint i = 0; i < nmem; i++) {
                vals[i] = other.vals[i];
            }
            return *self();
        }

        // Move Constructor
        DUAL Derived& operator=(Derived&& other)
        {
            for (luint i = 0; i < nmem; i++) {
                vals[i] = other.vals[i];
            }
            return *self();
        }

        // + operator
        DUAL Derived operator+(const Derived& prim) const
        {
            Derived result;
            for (luint i = 0; i < nmem; i++) {
                result.vals[i] = vals[i] + prim.vals[i];
            }
            return result;
        }

        // - operator
        DUAL Derived operator-(const Derived& prim) const
        {
            Derived result;
            for (luint i = 0; i < nmem; i++) {
                result.vals[i] = vals[i] - prim.vals[i];
            }
            return result;
        }

        // Scalar division
        DUAL Derived operator/(const real c) const
        {
            Derived result;
            for (luint i = 0; i < nmem; i++) {
                result.vals[i] = vals[i] / c;
            }
            return result;
        }

        // Scalar multiplication
        DUAL Derived operator*(const real c) const
        {
            Derived result;
            for (luint i = 0; i < nmem; i++) {
                result.vals[i] = vals[i] * c;
            }
            return result;
        }

        DUAL Derived& operator-=(const Derived& prim)
        {
            for (luint i = 0; i < nmem; i++) {
                vals[i] -= prim.vals[i];
            }
            return *self();
        }

        DUAL Derived& operator+=(const Derived& prim)
        {
            for (luint i = 0; i < nmem; i++) {
                vals[i] += prim.vals[i];
            }
            return *self();
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };
}   // namespace generic_hydro

// Forward declare the structs
template <int dim, Regime R>
struct anyConserved;

template <int dim, Regime R>
struct anyPrimitive;

template <int dim>
struct mag_four_vec {
  private:
    real lorentz, vdb;

  public:
    real zero, one, two, three;

    mag_four_vec() = default;

    ~mag_four_vec() = default;

    DUAL mag_four_vec(const anyPrimitive<dim, Regime::RMHD>& prim)
        : lorentz(prim.lorentz_factor()),
          vdb(prim.vdotb()),
          zero(lorentz * vdb),
          one(prim.b1() / lorentz + lorentz * prim.get_v1() * vdb),
          two(prim.b2() / lorentz + lorentz * prim.get_v2() * vdb),
          three(prim.b3() / lorentz + lorentz * prim.get_v3() * vdb)
    {
    }

    DUAL mag_four_vec(const mag_four_vec& c)
        : lorentz(c.lorentz),
          vdb(c.vdb),
          zero(c.zero),
          one(c.one),
          two(c.two),
          three(c.three)
    {
    }

    DUAL real inner_product() const
    {
        return -zero * zero + one * one + two * two + three * three;
    }

    DUAL constexpr real normal(const luint nhat) const
    {
        return nhat == 1 ? one : nhat == 2 ? two : three;
    }
};

template <int dim, Regime R>
struct anyConserved : generic_hydro::anyHydro<dim, anyConserved<dim, R>, R> {
    using generic_hydro::anyHydro<dim, anyConserved<dim, R>, R>::anyHydro;

    // Define accessors for the conserved variables
    DUAL real& dens() { return this->vals[0]; }

    DUAL constexpr real dens() const { return this->vals[0]; }

    DUAL real& m1() { return this->vals[1]; }

    DUAL constexpr real m1() const { return this->vals[1]; }

    DUAL real& m2()
    {
        if constexpr (dim > 1) {
            return this->vals[2];
        }
        else {
            static_assert(dim > 1, "m2 is not defined for dim = 1");
        }
    }

    DUAL constexpr real m2() const
    {
        if constexpr (dim > 1) {
            return this->vals[2];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& m3()
    {
        if constexpr (dim > 2) {
            return this->vals[3];
        }
        else {
            static_assert(dim < 3, "m3 is not defined for dim < 3");
        }
    }

    DUAL constexpr real m3() const
    {
        if constexpr (dim > 2) {
            return this->vals[3];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& nrg() { return this->vals[dim + 1]; }

    DUAL constexpr real nrg() const { return this->vals[dim + 1]; }

    DUAL real& b1()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 2];
        }
        else {
            static_assert(R == Regime::RMHD, "b1 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b1() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 2];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& b2()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 3];
        }
        else {
            static_assert(R == Regime::RMHD, "b2 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b2() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 3];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& b3()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 4];
        }
        else {
            static_assert(R == Regime::RMHD, "b3 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b3() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 4];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& chi() { return this->vals[dim + 2 + 3 * (R == Regime::RMHD)]; }

    DUAL constexpr real chi() const
    {
        return this->vals[dim + 2 + 3 * (R == Regime::RMHD)];
    }

    DUAL constexpr real total_energy() const
    {
        if constexpr (R == Regime::RMHD || R == Regime::SRHD) {
            return nrg() + dens();
        }
        else {
            return nrg();
        }
    }

    DUAL real& e1() { return b1(); };

    DUAL constexpr real e1() const { return b1(); }

    DUAL real& e2() { return b2(); };

    DUAL constexpr real e2() const { return b2(); }

    DUAL real& e3() { return b3(); };

    DUAL constexpr real e3() const { return b3(); }

    //=========================================================================
    DUAL constexpr real bsquared() const
    {
        return b1() * b1() + b2() * b2() + b3() * b3();
    }

    DUAL constexpr void calc_electric_field(const luint nhat)
    {
        if constexpr (R == Regime::RMHD) {
            if (nhat == 1) {
                e1() = 0.0;
                global::swap(e2(), e3());
                e3() *= -1.0;
            }
            else if (nhat == 2) {
                e2() = 0.0;
                global::swap(e1(), e3());
                e1() *= -1.0;
            }
            else {
                e3() = 0.0;
                global::swap(e2(), e1());
                e2() *= -1.0;
            }
        }
    }

    DUAL real momentum(const luint nhat) const
    {
        if (nhat == 1) {
            return m1();
        }
        else if (nhat == 2) {
            return m2();
        }
        else {
            return m3();
        }
    }

    DUAL real& momentum(const luint nhat = 1)
    {
        if (nhat == 1) {
            return m1();
        }
        if constexpr (dim > 1) {
            if (nhat == 2) {
                return m2();
            }
            if constexpr (dim > 2) {
                if (nhat == 3) {
                    return m3();
                }
            }
        }
        return m1();
    }

    DUAL real bcomponent(const luint nhat) const
    {
        if (nhat == 1) {
            return b1();
        }
        else if (nhat == 2) {
            return b2();
        }
        else {
            return b3();
        }
    }

    DUAL real& bcomponent(const luint nhat = 1)
    {
        if (nhat == 1) {
            return b1();
        }
        if constexpr (dim > 1) {
            if (nhat == 2) {
                return b2();
            }
        }
        if constexpr (dim > 2) {
            if (nhat == 3) {
                return b3();
            }
        }
        // throw an error if the dimension is not correct
        return b1();
    }

    DUAL real ecomponent(const luint nhat) const
    {
        if (nhat == 1) {
            return e1();
        }
        else if (nhat == 2) {
            return e2();
        }
        else {
            return e3();
        }
    }

    // change the -= overload if on an mhd run
    // to skip the magnetic fields
    DUAL anyConserved& operator-=(const anyConserved& cons)
    {
        if constexpr (R == Regime::RMHD) {
            for (luint i = 0; i < this->nmem; i++) {
                if (i < dim + 2 || i > dim + 4) {
                    this->vals[i] -= cons.vals[i];
                }
            }
        }
        else {
            for (luint i = 0; i < this->nmem; i++) {
                this->vals[i] -= cons.vals[i];
            }
        }
        return *this;
    }
};

template <int dim, Regime R>
struct anyPrimitive : generic_hydro::anyHydro<dim, anyPrimitive<dim, R>, R> {
    using generic_hydro::anyHydro<dim, anyPrimitive<dim, R>, R>::anyHydro;

    // Define accessors for the primitive variables
    DUAL real& rho() { return this->vals[0]; }

    DUAL constexpr real rho() const { return this->vals[0]; }

    DUAL real& v1() { return this->vals[1]; }

    DUAL constexpr real v1() const { return this->vals[1]; }

    DUAL real& v2()
    {
        if constexpr (dim > 1) {
            return this->vals[2];
        }
        else {
            static_assert(dim > 1, "v2 is not defined for dim = 1");
        }
    }

    DUAL constexpr real v2() const
    {
        if constexpr (dim > 1) {
            return this->vals[2];
        }
    }

    DUAL real& v3()
    {
        if constexpr (dim > 2) {
            return this->vals[3];
        }
        else {
            static_assert(dim > 2, "v3 is not defined for dim < 3");
        }
    }

    DUAL constexpr real v3() const
    {
        if constexpr (dim > 2) {
            return this->vals[3];
        }
        else {
            static_assert(dim > 2, "v3 is not defined for dim < 3");
        }
    }

    DUAL real& p() { return this->vals[dim + 1]; }

    DUAL constexpr real p() const { return this->vals[dim + 1]; }

    // Define accessors for bfield if RMHD
    DUAL real& b1()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 2];
        }
        else {
            static_assert(R == Regime::RMHD, "b1 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b1() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 2];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& b2()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 3];
        }
        else {
            static_assert(R == Regime::RMHD, "b2 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b2() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 3];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& b3()
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 4];
        }
        else {
            static_assert(R == Regime::RMHD, "b3 is not defined for non-RMHD");
        }
    }

    DUAL constexpr real b3() const
    {
        if constexpr (R == Regime::RMHD) {
            return this->vals[dim + 4];
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL real& chi() { return this->vals[dim + 2 + 3 * (R == Regime::RMHD)]; }

    DUAL constexpr real chi() const
    {
        return this->vals[dim + 2 + 3 * (R == Regime::RMHD)];
    }

    DUAL constexpr real alfven() const { return p(); }

    DUAL real& alfven() { return p(); }

    //=========================================================================

    DUAL constexpr real vsquared() const
    {
        if constexpr (dim == 1) {
            return v1() * v1();
        }
        else if constexpr (dim == 2) {
            return v1() * v1() + v2() * v2();
        }
        else {
            return v1() * v1() + v2() * v2() + v3() * v3();
        }
    }

    DUAL constexpr real get_v1() const
    {
        if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v1();
            }
            else {
                return v1() / std::sqrt(1.0 + vsquared());
            }
        }
        else {
            return v1();
        }
    }

    DUAL constexpr real get_v2() const
    {
        if constexpr (dim > 1) {
            if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
                if constexpr (global::VelocityType == global::Velocity::Beta) {
                    return v2();
                }
                else {
                    return v2() / std::sqrt(1.0 + vsquared());
                }
            }
            else {
                return v2();
            }
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real get_v3() const
    {
        if constexpr (dim > 2) {
            if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
                if constexpr (global::VelocityType == global::Velocity::Beta) {
                    return v3();
                }
                else {
                    return v3() / std::sqrt(1.0 + vsquared());
                }
            }
            else {
                return v3();
            }
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real vcomponent(const luint nhat) const
    {
        if constexpr (dim == 1) {
            if (nhat > 1) {
                return static_cast<real>(0.0);
            }
            return v1();
        }
        else if constexpr (dim == 2) {
            if (nhat > 2) {
                return static_cast<real>(0.0);
            }
            return (nhat == 1 ? v1() : v2());
        }
        else {
            return (nhat == 1 ? v1() : (nhat == 2) ? v2() : v3());
        }
    }

    DUAL constexpr real& vcomponent(const luint nhat)
    {
        if constexpr (dim == 1) {
            if (nhat > 1) {
                return v1();
            }
            return v1();
        }
        else if constexpr (dim == 2) {
            if (nhat > 2) {
                return v1();
            }
            return (nhat == 1 ? v1() : v2());
        }
        else {
            return (nhat == 1 ? v1() : (nhat == 2) ? v2() : v3());
        }
    }

    DUAL constexpr real lorentz_factor() const
    {
        if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
            return 1.0 / std::sqrt(1.0 - vsquared());
        }
        else {
            return 1.0;
        }
    }

    DUAL constexpr real lorentz_factor_squared() const
    {
        if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
            return 1.0 / (1.0 - vsquared());
        }
        else {
            return 1.0;
        }
    }

    DUAL constexpr real total_energy(const real gamma) const
    {
        if constexpr (R == Regime::NEWTONIAN) {
            return p() / (gamma - 1.0) + 0.5 * rho() * vsquared();
        }
        else if constexpr (R == Regime::SRHD) {
            return rho() * lorentz_factor_squared() * enthalpy(gamma) - p();
        }
        else {
            return rho() * lorentz_factor_squared() * enthalpy(gamma) - p() +
                   0.5 * (bsquared() + bsquared() + vsquared() * bsquared() -
                          vdotb() * vdotb());
        }
    }

    DUAL constexpr real enthalpy(real gamma) const
    {
        if constexpr (R == Regime::SRHD || R == Regime::RMHD) {
            return 1.0 + gamma * p() / (rho() * (gamma - 1.0));
        }
        else {
            return 1.0;
        }
    }

    DUAL constexpr real total_pressure() const
    {
        if constexpr (R == Regime::RMHD) {
            return p() + 0.5 * (bsquared() / lorentz_factor_squared() +
                                vdotb() * vdotb());
        }
        else {
            return p();
        }
    }

    DUAL constexpr real bcomponent(const luint nhat) const
    {
        if constexpr (R == Regime::RMHD) {
            return (
                nhat == 1     ? this->vals[dim + 2]
                : (nhat == 2) ? this->vals[dim + 3]
                              : this->vals[dim + 4]
            );
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real& bcomponent(const luint nhat)
    {
        if constexpr (R == Regime::RMHD) {
            return (
                nhat == 1     ? this->vals[dim + 2]
                : (nhat == 2) ? this->vals[dim + 3]
                              : this->vals[dim + 4]
            );
        }
        else {
            return this->vals[dim + 3];
        }
    }

    DUAL real ecomponent(luint nhat) const
    {
        if constexpr (R == Regime::RMHD) {
            if (nhat == 1) {
                return vcomponent(3) * b2() - vcomponent(2) * b3();
            }
            else if (nhat == 2) {
                return vcomponent(1) * b3() - vcomponent(3) * b1();
            }
            return vcomponent(2) * b1() - vcomponent(1) * b2();
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real vdotb() const
    {
        if constexpr (R == Regime::RMHD) {
            return vcomponent(1) * bcomponent(1) +
                   vcomponent(2) * bcomponent(2) +
                   vcomponent(3) * bcomponent(3);
        }
        else {
            return static_cast<real>(0.0);
        }
    }

    DUAL constexpr real bsquared() const
    {
        return b1() * b1() + b2() * b2() + b3() * b3();
    }

    DUAL anyConserved<dim, R> to_conserved(real gamma) const
    {
        if constexpr (R == Regime::NEWTONIAN) {
            const real rho = this->rho();
            const real v1  = vcomponent(1);
            const real v2  = vcomponent(2);
            const real v3  = vcomponent(3);
            const real p   = this->p();
            const real et  = p / (gamma - 1) + 0.5 * rho * vsquared();
            if constexpr (dim == 1) {
                return {rho, rho * v1, et};
            }
            else if constexpr (dim == 2) {
                return {rho, rho * v1, rho * v2, et};
            }
            else {
                return {rho, rho * v1, rho * v2, rho * v3, et};
            }
        }
        else if constexpr (R == Regime::SRHD) {
            const real rho = this->rho();
            const real v1  = vcomponent(1);
            const real v2  = vcomponent(2);
            const real v3  = vcomponent(3);
            const real p   = this->p();
            const real lf  = lorentz_factor();
            const real h   = enthalpy(gamma);
            const real d   = rho * lf;
            const real ed  = d * lf * h;
            if constexpr (dim == 1) {
                return {d, ed * v1, ed - p - d};
            }
            else if constexpr (dim == 2) {
                return {d, ed * v1, ed * v2, ed - p - d};
            }
            else {
                return {d, ed * v1, ed * v2, ed * v3, ed - p - d};
            }
        }
        else {
            const real rho = this->rho();
            const real v1  = vcomponent(1);
            const real v2  = vcomponent(2);
            const real v3  = vcomponent(3);
            const real pg  = this->p();
            const real b1  = bcomponent(1);
            const real b2  = bcomponent(2);
            const real b3  = bcomponent(3);
            const real lf  = lorentz_factor();
            const real h   = enthalpy(gamma);
            const real vdb = vdotb();
            const real bsq = bsquared();
            const real vsq = vsquared();
            const real d   = rho * lf;
            const real ed  = d * h * lf;

            return {
              d,
              (ed + bsq) * v1 - vdb * b1,
              (ed + bsq) * v2 - vdb * b2,
              (ed + bsq) * v3 - vdb * b3,
              ed - pg - d +
                  static_cast<real>(0.5) * (bsq + vsq * bsq - vdb * vdb),
              b1,
              b2,
              b3,
              d * chi()
            };
        }
    }

    DUAL anyConserved<dim, R> to_flux(const real gamma, const luint nhat) const
    {
        if constexpr (R == Regime::NEWTONIAN) {
            const real rho = this->rho();
            const real v1  = vcomponent(1);
            const real v2  = vcomponent(2);
            const real v3  = vcomponent(3);
            const real p   = this->p();
            const real vn  = nhat == 1 ? v1 : nhat == 2 ? v2 : v3;
            const real et  = p / (gamma - 1) + 0.5 * rho * vsquared();
            const real m1  = rho * v1;
            if constexpr (dim == 1) {
                return {
                  m1,
                  m1 * vn + kdelta(nhat, 1) * p,
                  (et + p) * vn,
                  rho * vn * chi()
                };
            }
            else if constexpr (dim == 2) {
                const real m2 = rho * v2;
                return {
                  rho * vn,
                  m1 * vn + kdelta(nhat, 1) * p,
                  m2 * vn + kdelta(nhat, 2) * p,
                  (et + p) * vn,
                  rho * vn * chi()
                };
            }
            else {
                const real m2 = rho * v2;
                const real m3 = rho * v3;
                return {
                  rho * vn,
                  m1 * vn + kdelta(nhat, 1) * p,
                  m2 * vn + kdelta(nhat, 2) * p,
                  m3 * vn + kdelta(nhat, 3) * p,
                  (et + p) * vn,
                  rho * vn * chi()
                };
            }
        }
        else if constexpr (R == Regime::SRHD) {
            const real rho = this->rho();
            const real v1  = vcomponent(1);
            const real v2  = vcomponent(2);
            const real v3  = vcomponent(3);
            const real p   = this->p();
            const real vn  = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
            const real lf  = lorentz_factor();

            const real h  = enthalpy(gamma);
            const real d  = rho * lf;
            const real ed = d * lf * h;
            const real s1 = ed * v1;
            const real s2 = ed * v2;
            const real s3 = ed * v3;
            const real mn = (nhat == 1) ? s1 : (nhat == 2) ? s2 : s3;
            if constexpr (dim == 1) {
                return {
                  d * vn,
                  s1 * vn + kdelta(nhat, 1) * p,
                  mn - d * vn,
                  d * vn * chi()
                };
            }
            else if constexpr (dim == 2) {
                return {
                  d * vn,
                  s1 * vn + kdelta(nhat, 1) * p,
                  s2 * vn + kdelta(nhat, 2) * p,
                  mn - d * vn,
                  d * vn * chi()
                };
            }
            else {
                return {
                  d * vn,
                  s1 * vn + kdelta(nhat, 1) * p,
                  s2 * vn + kdelta(nhat, 2) * p,
                  s3 * vn + kdelta(nhat, 3) * p,
                  mn - d * vn,
                  d * vn * chi()
                };
            }
        }
        else {
            const real rho   = this->rho();
            const real v1    = vcomponent(1);
            const real v2    = vcomponent(2);
            const real v3    = vcomponent(3);
            const real b1    = bcomponent(1);
            const real b2    = bcomponent(2);
            const real b3    = bcomponent(3);
            const real h     = enthalpy(gamma);
            const real lf    = lorentz_factor();
            const real invlf = 1.0 / lf;
            const real vdb   = vdotb();
            const real bsq   = bsquared();
            const real ptot  = total_pressure();
            const real vn    = (nhat == 1) ? v1 : (nhat == 2) ? v2 : v3;
            const real bn    = (nhat == 1) ? b1 : (nhat == 2) ? b2 : b3;
            const real d     = rho * lf;
            const real ed    = d * h * lf;
            const real m1    = (ed + bsq) * v1 - vdb * b1;
            const real m2    = (ed + bsq) * v2 - vdb * b2;
            const real m3    = (ed + bsq) * v3 - vdb * b3;
            const real mn    = (nhat == 1) ? m1 : (nhat == 2) ? m2 : m3;
            const auto bmu   = mag_fourvec_t(this);
            const real ind1  = (nhat == 1) ? 0.0 : vn * b1 - v1 * bn;
            const real ind2  = (nhat == 2) ? 0.0 : vn * b2 - v2 * bn;
            const real ind3  = (nhat == 3) ? 0.0 : vn * b3 - v3 * bn;
            return {
              d * vn,
              m1 * vn + kdelta(nhat, 1) * ptot - bn * bmu.one * invlf,
              m2 * vn + kdelta(nhat, 2) * ptot - bn * bmu.two * invlf,
              m3 * vn + kdelta(nhat, 3) * ptot - bn * bmu.three * invlf,
              mn - d * vn,
              ind1,
              ind2,
              ind3,
              d * vn * chi()
            };
        }
    }
};

//=======================================================
//                        NEWTONIAN
//=======================================================
namespace hydro1d {
    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, p, chi;
    };

    struct Eigenvals {
        real aL, aR, aStar, pStar;

        Eigenvals() = default;

        ~Eigenvals() = default;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}

        DUAL Eigenvals(real aL, real aR, real aStar, real pStar)
            : aL(aL), aR(aR), aStar(aStar), pStar(pStar)
        {
        }
    };

}   // namespace hydro1d

namespace hydro2d {
    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, v2, p, chi;
    };

    struct Eigenvals {
        Eigenvals() = default;

        ~Eigenvals() = default;

        real aL, aR, csL, csR, aStar, pStar;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}

        DUAL
        Eigenvals(real aL, real aR, real csL, real csR, real aStar, real pStar)
            : aL(aL), aR(aR), csL(csL), csR(csR), aStar(aStar), pStar(pStar)
        {
        }
    };

}   // namespace hydro2d

namespace hydro3d {
    struct PrimitiveSOA {
        std::vector<real> rho, v1, v2, v3, p, chi;
        PrimitiveSOA()  = default;
        ~PrimitiveSOA() = default;
    };

    struct Eigenvals {
        real aL, aR, csL, csR, aStar, pStar;
        Eigenvals()  = default;
        ~Eigenvals() = default;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}

        DUAL
        Eigenvals(real aL, real aR, real csL, real csR, real aStar, real pStar)
            : aL(aL), aR(aR), csL(csL), csR(csR), aStar(aStar), pStar(pStar)
        {
        }
    };

}   // namespace hydro3d

//=============================================
//                SRHD
//=============================================

namespace sr1d {
    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, p, chi;
    };

    struct Eigenvals {
        real aL, aR, csL, csR;

        Eigenvals() = default;

        ~Eigenvals() = default;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR), csL(0.0), csR(0.0) {}

        DUAL Eigenvals(real aL, real aR, real csL, real csR)
            : aL(aL), aR(aR), csL(csL), csR(csR)
        {
        }
    };

}   // namespace sr1d

namespace sr2d {
    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, v2, p, chi;
    };

    struct Eigenvals {
        Eigenvals() = default;

        ~Eigenvals() = default;

        real aL, aR, csL, csR;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}

        DUAL Eigenvals(real aL, real aR, real csL, real csR)
            : aL(aL), aR(aR), csL(csL), csR(csR)
        {
        }
    };

}   // namespace sr2d

namespace sr3d {
    struct PrimitiveSOA {
        PrimitiveSOA()  = default;
        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, v2, v3, p, chi;
    };

    struct Eigenvals {
        real aL, aR, csL, csR;
        Eigenvals()  = default;
        ~Eigenvals() = default;

        DUAL Eigenvals(real aL, real aR) : aL(aL), aR(aR), csL(0.0), csR(0.0) {}

        DUAL Eigenvals(real aL, real aR, real csL, real csR)
            : aL(aL), aR(aR), csL(csL), csR(csR)
        {
        }
    };

}   // namespace sr3d

//================================
//               RMHD
//================================
namespace rmhd {
    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, v2, v3, p, b1, b2, b3, chi;
    };

    struct Eigenvals {
        real afL, afR;

        Eigenvals() = default;

        ~Eigenvals() = default;

        DUAL Eigenvals(real afL, real afR) : afL(afL), afR(afR) {}

        // DUAL Eigenvals(real afL, real afR,
        // real asL, real asR, real csL, real csR) : afL(afL),
        // afR(afR), asL(asL), asR(asR), csL(csL), csR(csR) {}
    };
}   // namespace rmhd

#endif