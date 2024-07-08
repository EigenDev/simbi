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
#include <cmath>
#include <vector>

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
    template <int dim, typename Derived>
    struct Primitive {
    };

    template <typename Derived>
    struct Primitive<1, Derived> {
        real rho, v1, p, chi;

        // Default Destructor
        ~Primitive() = default;

        // Default Constructor
        Primitive() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            rho = other.rho;
            v1  = other.v1;
            p   = other.p;
            chi = other.chi;
            return *self();
        }

        DUAL Primitive(real rho, real v1, real p)
            : rho(rho), v1(v1), p(p), chi(0.0)
        {
        }

        DUAL Primitive(real rho, real v1, real p, real chi)
            : rho(rho), v1(v1), p(p), chi(chi)
        {
        }

        DUAL Primitive(const Primitive& prim)
            : rho(prim.rho), v1(prim.v1), p(prim.p), chi(prim.chi)
        {
        }

        DUAL Derived operator+(const Derived& prim) const
        {
            return Derived(
                rho + prim.rho,
                v1 + prim.v1,
                p + prim.p,
                chi + prim.chi
            );
        }

        DUAL Derived operator-(const Derived& prim) const
        {
            return Derived(
                rho - prim.rho,
                v1 - prim.v1,
                p - prim.p,
                chi - prim.chi
            );
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(rho / c, v1 / c, p / c, chi / c);
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(rho * c, v1 * c, p * c, chi * c);
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

    template <typename Derived>
    struct Primitive<2, Derived> {
        real rho, v1, v2, p, chi;

        // Default Constructor
        Primitive() = default;

        // Default Destructor
        ~Primitive() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            rho = other.rho;
            v1  = other.v1;
            v2  = other.v2;
            p   = other.p;
            chi = other.chi;
            return *self();
        }

        DUAL Primitive(real rho, real v1, real v2, real p)
            : rho(rho), v1(v1), v2(v2), p(p), chi(0.0)
        {
        }

        DUAL Primitive(real rho, real v1, real v2, real p, real chi)
            : rho(rho), v1(v1), v2(v2), p(p), chi(chi)
        {
        }

        DUAL Primitive(const Primitive& prims)
            : rho(prims.rho),
              v1(prims.v1),
              v2(prims.v2),
              p(prims.p),
              chi(prims.chi)
        {
        }

        DUAL Derived operator+(const Derived& prims) const
        {
            return Derived(
                rho + prims.rho,
                v1 + prims.v1,
                v2 + prims.v2,
                p + prims.p,
                chi + prims.chi
            );
        }

        DUAL Derived operator-(const Derived& prims) const
        {
            return Derived(
                rho - prims.rho,
                v1 - prims.v1,
                v2 - prims.v2,
                p - prims.p,
                chi - prims.chi
            );
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(rho * c, v1 * c, v2 * c, p * c, chi * c);
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(rho / c, v1 / c, v2 / c, p / c, chi / c);
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

    template <typename Derived>
    struct Primitive<3, Derived> {
        real rho, v1, v2, v3, p, chi;

        // Default Constructor
        Primitive() = default;

        // Default Destructor
        ~Primitive() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            rho = other.rho;
            v1  = other.v1;
            v2  = other.v2;
            v3  = other.v3;
            p   = other.p;
            chi = other.chi;
            return *self();
        }

        DUAL Primitive(real rho, real v1, real v2, real v3, real p)
            : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(0.0)
        {
        }

        DUAL Primitive(real rho, real v1, real v2, real v3, real p, real chi)
            : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(chi)
        {
        }

        DUAL Primitive(const Primitive& prims)
            : rho(prims.rho),
              v1(prims.v1),
              v2(prims.v2),
              v3(prims.v3),
              p(prims.p),
              chi(prims.chi)
        {
        }

        DUAL Derived operator+(const Derived& prims) const
        {
            return Derived(
                rho + prims.rho,
                v1 + prims.v1,
                v2 + prims.v2,
                v3 + prims.v3,
                p + prims.p,
                chi + prims.chi
            );
        }

        DUAL Derived operator-(const Derived& prims) const
        {
            return Derived(
                rho - prims.rho,
                v1 - prims.v1,
                v2 - prims.v2,
                v3 - prims.v3,
                p - prims.p,
                chi - prims.chi
            );
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(rho * c, v1 * c, v2 * c, v3 * c, p * c, chi * c);
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(rho / c, v1 / c, v2 / c, v3 / c, p / c, chi / c);
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

    template <int dim, typename Derived>
    struct Conserved {
    };

    template <typename Derived>
    struct Conserved<1, Derived> {
        real den, m1, nrg, chi;

        // Default Destructor
        ~Conserved() = default;

        // Default Constructor
        Conserved() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            den = other.den;
            m1  = other.m1;
            nrg = other.nrg;
            chi = other.chi;
            return *self();
        }

        DUAL Conserved(real den, real m1, real nrg)
            : den(den), m1(m1), nrg(nrg), chi(0.0)
        {
        }

        DUAL Conserved(real den, real m1, real nrg, real chi)
            : den(den), m1(m1), nrg(nrg), chi(chi)
        {
        }

        DUAL Conserved(const Conserved& prim)
            : den(prim.den), m1(prim.m1), nrg(prim.nrg), chi(prim.chi)
        {
        }

        DUAL Derived operator+(const Derived& prim) const
        {
            return Derived(
                den + prim.den,
                m1 + prim.m1,
                nrg + prim.nrg,
                chi + prim.chi
            );
        }

        DUAL Derived operator-(const Derived& prim) const
        {
            return Derived(
                den - prim.den,
                m1 - prim.m1,
                nrg - prim.nrg,
                chi - prim.chi
            );
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(den / c, m1 / c, nrg / c, chi / c);
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(den * c, m1 * c, nrg * c, chi * c);
        }

        DUAL Derived& operator-=(const Derived& cons)
        {
            den -= cons.den;
            m1 -= cons.m1;
            nrg -= cons.nrg;
            chi -= cons.chi;
            return *self();
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

    template <typename Derived>
    struct Conserved<2, Derived> {
        real den, m1, m2, nrg, chi;

        // Default Constructor
        Conserved() = default;

        // Default Destructor
        ~Conserved() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            den = other.den;
            m1  = other.m1;
            m2  = other.m2;
            nrg = other.nrg;
            chi = other.chi;
            return *self();
        }

        DUAL Conserved(real den, real m1, real m2, real nrg)
            : den(den), m1(m1), m2(m2), nrg(nrg), chi(0.0)
        {
        }

        DUAL Conserved(real den, real m1, real m2, real nrg, real chi)
            : den(den), m1(m1), m2(m2), nrg(nrg), chi(chi)
        {
        }

        DUAL Conserved(const Conserved& prims)
            : den(prims.den),
              m1(prims.m1),
              m2(prims.m2),
              nrg(prims.nrg),
              chi(prims.chi)
        {
        }

        DUAL Derived operator+(const Derived& cons) const
        {
            return Derived(
                den + cons.den,
                m1 + cons.m1,
                m2 + cons.m2,
                nrg + cons.nrg,
                chi + cons.chi
            );
        }

        DUAL Derived operator-(const Derived& cons) const
        {
            return Derived(
                den - cons.den,
                m1 - cons.m1,
                m2 - cons.m2,
                nrg - cons.nrg,
                chi - cons.chi
            );
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(den * c, m1 * c, m2 * c, nrg * c, chi * c);
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(den / c, m1 / c, m2 / c, nrg / c, chi / c);
        }

        DUAL Derived& operator-=(const Derived& cons)
        {
            den -= cons.den;
            m1 -= cons.m1;
            m2 -= cons.m2;
            nrg -= cons.nrg;
            chi -= cons.chi;
            return *self();
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

    template <typename Derived>
    struct Conserved<3, Derived> {
        real den, m1, m2, m3, nrg, chi;

        // Default Constructor
        Conserved() = default;

        // Default Destructor
        ~Conserved() = default;

        // Copy-Assignment Constructor
        DUAL Derived& operator=(const Derived& other)
        {
            den = other.den;
            m1  = other.m1;
            m2  = other.m2;
            m3  = other.m3;
            nrg = other.nrg;
            chi = other.chi;
            return *self();
        }

        DUAL Conserved(real den, real m1, real m2, real m3, real nrg)
            : den(den), m1(m1), m2(m2), m3(m3), nrg(nrg), chi(0.0)
        {
        }

        DUAL Conserved(real den, real m1, real m2, real m3, real nrg, real chi)
            : den(den), m1(m1), m2(m2), m3(m3), nrg(nrg), chi(chi)
        {
        }

        DUAL Conserved(const Conserved& prims)
            : den(prims.den),
              m1(prims.m1),
              m2(prims.m2),
              m3(prims.m3),
              nrg(prims.nrg),
              chi(prims.chi)
        {
        }

        DUAL Derived operator+(const Derived& prims) const
        {
            return Derived(
                den + prims.den,
                m1 + prims.m1,
                m2 + prims.m2,
                m3 + prims.m3,
                nrg + prims.nrg,
                chi + prims.chi
            );
        }

        DUAL Derived operator-(const Derived& prims) const
        {
            return Derived(
                den - prims.den,
                m1 - prims.m1,
                m2 - prims.m2,
                m3 - prims.m3,
                nrg - prims.nrg,
                chi - prims.chi
            );
        }

        DUAL Derived operator*(const real c) const
        {
            return Derived(den * c, m1 * c, m2 * c, m3 * c, nrg * c, chi * c);
        }

        DUAL Derived operator/(const real c) const
        {
            return Derived(den / c, m1 / c, m2 / c, m3 / c, nrg / c, chi / c);
        }

        DUAL Derived& operator-=(const Derived& cons)
        {
            den -= cons.den;
            m1 -= cons.m1;
            m2 -= cons.m2;
            m3 -= cons.m3;
            nrg -= cons.nrg;
            chi -= cons.chi;
            return *self();
        }

      private:
        DUAL Derived* self() { return static_cast<Derived*>(this); }
    };

}   // namespace generic_hydro

//=======================================================
//                        NEWTONIAN
//=======================================================
namespace hydro1d {
    struct Primitive : generic_hydro::Primitive<1, Primitive> {
        using generic_hydro::Primitive<1, Primitive>::Primitive;

        DUAL constexpr real get_v() const { return v1; }

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            if (nhat > 1) {
                return 0;
            }
            return v1;
        }

        DUAL real get_energy_density(real gamma) const
        {
            return p / (gamma - 1.0) + 0.5 * (rho * v1 * v1);
        }
    };

    struct Conserved : generic_hydro::Conserved<1, Conserved> {
        using generic_hydro::Conserved<1, Conserved>::Conserved;

        DUAL constexpr real& momentum() { return m1; }

        DUAL constexpr real momentum(const luint nhat) const
        {
            if (nhat == 1) {
                return m1;
            }
            return 0;
        }
    };

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
    struct Conserved : generic_hydro::Conserved<2, Conserved> {
        using generic_hydro::Conserved<2, Conserved>::Conserved;

        DUAL constexpr real momentum(const luint nhat) const
        {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? m1 : m2);
        }

        DUAL constexpr real& momentum(const luint nhat)
        {
            return (nhat == 1 ? m1 : m2);
        }
    };

    struct Primitive : generic_hydro::Primitive<2, Primitive> {
        using generic_hydro::Primitive<2, Primitive>::Primitive;

        DUAL constexpr real get_v1() const { return v1; }

        DUAL constexpr real get_v2() const { return v2; }

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? v1 : v2);
        }

        DUAL real get_energy_density(real gamma) const
        {
            return p / (gamma - 1) + 0.5 * (rho * (v1 * v1 + v2 * v2));
        }
    };

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
    struct Conserved : generic_hydro::Conserved<3, Conserved> {
        using generic_hydro::Conserved<3, Conserved>::Conserved;

        DUAL constexpr real momentum(const luint nhat) const
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }

        DUAL constexpr real& momentum(const luint nhat)
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }
    };

    struct Primitive : generic_hydro::Primitive<3, Primitive> {
        using generic_hydro::Primitive<3, Primitive>::Primitive;

        DUAL constexpr real get_v1() const { return v1; }

        DUAL constexpr real get_v2() const { return v2; }

        DUAL constexpr real get_v3() const { return v3; }

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            return (nhat == 1 ? v1 : (nhat == 2) ? v2 : v3);
        }

        DUAL real get_energy_density(real gamma) const
        {
            return p / (gamma - 1) +
                   0.5 * (rho * (v1 * v1 + v2 * v2 + v3 * v3));
        }
    };

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
    struct Primitive : generic_hydro::Primitive<1, Primitive> {
        using generic_hydro::Primitive<1, Primitive>::Primitive;

        DUAL constexpr real get_v() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v1;
            }
            else {
                return v1 / std::sqrt(1.0 + v1 * v1);
            }
        }

        DUAL constexpr real lorentz_factor() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / std::sqrt(1.0 - v1 * v1);
            }
            else {
                return std::sqrt(1.0 + v1 * v1);
            }
        }

        DUAL constexpr real lorentz_factor_squared() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / (1.0 - v1 * v1);
            }
            else {
                return (1.0 + v1 * v1);
            }
        }

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            if (nhat == 1) {
                return get_v();
            }
            return 0.0;
        }

        DUAL real get_enthalpy(real gamma) const
        {
            return 1.0 + gamma * p / (rho * (gamma - 1.0));
        }
    };

    struct Conserved : generic_hydro::Conserved<1, Conserved> {
        using generic_hydro::Conserved<1, Conserved>::Conserved;

        DUAL constexpr real& momentum() { return m1; }

        DUAL constexpr real momentum(const luint nhat) const
        {
            if (nhat == 1) {
                return m1;
            }
            return 0.0;
        }
    };

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
    struct Conserved : generic_hydro::Conserved<2, Conserved> {
        using generic_hydro::Conserved<2, Conserved>::Conserved;

        DUAL constexpr real momentum(const luint nhat) const
        {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? m1 : m2);
        }

        DUAL constexpr real& momentum(const luint nhat)
        {
            return (nhat == 1 ? m1 : m2);
        }
    };

    struct Primitive : generic_hydro::Primitive<2, Primitive> {
        using generic_hydro::Primitive<2, Primitive>::Primitive;

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            if (nhat > 2) {
                return 0.0;
            }
            return (nhat == 1 ? get_v1() : get_v2());
        }

        DUAL real lorentz_factor() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2));
            }
            else {
                return std::sqrt(1.0 + (v1 * v1 + v2 * v2));
            }
        }

        DUAL real lorentz_factor_squared() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / (1.0 - (v1 * v1 + v2 * v2));
            }
            else {
                return (1.0 + (v1 * v1 + v2 * v2));
            }
        }

        DUAL constexpr real get_v1() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v1;
            }
            else {
                return v1 / std::sqrt(1.0 + v1 * v1 + v2 * v2);
            }
        }

        DUAL constexpr real get_v2() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v2;
            }
            else {
                return v2 / std::sqrt(1.0 + v1 * v1 + v2 * v2);
            }
        }

        DUAL real get_enthalpy(real gamma) const
        {
            return 1.0 + gamma * p / (rho * (gamma - 1.0));
        }
    };

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
    struct Conserved : generic_hydro::Conserved<3, Conserved> {
        using generic_hydro::Conserved<3, Conserved>::Conserved;

        DUAL constexpr real momentum(const luint nhat) const
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }

        DUAL constexpr real& momentum(const luint nhat)
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }
    };

    struct Primitive : generic_hydro::Primitive<3, Primitive> {
        using generic_hydro::Primitive<3, Primitive>::Primitive;

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            return nhat == 1 ? get_v1() : (nhat == 2) ? get_v2() : get_v3();
        }

        DUAL real lorentz_factor() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
            }
            else {
                return std::sqrt(1.0 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        DUAL real lorentz_factor_squared() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / (1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
            }
            else {
                return (1.0 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        DUAL constexpr real get_v1() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v1;
            }
            else {
                return v1 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL constexpr real get_v2() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v2;
            }
            else {
                return v2 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL constexpr real get_v3() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v3;
            }
            else {
                return v3 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL real get_enthalpy(real gamma) const
        {
            return 1.0 + gamma * p / (rho * (gamma - 1.0));
        }
    };

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
    template <int dim>
    struct AnyConserved {
        real den, m1, m2, m3, nrg, b1, b2, b3, chi;

        AnyConserved() = default;

        ~AnyConserved() = default;

        DUAL AnyConserved(real den, real m1, real nrg, real b1)
            : den(den),
              m1(m1),
              m2(0.0),
              m3(0.0),
              nrg(nrg),
              b1(b1),
              b2(0.0),
              b3(0.0),
              chi(0.0)
        {
        }

        DUAL AnyConserved(real den, real m1, real nrg, real b1, real chi)
            : den(den),
              m1(m1),
              m2(0.0),
              m3(0.0),
              nrg(nrg),
              b1(b1),
              b2(0.0),
              b3(0.0),
              chi(chi)
        {
        }

        DUAL
        AnyConserved(real den, real m1, real m2, real nrg, real b1, real b2)
            : den(den),
              m1(m1),
              m2(m2),
              m3(0.0),
              nrg(nrg),
              b1(b1),
              b2(b2),
              b3(0.0),
              chi(0.0)
        {
        }

        DUAL AnyConserved(
            real den,
            real m1,
            real m2,
            real nrg,
            real b1,
            real b2,
            real chi
        )
            : den(den),
              m1(m1),
              m2(m2),
              m3(0.0),
              nrg(nrg),
              b1(b1),
              b2(b2),
              b3(0.0),
              chi(chi)
        {
        }

        DUAL AnyConserved(
            real den,
            real m1,
            real m2,
            real m3,
            real nrg,
            real b1,
            real b2,
            real b3
        )
            : den(den),
              m1(m1),
              m2(m2),
              m3(m3),
              nrg(nrg),
              b1(b1),
              b2(b2),
              b3(b3),
              chi(0.0)
        {
        }

        DUAL AnyConserved(
            real den,
            real m1,
            real m2,
            real m3,
            real nrg,
            real b1,
            real b2,
            real b3,
            real chi
        )
            : den(den),
              m1(m1),
              m2(m2),
              m3(m3),
              nrg(nrg),
              b1(b1),
              b2(b2),
              b3(b3),
              chi(chi)
        {
        }

        DUAL AnyConserved(const AnyConserved& u)
            : den(u.den),
              m1(u.m1),
              m2(u.m2),
              m3(u.m3),
              nrg(u.nrg),
              b1(u.b1),
              b2(u.b2),
              b3(u.b3),
              chi(u.chi)
        {
        }

        DUAL AnyConserved operator+(const AnyConserved& p) const
        {
            return AnyConserved(
                den + p.den,
                m1 + p.m1,
                m2 + p.m2,
                m3 + p.m3,
                nrg + p.nrg,
                b1 + p.b1,
                b2 + p.b2,
                b3 + p.b3,
                chi + p.chi
            );
        }

        DUAL AnyConserved operator-(const AnyConserved& p) const
        {
            return AnyConserved(
                den - p.den,
                m1 - p.m1,
                m2 - p.m2,
                m3 - p.m3,
                nrg - p.nrg,
                b1 - p.b1,
                b2 - p.b2,
                b3 - p.b3,
                chi - p.chi
            );
        }

        DUAL AnyConserved operator*(const real c) const
        {
            return AnyConserved(
                den * c,
                m1 * c,
                m2 * c,
                m3 * c,
                nrg * c,
                b1 * c,
                b2 * c,
                b3 * c,
                chi * c
            );
        }

        DUAL AnyConserved operator/(const real c) const
        {
            return AnyConserved(
                den / c,
                m1 / c,
                m2 / c,
                m3 / c,
                nrg / c,
                b1 / c,
                b2 / c,
                b3 / c,
                chi / c
            );
        }

        DUAL AnyConserved& operator+=(const AnyConserved& cons)
        {
            den += cons.den;
            m1 += cons.m1;
            m2 += cons.m2;
            m3 += cons.m3;
            nrg += cons.nrg;
            b1 += cons.b1;
            b2 += cons.b2;
            b3 += cons.b3;
            chi += cons.chi;
            return *this;
        }

        DUAL AnyConserved& operator-=(const AnyConserved& cons)
        {
            den -= cons.den;
            m1 -= cons.m1;
            m2 -= cons.m2;
            m3 -= cons.m3;
            nrg -= cons.nrg;
            // b1 -= cons.b1;
            // b2 -= cons.b2;
            // b3 -= cons.b3;
            chi -= cons.chi;
            return *this;
        }

        DUAL AnyConserved& operator*=(const real c)
        {
            den *= c;
            m1 *= c;
            m2 *= c;
            m3 *= c;
            nrg *= c;
            b1 *= c;
            b2 *= c;
            b3 *= c;
            chi *= c;
            return *this;
        }

        DUAL real total_energy() const { return den + nrg; }

        DUAL constexpr real momentum(const luint nhat) const
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }

        DUAL constexpr real& momentum(const luint nhat)
        {
            return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3);
        }

        DUAL constexpr real& momentum() { return m1; }

        DUAL constexpr real bcomponent(const luint nhat) const
        {
            return (nhat == 1 ? b1 : (nhat == 2) ? b2 : b3);
        }

        DUAL constexpr real& bcomponent(const luint nhat)
        {
            return (nhat == 1 ? b1 : (nhat == 2) ? b2 : b3);
        }

        //-------- E-field accessors ---------
        // constexpr real e1() { return b1; }
        DUAL constexpr real& e1() { return b1; }

        // constexpr real e2() { return b2; }
        DUAL constexpr real& e2() { return b2; }

        // constexpr real e3() { return b3; }
        DUAL constexpr real& e3() { return b3; }

        DUAL constexpr real ecomponent(luint nhat) const
        {
            if (nhat == 1) {
                return b1;
            }
            else if (nhat == 2) {
                return b2;
            }
            else {
                return b3;
            }
        }

        DUAL void calc_electric_field(const luint nhat)
        {
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
    };

    template <int dim>
    struct AnyPrimitive {
        real rho, v1, v2, v3, p, b1, b2, b3, chi;

        AnyPrimitive() = default;

        ~AnyPrimitive() = default;

        DUAL AnyPrimitive& operator=(const AnyPrimitive& other
        )   // III. copy assignment
        {
            if (this == &other) {
                return *this;
            }

            rho = other.rho;
            v1  = other.v1;
            v2  = other.v2;
            v3  = other.v3;
            p   = other.p;
            b1  = other.b1;
            b2  = other.b2;
            b3  = other.b3;
            chi = other.chi;

            return *this;
        }

        DUAL AnyPrimitive(real rho, real v1, real p, real b1)
            : rho(rho),
              v1(v1),
              v2(0.0),
              v3(0.0),
              p(p),
              b1(b1),
              b2(0.0),
              b3(0.0),
              chi(0.0)
        {
        }

        DUAL AnyPrimitive(real rho, real v1, real p, real b1, real chi)
            : rho(rho),
              v1(v1),
              v2(0.0),
              v3(0.0),
              p(p),
              b1(b1),
              b2(0.0),
              b3(0.0),
              chi(chi)
        {
        }

        DUAL AnyPrimitive(real rho, real v1, real v2, real p, real b1, real b2)
            : rho(rho),
              v1(v1),
              v2(v2),
              v3(0.0),
              p(p),
              b1(b1),
              b2(b2),
              b3(0.0),
              chi(0.0)
        {
        }

        DUAL AnyPrimitive(
            real rho,
            real v1,
            real v2,
            real p,
            real b1,
            real b2,
            real chi
        )
            : rho(rho),
              v1(v1),
              v2(v2),
              v3(0.0),
              p(p),
              b1(b1),
              b2(b2),
              b3(0.0),
              chi(chi)
        {
        }

        DUAL AnyPrimitive(
            real rho,
            real v1,
            real v2,
            real v3,
            real p,
            real b1,
            real b2,
            real b3
        )
            : rho(rho),
              v1(v1),
              v2(v2),
              v3(v3),
              p(p),
              b1(b1),
              b2(b2),
              b3(b3),
              chi(0.0)
        {
        }

        DUAL AnyPrimitive(
            real rho,
            real v1,
            real v2,
            real v3,
            real p,
            real b1,
            real b2,
            real b3,
            real chi
        )
            : rho(rho),
              v1(v1),
              v2(v2),
              v3(v3),
              p(p),
              b1(b1),
              b2(b2),
              b3(b3),
              chi(chi)
        {
        }

        DUAL AnyPrimitive(const AnyPrimitive& c)
            : rho(c.rho),
              v1(c.v1),
              v2(c.v2),
              v3(c.v3),
              p(c.p),
              b1(c.b1),
              b2(c.b2),
              b3(c.b3),
              chi(c.chi)
        {
        }

        DUAL AnyPrimitive operator+(const AnyPrimitive& e) const
        {
            return AnyPrimitive(
                rho + e.rho,
                v1 + e.v1,
                v2 + e.v2,
                v3 + e.v3,
                p + e.p,
                b1 + e.b1,
                b2 + e.b2,
                b3 + e.b3,
                chi + e.chi
            );
        }

        DUAL AnyPrimitive operator-(const AnyPrimitive& e) const
        {
            return AnyPrimitive(
                rho - e.rho,
                v1 - e.v1,
                v2 - e.v2,
                v3 - e.v3,
                p - e.p,
                b1 - e.b1,
                b2 - e.b2,
                b3 - e.b3,
                chi - e.chi
            );
        }

        DUAL AnyPrimitive operator*(const real c) const
        {
            return AnyPrimitive(
                rho * c,
                v1 * c,
                v2 * c,
                v3 * c,
                p * c,
                b1 * c,
                b2 * c,
                b3 * c,
                chi * c
            );
        }

        DUAL AnyPrimitive operator/(const real c) const
        {
            return AnyPrimitive(
                rho / c,
                v1 / c,
                v2 / c,
                v3 / c,
                p / c,
                b1 / c,
                b2 / c,
                b3 / c,
                chi / c
            );
        }

        DUAL AnyPrimitive& operator+=(const AnyPrimitive& prims)
        {
            rho += prims.rho;
            v1 += prims.v1;
            v2 += prims.v2;
            v3 += prims.v3;
            p += prims.p;
            b1 += prims.b1;
            b2 += prims.b2;
            b3 += prims.b3;
            chi += prims.chi;
            return *this;
        }

        DUAL AnyPrimitive& operator-=(const AnyPrimitive& prims)
        {
            rho -= prims.rho;
            v1 -= prims.v1;
            v2 -= prims.v2;
            v3 -= prims.v3;
            p -= prims.p;
            b1 -= prims.b1;
            b2 -= prims.b2;
            b3 -= prims.b3;
            chi -= prims.chi;
            return *this;
        }

        DUAL AnyPrimitive& operator*=(const real c)
        {
            rho *= c;
            v1 *= c;
            v2 *= c;
            v3 *= c;
            p *= c;
            b1 *= c;
            b2 *= c;
            b3 *= c;
            chi *= c;
            return *this;
        }

        DUAL constexpr real vcomponent(const luint nhat) const
        {
            return nhat == 1 ? get_v1() : (nhat == 2) ? get_v2() : get_v3();
        }

        DUAL constexpr real& vcomponent(const luint nhat)
        {
            return nhat == 1 ? v1 : (nhat == 2) ? v2 : v3;
        }

        DUAL constexpr real bcomponent(const luint nhat) const
        {
            return nhat == 1 ? b1 : (nhat == 2) ? b2 : b3;
        }

        DUAL constexpr real& bcomponent(const luint nhat)
        {
            return nhat == 1 ? b1 : (nhat == 2) ? b2 : b3;
        }

        DUAL constexpr real get_v1() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v1;
            }
            else {
                return v1 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL constexpr real get_v2() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v2;
            }
            else {
                return v2 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL constexpr real get_v3() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return v3;
            }
            else {
                return v3 / std::sqrt(1.0 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        DUAL constexpr real lorentz_factor() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / std::sqrt(1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
            }
            else {
                return std::sqrt(1.0 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        DUAL constexpr real lorentz_factor_squared() const
        {
            if constexpr (global::VelocityType == global::Velocity::Beta) {
                return 1.0 / (1.0 - (v1 * v1 + v2 * v2 + v3 * v3));
            }
            else {
                return (1.0 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        DUAL real gas_enthalpy(real gamma) const
        {
            return 1.0 + gamma * p / (rho * (gamma - 1.0));
        }

        DUAL real vdotb() const { return (v1 * b1 + v2 * b2 + v3 * b3); }

        DUAL real bsquared() const { return (b1 * b1 + b2 * b2 + b3 * b3); }

        DUAL real total_pressure() const
        {
            return p + 0.5 * (bsquared() / lorentz_factor_squared() +
                              vdotb() * vdotb());
        }

        DUAL real total_enthalpy(const real gamma) const
        {
            return gas_enthalpy(gamma) + bsquared() / lorentz_factor_squared() +
                   vdotb() * vdotb();
        }

        DUAL real vsquared() const { return v1 * v1 + v2 * v2 + v3 * v3; }

        DUAL real ecomponent(luint nhat) const
        {
            if (nhat == 1) {
                return v3 * b2 - v2 * b3;
            }
            else if (nhat == 2) {
                return v1 * b3 - v3 * b1;
            }
            return v2 * b1 - v1 * b2;
        }
    };

    template <int dim>
    struct mag_four_vec {
      private:
        real lorentz, vdb;

      public:
        real zero, one, two, three;

        mag_four_vec() = default;

        ~mag_four_vec() = default;

        DUAL mag_four_vec(const AnyPrimitive<dim>& prim)
            : lorentz(prim.lorentz_factor()),
              vdb(prim.vdotb()),
              zero(lorentz * vdb),
              one(prim.b1 / lorentz + lorentz * prim.get_v1() * vdb),
              two(prim.b2 / lorentz + lorentz * prim.get_v2() * vdb),
              three(prim.b3 / lorentz + lorentz * prim.get_v3() * vdb)
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

    struct PrimitiveSOA {
        PrimitiveSOA() = default;

        ~PrimitiveSOA() = default;

        std::vector<real> rho, v1, v2, v3, p, b1, b2, b3, chi;
    };

    struct Eigenvals {
        real afL, afR, csL, csR;

        Eigenvals() = default;

        ~Eigenvals() = default;

        DUAL Eigenvals(real afL, real afR) : afL(afL), afR(afR) {}

        DUAL Eigenvals(real afL, real afR, real csL, real csR)
            : afL(afL), afR(afR), csL(csL), csR(csR)
        {
        }

        // DUAL Eigenvals(real afL, real afR,
        // real asL, real asR, real csL, real csR) : afL(afL),
        // afR(afR), asL(asL), asR(asR), csL(csL), csR(csR) {}
    };
}   // namespace rmhd

#endif