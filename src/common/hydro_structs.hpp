/*
* Houses the different 1D struct members Conserved, Primitives, Eigenvals
* for ease of access and organization. All definitions within header files
* to ensure inlining by the compiler.
*/

#ifndef HYDRO_STRUCTS_HPP
#define HYDRO_STRUCTS_HPP 

#include <cmath>
#include <vector>
#include "enums.hpp"
#include "build_options.hpp"
//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------
struct PrimData
{
    std::vector<real> rho, v1, v2, v3, p, b1, b2, b3, chi;
};

struct DataWriteMembers
{
    int nx, ny, nz, xactive_zones, yactive_zones, zactive_zones, chkpt_idx, dimensions;
    bool linspace, first_order, using_fourvelocity, mesh_motion;
    real t, ad_gamma;
    real x1min, x1max, x2min, x2max, x3min, x3max, dt;
    std::string coord_system, regime, x1_cell_spacing, x2_cell_spacing, x3_cell_spacing;
    std::vector<real> x1, x2, x3;
    std::vector<std::string> boundary_conditions;

    DataWriteMembers():
    nx(1),
    ny(1),
    nz(1),
    x1min(0.0),
    x1max(0.0),
    x2min(0.0),
    x2max(0.0),
    x3min(0.0),
    x3max(0.0)
    {}
};

struct InitialConditions {
    real tstart, chkpt_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend;
    luint nx, ny, nz, chkpt_idx;
    bool first_order, quirk_smoothing, constant_sources;
    std::vector<std::vector<real>> sources, gsources, bsources;
    std::vector<bool> object_cells; 
    std::string data_directory, coord_system, solver, x1_cell_spacing, x2_cell_spacing, x3_cell_spacing, regime;
    std::vector<std::string> boundary_conditions;
    std::vector<std::vector<real>> boundary_sources;
    std::vector<real> x1, x2, x3;
};


//=======================================================
//                        NEWTONIAN
//=======================================================      
namespace hydro1d {
    struct Primitive {
        real rho, v1, p, chi;
        GPU_CALLABLE_MEMBER Primitive () {}
        GPU_CALLABLE_MEMBER ~Primitive () {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real p) : rho(rho), v1(v1), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real p, real chi) : rho(rho), v1(v1), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &prim) : rho(prim.rho), v1(prim.v1), p(prim.p), chi(prim.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v1 + prim.v1, p + prim.p, chi + prim.chi); }
        GPU_CALLABLE_MEMBER Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v1 - prim.v1, p - prim.p, chi - prim.chi); }
        GPU_CALLABLE_MEMBER Primitive operator /(const real c) const {return Primitive(rho/c, v1/c, p/c, chi/c); }
        GPU_CALLABLE_MEMBER Primitive operator *(const real c) const {return Primitive(rho*c, v1*c, p*c, chi*c); }
        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho -= prims.rho;
            v1  -= prims.v1;
            p   -= prims.p;
            chi -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho *= c;
            v1  *= c;
            p   *= c;
            chi *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v() const {
            return v1;
        }

        GPU_CALLABLE_MEMBER constexpr real vcomponent(const int nhat) const {
            if (nhat > 1) {
                return 0;
            }
            return v1;
        }

        GPU_CALLABLE_MEMBER
        real get_energy_density(real gamma) const {
            return p / (gamma - 1) + 0.5 * (rho * v1 * v1);
        }
    };

    struct Conserved {
        real rho, m1, e_dens, chi;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {};
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real e_dens) : rho(rho), m1(m1), e_dens(e_dens), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real e_dens, real chi) : rho(rho), m1(m1), e_dens(e_dens), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : rho(cons.rho), m1(cons.m1), e_dens(cons.e_dens), chi(cons.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(rho + cons.rho, m1 + cons.m1, e_dens + cons.e_dens, chi + cons.chi); }
        GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(rho - cons.rho, m1 - cons.m1, e_dens - cons.e_dens, chi - cons.chi); }
        GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(rho/c, m1/c, e_dens/c, chi/c); }
        GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(rho*c, m1*c, e_dens*c, chi*c); }
        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            rho    += cons.rho;
            m1     += cons.m1;
            e_dens += cons.e_dens;
            chi    += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            rho    -= cons.rho;
            m1     -= cons.m1;
            e_dens -= cons.e_dens;
            chi    -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            rho    *= c;
            m1     *= c;
            e_dens *= c;
            chi    *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real& momentum() {
            return m1;
        }

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
            if (nhat == 1) {
                return m1;
            }
            return 0;
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, p, chi;
    };

    struct Eigenvals {
        real aL, aR, aStar, pStar;
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real aStar, real pStar) : aL(aL), aR(aR), aStar(aStar), pStar(pStar) {}
    };

} // end hydro1d

namespace hydro2d {
    struct Conserved
    {
        real rho, m1, m2, e_dens, chi;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real m2, real e_dens) :  rho(rho), m1(m1), m2(m2), e_dens(e_dens), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real m2, real e_dens, real chi) :  rho(rho), m1(m1), m2(m2), e_dens(e_dens), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), e_dens(u.e_dens), chi(u.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(rho+p.rho, m1+p.m1, m2+p.m2, e_dens+p.e_dens, chi+p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(rho-p.rho, m1-p.m1, m2-p.m2, e_dens-p.e_dens, chi-p.chi); }
        GPU_CALLABLE_MEMBER Conserved operator * (const Conserved &p)  const { return Conserved(rho*p.rho, m1*p.m1, m2*p.m2, e_dens*p.e_dens, chi*p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(rho*c, m1*c, m2*c, e_dens*c , chi*c); }
        GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(rho/c, m1/c, m2/c, e_dens/c , chi/c); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            rho     += cons.rho;
            m1      += cons.m1;
            m2      += cons.m2;
            e_dens  += cons.e_dens;
            chi     += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            rho    -= cons.rho;
            m1     -= cons.m1;
            m2     -= cons.m2;
            e_dens -= cons.e_dens;
            chi    -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            rho     *= c;
            m1      *= c;
            m2      *= c;
            e_dens  *= c;
            chi     *= c;
            return *this;
        }
        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? m1 : m2); 
        }
        GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {return (nhat == 1 ? m1 : m2 ); }

    };

    struct Primitive {
        real rho, v1, v2, p, chi;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p) : rho(rho), v1(v1), v2(v2), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p, real chi) : rho(rho), v1(v1), v2(v2), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p), chi(c.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p, chi+e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p, chi-e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator * (const real c)        const { return Primitive(rho*c, v1*c, v2*c, p*c, chi*c ); }
        GPU_CALLABLE_MEMBER Primitive operator / (const real c)        const { return Primitive(rho/c, v1/c, v2/c, p/c, chi/c ); }

        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v1     -= prims.v1;
            v2     -= prims.v2;
            p      -= prims.p;
            chi    -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho  *= c;
            v1   *= c;
            v1   *= c;
            p    *= c;
            chi  *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v1() const {
            return v1;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v2() const {
            return v2;
        }
        
        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? v1 : v2); 
        }

        GPU_CALLABLE_MEMBER
        real get_energy_density(real gamma) const {
            return p / (gamma - 1) + 0.5 * (rho * (v1 * v1 + v2 * v2));
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR, csL, csR, aStar, pStar;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR, real aStar, real pStar) : aL(aL), aR(aR), csL(csL), csR(csR), aStar(aStar), pStar(pStar) {}
    };

} // end hydro2d

namespace hydro3d {
    struct Conserved
    {
        real rho, m1, m2, m3, e_dens, chi;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real m2, real m3, real e_dens) :  rho(rho), m1(m1), m2(m2), m3(m3), e_dens(e_dens), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real m2, real m3, real e_dens, real chi) :  rho(rho), m1(m1), m2(m2), m3(m3), e_dens(e_dens), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), e_dens(u.e_dens), chi(u.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(rho+p.rho, m1+p.m1, m2+p.m2, m3 + p.m3, e_dens+p.e_dens, chi+p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(rho-p.rho, m1-p.m1, m2-p.m2, m3 - p.m3, e_dens-p.e_dens, chi-p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(rho*c, m1*c, m2*c, m3 * c, e_dens*c, chi*c); }
        GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(rho/c, m1/c, m2/c, m3 / c, e_dens/c, chi/c); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            rho       += cons.rho;
            m1        += cons.m1;
            m2        += cons.m2;
            m3        += cons.m3;
            e_dens    += cons.e_dens;
            chi       += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            rho       -= cons.rho;
            m1        -= cons.m1;
            m2        -= cons.m2;
            m3        -= cons.m3;
            e_dens    -= cons.e_dens;
            chi       -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            rho    -= c;
            m1     -= c;
            m2     -= c;
            m3     -= c;
            e_dens -= c;
            chi    -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3); }
        GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3); }
    };

    struct Primitive {
        real rho, v1, v2, v3, p, chi;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p, real chi) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), v3(c.v3), p(c.p), chi(c.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2,v3+e.v3, p+e.p, chi+e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2,v3-e.v3, p-e.p, chi-e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator * (const real c)      const { return Primitive(rho*c, v1*c, v2*c,v3*c, p*c, chi*c ); }
        GPU_CALLABLE_MEMBER Primitive operator / (const real c)      const { return Primitive(rho/c, v1/c, v2/c,v3/c, p/c, chi/c ); }

        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            v3     += prims.v3;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v1     -= prims.v1;
            v2     -= prims.v2;
            v3     -= prims.v3;
            p      -= prims.p;
            chi    -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho    -= c;
            v1     -= c;
            v2     -= c;
            v3     -= c;
            p      -= c;
            chi    -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER  constexpr real get_v1() const {
            return v1;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v2() const {
            return v2;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v3() const {
            return v3;
        }

        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {
            return (nhat == 1 ? v1 : (nhat == 2) ? v2 : v3); 
        }

        GPU_CALLABLE_MEMBER
        real get_energy_density(real gamma) const {
            return p / (gamma - 1) + 0.5 * (rho * (v1 * v1 + v2 * v2 + v3 * v3));
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, v3, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR, csL, csR, aStar, pStar;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR, real aStar, real pStar) : aL(aL), aR(aR), csL(csL), csR(csR), aStar(aStar), pStar(pStar) {}
    };

} // end hydro3d


//=============================================
//                SRHD
//=============================================


namespace sr1d {
    struct Primitive {
        real rho, v1, p, chi;
        GPU_CALLABLE_MEMBER Primitive () {}
        GPU_CALLABLE_MEMBER ~Primitive () {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p) : rho(rho), v1(v), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p, real chi) : rho(rho), v1(v), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &prim) : rho(prim.rho), v1(prim.v1), p(prim.p), chi(prim.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v1 + prim.v1, p + prim.p, chi + prim.chi); }
        GPU_CALLABLE_MEMBER Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v1 - prim.v1, p - prim.p, chi - prim.chi); }
        GPU_CALLABLE_MEMBER Primitive operator /(const real c) const {return Primitive(rho/c, v1/c, p/c, chi/c); }
        GPU_CALLABLE_MEMBER Primitive operator *(const real c) const {return Primitive(rho*c, v1*c, p*c, chi*c); }
        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v1     -= prims.v1;
            p      -= prims.p;
            chi    -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho    *= c;
            v1     *= c;
            p      *= c;
            chi    *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v1;
            } else {
                return v1 / std::sqrt(1 + v1 * v1);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - v1 * v1);
            } else {
                return std::sqrt(1 + v1 * v1);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real vcomponent(const int nhat) const {
            if (nhat == 1) {
                return get_v();
            }
            return 0;
        }

        GPU_CALLABLE_MEMBER
        real get_enthalpy(real gamma) const {
            return 1 + gamma * p /(rho * (gamma - 1));
        }
    };

    struct Conserved {
        real d, s1, tau, chi;
        GPU_CALLABLE_MEMBER ~Conserved() {};
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real tau) : d(d), s1(s1), tau(tau), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real tau, real chi) : d(d), s1(s1), tau(tau), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : d(cons.d), s1(cons.s1), tau(cons.tau), chi(cons.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(d + cons.d, s1 + cons.s1, tau + cons.tau, chi + cons.chi); }
        GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(d - cons.d, s1 - cons.s1, tau - cons.tau, chi - cons.chi); }
        GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(d/c, s1/c, tau/c, chi/c); }
        GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(d*c, s1*c, tau*c, chi*c); }
        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            d      += cons.d;
            s1     += cons.s1;
            tau    += cons.tau;
            chi    += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            d      -= cons.d;
            s1     -= cons.s1;
            tau    -= cons.tau;
            chi    -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            d    *= c;
            s1   *= c;
            tau  *= c;
            chi  *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real& momentum() {
            return s1;
        }
        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
            if (nhat == 1) {
                return s1;
            }
            return 0;
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, p, chi;
    };

    struct Eigenvals {
        real aL, aR, csL, csR;
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR), csL(0), csR(0) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR) : aL(aL), aR(aR), csL(csL), csR(csR) {}
    };

} // end sr1d

namespace sr2d {
    struct Conserved
    {
        real d, s1, s2, tau, chi;
        
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real tau) :  d(d), s1(s1), s2(s2), tau(tau), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real tau, real chi) :  d(d), s1(s1), s2(s2), tau(tau), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : d(u.d), s1(u.s1), s2(u.s2), tau(u.tau), chi(u.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(d+p.d, s1+p.s1, s2+p.s2, tau+p.tau, chi+p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(d-p.d, s1-p.s1, s2-p.s2, tau-p.tau, chi-p.chi); }
        GPU_CALLABLE_MEMBER Conserved operator * (const Conserved &p)  const { return Conserved(d*p.d, s1*p.s1, s2*p.s2, tau*p.tau, chi*p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(d*c, s1*c, s2*c, tau*c , chi*c); }
        GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(d/c, s1/c, s2/c, tau/c , chi/c); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            d      += cons.d;
            s1     += cons.s1;
            s2     += cons.s2;
            tau    += cons.tau;
            chi    += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            d      -= cons.d;
            s1     -= cons.s1;
            s2     -= cons.s2;
            tau    -= cons.tau;
            chi    -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            d    *= c;
            s1   *= c;
            s2   *= c;
            tau  *= c;
            chi  *= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? s1 : s2); 
        }
        GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {
            return (nhat == 1 ? s1 : s2); 
        }
    };

    struct Primitive {
        real rho, v1, v2, p, chi;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p) : rho(rho), v1(v1), v2(v2), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p, real chi) : rho(rho), v1(v1), v2(v2), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p), chi(c.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p, chi+e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p, chi-e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator * (const real c)        const { return Primitive(rho*c, v1*c, v2*c, p*c, chi*c ); }
        GPU_CALLABLE_MEMBER Primitive operator / (const real c)        const { return Primitive(rho/c, v1/c, v2/c, p/c, chi/c ); }

        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho  *= c;
            v1   *= c;
            v2   *= c;
            p    *= c;
            chi  *= c;
            return *this;
        }
        
        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {
            if (nhat > 2) {
                return 0;
            }
            return (nhat == 1 ? get_v1() : get_v2()); 
        }

        GPU_CALLABLE_MEMBER real calc_lorentz_gamma() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2));
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v1() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v1;
            } else {
                return v1 / std::sqrt(1 + v1 * v1 + v2 * v2);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v2() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v2;
            } else {
                return v2 / std::sqrt(1 + v1 * v1 + v2 * v2);
            }
        } 

        GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2));
            }
        }

        GPU_CALLABLE_MEMBER
        real get_enthalpy(real gamma) const {
            return 1 + gamma * p /(rho * (gamma - 1));
        }

    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR, csL, csR;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR) : aL(aL), aR(aR), csL(csL), csR(csR) {}
    };

} // end sr2d

namespace sr3d {
    struct Conserved
    {
        real d, s1, s2, s3, tau, chi;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real s3, real tau) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real s3, real tau, real chi) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : d(u.d), s1(u.s1), s2(u.s2), s3(u.s3), tau(u.tau), chi(u.chi) {}
        GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(d+p.d, s1+p.s1, s2+p.s2, s3 + p.s3, tau+p.tau, chi+p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(d-p.d, s1-p.s1, s2-p.s2, s3 - p.s3, tau-p.tau, chi-p.chi); }  
        GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(d*c, s1*c, s2*c, s3 * c, tau*c ,chi*c); }
        GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(d/c, s1/c, s2/c, s3 / c, tau/c ,chi/c); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            d      += cons.d;
            s1     += cons.s1;
            s2     += cons.s2;
            s3     += cons.s3;
            tau    += cons.tau;
            chi    += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            d      -= cons.d;
            s1     -= cons.s1;
            s2     -= cons.s2;
            s3     -= cons.s3;
            tau    -= cons.tau;
            chi    -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
            d      -= c;
            s1     -= c;
            s2     -= c;
            s3     -= c;
            tau    -= c;
            chi    -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {return (nhat == 1 ? s1 : (nhat == 2) ? s2 : s3); }
        GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {return (nhat == 1 ? s1 : (nhat == 2) ? s2 : s3); }
    };

    struct Primitive {
        real rho, v1, v2, v3, p, chi;
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p, real chi) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), v3(c.v3), p(c.p), chi(c.chi) {}
        GPU_CALLABLE_MEMBER Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2,v3+e.v3, p+e.p, chi+e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2,v3-e.v3, p-e.p, chi-e.chi); }  
        GPU_CALLABLE_MEMBER Primitive operator * (const real c)      const { return Primitive(rho*c, v1*c, v2*c,v3*c, p*c, chi*c ); }
        GPU_CALLABLE_MEMBER Primitive operator / (const real c)      const { return Primitive(rho/c, v1/c, v2/c,v3/c, p/c, chi/c ); }

        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            v3     += prims.v3;
            p      += prims.p;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v1     -= prims.v1;
            v2     -= prims.v2;
            v3     -= prims.v3;
            p      -= prims.p;
            chi    -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
            rho    -= c;
            v1     -= c;
            v2     -= c;
            v3     -= c;
            p      -= c;
            chi    -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {
            return nhat == 1 ? get_v1() : (nhat == 2) ? get_v2() : get_v3();
        }

        GPU_CALLABLE_MEMBER real calc_lorentz_gamma() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v1() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v1;
            } else {
                return v1 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v2() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v2;
            } else {
                return v2 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v3() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v3;
            } else {
                return v3 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }


        GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        GPU_CALLABLE_MEMBER
        real get_enthalpy(real gamma) const {
            return 1 + gamma * p /(rho * (gamma - 1));
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, v3, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR, csL, csR;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR), csL(0), csR(0) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real csL, real csR) : aL(aL), aR(aR), csL(csL), csR(csR) {}
    };

} // end sr3d 

//================================
//               RMHD
//================================

// namespace rmhd1d {
//     struct Primitive {
//         real rho, v1, p, b1, chi;
//         GPU_CALLABLE_MEMBER Primitive () {}
//         GPU_CALLABLE_MEMBER ~Primitive () {}
//         GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p, real b1) : rho(rho), v1(v), p(p), b1(b1), chi(0) {}
//         GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p, real b1, real chi) : rho(rho), v1(v), p(p), b1(b1), chi(chi) {}
//         GPU_CALLABLE_MEMBER Primitive(const Primitive &prim) : rho(prim.rho), v1(prim.v1), p(prim.p), b1(prim.b1), chi(prim.chi) {}
//         GPU_CALLABLE_MEMBER Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v1 + prim.v1, p + prim.p, b1 + prim.b1, chi + prim.chi); }
//         GPU_CALLABLE_MEMBER Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v1 - prim.v1, p - prim.p, b1 - prim.b1, chi - prim.chi); }
//         GPU_CALLABLE_MEMBER Primitive operator /(const real c) const {return Primitive(rho/c, v1/c, p/c, b1/c, chi/c); }
//         GPU_CALLABLE_MEMBER Primitive operator *(const real c) const {return Primitive(rho*c, v1*c, p*c, b1*c, chi*c); }
//         GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
//             rho    += prims.rho;
//             v1     += prims.v1;
//             p      += prims.p;
//             b1     += prims.b1;
//             chi    += prims.chi;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
//             rho    -= prims.rho;
//             v1     -= prims.v1;
//             p      -= prims.p;
//             b1     -= prims.b1;
//             chi    -= prims.chi;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
//             rho    *= c;
//             v1     *= c;
//             p      *= c;
//             b1     *= c;
//             chi    *= c;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER constexpr real get_v() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return v1;
//             } else {
//                 return v1 / std::sqrt(1 + v1 * v1);
//             }
//         }

//         GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return 1 / std::sqrt(1 - v1 * v1);
//             } else {
//                 return std::sqrt(1 + v1 * v1);
//             }
//         }

//         GPU_CALLABLE_MEMBER constexpr real lorentz_factor_squared() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return 1 / (1 - v1 * v1);
//             } else {
//                 return (1 + v1 * v1);
//             }
//         }

//         GPU_CALLABLE_MEMBER constexpr real vcomponent(const int nhat) const {
//             if (nhat == 1) {
//                 return get_v();
//             }
//             return 0;
//         }

//         GPU_CALLABLE_MEMBER constexpr real bcomponent(const int nhat) const {
//             if (nhat == 1) {
//                 return b1;
//             }
//             return 0;
//         }

//         GPU_CALLABLE_MEMBER
//         real gas_enthalpy(real gamma) const {
//             return 1 + gamma * p /(rho * (gamma - 1));
//         }

//         GPU_CALLABLE_MEMBER
//         real vdotb() const {
//             return (v1 * b1);
//         }

//         GPU_CALLABLE_MEMBER
//         real bsquared() const {
//             return (b1 * b1);
//         }

//         GPU_CALLABLE_MEMBER
//         real total_pressure() const {
//             return p + 0.5 * bsquared() / lorentz_factor_squared() + vdotb() * vdotb();
//         }

//         GPU_CALLABLE_MEMBER
//         real total_enthalpy(const real gamma) const {
//             return gas_enthalpy(gamma) + bsquared() / lorentz_factor_squared() + vdotb() * vdotb();
//         }

//         GPU_CALLABLE_MEMBER
//         real vsquared() const {
//             return v1 * v1;
//         }
//     };

//     struct Conserved {
//         real d, s1, tau, b1, chi;
//         GPU_CALLABLE_MEMBER ~Conserved() {};
//         GPU_CALLABLE_MEMBER Conserved() {}
//         GPU_CALLABLE_MEMBER Conserved(real d, real s1, real tau, real b1) : d(d), s1(s1), tau(tau), b1(b1), chi(0) {}
//         GPU_CALLABLE_MEMBER Conserved(real d, real s1, real tau, real b1, real chi) : d(d), s1(s1), tau(tau), b1(b1), chi(chi) {}
//         GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : d(cons.d), s1(cons.s1), tau(cons.tau), chi(cons.chi) {}
//         GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(d + cons.d, s1 + cons.s1, tau + cons.tau, b1 + cons.b1, chi + cons.chi); }
//         GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(d - cons.d, s1 - cons.s1, tau - cons.tau, b1 - cons.b1, chi - cons.chi); }
//         GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(d/c, s1/c, tau/c, b1/c, chi/c); }
//         GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(d*c, s1*c, tau*c, b1*c, chi*c); }
//         GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
//             d    += cons.d;
//             s1   += cons.s1;
//             tau  += cons.tau;
//             b1   += cons.b1;
//             chi  += cons.chi;
//             return *this;
//         }
//         GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
//             d    -= cons.d;
//             s1   -= cons.s1;
//             tau  -= cons.tau;
//             b1   -= cons.b1;
//             chi  -= cons.chi;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
//             d   *= c;
//             s1  *= c;
//             tau *= c;
//             b1  *= c;
//             chi *= c;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER constexpr real& momentum() {
//             return s1;
//         }
//         GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
//             if (nhat == 1) {
//                 return b1;
//             }
//             return 0;
//         }

//         GPU_CALLABLE_MEMBER constexpr real& bcomponent() {
//             return s1;
//         }
//         GPU_CALLABLE_MEMBER constexpr real bcomponent(const int nhat) const {
//             if (nhat == 1) {
//                 return b1;
//             }
//             return 0;
//         }

//         GPU_CALLABLE_MEMBER real total_energy() {
//             return d + tau;
//         }
//     };

//     struct mag_four_vec {
//         real lorentz, vdb, zero, one;
//         GPU_CALLABLE_MEMBER mag_four_vec() {}
//         GPU_CALLABLE_MEMBER ~mag_four_vec() {}
//         // GPU_CALLABLE_MEMBER mag_four_vec(real zero, real one):
//         // zero(zero),
//         // one(one)
//         // {

//         // }

//         GPU_CALLABLE_MEMBER mag_four_vec(const Primitive &prim) 
//         : lorentz(prim.lorentz_factor()),
//           vdb(prim.vdotb()),
//           zero(lorentz * vdb),
//           one(prim.b1 / lorentz + lorentz * prim.get_v() * vdb)
//           {}
//         GPU_CALLABLE_MEMBER mag_four_vec(const mag_four_vec &c) : lorentz(c.lorentz), vdb(c.vdb), zero(c.zero), one(c.one) {}
//         // GPU_CALLABLE_MEMBER mag_four_vec operator + (const mag_four_vec &e)  const { return mag_four_vec(zero+e.zero, one+e.one); }  
//         // GPU_CALLABLE_MEMBER mag_four_vec operator - (const mag_four_vec &e)  const { return mag_four_vec(zero-e.zero, one-e.one); }  
//         // GPU_CALLABLE_MEMBER mag_four_vec operator * (const real c)      const { return mag_four_vec(zero*c, one*c); }
//         // GPU_CALLABLE_MEMBER mag_four_vec operator / (const real c)      const { return mag_four_vec(zero/c, one/c); }
//         GPU_CALLABLE_MEMBER real inner_product() const {
//             return -zero * zero + one * one;
//         }
//         GPU_CALLABLE_MEMBER constexpr real normal(const luint nhat) const {
//             if (nhat > 1) {
//                 return 0;
//             }
//             return one;
//         }
//     };

//     struct PrimitiveSOA {
//         PrimitiveSOA() {}
//         ~PrimitiveSOA() {}
//         std::vector<real> rho, v1, p, b1, chi;
//     };

//     struct Eigenvals{
//         real afL, afR, csL, csR;
//         GPU_CALLABLE_MEMBER Eigenvals() {}
//         GPU_CALLABLE_MEMBER ~Eigenvals() {}
//         GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR) : afL(afL), afR(afR) {}
//         GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real csL, real csR) : afL(afL), afR(afR), csL(csL), csR(csR) {}
//         // GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real asL, real asR, real csL, real csR) : afL(afL), afR(afR), asL(asL), asR(asR), csL(csL), csR(csR) {}
//     };

// } // end rmhd1d

// namespace rmhd2d {
//     struct Conserved
//     {
//         real d, s1, s2, tau, b1, b2, chi;
        
//         GPU_CALLABLE_MEMBER Conserved() {}
//         GPU_CALLABLE_MEMBER ~Conserved() {}
//         GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real tau, real b1, real b2) :  d(d), s1(s1), s2(s2), tau(tau), b1(b1), b2(b2), chi(0) {}
//         GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real tau, real b1, real b2, real chi) :  d(d), s1(s1), s2(s2), tau(tau), b1(b1), b2(b2), chi(chi) {}
//         GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : d(u.d), s1(u.s1), s2(u.s2), tau(u.tau), chi(u.chi) {}
//         GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(d+p.d, s1+p.s1, s2+p.s2, tau+p.tau, b1 + p.b1, b2 + p.b2, chi+p.chi); }  
//         GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(d-p.d, s1-p.s1, s2-p.s2, tau-p.tau, b1 - p.b1, b2 - p.b2, chi-p.chi); }
//         GPU_CALLABLE_MEMBER Conserved operator * (const Conserved &p)  const { return Conserved(d*p.d, s1*p.s1, s2*p.s2, tau*p.tau, b1 * p.b1, b2 * p.b2, chi*p.chi); }  
//         GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(d*c, s1*c, s2*c, tau*c , b1*c, b2*c, chi*c); }
//         GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(d/c, s1/c, s2/c, tau/c , b1/c, b2/c, chi/c); }

//         GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
//             d   += cons.d;
//             s1  += cons.s1;
//             s2  += cons.s2;
//             tau += cons.tau;
//             b1  += cons.b1;
//             b2  += cons.b2;
//             chi += cons.chi;
//             return *this;
//         }
//         GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
//             d   -= cons.d;
//             s1  -= cons.s1;
//             s2  -= cons.s2;
//             tau -= cons.tau;
//             b1  -= cons.b1;
//             b2  -= cons.b2;
//             chi -= cons.chi;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER Conserved & operator *=(const real c) {
//             d   *= c;
//             s1  *= c;
//             s2  *= c;
//             tau *= c;
//             b1  *= c;
//             b2  *= c;
//             chi *= c;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {
//             if (nhat > 2) {
//                 return 0;
//             }
//             return (nhat == 1 ? s1 : s2); 
//         }
//         GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {
//             return (nhat == 1 ? s1 : s2); 
//         }

//         GPU_CALLABLE_MEMBER constexpr real bcomponent(const int nhat) const {
//             if (nhat > 2) {
//                 return 0;
//             }
//             return (nhat == 1 ? b1 : b2); 
//         }
//         GPU_CALLABLE_MEMBER constexpr real& bcomponent(const int nhat) {
//             return (nhat == 1 ? b1 : b2); 
//         }

//         GPU_CALLABLE_MEMBER real total_energy() {
//             return d + tau;
//         }
//     };

//     struct Primitive {
//         real rho, v1, v2, p, b1, b2, chi;
//         GPU_CALLABLE_MEMBER Primitive() {}
//         GPU_CALLABLE_MEMBER ~Primitive() {}
//         GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p, real b1, real b2) : rho(rho), v1(v1), v2(v2), p(p), b1(b1), b2(b2), chi(0) {}
//         GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real p, real b1, real b2, real chi) : rho(rho), v1(v1), v2(v2), p(p), b1(b1), b2(b2), chi(chi) {}
//         GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p), b1(c.b1), b2(c.b2), chi(c.chi) {}
//         GPU_CALLABLE_MEMBER Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p, b1 + e.b1, b2 + e.b2, chi+e.chi); }  
//         GPU_CALLABLE_MEMBER Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p, b1 - e.b1, b2 - e.b2, chi-e.chi); }  
//         GPU_CALLABLE_MEMBER Primitive operator * (const real c)        const { return Primitive(rho*c, v1*c, v2*c, p*c, b1*c, b2*c, chi*c ); }
//         GPU_CALLABLE_MEMBER Primitive operator / (const real c)        const { return Primitive(rho/c, v1/c, v2/c, p/c, b1/c, b2/c, chi/c ); }

//         GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
//             rho    += prims.rho;
//             v1     += prims.v1;
//             v2     += prims.v2;
//             p      += prims.p;
//             b1     += prims.b1;
//             b2     += prims.b2;
//             chi    += prims.chi;
//             return *this;
//         }

//         GPU_CALLABLE_MEMBER Primitive & operator *=(const real c) {
//             rho  *= c;
//             v1   *= c;
//             v2   *= c;
//             p    *= c;
//             b1   *= c;
//             b2   *= c;
//             chi  *= c;
//             return *this;
//         }
        
//         GPU_CALLABLE_MEMBER
//         constexpr real vcomponent(const unsigned nhat) const {
//             if (nhat > 2) {
//                 return 0;
//             }
//             return (nhat == 1 ? get_v1() : get_v2()); 
//         }

//         GPU_CALLABLE_MEMBER
//         constexpr real bcomponent(const unsigned nhat) const {
//             if (nhat > 2) {
//                 return 0;
//             }
//             return (nhat == 1) ? b1 : b2; 
//         }

//         GPU_CALLABLE_MEMBER constexpr real get_v1() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return v1;
//             } else {
//                 return v1 / std::sqrt(1 + v1 * v1 + v2 * v2);
//             }
//         }

//         GPU_CALLABLE_MEMBER constexpr real get_v2() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return v2;
//             } else {
//                 return v2 / std::sqrt(1 + v1 * v1 + v2 * v2);
//             }
//         } 

//         GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));
//             } else {
//                 return std::sqrt(1 + (v1 * v1 + v2 * v2));
//             }
//         }

//         GPU_CALLABLE_MEMBER constexpr real lorentz_factor_squared() const {
//             if constexpr(global::VelocityType == global::Velocity::Beta) {
//                 return 1 / (1 - (v1 * v1 + v2 * v2));
//             } else {
//                 return (1 + (v1 * v1 + v2 * v2));
//             }
//         }

//         GPU_CALLABLE_MEMBER
//         real gas_enthalpy(real gamma) const {
//             return 1 + gamma * p /(rho * (gamma - 1));
//         }

//         GPU_CALLABLE_MEMBER
//         real vdotb() const {
//             return (v1 * b1 + v2 * b2);
//         }

//         GPU_CALLABLE_MEMBER
//         real bsquared() const {
//             return (b1 * b1 + b2 * b2);
//         }

//         GPU_CALLABLE_MEMBER
//         real total_pressure() const {
//             return p + 0.5 * bsquared() / lorentz_factor_squared() + vdotb() * vdotb();
//         }

//         GPU_CALLABLE_MEMBER
//         real total_enthalpy(const real gamma) const {
//             return gas_enthalpy(gamma) + bsquared() / lorentz_factor_squared() + vdotb() * vdotb();
//         }


//         GPU_CALLABLE_MEMBER
//         real vsquared() const {
//             return v1 * v1 + v2 * v2 ;
//         }

//     };

//     struct mag_four_vec {
//         real lorentz, vdb, zero, one, two;
//         GPU_CALLABLE_MEMBER mag_four_vec() {}
//         GPU_CALLABLE_MEMBER ~mag_four_vec() {}
//         // GPU_CALLABLE_MEMBER mag_four_vec(real zero, real one, real two):
//         // zero(zero),
//         // one(one),
//         // two(two)
//         // {

//         // }

//         GPU_CALLABLE_MEMBER mag_four_vec(const Primitive &prim) 
//         : lorentz(prim.lorentz_factor()),
//           vdb(prim.vdotb()),
//           zero(lorentz * vdb),
//           one(prim.b1 / lorentz + lorentz * prim.get_v1() * vdb),
//           two(prim.b2 / lorentz + lorentz * prim.get_v2() * vdb)
//           {}
//         GPU_CALLABLE_MEMBER mag_four_vec(const mag_four_vec &c) : lorentz(c.lorentz), vdb(c.vdb), zero(c.zero), one(c.one), two(c.two) {}
//         // GPU_CALLABLE_MEMBER mag_four_vec operator + (const mag_four_vec &e)  const { return mag_four_vec(zero+e.zero, one+e.one, two+e.two); }  
//         // GPU_CALLABLE_MEMBER mag_four_vec operator - (const mag_four_vec &e)  const { return mag_four_vec(zero-e.zero, one-e.one, two-e.two); }  
//         // GPU_CALLABLE_MEMBER mag_four_vec operator * (const real c)      const { return mag_four_vec(zero*c, one*c, two*c); }
//         // GPU_CALLABLE_MEMBER mag_four_vec operator / (const real c)      const { return mag_four_vec(zero/c, one/c, two/c); }
//         GPU_CALLABLE_MEMBER real inner_product() const {
//             return -zero * zero + one * one + two * two;
//         }
//         GPU_CALLABLE_MEMBER constexpr real normal(const luint nhat) const {
//             if (nhat > 2) {
//                 return 0;
//             }
//             return nhat == 1 ? one : two;
//         }
//     };

//     struct PrimitiveSOA {
//         PrimitiveSOA() {}
//         ~PrimitiveSOA() {}
//         std::vector<real> rho, v1, v2, p, b1, b2, chi;
//     };
    
//     struct Eigenvals{
//         real afL, afR, csL, csR;
//         GPU_CALLABLE_MEMBER Eigenvals() {}
//         GPU_CALLABLE_MEMBER ~Eigenvals() {}
//         GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR) : afL(afL), afR(afR) {}
//         GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real csL, real csR) : afL(afL), afR(afR), csL(csL), csR(csR) {}
//         // GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real asL, real asR, real csL, real csR) : afL(afL), afR(afR), asL(asL), asR(asR), csL(csL), csR(csR) {}
//     };

// } // end rmhd2d

namespace rmhd {
    template<int dim>
    struct AnyConserved
    {
        real d, s1, s2, s3, tau, b1, b2, b3, chi;
        GPU_CALLABLE_MEMBER AnyConserved() {}
        GPU_CALLABLE_MEMBER ~AnyConserved() {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real tau, real b1) : d(d), s1(s1), s2(0), s3(0), tau(tau), b1(b1), b2(0), b3(0), chi(0) {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real tau, real b1, real chi) : d(d), s1(s1), s2(0), s3(0), tau(tau), b1(b1), b2(0), b3(0), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real s2, real tau, real b1, real b2) : d(d), s1(s1), s2(s2), s3(0), tau(tau), b1(b1), b2(b2), b3(0), chi(0) {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real s2, real tau, real b1, real b2, real chi) : d(d), s1(s1), s2(s2), s3(0), tau(tau), b1(b1), b2(b2), b3(0), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real s2, real s3, real tau, real b1, real b2, real b3) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), b1(b1), b2(b2), b3(b3), chi(0) {}
        GPU_CALLABLE_MEMBER AnyConserved(real d, real s1, real s2, real s3, real tau, real b1, real b2, real b3, real chi) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), b1(b1), b2(b2), b3(b3), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyConserved(const AnyConserved &u) : d(u.d), s1(u.s1), s2(u.s2), s3(u.s3), tau(u.tau), b1(u.b1), b2(u.b2), b3(u.b3), chi(u.chi) {}
        GPU_CALLABLE_MEMBER AnyConserved operator + (const AnyConserved &p)  const { return AnyConserved(d+p.d, s1+p.s1, s2+p.s2, s3 + p.s3, tau+p.tau, b1 + p.b1, b2 + p.b2, b3 + p.b3, chi+p.chi); }  
        GPU_CALLABLE_MEMBER AnyConserved operator - (const AnyConserved &p)  const { return AnyConserved(d-p.d, s1-p.s1, s2-p.s2, s3 - p.s3, tau-p.tau, b1 - p.b1, b2 - p.b2, b3 - p.b3, chi-p.chi); }  
        GPU_CALLABLE_MEMBER AnyConserved operator * (const real c)      const { return AnyConserved(d*c, s1*c, s2*c, s3 * c, tau*c , b1*c, b2*c, b3*c, chi*c); }
        GPU_CALLABLE_MEMBER AnyConserved operator / (const real c)      const { return AnyConserved(d/c, s1/c, s2/c, s3 / c, tau/c , b1/c, b2/c, b3/c, chi/c); }

        GPU_CALLABLE_MEMBER AnyConserved & operator +=(const AnyConserved &cons) {
            d   += cons.d;
            s1  += cons.s1;
            s2  += cons.s2;
            s3  += cons.s3;
            tau += cons.tau;
            b1  += cons.b1;
            b2  += cons.b2;
            b3  += cons.b3;
            chi += cons.chi;
            return *this;
        }
        GPU_CALLABLE_MEMBER AnyConserved & operator -=(const AnyConserved &cons) {
            d   -= cons.d;
            s1  -= cons.s1;
            s2  -= cons.s2;
            s3  -= cons.s3;
            tau -= cons.tau;
            b1  -= cons.b1;
            b2  -= cons.b2;
            b3  -= cons.b3;
            chi -= cons.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER AnyConserved & operator *=(const real c) {
            d   -= c;
            s1  -= c;
            s2  -= c;
            s3  -= c;
            tau -= c;
            b1  -= c;
            b2  -= c;
            b3  -= c;
            chi -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER real total_energy() {
            return d + tau;
        }

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {return (nhat == 1 ? s1 : (nhat == 2) ? s2 : s3); }
        GPU_CALLABLE_MEMBER constexpr real& momentum(const int nhat) {return (nhat == 1 ? s1 : (nhat == 2) ? s2 : s3); }
        GPU_CALLABLE_MEMBER constexpr real& momentum() {return s1; }
        GPU_CALLABLE_MEMBER constexpr real  bcomponent(const int nhat) const {return (nhat == 1 ? b1 : (nhat == 2) ? b2 : b3); }
        GPU_CALLABLE_MEMBER constexpr real& bcomponent(const int nhat) {return (nhat == 1 ? b1 : (nhat == 2) ? b2 : b3); }
    };

    template<int dim>
    struct AnyPrimitive {
        real rho, v1, v2, v3, p, b1, b2, b3, chi;
        GPU_CALLABLE_MEMBER AnyPrimitive() {}
        GPU_CALLABLE_MEMBER ~AnyPrimitive() {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real p, real b1) : rho(rho), v1(v1), v2(0), v3(0), p(p), b1(b1), b2(0), b3(0), chi(0) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real p, real b1, real chi) : rho(rho), v1(v1), v2(0), v3(0), p(p), b1(b1), b2(0), b3(0), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real v2, real p, real b1, real b2) : rho(rho), v1(v1), v2(v2), v3(0), p(p), b1(b1), b2(b2), b3(0), chi(0) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real v2, real p, real b1, real b2, real chi) : rho(rho), v1(v1), v2(v2), v3(0), p(p), b1(b1), b2(b2), b3(0), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real v2, real v3, real p, real b1, real b2, real b3) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), b1(b1), b2(b2), b3(b3), chi(0) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(real rho, real v1, real v2, real v3, real p, real b1, real b2, real b3, real chi) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), b1(b1), b2(b2), b3(b3), chi(chi) {}
        GPU_CALLABLE_MEMBER AnyPrimitive(const AnyPrimitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), v3(c.v3), p(c.p), b1(c.b1), b2(c.b2), b3(c.b3), chi(c.chi) {}
        GPU_CALLABLE_MEMBER AnyPrimitive operator + (const AnyPrimitive &e)  const { return AnyPrimitive(rho+e.rho, v1+e.v1, v2+e.v2,v3+e.v3, p+e.p, b1 + e.b1, b2 + e.b2, b3 + e.b3, chi+e.chi); }  
        GPU_CALLABLE_MEMBER AnyPrimitive operator - (const AnyPrimitive &e)  const { return AnyPrimitive(rho-e.rho, v1-e.v1, v2-e.v2,v3-e.v3, p-e.p, b1 - e.b1, b2 - e.b2, b3 - e.b3, chi-e.chi); }  
        GPU_CALLABLE_MEMBER AnyPrimitive operator * (const real c)      const { return AnyPrimitive(rho*c, v1*c, v2*c,v3*c, p*c, b1*c, b2*c, b3*c, chi*c ); }
        GPU_CALLABLE_MEMBER AnyPrimitive operator / (const real c)      const { return AnyPrimitive(rho/c, v1/c, v2/c,v3/c, p/c, b1/c, b2/c, b3/c, chi/c ); }

        GPU_CALLABLE_MEMBER AnyPrimitive & operator +=(const AnyPrimitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            v3     += prims.v3;
            p      += prims.p;
            b1     += prims.b1;
            b2     += prims.b2;
            b3     += prims.b3;
            chi    += prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER AnyPrimitive & operator -=(const AnyPrimitive &prims) {
            rho    -= prims.rho;
            v1     -= prims.v1;
            v2     -= prims.v2;
            v3     -= prims.v3;
            p      -= prims.p;
            b1     -= prims.b1;
            b2     -= prims.b2;
            b3     -= prims.b3;
            chi    -= prims.chi;
            return *this;
        }

        GPU_CALLABLE_MEMBER AnyPrimitive & operator *=(const real c) {
            rho    -= c;
            v1     -= c;
            v2     -= c;
            v3     -= c;
            p      -= c;
            b1     -= c;
            b2     -= c;
            b3     -= c;
            chi    -= c;
            return *this;
        }

        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {
            return nhat == 1 ? get_v1() : (nhat == 2) ? get_v2() : get_v3();
        }

        GPU_CALLABLE_MEMBER
        constexpr real bcomponent(const unsigned nhat) const {
            return nhat == 1 ? b1: (nhat == 2) ? b2 : b3;
        }

        GPU_CALLABLE_MEMBER constexpr real get_v1() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v1;
            } else {
                return v1 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v2() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v2;
            } else {
                return v2 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }

        GPU_CALLABLE_MEMBER constexpr real get_v3() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return v3;
            } else {
                return v3 / std::sqrt(1 + v1 * v1 + v2 * v2 + v3 * v3);
            }
        }


        GPU_CALLABLE_MEMBER constexpr real lorentz_factor() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        GPU_CALLABLE_MEMBER constexpr real lorentz_factor_squared() const {
            if constexpr(global::VelocityType == global::Velocity::Beta) {
                return 1 / (1 - (v1 * v1 + v2 * v2 + v3 * v3));
            } else {
                return (1 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
        }

        GPU_CALLABLE_MEMBER
        real gas_enthalpy(real gamma) const {
            return 1 + gamma * p /(rho * (gamma - 1));
        }

        GPU_CALLABLE_MEMBER
        real vdotb() const {
            return (v1 * b1 + v2 * b2 + v3 * b3);
        }

        GPU_CALLABLE_MEMBER
        real bsquared() const {
            return (b1 * b1 + b2 * b2 + b3 * b3);
        }

        GPU_CALLABLE_MEMBER
        real total_pressure() const {
            return p + 0.5 * (bsquared() / lorentz_factor_squared() + vdotb() * vdotb());
        }

        GPU_CALLABLE_MEMBER
        real total_enthalpy(const real gamma) const {
            return gas_enthalpy(gamma) + bsquared() / lorentz_factor_squared() + vdotb() * vdotb();
        }

        GPU_CALLABLE_MEMBER
        real vsquared() const {
            return v1 * v1 + v2 * v2 + v3 * v3;
        }
    };

    template<int dim>
    struct mag_four_vec {
        real lorentz, vdb, zero, one, two, three;
        GPU_CALLABLE_MEMBER mag_four_vec() {}
        GPU_CALLABLE_MEMBER ~mag_four_vec() {}
        // GPU_CALLABLE_MEMBER mag_four_vec(real zero, real one, real two, real three):
        // zero(zero),
        // one(one),
        // two(two),
        // three(three)
        // {

        // }

        GPU_CALLABLE_MEMBER mag_four_vec(const AnyPrimitive<dim> &prim) 
        : lorentz(prim.lorentz_factor()),
          vdb(prim.vdotb()),
          zero(lorentz * vdb),
          one(prim.b1 / lorentz + lorentz * prim.get_v1() * vdb),
          two(prim.b2 / lorentz + lorentz * prim.get_v2() * vdb),
          three(prim.b3 / lorentz + lorentz * prim.get_v3() * vdb) 
          {}
        GPU_CALLABLE_MEMBER mag_four_vec(const mag_four_vec &c) : lorentz(c.lorentz), vdb(c.vdb), zero(c.zero), one(c.one), two(c.two), three(c.three) {}
        // GPU_CALLABLE_MEMBER mag_four_vec operator + (const mag_four_vec &e)  const { return mag_four_vec(zero+e.zero, one+e.one, two+e.two,three+e.three); }  
        // GPU_CALLABLE_MEMBER mag_four_vec operator - (const mag_four_vec &e)  const { return mag_four_vec(zero-e.zero, one-e.one, two-e.two,three-e.three); }  
        // GPU_CALLABLE_MEMBER mag_four_vec operator * (const real c)      const { return mag_four_vec(zero*c, one*c, two*c,three*c); }
        // GPU_CALLABLE_MEMBER mag_four_vec operator / (const real c)      const { return mag_four_vec(zero/c, one/c, two/c,three/c); }
        GPU_CALLABLE_MEMBER real inner_product() const {
            return -zero * zero + one * one + two * two + three * three;
        }
        GPU_CALLABLE_MEMBER constexpr real normal(const luint nhat) const {
            return nhat == 1 ? one : nhat == 2 ? two : three;
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, v3, p, b1, b2, b3, chi;
    };
    
    struct Eigenvals{
        real afL, afR, csL, csR;
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR) : afL(afL), afR(afR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real csL, real csR) : afL(afL), afR(afR), csL(csL), csR(csR) {}
        // GPU_CALLABLE_MEMBER Eigenvals(real afL, real afR, real asL, real asR, real csL, real csR) : afL(afL), afR(afR), asL(asL), asR(asR), csL(csL), csR(csR) {}
    };
} // end rmhd

#endif 