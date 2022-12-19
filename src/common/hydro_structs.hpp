/*
* Houses the different 1D struct members Conserved, Primitives, Eigenvals
* for ease of access and organization. All definitions within header files
* to ensure inlining by the compiler.
*/

#ifndef HYDRO_STRUCTS_HPP
#define HYDRO_STRUCTS_HPP 

#include <vector>
#include "enums.hpp"
#include "build_options.hpp"
//---------------------------------------------------------------------------------------------------------
//  HELPER-GLOBAL-STRUCTS
//---------------------------------------------------------------------------------------------------------
struct PrimData
{
    std::vector<real> rho, v1, v2, v3, p, v, chi;
};

struct DataWriteMembers
{
    real t, ad_gamma;
    real x1min, x1max, x2min, x2max, zmin, zmax, dt;
    int nx, ny, nz, xactive_zones, yactive_zones, zactive_zones, chkpt_idx;
    bool linspace, first_order, using_fourvelocity, mesh_motion;
    std::string coord_system, boundarycond, regime;
    std::vector<real> x1, x2, x3;
};

namespace hydro1d {
    struct Primitive {
        real rho, v, p;
        GPU_CALLABLE_MEMBER Primitive () {}
        GPU_CALLABLE_MEMBER ~Primitive () {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p) : rho(rho), v(v), p(p) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}
        GPU_CALLABLE_MEMBER Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }
        GPU_CALLABLE_MEMBER Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }
        GPU_CALLABLE_MEMBER Primitive operator /(const real c) const {return Primitive(rho/c, v/c, p/c); }
        GPU_CALLABLE_MEMBER Primitive operator *(const real c) const {return Primitive(rho*c, v*c, p*c); }
        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v      += prims.v;
            p      += prims.p;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v      -= prims.v;
            p      -= prims.p;
            return *this;
        }
    };

    struct Conserved {
        real rho, m, e_dens;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {};
        GPU_CALLABLE_MEMBER Conserved(real rho, real m, real e_dens) : rho(rho), m(m), e_dens(e_dens) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : rho(cons.rho), m(cons.m), e_dens(cons.e_dens) {}
        GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(rho + cons.rho, m + cons.m, e_dens + cons.e_dens); }
        GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(rho - cons.rho, m - cons.m, e_dens - cons.e_dens); }
        GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(rho/c, m/c, e_dens/c); }
        GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(rho*c, m*c, e_dens*c); }
        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            rho       += cons.rho;
            m         += cons.m;
            e_dens    += cons.e_dens;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            rho    -= cons.rho;
            m      -= cons.m;
            e_dens -= cons.e_dens;
            return *this;
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v, p;
    };

    struct Eigenvals {
        real aL, aR, aStar, pStar;
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real aStar, real pStar) : aL(aL), aR(aR), aStar(aStar), pStar(pStar) {}
    };

}

namespace sr1d {
    struct Primitive {
        real rho, v, p;
        GPU_CALLABLE_MEMBER Primitive () {}
        GPU_CALLABLE_MEMBER ~Primitive () {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v, real p) : rho(rho), v(v), p(p) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}
        GPU_CALLABLE_MEMBER Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }
        GPU_CALLABLE_MEMBER Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }
        GPU_CALLABLE_MEMBER Primitive operator /(const real c) const {return Primitive(rho/c, v/c, p/c); }
        GPU_CALLABLE_MEMBER Primitive operator *(const real c) const {return Primitive(rho*c, v*c, p*c); }
        GPU_CALLABLE_MEMBER Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v      += prims.v;
            p      += prims.p;
            return *this;
        }

        GPU_CALLABLE_MEMBER Primitive & operator -=(const Primitive &prims) {
            rho    -= prims.rho;
            v      -= prims.v;
            p      -= prims.p;
            return *this;
        }
    };

    struct Conserved {
        real d, s, tau;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {};
        GPU_CALLABLE_MEMBER Conserved(real d, real s, real tau) : d(d), s(s), tau(tau) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : d(cons.d), s(cons.s), tau(cons.tau) {}
        GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(d + cons.d, s + cons.s, tau + cons.tau); }
        GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(d - cons.d, s - cons.s, tau - cons.tau); }
        GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(d/c, s/c, tau/c); }
        GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(d*c, s*c, tau*c); }
        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            d      += cons.d;
            s      += cons.s;
            tau    += cons.tau;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            d      -= cons.d;
            s      -= cons.s;
            tau    -= cons.tau;
            return *this;
        }
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v, p;
    };

    struct ConservedArray {
        ConservedArray() {}
        ~ConservedArray() {}
        std::vector<real> d, s, tau;
    };
    struct Eigenvals {
        real aL, aR;
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
    };

}

namespace sr2d {
    struct Conserved
    {
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        real d, s1, s2, tau, chi;
        
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

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {return (nhat == 1 ? s1 : s2); }
    };

    struct Primitive {
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        real rho, v1, v2, p, chi;

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
        
        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {return (nhat == 1 ? v1 : v2); }

        GPU_CALLABLE_MEMBER inline real calc_lorentz_gamma() const {
            if constexpr(VelocityType == Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2));
            }
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

}

namespace hydro2d {
    struct Conserved
    {
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        real rho, m1, m2, e_dens, chi;
        
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

        GPU_CALLABLE_MEMBER constexpr real momentum(const int nhat) const {return (nhat == 1 ? m1 : m2); }
    };

    struct Primitive {
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        real rho, v1, v2, p, chi;

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
        
        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {return (nhat == 1 ? v1 : v2); }

    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR, cL, cR, aStar, pStar;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real cL, real cR, real aStar, real pStar) : aL(aL), aR(aR), cL(cL), cR(cR), aStar(aStar), pStar(pStar) {}
    };

}


namespace sr3d {
    struct Conserved
    {
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        real d, s1, s2, s3, tau, chi;

        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real s3, real tau) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), chi(0) {}
        GPU_CALLABLE_MEMBER Conserved(real d, real s1, real s2, real s3, real tau, real chi) :  d(d), s1(s1), s2(s2), s3(s3), tau(tau), chi(chi) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : d(u.d), s1(u.s1), s2(u.s2), tau(u.tau), chi(u.chi) {}
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

        GPU_CALLABLE_MEMBER real momentum(const int nhat) const {return (nhat == 1 ? s1 : (nhat == 2) ? s2 : s3); }
    };

    struct Primitive {
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        real rho, v1, v2, v3, p, chi;

        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p, real chi) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p), chi(c.chi) {}
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

        GPU_CALLABLE_MEMBER
        constexpr real vcomponent(const unsigned nhat) const {return (nhat == 1 ? v1 : (nhat == 2) ? v2 : v3); }

        GPU_CALLABLE_MEMBER inline real calc_lorentz_gamma() const {
            if constexpr(VelocityType == Velocity::Beta) {
                return 1 / std::sqrt(1 - (v1 * v1 + v2 * v2 + v3 * v3));
            } else {
                return std::sqrt(1 + (v1 * v1 + v2 * v2 + v3 * v3));
            }
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
        real aL, aR, cL, cR;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR), cL(0), cR(0) {}
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR, real cL, real cR) : aL(aL), aR(aR), cL(cL), cR(cR) {}
    };

} // end sr3d 

namespace hydro3d {
    struct Conserved
    {
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() {}
        real rho, m1, m2, m3, e_dens;

        GPU_CALLABLE_MEMBER Conserved(real rho, real m1, real m2, real m3, real e_dens) :  rho(rho), m1(m1), m2(m2), m3(m3), e_dens(e_dens) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), e_dens(u.e_dens) {}
        GPU_CALLABLE_MEMBER Conserved operator + (const Conserved &p)  const { return Conserved(rho+p.rho, m1+p.m1, m2+p.m2, m3 + p.m3, e_dens+p.e_dens); }  
        GPU_CALLABLE_MEMBER Conserved operator - (const Conserved &p)  const { return Conserved(rho-p.rho, m1-p.m1, m2-p.m2, m3 - p.m3, e_dens-p.e_dens); }  
        GPU_CALLABLE_MEMBER Conserved operator * (const real c)      const { return Conserved(rho*c, m1*c, m2*c, m3 * c, e_dens*c ); }
        GPU_CALLABLE_MEMBER Conserved operator / (const real c)      const { return Conserved(rho/c, m1/c, m2/c, m3 / c, e_dens/c ); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            rho       += cons.rho;
            m1        += cons.m1;
            m2        += cons.m2;
            m3        += cons.m3;
            e_dens    += cons.e_dens;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            rho       -= cons.rho;
            m1        -= cons.m1;
            m2        -= cons.m2;
            m3        -= cons.m3;
            e_dens    -= cons.e_dens;
            return *this;
        }

        GPU_CALLABLE_MEMBER real momentum(const int nhat) const {return (nhat == 1 ? m1 : (nhat == 2) ? m2 : m3); }
    };

    struct Primitive {
        GPU_CALLABLE_MEMBER Primitive() {}
        GPU_CALLABLE_MEMBER ~Primitive() {}
        real rho, v1, v2, v3, p, chi;

        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(0) {}
        GPU_CALLABLE_MEMBER Primitive(real rho, real v1, real v2, real v3, real p, real chi) : rho(rho), v1(v1), v2(v2), v3(v3), p(p), chi(chi) {}
        GPU_CALLABLE_MEMBER Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p), chi(c.chi) {}
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
    };

    struct PrimitiveSOA {
        PrimitiveSOA() {}
        ~PrimitiveSOA() {}
        std::vector<real> rho, v1, v2, v3, p, chi;
    };
    
    struct Eigenvals{
        GPU_CALLABLE_MEMBER Eigenvals() {}
        GPU_CALLABLE_MEMBER ~Eigenvals() {}
        real aL, aR;
        GPU_CALLABLE_MEMBER Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
    };

} // end hydro3d

#endif 