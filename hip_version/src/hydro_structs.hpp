/*
* Houses the different 1D struct members Conserved, Primitives, Eigenvals
* for ease of access and organization. All definitions within header files
* to ensure inlining by the compiler.
*/

#ifndef STRUCTS1D_HPP
#define STRUCTS1D_HPP 

#include <vector>
#include "config.hpp"

#ifdef __HIPCC__
#define GPU_CALLABLE_MEMBER __host__ __device__
#else
#define GPU_CALLABLE_MEMBER 
#endif 


namespace hydro1d {
    struct Primitive {
        real rho, v, p;
        Primitive () {}
        ~Primitive () {}
        Primitive(real rho, real v, real p) : rho(rho), v(v), p(p) {}
        Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}

        Primitive operator + (const Primitive &prim) const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }
        Primitive operator - (const Primitive &prim) const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }
        Primitive operator / (const real c)        const {return Primitive(rho/c, v/c, p/c); }
        Primitive operator * (const real c)        const {return Primitive(rho*c, v*c, p*c); }

        Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v      += prims.v;
            p      += prims.p;
            return *this;
        }

        Primitive & operator -=(const Primitive &prims) {
            rho    += prims.rho;
            v      += prims.v;
            p      += prims.p;
            return *this;
        }
    };

    struct Conserved {
        real rho, m, e_dens;
        Conserved() {}
        ~Conserved() {}
        Conserved(real rho, real m, real e_dens) : rho(rho), m(m), e_dens(e_dens) {}
        Conserved(const Conserved &cons) : rho(cons.rho), m(cons.m), e_dens(cons.e_dens) {}

        Conserved operator   + (const Conserved &cons) const {return Conserved(rho + cons.rho, m + cons.m, e_dens + cons.e_dens); }
        Conserved operator   - (const Conserved &cons) const {return Conserved(rho - cons.rho, m - cons.m, e_dens - cons.e_dens); }
        Conserved operator   / (const real c) const {return Conserved(rho/c, m/c, e_dens/c); }
        Conserved operator   * (const real c)  const {return Conserved(rho*c, m*c, e_dens*c); }
        Conserved & operator +=(const Conserved &cons) {
            rho    += cons.rho;
            m      += cons.m;
            e_dens += cons.e_dens;
            return *this;
        }
        Conserved & operator -=(const Conserved &cons) {
            rho    -= cons.rho;
            m      -= cons.m;
            e_dens -= cons.e_dens;
            return *this;
        }

    };

    Conserved operator * (const real c, const Conserved &cons);

    Conserved operator - (const Conserved &cons);

    struct PrimitiveArray {
        PrimitiveArray() {}
        ~PrimitiveArray() {}
        std::vector<real> rho, v, p;
    };

    struct Eigenvals {
        Eigenvals() {}
        ~Eigenvals() {}
        real aL, aR;
        real aStar, pStar;
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
            rho    += prims.rho;
            v      += prims.v;
            p      += prims.p;
            return *this;
        }
    };

    struct Conserved {
        real D, S, tau;
        GPU_CALLABLE_MEMBER Conserved() {}
        GPU_CALLABLE_MEMBER ~Conserved() = default;
        GPU_CALLABLE_MEMBER Conserved(real D, real S, real tau) : D(D), S(S), tau(tau) {}
        GPU_CALLABLE_MEMBER Conserved(const Conserved &cons) : D(cons.D), S(cons.S), tau(cons.tau) {}

        GPU_CALLABLE_MEMBER Conserved operator +(const Conserved &cons) const {return Conserved(D + cons.D, S + cons.S, tau + cons.tau); }

        GPU_CALLABLE_MEMBER Conserved operator -(const Conserved &cons) const {return Conserved(D - cons.D, S - cons.S, tau - cons.tau); }

        GPU_CALLABLE_MEMBER Conserved operator /(const real c) const {return Conserved(D/c, S/c, tau/c); }

        GPU_CALLABLE_MEMBER Conserved operator *(const real c) const {return Conserved(D*c, S*c, tau*c); }

        GPU_CALLABLE_MEMBER Conserved & operator +=(const Conserved &cons) {
            D      += cons.D;
            S      += cons.S;
            tau    += cons.tau;
            return *this;
        }
        GPU_CALLABLE_MEMBER Conserved & operator -=(const Conserved &cons) {
            D      -= cons.D;
            S      -= cons.S;
            tau    -= cons.tau;
            return *this;
        }
    };

    struct PrimitiveArray {
        PrimitiveArray() {}
        ~PrimitiveArray() {}
        std::vector<real> rho, v, p;
    };

    struct ConservedArray {
        ConservedArray() {}
        ~ConservedArray() {}
        std::vector<real> D, S, tau;
    };
    struct Eigenvals {
        real aL, aR;
        Eigenvals() {}
        ~Eigenvals() {}
    };

}

namespace sr2d {
    struct Conserved
    {
        Conserved() {}
        ~Conserved() {}
        real D, S1, S2, tau;

        Conserved(real D, real S1, real S2, real tau) :  D(D), S1(S1), S2(S2), tau(tau) {}
        Conserved(const Conserved &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau) {}
        Conserved operator + (const Conserved &p)  const { return Conserved(D+p.D, S1+p.S1, S2+p.S2, tau+p.tau); }  
        Conserved operator - (const Conserved &p)  const { return Conserved(D-p.D, S1-p.S1, S2-p.S2, tau-p.tau); }  
        Conserved operator * (const real c)      const { return Conserved(D*c, S1*c, S2*c, tau*c ); }
        Conserved operator / (const real c)      const { return Conserved(D/c, S1/c, S2/c, tau/c ); }

        Conserved & operator +=(const Conserved &cons) {
            D      += cons.D;
            S1     += cons.S1;
            S2     += cons.S2;
            tau    += cons.tau;
            return *this;
        }
        Conserved & operator -=(const Conserved &cons) {
            D      += cons.D;
            S1     += cons.S1;
            S2     += cons.S2;
            tau    += cons.tau;
            return *this;
        }

        real momentum(const int nhat) const {return (nhat == 1 ? S1 : S2); }
    };

    struct Primitive {
        Primitive() {}
        ~Primitive() {}
        real rho, v1, v2, p;

        Primitive(real rho, real v1, real v2, real p) : rho(rho), v1(v1), v2(v2), p(p) {}
        Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p) {}
        Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p); }  
        Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p); }  
        Primitive operator * (const real c)      const { return Primitive(rho*c, v1*c, v2*c, p*c ); }
        Primitive operator / (const real c)      const { return Primitive(rho/c, v1/c, v2/c, p/c ); }

        Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            return *this;
        }

        Primitive & operator -=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            return *this;
        }
    };

    struct PrimitiveData {
        PrimitiveData() {}
        ~PrimitiveData() {}
        std::vector<real> rho, v1, v2, p;
    };
    
    struct Eigenvals{
        Eigenvals() {}
        ~Eigenvals() {}
        real aL, aR;
        Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
    };

}

namespace hydro2d {
    struct Conserved
    {
        real rho, m1, m2, e_dens;
        Conserved() {}
        ~Conserved() {}
        Conserved(real rho, real m1, real m2, real e_dens) : rho(rho), m1(m1), m2(m2), e_dens(e_dens) {}  
        Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), e_dens(u.e_dens)    {}  
        Conserved operator + (const Conserved &p)  const { return Conserved(rho+p.rho, m1+p.m1, m2+p.m2, e_dens+p.e_dens); }  
        Conserved operator - (const Conserved &p)  const { return Conserved(rho-p.rho, m1-p.m1, m2-p.m2, e_dens-p.e_dens); }  
        Conserved operator * (const real c)      const { return Conserved(rho*c, m1*c, m2*c, e_dens*c ); }
        Conserved operator / (const real c)      const { return Conserved(rho/c, m1/c, m2/c, e_dens/c ); }
        Conserved & operator +=(const Conserved &cons) {
            rho       += cons.rho;
            m1        += cons.m1;
            m2        += cons.m2;
            e_dens    += cons.e_dens;
            return *this;
        }
        Conserved & operator -=(const Conserved &cons) {
            rho       += cons.rho;
            m1        += cons.m1;
            m2        += cons.m2;
            e_dens    += cons.e_dens;
            return *this;
        }

        real momentum(const int nhat) const { return (nhat == 1 ? m1 : m2); }
    };

    struct Primitive {
        Primitive() {}
        ~Primitive() {}
        real rho, v1, v2, p;

        Primitive(real rho, real v1, real v2, real p) : rho(rho), v1(v1), v2(v2), p(p) {}
        Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p) {}
        Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p); }  
        Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p); }  
        Primitive operator * (const real c)      const { return Primitive(rho*c, v1*c, v2*c, p*c ); }
        Primitive operator / (const real c)      const { return Primitive(rho/c, v1/c, v2/c, p/c ); }

        Primitive & operator +=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            return *this;
        }

        Primitive & operator -=(const Primitive &prims) {
            rho    += prims.rho;
            v1     += prims.v1;
            v2     += prims.v2;
            p      += prims.p;
            return *this;
        }

    };
    
    struct Eigenvals{
        Eigenvals() {}
        ~Eigenvals() {}
        real aL, aR, aStar, pStar;
        Eigenvals(real aL, real aR) : aL(aL), aR(aR) {}
        Eigenvals(real aL, real aR, real aStar, real pStar) : aL(aL), aR(aR), aStar(aStar), pStar(pStar) {}
    };

}


#endif 