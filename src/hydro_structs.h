/*
* Houses the different 1D struct members Conserved, Primitives, Eigenvals
* for ease of access and organization. All definitions within header files
* to ensure inlining by the compiler.
*/

#ifndef STRUCTS1D_H
#define STRUCTS1D_H 

#include <vector>

namespace hydro1d {
    struct Primitive {
        double rho, v, p;
        Primitive () {}
        ~Primitive () {}
        Primitive(double rho, double v, double p) : rho(rho), v(v), p(p) {}
        Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}

        Primitive operator + (const Primitive &prim) const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }
        Primitive operator - (const Primitive &prim) const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }
        Primitive operator / (const double c)        const {return Primitive(rho/c, v/c, p/c); }
        Primitive operator * (const double c)        const {return Primitive(rho*c, v*c, p*c); }
    };

    struct Conserved {
        double rho, m, e_dens;
        Conserved() {}
        ~Conserved() {}
        Conserved(double rho, double m, double e_dens) : rho(rho), m(m), e_dens(e_dens) {}
        Conserved(const Conserved &cons) : rho(cons.rho), m(cons.m), e_dens(cons.e_dens) {}

        Conserved operator   + (const Conserved &cons) const {return Conserved(rho + cons.rho, m + cons.m, e_dens + cons.e_dens); }
        Conserved operator   - (const Conserved &cons) const {return Conserved(rho - cons.rho, m - cons.m, e_dens - cons.e_dens); }
        Conserved operator   / (const double c) const {return Conserved(rho/c, m/c, e_dens/c); }
        Conserved operator   * (const double c)  const {return Conserved(rho*c, m*c, e_dens*c); }
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

    Conserved operator * (const double c, const Conserved &cons);

    Conserved operator - (const Conserved &cons);

    struct PrimitiveArray {
        PrimitiveArray() {}
        ~PrimitiveArray() {}
        std::vector<double> rho, v, p;
    };

    struct Eigenvals {
        Eigenvals() {}
        ~Eigenvals() {}
        double aL, aR;
        double aStar, pStar;
    };

}

namespace sr1d {
    struct Primitive {
        double rho, v, p;
        Primitive () {}
        ~Primitive () {}
        Primitive(double rho, double v, double p) : rho(rho), v(v), p(p) {}
        Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}
        Primitive operator +(const Primitive &prim) const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }

        Primitive operator -(const Primitive &prim) const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }

        Primitive operator /(const double c) const {return Primitive(rho/c, v/c, p/c); }

        Primitive operator *(const double c) const {return Primitive(rho*c, v*c, p*c); }
    };

    struct Conserved {
        double D, S, tau;
        Conserved() {}
        ~Conserved() {}
        
        Conserved(double D, double S, double tau) : D(D), S(S), tau(tau) {}
        Conserved(const Conserved &cons) : D(cons.D), S(cons.S), tau(cons.tau) {}

        Conserved operator +(const Conserved &cons) const {return Conserved(D + cons.D, S + cons.S, tau + cons.tau); }

        Conserved operator -(const Conserved &cons) const {return Conserved(D - cons.D, S - cons.S, tau - cons.tau); }

        Conserved operator /(const double c) const {return Conserved(D/c, S/c, tau/c); }

        Conserved operator *(const double c) const {return Conserved(D*c, S*c, tau*c); }

        Conserved & operator +=(const Conserved &cons) {
            D      += cons.D;
            S      += cons.S;
            tau    += cons.tau;
            return *this;
        }
        Conserved & operator -=(const Conserved &cons) {
            D      -= cons.D;
            S      -= cons.S;
            tau    -= cons.tau;
            return *this;
        }
    };

    struct PrimitiveArray {
        PrimitiveArray() {}
        ~PrimitiveArray() {}
        std::vector<double> rho, v, p;
    };

    struct ConservedArray {
        ConservedArray() {}
        ~ConservedArray() {}
        std::vector<double> D, S, tau;
    };
    struct Eigenvals {
        double aL, aR;
        Eigenvals() {}
        ~Eigenvals() {}
    };

}

namespace sr2d {
    struct Conserved
    {
        Conserved() {}
        ~Conserved() {}
        double D, S1, S2, tau;

        Conserved(double D, double S1, double S2, double tau) :  D(D), S1(S1), S2(S2), tau(tau) {}
        Conserved(const Conserved &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau) {}
        Conserved operator + (const Conserved &p)  const { return Conserved(D+p.D, S1+p.S1, S2+p.S2, tau+p.tau); }  
        Conserved operator - (const Conserved &p)  const { return Conserved(D-p.D, S1-p.S1, S2-p.S2, tau-p.tau); }  
        Conserved operator * (const double c)      const { return Conserved(D*c, S1*c, S2*c, tau*c ); }
        Conserved operator / (const double c)      const { return Conserved(D/c, S1/c, S2/c, tau/c ); }

        double momentum(const int nhat) const {return (nhat == 1 ? S1 : S2); }
    };

    struct Primitive {
        Primitive() {}
        ~Primitive() {}
        double rho, v1, v2, p;

        Primitive(double rho, double v1, double v2, double p) : rho(rho), v1(v1), v2(v2), p(p) {}
        Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p) {}
        Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p); }  
        Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p); }  
        Primitive operator * (const double c)      const { return Primitive(rho*c, v1*c, v2*c, p*c ); }
        Primitive operator / (const double c)      const { return Primitive(rho/c, v1/c, v2/c, p/c ); }
    };

    struct PrimitiveData {
        PrimitiveData() {}
        ~PrimitiveData() {}
        std::vector<double> rho, v1, v2, p;
    };
    
    struct Eigenvals{
        Eigenvals() {}
        ~Eigenvals() {}
        double aL, aR;
        Eigenvals(double aL, double aR) : aL(aL), aR(aR) {}
    };

}

namespace hydro2d {
    struct Conserved
    {
        double rho, m1, m2, e_dens;
        Conserved() {}
        ~Conserved() {}
        Conserved(double rho, double m1, double m2, double e_dens) : rho(rho), m1(m1), m2(m2), e_dens(e_dens) {}  
        Conserved(const Conserved &u) : rho(u.rho), m1(u.m1), m2(u.m2), e_dens(u.e_dens)    {}  
        Conserved operator + (const Conserved &p)  const { return Conserved(rho+p.rho, m1+p.m1, m2+p.m2, e_dens+p.e_dens); }  
        Conserved operator - (const Conserved &p)  const { return Conserved(rho-p.rho, m1-p.m1, m2-p.m2, e_dens-p.e_dens); }  
        Conserved operator * (const double c)      const { return Conserved(rho*c, m1*c, m2*c, e_dens*c ); }
        Conserved operator / (const double c)      const { return Conserved(rho/c, m1/c, m2/c, e_dens/c ); }

        double momentum(const int nhat) const { return (nhat == 1 ? m1 : m2); }
    };

    struct Primitive {
        Primitive() {}
        ~Primitive() {}
        double rho, v1, v2, p;

        Primitive(double rho, double v1, double v2, double p) : rho(rho), v1(v1), v2(v2), p(p) {}
        Primitive(const Primitive &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p) {}
        Primitive operator + (const Primitive &e)  const { return Primitive(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p); }  
        Primitive operator - (const Primitive &e)  const { return Primitive(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p); }  
        Primitive operator * (const double c)      const { return Primitive(rho*c, v1*c, v2*c, p*c ); }
        Primitive operator / (const double c)      const { return Primitive(rho/c, v1/c, v2/c, p/c ); }
    };
    
    struct Eigenvals{
        Eigenvals() {}
        ~Eigenvals() {}
        double aL, aR, aStar, pStar;
        Eigenvals(double aL, double aR) : aL(aL), aR(aR) {}
        Eigenvals(double aL, double aR, double aStar, double pStar) : aL(aL), aR(aR), aStar(aStar), pStar(pStar) {}
    };

}


#endif 