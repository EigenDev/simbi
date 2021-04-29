/*
* Implements the different XD struct members Conserved, Primitives, Eigenvals
* for ease of access and organization
*/

#include <hydro_structs.h>

namespace hydro1d
{
    Conserved operator * (const double c, const Conserved &cons) {return cons * c; };
    Conserved operator - (const Conserved &cons) {return Conserved(-cons.rho, -cons.m, -cons.e_dens); };
    
} // namespace hydro1d


namespace sr1d {
    //========================================================================================
    Primitive::Primitive () {}
    Primitive::~Primitive () {}
    Primitive::Primitive(double rho, double v, double p) : rho(rho), v(v), p(p) {}
    Primitive::Primitive(const Primitive &prim) : rho(prim.rho), v(prim.v), p(prim.p) {}

    Primitive Primitive::operator +(const Primitive &prim) 
                    const {return Primitive(rho + prim.rho, v + prim.v, p + prim.p); }

    Primitive Primitive::operator -(const Primitive &prim) 
                    const {return Primitive(rho - prim.rho, v - prim.v, p - prim.p); }

    Primitive Primitive::operator /(const double c) 
                    const {return Primitive(rho/c, v/c, p/c); }

    Primitive Primitive::operator *(const double c) 
                    const {return Primitive(rho*c, v*c, p*c); }

    //=======================================================================================
    Conserved::Conserved() {}
    Conserved::~Conserved() {}
    Conserved::Conserved(double D, double S, double tau) : D(D), S(S), tau(tau) {}
    Conserved::Conserved(const Conserved &cons) : D(cons.D), S(cons.S), tau(cons.tau) {}

    Conserved Conserved::operator +(const Conserved &cons) 
                    const {return Conserved(D + cons.D, S + cons.S, tau + cons.tau); }

    Conserved Conserved::operator -(const Conserved &cons) 
                    const {return Conserved(D - cons.D, S - cons.S, tau - cons.tau); }

    Conserved Conserved::operator /(const double c) 
                    const {return Conserved(D/c, S/c, tau/c); }

    Conserved Conserved::operator *(const double c) 
                    const {return Conserved(D*c, S*c, tau*c); }
    
    Conserved operator *(const double c, const Conserved &cons)
                 {return cons * c; }

    Conserved operator -(const Conserved &cons)
                 {return Conserved(-cons.D, -cons.S, -cons.tau); }

    //=========================================================================================
    Eigenvals::Eigenvals() {}
    Eigenvals::~Eigenvals() {}

    PrimitiveArray::PrimitiveArray() {}
    PrimitiveArray::~PrimitiveArray() {}

    ConservedArray::ConservedArray() {}
    ConservedArray::~ConservedArray() {}
}   //namespace sr1d
