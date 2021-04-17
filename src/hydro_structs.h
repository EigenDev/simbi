/*
* Houses the different 1D struct members Conserved, Primitives, Eigenvals
* for ease of access and organization
*/

#ifndef STRUCTS1D_H
#define STRUCTS1D_H 

#include <vector>

namespace hydro1d {
    struct Primitive {
        double rho, v, p;
        Primitive () {}
        ~Primitive () {}
    };

    struct PrimitiveArray {
        std::vector<double> rho, v, p;
    };

    struct Conserved {
        double rho, m, e_dens;
        double D, S, tau;
        Conserved() {}
        ~Conserved() {}
    };

    struct ConservedArray {
        std::vector<double> rho, m, e_dens;
        std::vector<double> D, S, tau;
    };

    struct Flux {
        double rho, m, e_dens;
        double D, S, tau;
        Flux() {}
        ~Flux() {}
    };

    struct Eigenvals {
        double aL, aR;
        double aLplus, aLminus, aRplus, aRminus;
        Eigenvals() {}
        ~Eigenvals() {}
    };

}

namespace hydro2d {
    /* Define Data Structures for the Fluid Properties. */
        struct Conserved
        {
            Conserved() {}
            ~Conserved() {}
            double D, S1, S2, tau;

            Conserved(double D, double S1, double S2, double tau) : D(D), S1(S1), S2(S2), tau(tau) {}  
            Conserved(const Conserved &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau)    {}  
            Conserved operator + (const Conserved &p)  const { return Conserved(D+p.D, S1+p.S1, S2+p.S2, tau+p.tau); }  
            Conserved operator - (const Conserved &p)  const { return Conserved(D-p.D, S1-p.S1, S2-p.S2, tau-p.tau); }  
            Conserved operator * (const double c)      const { return Conserved(D*c, S1*c, S2*c, tau*c ); }
            Conserved operator / (const double c)      const { return Conserved(D/c, S1/c, S2/c, tau/c ); }

            double momentum(int nhat)
            {
                return (nhat == 1 ? S1 : S2);
            }
        };

        struct Primitives {
            Primitives()  {}
            ~Primitives() {}
            double rho, v1, v2, p;

            Primitives(double rho, double v1, double v2, double p) : rho(rho), v1(v1), v2(v2), p(p) {}
            Primitives(const Primitives &c) : rho(c.rho), v1(c.v1), v2(c.v2), p(c.p) {}
            Primitives operator + (const Primitives &e)  const { return Primitives(rho+e.rho, v1+e.v1, v2+e.v2, p+e.p); }  
            Primitives operator - (const Primitives &e)  const { return Primitives(rho-e.rho, v1-e.v1, v2-e.v2, p-e.p); }  
            Primitives operator * (const double c)       const { return Primitives(rho*c, v1*c, v2*c, p*c ); }
            Primitives operator / (const double c)       const { return Primitives(rho/c, v1/c, v2/c, p/c ); }
        };
        
        struct Eigenvals{
            double aL;
            double aR;
        };
        struct ConserveData
        {
            ConserveData() {}
            ~ConserveData() {}
            std::vector<double> D, S1, S2, tau;

            ConserveData(std::vector<double>  &D,  std::vector<double>  &S1, 
                          std::vector<double>  &S2, std::vector<double>  &tau) : D(D), S1(S1), S2(S2), tau(tau) {} 

            ConserveData(const ConserveData &u) : D(u.D), S1(u.S1), S2(u.S2), tau(u.tau){} 
            Conserved operator[] (int i) const {return Conserved(D[i], S1[i], S2[i], tau[i]); }

            void swap( ConserveData &c) {
                this->D.swap(c.D); 
                this->S1.swap(c.S1);
                this->S2.swap(c.S2); 
                this->tau.swap(c.tau); 
                };
        };

        struct PrimitiveData
        {
            std::vector<double> rho, v1, v2, p;
        };

}


#endif 