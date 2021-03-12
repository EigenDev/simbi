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
    struct Primitive {
    double rho, v1, v2, p;
    Primitive () {}
    ~Primitive () {}
    };

    struct Conserved {
        double rho, m1, m2, e_dens;
        Conserved() {}
        ~Conserved() {}
    };

    struct Flux {
        double rho, m1, m2, e_dens;
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


#endif 