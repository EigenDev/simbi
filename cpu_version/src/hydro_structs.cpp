/*
* Implements the different XD struct members Conserved, Primitives, Eigenvals
* for ease of access and organization
*/

#include "hydro_structs.h"

namespace hydro1d
{
    Conserved operator * (const double c, const Conserved &cons) {return cons * c; };
    Conserved operator - (const Conserved &cons) {return Conserved(-cons.rho, -cons.m, -cons.e_dens); };
    
} // namespace hydro1d
