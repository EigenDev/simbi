/**
 * Implementation zone for 
 * the viscous diffusion header
 * file 
 * 
*/

#include "viscous_diff.hpp"
#include <cmath>
#include <iostream>

constexpr double ADIABATIC_GAMMA = 5.0/3.0;
constexpr double PRANDTL_NUMBER  = 3.0/4.0;
constexpr double C_AV = 0.50;
constexpr double C_TH = 1.0 ;//0.05;

simbi::ArtificialViscosity::ArtificialViscosity(){}
simbi::ArtificialViscosity::~ArtificialViscosity() {};




void simbi::ArtificialViscosity::av_flux_at_face(
    sr2d::Conserved &flux,
    const std::vector<sr2d::Primitive> &prims,
    const CLattice &coord_lattice,
    const int ii, 
    const int jj,
    const simbi::Face cell_face
)
{
    const int nx = coord_lattice.nx1zones;
    const int ny = coord_lattice.nx2zones;

    double du1_dx1, du1_dx2, du2_dx1, du2_dx2;
    double tau_11, tau_12, tau_22, tau_21, dh_dx1, dh_dx2; 
    double h_left, h_right, qx1, qx2, mu_av, inv_vol; 
    double si1, sj2, dv1_deta, dv2_deta, volAvg;
    double dv1_dxi, dv2_dxi; 
    double u1f, u2f;

    // Some bounds checking 
    const int il = ii - 1;
    const int ir = ii + 1;
    const int jl = jj - 1;
    const int jr = jj + 1;

    // Translate to the active coordinates for the
    // physical zone (excluding the ghost zones)
    
    const int xc = ii - 2;
    const int yc = jj - 2;
    const int xr = (xc < nx - 1) ? xc + 1 : xc;
    const int xl = (xc > 0)      ? xc - 1 : xc;
    const int yl = (yc > 0)      ? yc - 1 : yc;
    const int yr = (yc < ny - 1) ? yc + 1 : yc;

    const double dx1 = coord_lattice.dx1[xc];
    const double x1  = coord_lattice.x1mean[xc];
    const double dx2 = x1 * coord_lattice.dx2[yc];

    const double xvl = coord_lattice.x1vertices[xc];
    const double xvr = coord_lattice.x1vertices[xc + 1];
    const double dth = coord_lattice.dx2[yc];
    const double dVc = coord_lattice.dVc[yc * nx + xc];
    const double s1r = coord_lattice.s1_face_areas[(nx + 1) * yc + (xc + 1)];
    const double s1l = coord_lattice.s1_face_areas[(nx + 1) * yc + (xc + 0)];
    const double s2r = coord_lattice.s2_face_areas[(yc + 1) * nx + (xc + 0)];
    const double s2l = coord_lattice.s2_face_areas[(yc + 0) * nx + (xc + 0)];
    const double v1r = (prims[jj * nx + ir].v1 - prims[jj * nx + ii].v1 );
    const double v1l = (prims[jj * nx + ii].v1 - prims[jj * nx + il].v1 );
    const double v2r = (prims[jr * nx + ii].v2 - prims[jj * nx + ii].v2 );
    const double v2l = (prims[jj * nx + ii].v2 - prims[jl * nx + il].v2 );

    const double div_uvec  = 1/(dVc) * ( (v1r * s1r - v1l * s1r)  
                                       + (v2r * s2r - v2l * s2r) );
    const double pc        = prims[jj * nx + ii].p;
    const double rhoc      = prims[jj * nx + ii].rho;
    const double hc        = 1.0 + ADIABATIC_GAMMA * pc/(rhoc * (ADIABATIC_GAMMA - 1.0));
    const double h_mesh    = std::max(dx1, dx2) / std::sqrt(2.0);
    const double cs        = std::sqrt(ADIABATIC_GAMMA * pc / (rhoc * hc));
    const double cterm     = C_TH * cs / h_mesh;

    // std::cout << v1r << "\n";
    // std::cout << v1l << "\n";
    // std::cout << v2r << "\n";
    // std::cout << v2r << "\n";
    // std::cout << s2l << "\n";
    // std::cout << s2r << "\n";
    // std::cout << yc << ", " << xc << "\n";
    // std::cin.get();
    double meh;
    if (div_uvec > -cterm){
        flux = {0.0, 0.0, 0.0, 0.0};
        return;
    }
    switch (cell_face)
    {
        case simbi::Face::RIGHT:
                u1f     = 0.5 * (prims[jj * nx + ir].v1 + prims[jj * nx + ii].v1);
                u2f     = 0.5 * (prims[jj * nx + ir].v2 + prims[jj * nx + ii].v2);
                volAvg  = 0.5 * (coord_lattice.dVc[yc * nx + xc] + coord_lattice.dVc[yc * nx + xr]);
                inv_vol = 1.0 / volAvg; 

                //-----------Get surface areas
                si1 = coord_lattice.s1_face_areas[yc * (nx + 1) + xc + 1];
                sj2 = 0.25 * (coord_lattice.s2_face_areas[(yc + 0) * nx + xc] + 
                              coord_lattice.s2_face_areas[(yc + 1) * nx + xc] +
                              coord_lattice.s2_face_areas[(yc + 0) * nx + xr] +
                              coord_lattice.s2_face_areas[(yc + 1) * nx + xr]  );
                
                //----------V1 Calculations
                dv1_deta = 0.25 * ( prims[jr * nx + ii].v1 + 
                                    prims[jr * nx + ir].v1 -
                                    prims[jl * nx + ii].v1 - 
                                    prims[jl * nx + ir].v1);
                
                dv1_dxi = prims[jj * nx + ir].v1 - prims[jj * nx + ii].v1;

                du1_dx1 = inv_vol * (si1 * dv1_dxi);
                du1_dx2 = inv_vol * (sj2 * dv1_deta);


                //----------V2 Calculations
                dv2_deta = 0.25 * ( prims[jr * nx + ii].v2 + 
                                    prims[jr * nx + ir].v2 -
                                    prims[jl * nx + ii].v2 - 
                                    prims[jl * nx + ir].v2);

                dv2_dxi = prims[jj * nx + ir].v2 - prims[jj * nx + ii].v2;

                du2_dx1 = inv_vol * (si1 * dv2_dxi); 
                du2_dx2 = inv_vol * (sj2 * dv2_deta); 
    
                h_left   = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + ii].p/(prims[jj*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                h_right  = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + ir].p/(prims[jj*nx + ir].rho * (ADIABATIC_GAMMA - 1.0));
                dh_dx1  = inv_vol * (si1 * (h_right - h_left));
                meh     = (prims[jj * nx + ir].p + prims[jj * nx + ii].p)/(prims[jj * nx + ir].rho + prims[jj * nx + ii].rho);
                mu_av  = C_AV * rhoc * h_mesh * h_mesh * std::sqrt(div_uvec * div_uvec - cterm * cterm );
                tau_11 = mu_av * (2.0 * du1_dx1 - 2.0 * (u1f*u1f*du1_dx1 + u1f * u2f * du1_dx2) - (2.0/3.0) * (div_uvec) * (1. - u1f*u1f));
                tau_21 = mu_av * ((du2_dx1 + du1_dx2) - (u2f*u1f*du1_dx1 + u2f * u2f * du1_dx2)
                            - (u1f * u1f * du2_dx1 + u1f * u2f * du2_dx2) - (2.0/3.0) * (div_uvec) * (- u1f*u2f));
                qx1    = mu_av/ PRANDTL_NUMBER * dh_dx1 * (1.0 - 4*meh);
                flux   = {0.0, tau_11, tau_21, qx1 + tau_11 * u1f + tau_21 * u2f};
                


            break;
        
        case simbi::Face::LEFT:
                u1f     = 0.5 * (prims[jj * nx + ii].v1 + prims[jj * nx + il].v1);
                u2f     = 0.5 * (prims[jj * nx + ii].v2 + prims[jj * nx + il].v2);
                volAvg  = 0.5 * (coord_lattice.dVc[yc * nx + xc] + coord_lattice.dVc[yc * nx + xl]);
                inv_vol = 1.0 / volAvg; 

                
                si1 = coord_lattice.s1_face_areas[yc * (nx + 1) + xc];
                sj2 = 0.25 * (  coord_lattice.s2_face_areas[(yc + 0) * nx + xl] + 
                                coord_lattice.s2_face_areas[(yc + 1) * nx + xl] +
                                coord_lattice.s2_face_areas[(yc + 0) * nx + xc] +
                                coord_lattice.s2_face_areas[(yc + 1) * nx + xc]  );
                
                dv1_deta = 0.25 * ( prims[jr * nx + il].v1 + 
                                    prims[jr * nx + ii].v1 -
                                    prims[jl * nx + il].v1 - 
                                    prims[jl * nx + ii].v1);
                
                dv1_dxi = prims[jj * nx + ii].v1 - prims[jj * nx + il].v1;

                du1_dx1 = inv_vol * (si1 * dv1_dxi);
                du1_dx2 = inv_vol * (sj2 * dv1_deta);

                dv2_deta = 0.25 * ( prims[jr * nx + il].v2 + 
                                    prims[jr * nx + ii].v2 -
                                    prims[jl * nx + il].v2 - 
                                    prims[jl * nx + ii].v2);

                dv2_dxi = prims[jj * nx + ii].v2 - prims[jj * nx + il].v2;

                du2_dx1 = inv_vol * (si1 * dv2_dxi); 
                du2_dx2 = inv_vol * (sj2 * dv2_deta); 

                h_left   = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + il].p/(prims[jj*nx + il].rho * (ADIABATIC_GAMMA - 1.0));
                h_right  = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + ii].p/(prims[jj*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                dh_dx1  = inv_vol * (si1 * (h_right - h_left));
                meh     = (prims[jj * nx + ii].p + prims[jj * nx + il].p)/(prims[jj * nx + ii].rho + prims[jj * nx + il].rho);
                mu_av  = C_AV * rhoc * h_mesh * h_mesh * std::sqrt(div_uvec * div_uvec - cterm * cterm );
                tau_11 = - mu_av * (2.0 * du1_dx1 + 2.0 * (u1f*u1f*du1_dx1 + u1f * u2f * du1_dx2) - (2.0/3.0) * (div_uvec) * (1 - u1f*u1f));
                tau_21 = - mu_av * ((du2_dx1 + du1_dx2) + (u2f*u1f*du1_dx1 + u2f * u2f * du1_dx2)
                            + (u1f * u1f * du2_dx1 + u1f * u2f * du2_dx2) - (2.0/3.0) * (div_uvec) * (- u2f*u1f));
                qx1    = - mu_av/ PRANDTL_NUMBER * dh_dx1 * (1.0 - 4 * meh);

                // bite = true;
                std::cout << "div_u: " << div_uvec << "\n";
                std::cout << "old div_u: " << du1_dx1 + du2_dx2 << "\n";
                std::cout << "sound cs: " << cs << "\n";
                std::cout << "hmesh: " << h_mesh << "\n";
                std::cout << "yc: " << yc << "\n";
                std::cout << "xc: " << xc << "\n";
                std::cout << "rc: " << coord_lattice.x1ccenters[xc] << "\n";
                std::cout << "yc: " << coord_lattice.x2ccenters[yc] << "\n";
                std::cout << "VOLUME_AVG: " << volAvg << "\n";
                std::cout << "mu_av:  " << mu_av << "\n";
                std::cout << "c_av:  " << cterm << "\n";
                std::cout << "New tau11: " << tau_11 << "\n";
                std::cout << "Old tau11: " << mu_av * (2.0 * du1_dx1 - (2.0/3.0) * div_uvec) << "\n";
                std::cout << "heat flux: " << qx1 << "\n";
                std::cout << "tau 21: " << tau_21 << "\n";
                std::cout << "u1: " << u1f << "\n";
                std::cout << "u2: " << u2f << "\n";
                std::cout << "dh: " << dh_dx1 << "\n";
                std::cin.get();
                flux   = {0.0, tau_11, tau_21, qx1 + tau_11 * u1f + tau_21 * u2f};
            break;
    
        case simbi::Face::UPPER:
                u1f     = 0.5 * (prims[jr * nx + ii].v1 + prims[jj * nx + ii].v1);
                u2f     = 0.5 * (prims[jr * nx + ii].v2 + prims[jj * nx + ii].v2);
                volAvg  = 0.5 * (coord_lattice.dVc[yc * nx + xc] + coord_lattice.dVc[yr * nx + xc]);
                inv_vol = 1.0 / volAvg; 

                sj2 = coord_lattice.s2_face_areas[(yc + 1) * (nx + 0) + xc];
                si1 = 0.25 * (coord_lattice.s1_face_areas[yc * (nx + 1) + xc] + 
                              coord_lattice.s1_face_areas[yc * (nx + 1) + xc + 1] +
                              coord_lattice.s1_face_areas[yr * (nx + 1) + xc] +
                              coord_lattice.s1_face_areas[yr * (nx + 1) + xc + 1]  );
                
                dv1_deta = prims[jr * nx + ii].v1 - prims[jj * nx + ii].v1;

                dv1_dxi  = 0.25 * ( prims[jj * nx + ir].v1 + 
                                    prims[jr * nx + ir].v1 -
                                    prims[jj * nx + il].v1 - 
                                    prims[jr * nx + il].v1);
                
                

                du1_dx1 = inv_vol * (si1 * dv1_dxi);
                du1_dx2 = inv_vol * (sj2 * dv1_deta);

                dv2_deta = prims[jr * nx + ii].v2 - prims[jj * nx + ii].v2;

                dv2_dxi  = 0.25 * ( prims[jj * nx + ir].v2 + 
                                    prims[jr * nx + ir].v2 -
                                    prims[jj * nx + il].v2 - 
                                    prims[jr * nx + il].v2);

                du2_dx1 = inv_vol * (si1 * dv2_dxi); 
                du2_dx2 = inv_vol * (sj2 * dv2_deta); 

                h_left   = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + ii].p/(prims[jj*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                h_right  = 1.0 + ADIABATIC_GAMMA * prims[jr * nx + ii].p/(prims[jr*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                dh_dx2  = inv_vol * (sj2 * (h_right - h_left));
                meh     = (prims[jr * nx + ii].p + prims[jj * nx + ii].p)/(prims[jr * nx + ii].rho + prims[jj * nx + ii].rho);
                mu_av = C_AV * rhoc * h_mesh * h_mesh * std::sqrt(div_uvec * div_uvec - cterm * cterm );
                tau_22 = mu_av * (2.0 * du2_dx2 - 2.0 * (u2f*u1f*du2_dx1 + u2f * u2f * du2_dx2) - (2.0/3.0) * (div_uvec) * (1.0 - u2f*u2f));
                tau_12 = mu_av * ((du2_dx1 + du1_dx2) - (u1f*u1f*du2_dx1 + u1f * u2f * du2_dx2)
                            - u2f * u1f * du1_dx1 - u2f * u2f * du1_dx2 - (2.0/3.0) * (div_uvec) * (- u1f*u2f));
                qx2    = mu_av/ PRANDTL_NUMBER * dh_dx2 * (1.0 - 4 * meh);
                flux   = {0.0, tau_12, tau_22, qx2 + tau_12 * u1f + tau_22 * u2f};

            break;
        
        case simbi::Face::LOWER:
                u1f     = 0.5 * (prims[jj * nx + ii].v1 + prims[jl * nx + ii].v1);
                u2f     = 0.5 * (prims[jj * nx + ii].v2 + prims[jl * nx + ii].v2);
                volAvg  = 0.5 * (coord_lattice.dVc[yl * nx + xc] + coord_lattice.dVc[yc * nx + xc]);
                inv_vol = 1.0 / volAvg; 

                sj2 = coord_lattice.s2_face_areas[yc * (nx + 0) + xc];
                si1 = 0.25 * (coord_lattice.s1_face_areas[yl * (nx + 1) + xc] + 
                              coord_lattice.s1_face_areas[yl * (nx + 1) + xc + 1] +
                              coord_lattice.s1_face_areas[yc * (nx + 1) + xc] +
                              coord_lattice.s1_face_areas[yc * (nx + 1) + xc + 1]  );
                
                dv1_deta = prims[jj * nx + ii].v1 - prims[jl * nx + ii].v1;

                dv1_dxi  = 0.25 * ( prims[jl * nx + ir].v1 + 
                                    prims[jj * nx + ir].v1 -
                                    prims[jl * nx + il].v1 - 
                                    prims[jj * nx + il].v1);

                du1_dx1 = inv_vol * (si1 * dv1_dxi);
                du1_dx2 = inv_vol * (sj2 * dv1_deta);

                dv2_deta = prims[jj * nx + ii].v2 - prims[jl * nx + ii].v2;

                dv2_dxi  = 0.25 * ( prims[jl * nx + ir].v2 + 
                                    prims[jj * nx + ir].v2 -
                                    prims[jl * nx + il].v2 - 
                                    prims[jj * nx + il].v2);

                du2_dx1 = inv_vol * (si1 * dv2_dxi); 
                du2_dx2 = inv_vol * (sj2 * dv2_deta); 

                h_left   = 1.0 + ADIABATIC_GAMMA * prims[jl * nx + ii].p/(prims[jl*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                h_right  = 1.0 + ADIABATIC_GAMMA * prims[jj * nx + ii].p/(prims[jj*nx + ii].rho * (ADIABATIC_GAMMA - 1.0));
                dh_dx2  = inv_vol * (sj2 * (h_right - h_left));
                meh     = (prims[jj * nx + ii].p + prims[jl * nx + ii].p)/(prims[jj * nx + ii].rho + prims[jl * nx + ii].rho);
                mu_av  = C_AV * rhoc * h_mesh * h_mesh * std::sqrt(div_uvec * div_uvec - cterm * cterm );
                tau_22 = mu_av * (2.0 * du2_dx2 - 2.0 * (u2f*u1f*du2_dx1 + u2f * u2f * du2_dx2) - (2.0/3.0) * (div_uvec) * (1.0 - u2f*u2f));
                tau_12 = mu_av * ( (du2_dx1 + du1_dx2) - (u1f*u1f*du2_dx1 + u1f * u2f * du2_dx2)
                            - u2f * u1f * du1_dx1 - u2f * u2f * du1_dx2 - (2.0/3.0) * (div_uvec) * (- u1f*u2f) );
                qx2    = mu_av/ PRANDTL_NUMBER * dh_dx2 * (1.0 - 4 * meh);
                flux   = {0.0, tau_12, tau_22, qx2 + tau_12 * u1f + tau_22 * u2f};
            break;
    }
}

void simbi::ArtificialViscosity::calc_four_velocity(
    std::vector<sr2d::Primitive> &prims
)
{
    double v1, v2, lorentz_gamma;
    for(auto &prim: prims){
        v1 = prim.v1;
        v2 = prim.v2;
        lorentz_gamma = 1.0/std::sqrt(1 - (v1*v1 + v2*v2));
        prim.v1 = lorentz_gamma * v1;
        prim.v2 = lorentz_gamma * v2;
    }
}
void simbi::ArtificialViscosity::calc_artificial_visc(
    const std::vector<sr2d::Primitive> &prims,
    const CLattice &coord_lattice
)
{
    const int nx = coord_lattice.nx1zones;
    const int ny = coord_lattice.nx2zones;

    avFlux.resize(nx * ny);
    visc_prims.reserve(prims.size());
    visc_prims = prims;
    calc_four_velocity(visc_prims);

    double dV, right_surf, left_surf, upper_surf, lower_surf;

    sr2d::Conserved favl, favr, gavl, gavr;

    double yc, xc, yr;
    bite = false;
    for (int jj = 2; jj < ny + 2; jj++)
    {
        yc = jj - 2;
        yr = yc + 1;    
        for (int ii = 2; ii < nx + 2; ii++)
        {   
            xc = ii - 2;
            dV = coord_lattice.dVc[yc * nx + xc];
            // Viscous flux in x
            /* i-1/2 face */
            av_flux_at_face(favl, visc_prims, coord_lattice, ii, jj, simbi::Face::LEFT);
            
            /* i+1/2 face */
            av_flux_at_face(favr, visc_prims, coord_lattice, ii, jj, simbi::Face::RIGHT);

            /* j-1/2 face */
            av_flux_at_face(gavl, visc_prims, coord_lattice, ii, jj, simbi::Face::LOWER);

            /* j+1/2 face */
            av_flux_at_face(gavr, visc_prims, coord_lattice, ii, jj, simbi::Face::UPPER);
            
            right_surf = coord_lattice.s1_face_areas[yc * (nx + 1) + xc + 1];
            left_surf  = coord_lattice.s1_face_areas[yc * (nx + 1) + xc + 0];
            lower_surf = coord_lattice.s2_face_areas[yc * (nx + 0) + xc + 0];
            upper_surf = coord_lattice.s2_face_areas[yr * (nx + 0) + xc + 0];

            avFlux[yc * nx + xc] = 
                sr2d::Conserved{
                    (favr * right_surf - favl * left_surf)/dV 
                    + (gavr * upper_surf - gavl * lower_surf)/dV
                    }
                ;

            // std::cout << "Flux in S1: "  << avFlux[yc * nx + xc].S1 << "\n";
            // std::cout << "Flux in S2: "  << avFlux[yc * nx + xc].S2 << "\n";
            // std::cout << "Flux in tau: " << avFlux[yc * nx + xc].tau << "\n";
            // std::cout << "\n";
            // std::cin.get();
        }
        
    }
    


}