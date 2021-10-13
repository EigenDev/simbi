
void advance(
    SRHD *s,  
    const int sh_block_size,
    const int radius, 
    const simbi::Geometry geometry)
{
    int ii  = blockDim.x * blockIdx.x + threadIdx.x;
    int txa = threadIdx.x + radius;

    extern __shared__ Primitive prim_buff[];

    const int ibound                = s->i_bound;
    const int istart                = s->i_start;
    const real decay_constant       = s->decay_constant;
    const CLattice1D *coord_lattice = &(s->coord_lattice);
    const real dt                   = s->dt;
    const real plm_theta            = s->plm_theta;
    const int nx                    = s->nx;

    Conserved u_l, u_r;
    Conserved f_l, f_r, f1, f2;
    Primitive prims_l, prims_r;
    real rmean, dV, sL, sR, pc, dx;
    if ( (ii >= s->active_zones) ){
        prim_buff[txa] = s->gpu_prims[nx - 1]; 
        return;
    }

    for(auto ii = blockDim.x * blockIdx.x + threadIdx.x; i < s->active_zones; i += gridDim.x * blockDim.x)
    {
        const int ia = ii + radius;
        const bool inbounds = ia + BLOCK_SIZE < nx - 1;
        prim_buff[txa] = s->gpu_prims[ia];

        if (threadIdx.x < radius)
        {
            prim_buff[txa - radius]     = s->gpu_prims[ia - radius];
            prim_buff[txa + BLOCK_SIZE] = inbounds ? s->gpu_prims[ia + BLOCK_SIZE] : s->gpu_prims[nx - 1];  
        }
        __syncthreads();

        if (s->first_order)
        {
            {
                if (s->periodic)
                {
                    // Set up the left and right state interfaces for i+1/2
                    prims_l = prim_buff[txa];
                    prims_r = roll(prim_buff, txa + 1, sh_block_size);
                }
                else
                {
                    // Set up the left and right state interfaces for i+1/2
                    prims_l = prim_buff[txa];
                    prims_r = prim_buff[txa + 1];
                }
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);
                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

                // Calc HLL Flux at i+1/2 interface
                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Set up the left and right state interfaces for i-1/2
                if (s->periodic)
                {
                    prims_l = roll(prim_buff, txa - 1, sh_block_size);
                    prims_r = prim_buff[txa];
                }
                else
                {
                    prims_l = prim_buff[txa - 1];
                    prims_r = prim_buff[txa];
                }

                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);

                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

                // Calc HLL Flux at i-1/2 interface
                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[ii];
                    s->gpu_cons[ia].D   += dt * ( -(f1.D - f2.D)     / dx + s->gpu_sourceD[ii] );
                    s->gpu_cons[ia].S   += dt * ( -(f1.S - f2.S)     / dx + s->gpu_sourceS[ii] );
                    s->gpu_cons[ia].tau += dt * ( -(f1.tau - f2.tau) / dx + s->gpu_source0[ii] );

                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc = prim_buff[txa].p;
                    sL = coord_lattice->gpu_face_areas[ii + 0];
                    sR = coord_lattice->gpu_face_areas[ii + 1];
                    dV = coord_lattice->gpu_dV[ii];
                    rmean = coord_lattice->gpu_x1mean[ii];

                    s->gpu_cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[ii] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + 2.0 * pc / rmean +
                        s->gpu_sourceS[ii] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[ii] * decay_constant
                    } * dt;
                    break;
                }
                
            }
        }
        else
        {
            Primitive left_most, right_most, left_mid, right_mid, center;
            // if ( (unsigned)(ii - istart) < (ibound - istart))
            {
                if (s->periodic)
                {
                    // Declare the c[i-2],c[i-1],c_i,c[i+1], c[i+2] variables
                    left_most  = roll(prim_buff, txa - 2, sh_block_size);
                    left_mid   = roll(prim_buff, txa - 1, sh_block_size);
                    center     = prim_buff[txa];
                    right_mid  = roll(prim_buff, txa + 1, sh_block_size);
                    right_most = roll(prim_buff, txa + 2, sh_block_size);
                }
                else
                {
                    left_most  = prim_buff[txa - 2];
                    left_mid   = prim_buff[txa - 1];
                    center     = prim_buff[txa + 0];
                    right_mid  = prim_buff[txa + 1];
                    right_most = prim_buff[txa + 2];
                }

                // Compute the reconstructed primitives at the i+1/2 interface

                // Reconstructed left primitives vector
                prims_l.rho =
                    center.rho + 0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                plm_theta * (right_mid.rho - center.rho));

                prims_l.v = center.v + 0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_l.p = center.p + 0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Reconstructed right primitives vector
                prims_r.rho = right_mid.rho -
                                0.5 * minmod(plm_theta * (right_mid.rho - center.rho),
                                            0.5 * (right_most.rho - center.rho),
                                            plm_theta * (right_most.rho - right_mid.rho));

                prims_r.v =
                    right_mid.v - 0.5 * minmod(plm_theta * (right_mid.v - center.v),
                                                0.5 * (right_most.v - center.v),
                                                plm_theta * (right_most.v - right_mid.v));

                prims_r.p =
                    right_mid.p - 0.5 * minmod(plm_theta * (right_mid.p - center.p),
                                                0.5 * (right_most.p - center.p),
                                                plm_theta * (right_most.p - right_mid.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);
                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

                if (s->hllc)
                {
                    f1 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f1 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                // Do the same thing, but for the right side interface [i - 1/2]
                prims_l.rho =
                    left_mid.rho + 0.5 * minmod(plm_theta * (left_mid.rho - left_most.rho),
                                                0.5 * (center.rho - left_most.rho),
                                                plm_theta * (center.rho - left_mid.rho));

                prims_l.v =
                    left_mid.v + 0.5 * minmod(plm_theta * (left_mid.v - left_most.v),
                                                0.5 * (center.v - left_most.v),
                                                plm_theta * (center.v - left_mid.v));

                prims_l.p =
                    left_mid.p + 0.5 * minmod(plm_theta * (left_mid.p - left_most.p),
                                                0.5 * (center.p - left_most.p),
                                                plm_theta * (center.p - left_mid.p));

                prims_r.rho =
                    center.rho - 0.5 * minmod(plm_theta * (center.rho - left_mid.rho),
                                                0.5 * (right_mid.rho - left_mid.rho),
                                                plm_theta * (right_mid.rho - center.rho));

                prims_r.v = center.v - 0.5 * minmod(plm_theta * (center.v - left_mid.v),
                                                    0.5 * (right_mid.v - left_mid.v),
                                                    plm_theta * (right_mid.v - center.v));

                prims_r.p = center.p - 0.5 * minmod(plm_theta * (center.p - left_mid.p),
                                                    0.5 * (right_mid.p - left_mid.p),
                                                    plm_theta * (right_mid.p - center.p));

                // Calculate the left and right states using the reconstructed PLM
                // primitives
                u_l = s->prims2cons(prims_l);
                u_r = s->prims2cons(prims_r);
                f_l = s->prims2flux(prims_l);
                f_r = s->prims2flux(prims_r);

                if (s->hllc)
                {
                    f2 = s->calc_hllc_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }
                else
                {
                    f2 = s->calc_hll_flux(prims_l, prims_r, u_l, u_r, f_l, f_r);
                }

                switch (geometry)
                {
                case simbi::Geometry::CARTESIAN:
                    dx = coord_lattice->gpu_dx1[ii];
                    s->gpu_cons[ia].D   += 0.5 * dt * ( -(f1.D - f2.D)     / dx +  s->gpu_sourceD[ii] );
                    s->gpu_cons[ia].S   += 0.5 * dt * ( -(f1.S - f2.S)     / dx +  s->gpu_sourceS[ii] );
                    s->gpu_cons[ia].tau += 0.5 * dt * ( -(f1.tau - f2.tau) / dx  + s->gpu_source0[ii] );
                    break;
                
                case simbi::Geometry::SPHERICAL:
                    pc    = prim_buff[txa].p;
                    sL    = coord_lattice->gpu_face_areas[ii + 0];
                    sR    = coord_lattice->gpu_face_areas[ii + 1];
                    dV    = coord_lattice->gpu_dV[ii];
                    rmean = coord_lattice->gpu_x1mean[ii];

                    s->gpu_cons[ia] += Conserved{ 
                        -(sR * f1.D - sL * f2.D) / dV +
                        s->gpu_sourceD[ii] * decay_constant,

                        -(sR * f1.S - sL * f2.S) / dV + 2.0 * pc / rmean +
                        s->gpu_sourceS[ii] * decay_constant,

                        -(sR * f1.tau - sL * f2.tau) / dV +
                        s->gpu_source0[ii] * decay_constant
                    } * dt * 0.5;
                    break;
                }
            }
        }

    }
    
}