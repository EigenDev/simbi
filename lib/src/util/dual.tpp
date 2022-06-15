#include <iostream>
namespace simbi
{
    namespace dual
    {
        template<typename T, typename C, typename U>
        void DualSpace1D<T, C, U>::copyHostToDev(const U &host, U *device)
        {
            n     = host.nx;
            nreal = host.active_zones; 

            // Precompute byes
            cbytes  = n * sizeof(C);
            pbytes  = n * sizeof(T);
            rbytes  = n * sizeof(real);
            rrbytes = nreal * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,               cbytes);
            simbi::gpu::api::gpuMalloc(&host_prims,            pbytes);
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,   rbytes);
            simbi::gpu::api::gpuMalloc(&host_source0,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceD,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes);

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(), cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data()    , pbytes);
            simbi::gpu::api::copyHostToDevice(host_pressure_guess, host.pressure_guess.data() , rbytes);
            simbi::gpu::api::copyHostToDevice(host_source0, host.source0.data() , rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceD, host.sourceD.data() , rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS, host.sourceS.data() , rrbytes);

            // copy pointer to allocated device storage to device class
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,                sizeof(C*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims), &host_prims,             sizeof(T*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_pressure_guess), &host_pressure_guess, sizeof(real*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_source0), &host_source0,     sizeof(real*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceD), &host_sourceD,     sizeof(real*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS), &host_sourceS,     sizeof(real*));
            simbi::gpu::api::copyHostToDevice(&(device->dt_min), &host_dtmin,            sizeof(real*));

            simbi::gpu::api::copyHostToDevice(&(device->inFailureState), &host.inFailureState, sizeof(bool));
            simbi::gpu::api::copyHostToDevice(&(device->dt),        &host.dt      ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),     &host.gamma   ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->cfl)  ,     &host.cfl     ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),        &host.nx      ,            sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->idx_active),&host.idx_active,          sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),&host.active_zones,      sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->decay_constant), &host.decay_constant, sizeof(real));

            // reset_dt();
        };

        template<typename T, typename C>
        void DualSpace1D<T, C, Newtonian1D>::copyHostToDev(const Newtonian1D &host, Newtonian1D *device)
        {
            n     = host.nx;
            nreal = host.active_zones; 

            // Precompute byes
            cbytes  = n * sizeof(C);
            pbytes  = n * sizeof(T);
            rbytes  = n * sizeof(real);
            rrbytes = nreal * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,               cbytes);
            simbi::gpu::api::gpuMalloc(&host_prims,            pbytes);
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes);

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(), cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data()    , pbytes);

            // copy pointer to allocated device storage to device class
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,                sizeof(C*));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims), &host_prims,             sizeof(T*));

            simbi::gpu::api::copyHostToDevice(&(device->dt_min), &host_dtmin,            sizeof(real*));
            simbi::gpu::api::copyHostToDevice(&(device->dt),        &host.dt      ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),     &host.gamma   ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->cfl)  ,     &host.cfl     ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),        &host.nx      ,            sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->idx_active),        &host.idx_active,  sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),&host.active_zones,      sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->decay_constant), &host.decay_constant, sizeof(real));
        };
        
        template<typename T, typename C, typename U>
        void DualSpace1D<T,C,U>::reset_dt()
        {
            std::vector<real> null_vec(n);
            simbi::gpu::api::copyHostToDevice(host_dtmin, null_vec.data(), rbytes);
        };

        template<typename T, typename C, typename U>
        real DualSpace1D<T,C,U>::get_dt()
        {
            std::vector<real> dts(n, 1000.0);
            simbi::gpu::api::copyDevToHost(dts.data(), host_dtmin, rbytes);
            return dts[0];
        };

        template<typename T, typename C, typename U>
        void DualSpace1D<T, C, U>::copyDevToHost(const U *device, U &host)
        {
            simbi::gpu::api::copyDevToHost(host.cons.data(),  host_u0,    cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(), host_prims, pbytes);
        };

        template<typename T, typename C>
        void DualSpace1D<T, C, Newtonian1D>::copyDevToHost(const Newtonian1D *device, Newtonian1D &host)
        {

            simbi::gpu::api::copyDevToHost(host.cons.data(),  host_u0,    cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(), host_prims, pbytes);
        };





        //======================================================
        //                      2D
        //======================================================
        template<typename T, typename C, typename U>
        void DualSpace2D<T, C, U>::copyHostToDev(
            const U &host,
            U *device
        )
        {
            luint nx     = host.nx;
            luint ny     = host.ny;
            luint nxreal = host.xphysical_grid; 
            luint nyreal = host.yphysical_grid;

            luint nzones = nx * ny;
            luint nzreal = nxreal * nyreal;

            // Precompute byes
            luint cbytes  = nzones * sizeof(C);
            luint pbytes  = nzones * sizeof(T);
            luint rbytes  = nzones * sizeof(real);

            luint rrbytes  = nzreal * sizeof(real);
            luint r1bytes  = nxreal * sizeof(real);
            luint r2bytes  = nyreal * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,              cbytes  );
            simbi::gpu::api::gpuMalloc(&host_prims,           pbytes  );
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,  rbytes  );
            simbi::gpu::api::gpuMalloc(&host_dtmin,           rrbytes );

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(), cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data()    , pbytes);
            simbi::gpu::api::copyHostToDevice(host_pressure_guess, host.pressure_guess.data() , rbytes);

            // copy pointer to allocated device storage to device class
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,    sizeof(C *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims),&host_prims, sizeof(T *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_pressure_guess),  &host_pressure_guess, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->dt_min),       &host_dtmin, sizeof(real *));

            //===================================================
            // SOURCE TERM OFF-LOADING BRANCHES
            //===================================================
            if (!host.d_all_zeros)
            {
                this->d_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceD, rrbytes);
                simbi::gpu::api::copyHostToDevice(host_sourceD, host.sourceD.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceD),  &host_sourceD,  sizeof(real *));
            } 
            if(!host.s1_all_zeros)
            {
                this->s1_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceS1, rrbytes);
                simbi::gpu::api::copyHostToDevice(host_sourceS1, host.sourceS1.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS1), &host_sourceS1, sizeof(real *));
            }
            if (!host.s2_all_zeros)
            {
                this->s2_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceS2, rrbytes );
                simbi::gpu::api::copyHostToDevice(host_sourceS2, host.sourceS2.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS2), &host_sourceS2, sizeof(real *));
            } 
            if (!host.e_all_zeros)
            {
                this->e_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_source0, rrbytes );
                simbi::gpu::api::copyHostToDevice(host_source0, host.sourceTau.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceTau),&host_source0,  sizeof(real *));
            }

            simbi::gpu::api::copyHostToDevice(&(device->dt),          &host.dt      ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->plm_theta),   &host.plm_theta,   sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),       &host.gamma   ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->cfl)  ,       &host.cfl     ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),          &host.nx      ,    sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->ny),          &host.ny      ,    sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->i_bound),     &host.i_bound,     sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->i_start),     &host.i_start,     sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->j_bound),     &host.j_bound,     sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->j_start),     &host.j_start,     sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->idx_active),  &host.idx_active,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->decay_const), &host.decay_const, sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->xphysical_grid),&host.xphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->yphysical_grid),&host.yphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),  &host.active_zones,    sizeof(int));
            
        };

        template<typename T, typename C>
        void DualSpace2D<T, C, Newtonian2D>::copyHostToDev(
            const Newtonian2D &host,
            Newtonian2D *device
        )
        {
            luint nx     = host.nx;
            luint ny     = host.ny;
            luint nxreal = host.xphysical_grid; 
            luint nyreal = host.yphysical_grid;

            luint nzones = nx * ny;
            luint nzreal = nxreal * nyreal;

            // Precompute byes
            luint cbytes  = nzones * sizeof(C);
            luint pbytes  = nzones * sizeof(T);
            luint rbytes  = nzones * sizeof(real);

            luint rrbytes  = nzreal * sizeof(real);
            luint r1bytes  = nxreal * sizeof(real);
            luint r2bytes  = nyreal * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,              cbytes  );
            simbi::gpu::api::gpuMalloc(&host_prims,           pbytes  );
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes );

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(), cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data(), pbytes);

            // copy pointer to allocated device storage to device class
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,    sizeof(C *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims),&host_prims, sizeof(T *));
            simbi::gpu::api::copyHostToDevice(&(device->dt_min),   &host_dtmin,    sizeof(real *));

            //===================================================
            // SOURCE TERM OFF-LOADING BRANCHES
            //===================================================
            if (!host.rho_all_zeros)
            {
                this->rho_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceRho, rrbytes);
                simbi::gpu::api::copyHostToDevice(host_sourceRho, host.sourceRho.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceRho),  &host_sourceRho,  sizeof(real *));
            } 
            if(!host.m1_all_zeros)
            {
                this->m1_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceM1, rrbytes);
                simbi::gpu::api::copyHostToDevice(host_sourceM1, host.sourceM1.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceM1), &host_sourceM1, sizeof(real *));
            }
            if (!host.m2_all_zeros)
            {
                this->m2_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_sourceM2, rrbytes );
                simbi::gpu::api::copyHostToDevice(host_sourceM2, host.sourceM2.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceM2), &host_sourceM2, sizeof(real *));
            } 
            if (!host.e_all_zeros)
            {
                this->e_all_zeros = false;
                simbi::gpu::api::gpuMalloc(&host_source0, rrbytes );
                simbi::gpu::api::copyHostToDevice(host_source0, host.sourceE.data() , rrbytes);
                simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceE),&host_source0,  sizeof(real *));
            }

            simbi::gpu::api::copyHostToDevice(&(device->inFailureState), &host.inFailureState, sizeof(bool));
            simbi::gpu::api::copyHostToDevice(&(device->dt),          &host.dt      ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->plm_theta),   &host.plm_theta,   sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),       &host.gamma   ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->cfl)  ,       &host.cfl     ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),          &host.nx      ,    sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->ny),          &host.ny      ,    sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->idx_active),  &host.idx_active,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->decay_const), &host.decay_const, sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->xphysical_grid),&host.xphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->yphysical_grid),&host.yphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),  &host.active_zones,    sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->hllc), &host.hllc, sizeof(bool));
            
        };

        template<typename T, typename C, typename U>
        void DualSpace2D<T,C,U>::copyDevToHost(
            const U *device,
            U &host
        )
        {
            const luint nx     = host.nx;
            const luint ny     = host.ny;
            const luint cbytes = nx * ny * sizeof(C); 
            const luint pbytes = nx * ny * sizeof(T);

            simbi::gpu::api::copyDevToHost(host.cons.data(), host_u0, cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(),   host_prims , pbytes);
            
        };

        template<typename T, typename C>
        void DualSpace2D<T,C,Newtonian2D>::copyDevToHost(
            const  Newtonian2D *device,
            Newtonian2D &host
        )
        {
            const luint nx     = host.nx;
            const luint ny     = host.ny;
            const luint cbytes = nx * ny * sizeof(C); 
            const luint pbytes = nx * ny * sizeof(T);

            simbi::gpu::api::copyDevToHost(host.cons.data(), host_u0, cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(),   host_prims , pbytes);
            
        };

        //========================================================================
        template<typename T, typename U, typename V>
        void DualSpace3D<T, U, V>::copyHostToDev(
            const V &host,
            V *device
        )
        {
            luint nx     = host.nx;
            luint ny     = host.ny;
            luint nz     = host.nz;
            luint nxreal = host.xphysical_grid; 
            luint nyreal = host.yphysical_grid;
            luint nzreal = host.zphysical_grid;

            luint nzones      = nx * ny * nz;
            luint nzones_real = nxreal * nyreal * nzreal;

            // Precompute byes
            luint cbytes  = nzones * sizeof(U);
            luint pbytes  = nzones * sizeof(T);
            luint rbytes  = nzones * sizeof(real);

            luint rrbytes  = nzones_real * sizeof(real);
            luint r1bytes  = nxreal * sizeof(real);
            luint r2bytes  = nyreal * sizeof(real);
            luint r3bytes  = nzreal * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,              cbytes);
            simbi::gpu::api::gpuMalloc(&host_prims,           pbytes);
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,  rbytes);
            simbi::gpu::api::gpuMalloc(&host_source0,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceD,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS1,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS2,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS3,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes);

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(),                     cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data(),                    pbytes);
            simbi::gpu::api::copyHostToDevice(host_pressure_guess, host.pressure_guess.data() , rbytes);
            simbi::gpu::api::copyHostToDevice(host_source0, host.sourceTau.data() ,           rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceD, host.sourceD.data() ,              rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS1, host.sourceS1.data() ,           rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS2, host.sourceS2.data() ,           rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS3, host.sourceS3.data() ,           rrbytes);

            // // copy pointer to allocated device storage to device class 
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,                        sizeof(U *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims),     &host_prims,                sizeof(T *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_pressure_guess),  &host_pressure_guess, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceTau), &host_source0,              sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceD), &host_sourceD,                sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS1), &host_sourceS1,              sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS2), &host_sourceS2,              sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS3), &host_sourceS3,              sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->dt_min), &host_dtmin,                       sizeof(real *));

            simbi::gpu::api::copyHostToDevice(&(device->inFailureState), &host.inFailureState, sizeof(bool));
            simbi::gpu::api::copyHostToDevice(&(device->dt),            &host.dt      ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->plm_theta),     &host.plm_theta,       sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),         &host.gamma   ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->cfl)  ,         &host.cfl     ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),            &host.nx      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->ny),            &host.ny      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->nz),            &host.nz      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->idx_active),    &host.idx_active,      sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->decay_const),   &host.decay_const,     sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->xphysical_grid),&host.xphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->yphysical_grid),&host.yphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->zphysical_grid),&host.zphysical_grid,  sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),  &host.active_zones,    sizeof(int));
            
        }

        template<typename T, typename U, typename V>
        void DualSpace3D<T, U, V>::copyDevToHost(
            const V*device,
            V &host
        )
        {
            const luint nx     = host.nx;
            const luint ny     = host.ny;
            const luint nz     = host.nz;
            const luint cbytes = nx * ny * nz * sizeof(U); 
            const luint pbytes = nx * ny * nz * sizeof(T);

            simbi::gpu::api::copyDevToHost(host.cons.data(), host_u0, cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(), host_prims , pbytes);
            
        }
        
        
    } // namespace dual
    
} // namespace simbi
