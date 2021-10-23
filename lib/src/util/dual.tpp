namespace simbi
{
    namespace dual
    {
        template<typename T, typename C, typename U>
        void DualSpace1D<T, C, U>::copyHostToDev(const U &host, U *device)
        {
            int nz     = host.nx;
            int nzreal = host.active_zones; 

            // Precompute byes
            int cbytes  = nz * sizeof(C);
            int pbytes  = nz * sizeof(T);
            int rbytes  = nz * sizeof(real);


            int rrbytes = nzreal * sizeof(real);
            int fabytes = host.coord_lattice.face_areas.size() * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,               cbytes);
            simbi::gpu::api::gpuMalloc(&host_prims,            pbytes);
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,   rbytes);
            simbi::gpu::api::gpuMalloc(&host_dx1,             rrbytes);
            simbi::gpu::api::gpuMalloc(&host_dV ,             rrbytes);
            simbi::gpu::api::gpuMalloc(&host_x1m,             rrbytes);
            simbi::gpu::api::gpuMalloc(&host_fas,             fabytes);
            simbi::gpu::api::gpuMalloc(&host_source0,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceD,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes);
            simbi::gpu::api::gpuMalloc(&host_clattice, sizeof(CLattice1D));

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
            // ====================================================
            //          GEOMETRY DEEP COPY
            //=====================================================
            simbi::gpu::api::copyHostToDevice(host_dx1, host.coord_lattice.dx1.data() ,       rrbytes);
            simbi::gpu::api::copyHostToDevice(host_dV,  host.coord_lattice.dV.data(),         rrbytes);
            simbi::gpu::api::copyHostToDevice(host_fas, host.coord_lattice.face_areas.data(), fabytes);
            simbi::gpu::api::copyHostToDevice(host_x1m, host.coord_lattice.x1mean.data(),     rrbytes);

            // Now copy pointer to device directly
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx1), &host_dx1,        sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV), &host_dV,          sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x1mean),&host_x1m,      sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_face_areas), &host_fas, sizeof(real *));

            simbi::gpu::api::copyHostToDevice(&(device->dt),        &host.dt      ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),     &host.gamma   ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->CFL)  ,     &host.CFL     ,            sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),        &host.nx      ,            sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->active_zones),&host.active_zones,      sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->i_bound),   &host.i_bound,             sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->i_start),   &host.i_start,             sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->idx_shift), &host.idx_shift,           sizeof(int) );
            simbi::gpu::api::copyHostToDevice(&(device->decay_constant), &host.decay_constant, sizeof(real));
        };

        template<typename T, typename C, typename U>
        void DualSpace1D<T, C, U>::copyDevToHost(const U *device, U &host)
        {
            const int nz     = host.nx;
            const int cbytes = nz * sizeof(C); 
            const int pbytes = nz * sizeof(T); 

            simbi::gpu::api::copyDevToHost(host.cons.data(),  host_u0,    cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(), host_prims, pbytes);
        };

        template<typename T, typename C, typename U>
        void DualSpace2D<T, C, U>::copyHostToDev(
            const U &host,
            U *device
        )
        {
            int nx     = host.nx;
            int ny     = host.ny;
            int nxreal = host.xphysical_grid; 
            int nyreal = host.yphysical_grid;

            int nzones = nx * ny;
            int nzreal = nxreal * nyreal;

            // Precompute byes
            int cbytes  = nzones * sizeof(C);
            int pbytes  = nzones * sizeof(T);
            int rbytes  = nzones * sizeof(real);

            int rrbytes  = nzreal * sizeof(real);
            int r1bytes  = nxreal * sizeof(real);
            int r2bytes  = nyreal * sizeof(real);
            int fa1bytes = host.coord_lattice.x1_face_areas.size() * sizeof(real);
            int fa2bytes = host.coord_lattice.x2_face_areas.size() * sizeof(real);

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,              cbytes  );
            simbi::gpu::api::gpuMalloc(&host_prims,           pbytes  );
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,  rbytes  );
            simbi::gpu::api::gpuMalloc(&host_dx1,             r1bytes );
            simbi::gpu::api::gpuMalloc(&host_dx2,             r2bytes );
            simbi::gpu::api::gpuMalloc(&host_dV1,             r1bytes );
            simbi::gpu::api::gpuMalloc(&host_dV2,             r2bytes );
            simbi::gpu::api::gpuMalloc(&host_x1m,             r1bytes );
            simbi::gpu::api::gpuMalloc(&host_cot,             r2bytes );
            simbi::gpu::api::gpuMalloc(&host_fas1,            fa1bytes);
            simbi::gpu::api::gpuMalloc(&host_fas2,            fa2bytes);
            simbi::gpu::api::gpuMalloc(&host_source0,         rrbytes );
            simbi::gpu::api::gpuMalloc(&host_sourceD,         rrbytes );
            simbi::gpu::api::gpuMalloc(&host_sourceS1,        rrbytes );
            simbi::gpu::api::gpuMalloc(&host_sourceS2,        rrbytes );
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes );
            simbi::gpu::api::gpuMalloc(&host_clattice, sizeof(CLattice2D));

            //--------Copy the host resources to pointer variables on host
            simbi::gpu::api::copyHostToDevice(host_u0,    host.cons.data(), cbytes);
            simbi::gpu::api::copyHostToDevice(host_prims, host.prims.data()    , pbytes);
            simbi::gpu::api::copyHostToDevice(host_pressure_guess, host.pressure_guess.data() , rbytes);
            simbi::gpu::api::copyHostToDevice(host_source0, host.sourceTau.data() , rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceD, host.sourceD.data() , rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS1, host.sourceS1.data() , rrbytes);
            simbi::gpu::api::copyHostToDevice(host_sourceS2, host.sourceS2.data() , rrbytes);
            // copy pointer to allocated device storage to device class
            simbi::gpu::api::copyHostToDevice(&(device->gpu_cons), &host_u0,    sizeof(C *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_prims),&host_prims, sizeof(T *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_pressure_guess),  &host_pressure_guess, sizeof(real *));

            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceTau),&host_source0,  sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceD),  &host_sourceD,  sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS1), &host_sourceS1, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->gpu_sourceS2), &host_sourceS2, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->dt_min),       &host_dtmin,    sizeof(real *));

            // ====================================================
            //          GEOMETRY DEEP COPY
            //=====================================================
            simbi::gpu::api::copyHostToDevice(host_dx1, host.coord_lattice.dx1.data() , r1bytes);
            simbi::gpu::api::copyHostToDevice(host_dx2, host.coord_lattice.dx2.data() , r2bytes);
            // simbi::gpu::api::copyHostToDevice(host_dV1,  host.coord_lattice.dV1.data(), r1bytes);
            // simbi::gpu::api::copyHostToDevice(host_dV2,  host.coord_lattice.dV2.data(), r2bytes);
            // simbi::gpu::api::copyHostToDevice(host_fas1, host.coord_lattice.x1_face_areas.data() , fa1bytes);
            // simbi::gpu::api::copyHostToDevice(host_fas2, host.coord_lattice.x2_face_areas.data() , fa2bytes);
            simbi::gpu::api::copyHostToDevice(host_x1m, host.coord_lattice.x1mean.data(), r1bytes);
            // simbi::gpu::api::copyHostToDevice(host_cot, host.coord_lattice.cot.data(), r2bytes);

            // // Now copy pointer to device directly
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx1), &host_dx1, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx2), &host_dx2, sizeof(real *));
            // simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV1), &host_dV1, sizeof(real *));
            // simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV2), &host_dV2, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x1mean),&host_x1m, sizeof(real *));
            // simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_cot),&host_cot, sizeof(real *));
            // simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x1_face_areas), &host_fas1, sizeof(real *));
            // simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x2_face_areas), &host_fas2, sizeof(real *));

            simbi::gpu::api::copyHostToDevice(&(device->dt),          &host.dt      ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->plm_theta),   &host.plm_theta,   sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),       &host.gamma   ,    sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->CFL)  ,       &host.CFL     ,    sizeof(real));
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

        template<typename T, typename C, typename U>
        void DualSpace2D<T,C,U>::copyDevToHost(
            const U *device,
            U &host
        )
        {
            const int nx     = host.nx;
            const int ny     = host.ny;
            const int cbytes = nx * ny * sizeof(C); 
            const int pbytes = nx * ny * sizeof(T);

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
            int nx     = host.nx;
            int ny     = host.ny;
            int nz     = host.nz;
            int nxreal = host.xphysical_grid; 
            int nyreal = host.yphysical_grid;
            int nzreal = host.zphysical_grid;

            int nzones      = nx * ny * nz;
            int nzones_real = nxreal * nyreal * nzreal;

            // Precompute byes
            int cbytes  = nzones * sizeof(U);
            int pbytes  = nzones * sizeof(T);
            int rbytes  = nzones * sizeof(real);

            int rrbytes  = nzones_real * sizeof(real);
            int r1bytes  = nxreal * sizeof(real);
            int r2bytes  = nyreal * sizeof(real);
            int r3bytes  = nzreal * sizeof(real);
            int fa1bytes = host.coord_lattice.x1_face_areas.size() * sizeof(real);
            int fa2bytes = host.coord_lattice.x2_face_areas.size() * sizeof(real);
            int fa3bytes = host.coord_lattice.x3_face_areas.size() * sizeof(real);

            

            //--------Allocate the memory for pointer objects-------------------------
            simbi::gpu::api::gpuMalloc(&host_u0,              cbytes);
            simbi::gpu::api::gpuMalloc(&host_prims,           pbytes);
            simbi::gpu::api::gpuMalloc(&host_pressure_guess,  rbytes);
            simbi::gpu::api::gpuMalloc(&host_dx1,             r1bytes);
            simbi::gpu::api::gpuMalloc(&host_dx2,             r2bytes);
            simbi::gpu::api::gpuMalloc(&host_dx3,             r3bytes);
            simbi::gpu::api::gpuMalloc(&host_dV1,             r1bytes);
            simbi::gpu::api::gpuMalloc(&host_dV2,             r2bytes);
            simbi::gpu::api::gpuMalloc(&host_dV3,             r3bytes);
            simbi::gpu::api::gpuMalloc(&host_x1m,             r1bytes);
            simbi::gpu::api::gpuMalloc(&host_cot,             r2bytes);
            simbi::gpu::api::gpuMalloc(&host_sin,             r2bytes);
            simbi::gpu::api::gpuMalloc(&host_fas1,            fa1bytes);
            simbi::gpu::api::gpuMalloc(&host_fas2,            fa2bytes);
            simbi::gpu::api::gpuMalloc(&host_fas3,            fa3bytes);
            simbi::gpu::api::gpuMalloc(&host_source0,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceD,         rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS1,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS2,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_sourceS3,        rrbytes);
            simbi::gpu::api::gpuMalloc(&host_dtmin,            rbytes);
            simbi::gpu::api::gpuMalloc(&host_clattice, sizeof(CLattice3D));

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
            // ====================================================
            //          GEOMETRY DEEP COPY
            //=====================================================
            simbi::gpu::api::copyHostToDevice(host_dx1, host.coord_lattice.dx1.data() ,             r1bytes);
            simbi::gpu::api::copyHostToDevice(host_dx2, host.coord_lattice.dx2.data() ,             r2bytes);
            simbi::gpu::api::copyHostToDevice(host_dx3, host.coord_lattice.dx3.data() ,             r3bytes);
            simbi::gpu::api::copyHostToDevice(host_dV1,  host.coord_lattice.dV1.data(),             r1bytes);
            simbi::gpu::api::copyHostToDevice(host_dV2,  host.coord_lattice.dV2.data(),             r2bytes);
            simbi::gpu::api::copyHostToDevice(host_fas1, host.coord_lattice.x1_face_areas.data() , fa1bytes);
            simbi::gpu::api::copyHostToDevice(host_fas2, host.coord_lattice.x2_face_areas.data() , fa2bytes);
            simbi::gpu::api::copyHostToDevice(host_fas3, host.coord_lattice.x3_face_areas.data() , fa3bytes);
            simbi::gpu::api::copyHostToDevice(host_x1m, host.coord_lattice.x1mean.data(),           r1bytes);
            simbi::gpu::api::copyHostToDevice(host_cot, host.coord_lattice.cot.data(),              r2bytes);
            simbi::gpu::api::copyHostToDevice(host_sin, host.coord_lattice.sin.data(),              r2bytes);

            // Now copy pointer to device directly
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx1), &host_dx1, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx2), &host_dx2, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dx3), &host_dx3, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV1), &host_dV1, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV2), &host_dV2, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_dV3), &host_dV3, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x1mean),&host_x1m, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_cot),&host_cot, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_sin),&host_sin, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x1_face_areas), &host_fas1, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x2_face_areas), &host_fas2, sizeof(real *));
            simbi::gpu::api::copyHostToDevice(&(device->coord_lattice.gpu_x3_face_areas), &host_fas3, sizeof(real *));

            simbi::gpu::api::copyHostToDevice(&(device->dt),            &host.dt      ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->plm_theta),     &host.plm_theta,       sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->gamma),         &host.gamma   ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->CFL)  ,         &host.CFL     ,        sizeof(real));
            simbi::gpu::api::copyHostToDevice(&(device->nx),            &host.nx      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->ny),            &host.ny      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->nz),            &host.nz      ,        sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->i_bound),       &host.i_bound,         sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->i_start),       &host.i_start,         sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->j_bound),       &host.j_bound,         sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->j_start),       &host.j_start,         sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->k_bound),       &host.k_bound,         sizeof(int));
            simbi::gpu::api::copyHostToDevice(&(device->k_start),       &host.k_start,         sizeof(int));
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
            const int nx     = host.nx;
            const int ny     = host.ny;
            const int nz     = host.nz;
            const int cbytes = nx * ny * nz * sizeof(U); 
            const int pbytes = nx * ny * nz * sizeof(T);

            simbi::gpu::api::copyDevToHost(host.cons.data(), host_u0, cbytes);
            simbi::gpu::api::copyDevToHost(host.prims.data(), host_prims , pbytes);
            
        }
        
        
    } // namespace dual
    
} // namespace simbi
