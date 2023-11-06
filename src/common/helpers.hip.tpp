
#include "util/parallel_for.hpp"
namespace simbi{
    namespace helpers
    {
        template<typename ... Args>
        std::string string_format( const std::string& format, Args ... args )
        {
            size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
            if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
            std::unique_ptr<char[]> buf( new char[ size ] ); 
            snprintf( buf.get(), size, format.c_str(), args ... );
            return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
        }

        template<typename T, typename U>
        typename std::enable_if<is_3D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v1   = from->v1;
            to->v2   = from->v2;
            to->v3   = from->v3;
            to->p    = from->p;
            to->chi  = from->chi;
        }

        //Handle 2D primitive arrays whether SR or Newtonian
        template<typename T, typename U>
        typename std::enable_if<is_2D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v1   = from->v1;
            to->v2   = from->v2;
            to->p    = from->p;
            to->chi  = from->chi;
        }

        template<typename T, typename U>
        typename std::enable_if<is_1D_primitive<U>::value>::type
        writeToProd(T *from, PrimData *to){
            to->rho  = from->rho;
            to->v    = from->v;
            to->p    = from->p;
        }

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_3D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.v3.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.v3.push_back(p[i].v3);
                sprims.p.push_back(p[i].p);
                sprims.chi.push_back(p[i].chi);
            }
            
            return sprims;
        }

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_2D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.p.push_back(p[i].p);
                sprims.chi.push_back(p[i].chi);
            }
            
            return sprims;
        }

        template<typename T , typename U, typename arr_type>
        typename std::enable_if<is_1D_primitive<U>::value, T>::type
        vec2struct(const arr_type &p){
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v.reserve(nzones);
            sprims.p.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v.push_back(p[i].v1);
                sprims.p.push_back(p[i].p);
            }
            
            return sprims;
        }

        template<typename Prim_type, int Ndim, typename Sim_type>
        void write_to_file(
            Sim_type &sim_state_host, 
            DataWriteMembers &setup,
            const std::string data_directory,
            const real t, 
            const real t_interval, 
            const real chkpt_interval, 
            const luint chkpt_zone_label)
        {
            sim_state_host.prims.copyFromGpu();
            sim_state_host.cons.copyFromGpu();
            setup.x1max = sim_state_host.x1max;
            setup.x1min = sim_state_host.x1min;

            PrimData prods;
            static auto step                = sim_state_host.init_chkpt_idx;
            static auto tbefore             = sim_state_host.tstart;
            static lint tchunk_order_of_mag = 2;
            const auto time_order_of_mag    = std::floor(std::log10(t));
            if (time_order_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }

            // Transform vector of primitive structs to struct of primitive vectors
            auto transfer_prims = vec2struct<Prim_type, typename Sim_type::primitive_t>(sim_state_host.prims);
            writeToProd<Prim_type, typename Sim_type::primitive_t>(&transfer_prims, &prods);
            std::string tnow;
            if (sim_state_host.dlogt != 0)
            {
                const auto time_order_of_mag = std::floor(std::log10(step));
                if (time_order_of_mag > tchunk_order_of_mag) {
                    tchunk_order_of_mag += 1;
                }
                tnow = create_step_str(step, tchunk_order_of_mag);
            } else {
                tnow = create_step_str(t_interval, tchunk_order_of_mag);
            }
            if (t_interval == INFINITY) {
                tnow = "interrupted";
            }
            const auto filename = string_format("%d.chkpt." + tnow + ".h5", chkpt_zone_label);

            setup.t             = t;
            setup.dt            = t - tbefore;
            setup.chkpt_idx     = step;
            tbefore             = t;
            step++;
            write_hdf5(data_directory, filename, prods, setup, Ndim, sim_state_host.total_zones);
        }

            
        template<typename T, typename U>
        void config_ghosts1D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int grid_size,
            const bool first_order, 
            const simbi::BoundaryCondition* boundary_conditions,
            const U *outer_zones,
            const U* inflow_zones) 
        {
            simbi::parallel_for(p, 0, 1, [=] GPU_LAMBDA (const int gid) {
                if (first_order){                
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::INFLOW:
                        cons[0] = inflow_zones[0];
                        break;
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[0]   = cons[1];
                        cons[0].momentum() *= -1;
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[0] = cons[grid_size - 2];
                    default:
                        cons[0] = cons[1];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::INFLOW:
                        cons[grid_size - 1] = inflow_zones[1];
                        break;
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[grid_size - 1]   = cons[grid_size - 2];
                        cons[grid_size - 1].momentum() *= -1;
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[grid_size - 1] = cons[1];
                    default:
                        cons[grid_size - 1] = cons[grid_size - 2];
                        break;
                    }

                    if (outer_zones) {
                        cons[grid_size - 1] = outer_zones[0];
                    }
                } else {
                    
                    switch (boundary_conditions[0])
                    {
                    case simbi::BoundaryCondition::INFLOW:
                        cons[0] = inflow_zones[0];
                        cons[1] = inflow_zones[1];
                        break;
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[0]   = cons[3];
                        cons[1]   = cons[2];
                        cons[0].momentum() *= -1;
                        cons[1].momentum() *= -1;
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[0] = cons[grid_size - 4];
                        cons[1] = cons[grid_size - 3];
                    default:
                        cons[0] = cons[2];
                        cons[1] = cons[2];
                        break;
                    }

                    switch (boundary_conditions[1])
                    {
                    case simbi::BoundaryCondition::INFLOW:
                        cons[grid_size - 1] = inflow_zones[0];
                        cons[grid_size - 2] = inflow_zones[0];
                        break;
                    case simbi::BoundaryCondition::REFLECTING:
                        cons[grid_size - 1]   = cons[grid_size - 4];
                        cons[grid_size - 2]   = cons[grid_size - 3];
                        cons[grid_size - 1].momentum() *= -1;
                        cons[grid_size - 2].momentum() *= -1;
                        break;
                    case simbi::BoundaryCondition::PERIODIC:
                        cons[grid_size - 1] = cons[3];
                        cons[grid_size - 2] = cons[2];
                    default:
                        cons[grid_size - 1] = cons[grid_size - 3];
                        cons[grid_size - 2] = cons[grid_size - 3];
                        break;
                    }

                    if (outer_zones) {
                        cons[grid_size - 1] = outer_zones[0];
                        cons[grid_size - 2] = outer_zones[0];
                    }
                }
            });
        };

        template<typename T, typename U>
        void config_ghosts2D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int x1grid_size, 
            const int x2grid_size, 
            const bool first_order,
            const simbi::Geometry geometry,
            const simbi::BoundaryCondition *boundary_conditions,
            const U *outer_zones,
            const U *boundary_zones,
            const bool half_sphere)
        {
            const int extent = p.get_full_extent();
            const int sx = (col_maj) ? 1 : x1grid_size;
            const int sy = (col_maj) ? x2grid_size : 1;
            simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
                const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x: gid % x1grid_size;
                const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y: gid / x1grid_size;
                if (first_order){
                    if(jj < x2grid_size - 2) {
                        switch (boundary_conditions[0])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(jj + 1) * sx + 0 * sy]     =  cons[(jj + 1) * sx + 1 * sy];
                            cons[(jj + 1) * sx + 0 * sy].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(jj + 1) * sx + 0 * sy] =  boundary_zones[0];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(jj + 1) * sx +  0 * sy] = cons[(jj + 1) * sx +  (x1grid_size - 2) * sy];
                            break;
                        default:
                            cons[(jj + 1) * sx +  0 * sy] = cons[(jj + 1) * sx +  1 * sy];
                            break;
                        }

                        switch (boundary_conditions[1])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(jj + 1) * sx + (x1grid_size - 1) * sy]     = cons[(jj + 1) * sx + (x1grid_size - 2) * sy];
                            cons[(jj + 1) * sx + (x1grid_size - 1) * sy].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = boundary_zones[1];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = cons[(jj + 1) * sx +  1 * sy];
                            break;
                        default:
                            cons[(jj + 1) * sx + (x1grid_size - 1) * sy] = cons[(jj + 1) * sx +  (x1grid_size - 2) * sy];
                            break;
                        }
                    }
                    // Fix the ghost zones at the x2 boundaries
                    if (ii < x1grid_size - 2) {
                        switch (geometry)
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                            cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                            if (half_sphere) {
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy].momentum(2) *= -1;
                            }
                            break;
                        case simbi::Geometry::PLANAR_CYLINDRICAL:
                            cons[0 * sx + (ii + 1) * sy]  = cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                            cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[1 * sx + (ii + 1) * sy];
                            break;
                        default:
                            switch (boundary_conditions[2])
                                {
                                case simbi::BoundaryCondition::REFLECTING:
                                    cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                    cons[0 * sx + (ii + 1) * sy].momentum(2)  *= -1;
                                    break;
                                case simbi::BoundaryCondition::INFLOW:
                                    cons[0 * sx + (ii + 1) * sy] = boundary_zones[2];
                                    break;
                                case simbi::BoundaryCondition::PERIODIC:
                                    cons[0 * sx + (ii + 1) * sy]  = cons[(x1grid_size - 2) * sx + (ii + 1) * sy];
                                    break;
                                default:
                                    cons[0 * sx + (ii + 1) * sy]  = cons[1 * sx + (ii + 1) * sy];
                                    break;
                                }

                            switch (boundary_conditions[3])
                                {
                                case simbi::BoundaryCondition::REFLECTING:
                                    cons[(x2grid_size - 1) * sx + (ii + 1) * sy]    =   cons[(x2grid_size - 2) * sx + (ii + 1) * sy];
                                    cons[(x2grid_size - 1) * sx + (ii + 1) * sy].momentum(2) *= -1;
                                    break;
                                case simbi::BoundaryCondition::INFLOW:
                                    cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = boundary_zones[3];
                                    break;
                                case simbi::BoundaryCondition::PERIODIC:
                                    cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[1 * sx + (ii + 1) * sy];
                                    break;
                                default:
                                    // Fix the ghost zones at the x1 boundaries
                                    cons[(x2grid_size - 1) * sx + (ii + 1) * sy] = cons[(x2grid_size - 2) * sx +  (ii + 1) * sy];
                                    break;
                                }
                            
                            break;
                        } // end switch
                    }
                } else {
                    if(jj < x2grid_size - 4) {
                        // Fix the ghost zones at the x1 boundaries
                        cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                        switch (boundary_conditions[0]) {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  3 * sy];
                            cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];

                            cons[(jj + 2) * sx + 0 * sy].momentum(1) *= -1;
                            cons[(jj + 2) * sx + 1 * sy].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(jj + 2) * sx +  0 * sy]   = boundary_zones[0];
                            cons[(jj + 2) * sx +  1 * sy]   = boundary_zones[0];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                            cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                            break;
                        default:
                            cons[(jj + 2) * sx +  0 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                            cons[(jj + 2) * sx +  1 * sy]   = cons[(jj + 2) * sx +  2 * sy];
                            break;
                        }

                        switch (boundary_conditions[1]) {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(jj + 2) * sx +  (x1grid_size - 1) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 4) * sy];
                            cons[(jj + 2) * sx +  (x1grid_size - 2) * sy]   = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];

                            cons[(jj + 2) * sx + (x1grid_size - 1) * sy].momentum(1) *= -1;
                            cons[(jj + 2) * sx + (x1grid_size - 2) * sy].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(jj + 2) * sx +  0 * sy]   = boundary_zones[1];
                            cons[(jj + 2) * sx +  1 * sy]   = boundary_zones[1];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  3 * sy];
                            cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  2 * sy];
                            break;
                        default:
                            cons[(jj + 2) * sx +  (x1grid_size - 1) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                            cons[(jj + 2) * sx +  (x1grid_size - 2) * sy] = cons[(jj + 2) * sx +  (x1grid_size - 3) * sy];
                            break;
                        }
                    }

                    // Fix the ghost zones at the x3 boundaries
                    if (ii < x1grid_size - 4) {
                        switch (geometry) 
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                            cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                            cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                            if (half_sphere) {
                                cons[(x2grid_size - 1) * sx + (ii + 2) * sy].momentum(2) *= -1;
                                cons[(x2grid_size - 2) * sx + (ii + 2) * sy].momentum(2) *= -1;
                            }
                            break;
                        case simbi::Geometry::PLANAR_CYLINDRICAL:
                            cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                            cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                            cons[(x2grid_size - 1) * sx + (ii + 2) * sy] = cons[2 * sx + (ii + 2) * sy];
                            cons[(x2grid_size - 2) * sx + (ii + 2) * sy] = cons[3 * sx + (ii + 2) * sy];
                            break;
                        default:
                                switch (boundary_conditions[2]) {
                                case simbi::BoundaryCondition::REFLECTING:
                                    cons[0 * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                    cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                    cons[0 * sx + (ii + 2) * sy].momentum(2)  *= -1;
                                    cons[1 * sx + (ii + 2) * sy].momentum(2)  *= -1;
                                    break;
                                case simbi::BoundaryCondition::INFLOW:
                                    cons[0 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                    cons[1 * sx +  (ii + 2) * sy] = boundary_zones[2];
                                    break;
                                case simbi::BoundaryCondition::PERIODIC:
                                    cons[0 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                    cons[1 * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                    break;
                                default:
                                    cons[0 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                    cons[1 * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                    break;
                                }

                                switch (boundary_conditions[3]) {
                                case simbi::BoundaryCondition::REFLECTING:
                                    cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 4) * sx + (ii + 2) * sy];
                                    cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                    cons[(x2grid_size - 1) * sx + (ii + 2) * sy].momentum(2)  *= -1;
                                    cons[(x2grid_size - 2) * sx + (ii + 2) * sy].momentum(2)  *= -1;
                                    break;
                                case simbi::BoundaryCondition::INFLOW:
                                    cons[0 * sx +  (ii + 2) * sy] = boundary_zones[3];
                                    cons[1 * sx +  (ii + 2) * sy] = boundary_zones[3];
                                    break;
                                case simbi::BoundaryCondition::PERIODIC:
                                    cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[3 * sx + (ii + 2) * sy];
                                    cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[2 * sx + (ii + 2) * sy];
                                    break;
                                default:
                                    cons[(x2grid_size - 1) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                    cons[(x2grid_size - 2) * sx + (ii + 2) * sy]  = cons[(x2grid_size - 3) * sx + (ii + 2) * sy];
                                    break;
                                }
                            break;
                        } // end switch
                    }
                }
            });
        };

        template<typename T, typename U>
        void config_ghosts3D(
            const ExecutionPolicy<> p,
            T *cons, 
            const int x1grid_size, 
            const int x2grid_size,
            const int x3grid_size, 
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* inflow_zones,
            const bool half_sphere,
            const simbi::Geometry geometry)
        {
            const int extent = p.get_full_extent();
            const int sx = x1grid_size;
            const int sy = x2grid_size;
            simbi::parallel_for(p, 0, extent, [=] GPU_LAMBDA (const int gid) {
                const int kk = (BuildPlatform == Platform::GPU) ? blockDim.z * blockIdx.z + threadIdx.z : simbi::helpers::get_height(gid, x1grid_size, x2grid_size);
                const int jj = (BuildPlatform == Platform::GPU) ? blockDim.y * blockIdx.y + threadIdx.y : simbi::helpers::get_row(gid, x1grid_size, x2grid_size, kk);
                const int ii = (BuildPlatform == Platform::GPU) ? blockDim.x * blockIdx.x + threadIdx.x : simbi::helpers::get_column(gid, x1grid_size, x2grid_size, kk);

                if (first_order){
                    if(jj < x2grid_size - 2 && kk < x3grid_size - 2) {
                        
                        switch (boundary_conditions[0])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = inflow_zones[0];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                            break;
                        default:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + 0] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                            break;
                        }

                        switch (boundary_conditions[1])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = inflow_zones[1];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + 1];
                            break;
                        default:
                            cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 1)] = cons[(kk + 1) * sx * sy + (jj + 1) * sx + (x1grid_size - 2)];
                            break;
                        }
                    }
                    // Fix the ghost zones at the x2 boundaries
                    if (ii < x1grid_size - 2 && kk < x3grid_size - 2) {
                        switch (geometry)
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]                 = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                            
                            if (half_sphere) {
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)].momentum(2) *= -1;
                            }
                            break;
                        case simbi::Geometry::CYLINDRICAL:
                            cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]                 = cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                            cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                            break;
                        default:
                            switch (boundary_conditions[2])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)]     = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                                cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)].momentum(2) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = inflow_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)];
                                break;
                            default:
                                cons[(kk + 1) * sx * sy + 0 * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                                break;
                            }

                            switch (boundary_conditions[3])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)]    =  cons[(kk + 1) * sx * sy + (x2grid_size - 2) * sx + (ii + 1)];
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)].momentum(2) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = inflow_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + 1 * sx + (ii + 1)];
                                break;
                            default:
                                cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)] = cons[(kk + 1) * sx * sy + (x2grid_size - 1) * sx + (ii + 1)];
                                break;
                            }
                            break;
                        }
                    }

                    // Fix the ghost zones at the x3 boundaries
                    if (jj < x2grid_size - 2 && ii < x1grid_size - 2) {
                        switch (geometry)
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)]                 = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                            cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                            break;
                        default:
                            switch (boundary_conditions[4])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)]    =   cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                                cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)].momentum(3) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = inflow_zones[4];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                                break;
                            default:
                                cons[0 * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                                break;
                            }
                            switch (boundary_conditions[5])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)]    =   cons[(x2grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                                cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)].momentum(3) *= - 1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = inflow_zones[5];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[1 * sx * sy + (jj + 1) * sx + (ii + 1)];
                                break;
                            default:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 1) * sx + (ii + 1)] = cons[(x3grid_size - 2) * sx * sy + (jj + 1) * sx + (ii + 1)];
                                break;
                            }

                            break;
                        }
                        
                    }

                } else {
                    if(jj < x2grid_size - 4 && kk < x3grid_size - 4) {
                        
                        switch (boundary_conditions[0])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 3];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0].momentum(1) *= -1;
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = inflow_zones[0];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = inflow_zones[0];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 4)];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                            break;
                        default:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 0] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + 1] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                            break;
                        }

                        switch (boundary_conditions[1])
                        {
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 4)];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)].momentum(1) *= -1;
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)].momentum(1) *= -1;
                            break;
                        case simbi::BoundaryCondition::INFLOW:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = inflow_zones[1];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = inflow_zones[1];
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 3];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + 2];
                            break;
                        default:
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 1)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                            cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 2)] = cons[(kk + 2) * sx * sy + (jj + 2) * sx + (x1grid_size - 3)];
                            break;
                        }
                    }
                    // Fix the ghost zones at the x2 boundaries
                    if (ii < x1grid_size - 4 && kk < x3grid_size - 4) {
                        switch (geometry)
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            if (half_sphere) {
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)].momentum(2) *= -1;
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)].momentum(2) *= -1;
                            }
                            break;
                        case simbi::Geometry::CYLINDRICAL:
                            cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]                 = cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)];                        
                            cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                            cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                            break;
                        default:
                            switch (boundary_conditions[2])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)]     =   cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)]     =   cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)].momentum(2)  *= -1;
                                cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)].momentum(2)  *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = inflow_zones[2];
                                cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = inflow_zones[2];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                                break;
                            default:
                                cons[(kk + 2) * sx * sy + 0 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + 2 * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + 1 * sx + (ii + 2)] = cons[(kk + 2)* sx * sy + 2 * sx + (ii + 2)];
                                break;
                            }

                            switch (boundary_conditions[3])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 4) * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)].momentum(2) *= -1;
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)].momentum(2) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = inflow_zones[3];
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = inflow_zones[3];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 3 * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)] = cons[(kk + 2) * sx * sy + 2 * sx + (ii + 2)];
                                break;
                            default:
                                cons[(kk + 2) * sx * sy + (x2grid_size - 1) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                                cons[(kk + 2) * sx * sy + (x2grid_size - 2) * sx + (ii + 2)]    =   cons[(kk + 2) * sx * sy + (x2grid_size - 3) * sx + (ii + 2)];
                                break;
                            }
                            break;
                        }
                    }

                    // Fix the ghost zones at the x3 boundaries
                    if (jj < x2grid_size - 4 && ii < x1grid_size - 4) {
                        switch (geometry)
                        {
                        case simbi::Geometry::SPHERICAL:
                            cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                            break;
                        default:
                            switch (boundary_conditions[4])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)].momentum(3) *= -1;
                                cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)].momentum(3) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[4];
                                cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[4];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                break;
                            default:
                                cons[0 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[1 * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                break;
                            }
                            switch (boundary_conditions[5])
                            {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)]    =   cons[(x3grid_size - 4) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)]    =   cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)].momentum(3) *= -1;
                                cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)].momentum(3) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[5];
                                cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = inflow_zones[5];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[3 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[2 * sx * sy + (jj + 2) * sx + (ii + 2)];
                                break;
                            default:
                                cons[(x3grid_size - 1) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                cons[(x3grid_size - 2) * sx * sy + (jj + 2) * sx + (ii + 2)] = cons[(x3grid_size - 3) * sx * sy + (jj + 2) * sx + (ii + 2)];
                                break;
                            }
                            break;
                        }
                    }
                }
            });
        };

        template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_1D_primitive<T>::value>::type 
        compute_dt(U *self, const V* prim_buffer, real* dt_min)
        {
            #if GPU_CODE
            real vPlus, vMinus;
            int ii   = blockDim.x * blockIdx.x + threadIdx.x;
            int aid  = ii + self->radius;
            if (ii < self->active_zones)
            {
                const real rho = prim_buffer[aid].rho;
                const real p   = prim_buffer[aid].p;
                const real v   = prim_buffer[aid].get_v();
            
                if constexpr(is_relativistic<T>::value)
                {
                    if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        real h   = 1 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs  = std::sqrt(self->gamma * p / (rho * h));
                        vPlus  = (v + cs) / (1 + v * cs);
                        vMinus = (v - cs) / (1 - v * cs);
                    } else {
                        vPlus  = 1;
                        vMinus = 1;
                    }
                } else {
                    const real cs = std::sqrt(self->gamma * p / rho );
                    vPlus         = (v + cs);
                    vMinus        = (v - cs);
                }
                const real x1l    = self->get_x1face(ii, 0);
                const real x1r    = self->get_x1face(ii, 1);
                const real dx1    = x1r - x1l;
                const real vfaceL = (self->geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1l * self->hubble_param;
                const real vfaceR = (self->geometry == simbi::Geometry::CARTESIAN) ? self->hubble_param : x1r * self->hubble_param;
                const real cfl_dt = dx1 / (helpers::my_max(std::abs(vPlus + vfaceR), std::abs(vMinus + vfaceL)));
                dt_min[ii]        = self->cfl * cfl_dt;
            }
            #endif
        }

        template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_2D_primitive<T>::value>::type 
        compute_dt(U *self, 
        const V* prim_buffer,
        real* dt_min,
        const simbi::Geometry geometry)
        {
            #if GPU_CODE
            real cfl_dt, v1p, v1m, v2p, v2m;
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint ia  = ii + self->idx_active;
            const luint ja  = jj + self->idx_active;
            const luint aid = (col_maj) ? ia * self-> ny + ja : ja * self->nx + ia;
            if ((ii < self->xactive_grid) && (jj < self->yactive_grid))
            {
                real plus_v1 , plus_v2 , minus_v1, minus_v2;
                const real rho  = prim_buffer[aid].rho;
                const real p    = prim_buffer[aid].p;
                const real v1   = prim_buffer[aid].get_v1();
                const real v2   = prim_buffer[aid].get_v2();

                if constexpr(is_relativistic<T>::value)
                {
                    if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        real h   = 1 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs  = std::sqrt(self->gamma * p / (rho * h));
                        plus_v1  = (v1 + cs) / (1 + v1 * cs);
                        plus_v2  = (v2 + cs) / (1 + v2 * cs);
                        minus_v1 = (v1 - cs) / (1 - v1 * cs);
                        minus_v2 = (v2 - cs) / (1 - v2 * cs);
                    } else {
                        plus_v1  = 1;
                        plus_v2  = 1;
                        minus_v1 = 1;
                        minus_v2 = 1;
                    }
                } else {
                    real cs  = std::sqrt(self->gamma * p / rho);
                    plus_v1  = (v1 + cs);
                    plus_v2  = (v2 + cs);
                    minus_v1 = (v1 - cs);
                    minus_v2 = (v2 - cs);
                }

                v1p = std::abs(plus_v1);
                v1m = std::abs(minus_v1);
                v2p = std::abs(plus_v2);
                v2m = std::abs(minus_v2);
                switch (geometry)
                {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = helpers::my_min(self->dx1 / (helpers::my_max(v1p, v1m)),
                                                self->dx2 / (helpers::my_max(v2m, v2m)));
                        break;
                    
                    case simbi::Geometry::SPHERICAL:
                    {
                        // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                        const real rl           = self->get_x1face(ii, 0);
                        const real rr           = self->get_x1face(ii, 1);
                        const real tl           = self->get_x2face(jj, 0);  
                        const real tr           = self->get_x2face(jj, 1); 
                        if (self->mesh_motion)
                        {
                            const real vfaceL   = rl * self->hubble_param;
                            const real vfaceR   = rr * self->hubble_param;
                            v1p = std::abs(plus_v1  - vfaceR);
                            v1m = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean        = 0.75 * (rr * rr * rr * rr - rl * rl * rl *rl) / (rr * rr * rr - rl * rl * rl);
                        cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                        rmean * (tr - tl) / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                    {
                        // Compute avg spherical distance 3/4 *(rf^4 - ri^4)/(rf^3 - ri^3)
                        const real rl           = self->get_x1face(ii, 0);
                        const real rr           = self->get_x1face(ii, 1);
                        const real tl           = self->get_x2face(jj, 0);  
                        const real tr           = self->get_x2face(jj, 1); 
                        if (self->mesh_motion)
                        {
                            const real vfaceL   = rl * self->hubble_param;
                            const real vfaceR   = rr * self->hubble_param;
                            v1p = std::abs(plus_v1  - vfaceR);
                            v1m = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean        = (2.0 / 3.0) * (rr * rr * rr - rl * rl * rl) / (rr * rr - rl * rl);
                        cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                        rmean * (tr - tl) / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                    case simbi::Geometry::AXIS_CYLINDRICAL:
                    {
                        const real rl           = self->get_x1face(ii, 0);
                        const real rr           = self->get_x1face(ii, 1);
                        const real zl           = self->get_x2face(jj, 0);  
                        const real zr           = self->get_x2face(jj, 1); 
                        if (self->mesh_motion)
                        {
                            const real vfaceL   = rl * self->hubble_param;
                            const real vfaceR   = rr * self->hubble_param;
                            v1p = std::abs(plus_v1  - vfaceR);
                            v1m = std::abs(minus_v1 - vfaceL);
                        }
                        cfl_dt = helpers::my_min((rr - rl) / (helpers::my_max(v1p, v1m)),
                                                (zr - zl) / (helpers::my_max(v2p, v2m)));
                        break;
                    }
                    // TODO: Implement
                } // end switch
                dt_min[jj * self->xactive_grid + ii] = self->cfl * cfl_dt;
            }
            #endif
        }

        template<typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        GPU_LAUNCHABLE  typename std::enable_if<is_3D_primitive<T>::value>::type 
        compute_dt(U *self, 
        const V* prim_buffer,
        real *dt_min,
        const simbi::Geometry geometry)
        {
            #if GPU_CODE
            real cfl_dt;
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint kk  = blockDim.z * blockIdx.z + threadIdx.z;
            const luint ia  = ii + self->idx_active;
            const luint ja  = jj + self->idx_active;
            const luint ka  = kk + self->idx_active;
            const luint aid = (col_maj) ? ia * self-> ny + ja : ka * self->nx * self->ny + ja * self->nx + ia;
            if ((ii < self->xactive_grid) && (jj < self->yactive_grid) && (kk < self->zactive_grid))
            {
                real plus_v1 , plus_v2 , minus_v1, minus_v2, plus_v3, minus_v3;

                if constexpr(is_relativistic<T>::value)
                {
                    if constexpr(dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        const real rho  = prim_buffer[aid].rho;
                        const real p    = prim_buffer[aid].p;
                        const real v1   = prim_buffer[aid].get_v1();
                        const real v2   = prim_buffer[aid].get_v2();
                        const real v3   = prim_buffer[aid].get_v3();

                        real h   = 1 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs  = std::sqrt(self->gamma * p / (rho * h));
                        plus_v1  = (v1 + cs) / (1 + v1 * cs);
                        plus_v2  = (v2 + cs) / (1 + v2 * cs);
                        plus_v3  = (v3 + cs) / (1 + v3 * cs);
                        minus_v1 = (v1 - cs) / (1 - v1 * cs);
                        minus_v2 = (v2 - cs) / (1 - v2 * cs);
                        minus_v3 = (v3 - cs) / (1 - v3 * cs);
                    } else {
                        plus_v1  = 1;
                        plus_v2  = 1;
                        plus_v3  = 1;
                        minus_v1 = 1;
                        minus_v2 = 1;
                        minus_v3 = 1;
                    }
                } else {
                    const real rho  = prim_buffer[aid].rho;
                    const real p    = prim_buffer[aid].p;
                    const real v1   = prim_buffer[aid].get_v1();
                    const real v2   = prim_buffer[aid].get_v2();
                    const real v3   = prim_buffer[aid].get_v3();
                    
                    real cs  = std::sqrt(self->gamma * p / rho);
                    plus_v1  = (v1 + cs);
                    plus_v2  = (v2 + cs);
                    plus_v3  = (v3 + cs);
                    minus_v1 = (v1 - cs);
                    minus_v2 = (v2 - cs);
                    minus_v3 = (v3 - cs);
                }

                const auto x1l = self->get_x1face(ii, 0);
                const auto x1r = self->get_x1face(ii, 1);
                const auto dx1 = x1r - x1l; 
                const auto x2l = self->get_x2face(jj, 0);
                const auto x2r = self->get_x2face(jj, 1);
                const auto dx2 = x2r - x2l; 
                const auto x3l = self->get_x3face(kk, 0);
                const auto x3r = self->get_x3face(kk, 1);
                const auto dx3 = x3r - x3l; 
                switch (geometry)
                {
                    case simbi::Geometry::CARTESIAN:
                    {

                        cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                                dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                                                dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                        break;
                    }
                    case simbi::Geometry::SPHERICAL:
                    {     
                        const real rmean = static_cast<real>(0.75) * (x1r * x1r * x1r * x1r - x1l * x1l * x1l *x1l) / (x1r * x1r * x1r - x1l * x1l * x1l);
                        const real th    = static_cast<real>(0.5) * (x2l + x2r);
                        cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                        rmean * dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                        rmean * std::sin(th) * dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                        break;
                    }
                    case simbi::Geometry::CYLINDRICAL:
                    {    
                        const real rmean = static_cast<real>(2.0 / 3.0) * (x1r * x1r * x1r - x1l * x1l * x1l) / (x1r * x1r - x1l * x1l);
                        const real th    = static_cast<real>(0.5) * (x2l + x2r);
                        cfl_dt = helpers::my_min3(dx1 / (helpers::my_max(std::abs(plus_v1), std::abs(minus_v1))),
                                        rmean * dx2 / (helpers::my_max(std::abs(plus_v2), std::abs(minus_v2))),
                                                dx3 / (helpers::my_max(std::abs(plus_v3), std::abs(minus_v3))));
                        break;
                    }
                } // end switch
                
                dt_min[kk * self->xactive_grid * self->yactive_grid + jj * self->xactive_grid + ii] = self->cfl * cfl_dt;
            }
            #endif
        }

        template<int dim, typename T>
        GPU_LAUNCHABLE void deviceReduceKernel(T *self, real *dt_min, lint nmax) {
            #if GPU_CODE
            real min = INFINITY;
            luint ii   = blockIdx.x * blockDim.x + threadIdx.x;
            luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
            luint kk   = blockIdx.z * blockDim.z + threadIdx.z;
            luint tid  = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
            luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
            luint nt   = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
            luint gid;
            if constexpr(dim == 1) {
                gid = ii;
            } else if constexpr(dim == 2) {
                gid  = self->xactive_grid * jj + ii;
            } else if constexpr(dim == 3) {
                gid  = self->yactive_grid * self->xactive_grid * kk + self->xactive_grid * jj + ii;
            }
            // reduce multiple elements per thread
            for (luint i = gid; i < nmax; i += nt) {
                min = helpers::my_min(dt_min[i], min);
            }
            min = blockReduceMin(min);
            if (tid==0) {
                dt_min[bid] = min;
                self->dt    = dt_min[0];
            }
            #endif 
        };

        template<int dim, typename T>
        GPU_LAUNCHABLE void deviceReduceWarpAtomicKernel(T *self, real *dt_min, lint nmax) {
            #if GPU_CODE
            real min = INFINITY;
            luint ii   = blockIdx.x * blockDim.x + threadIdx.x;
            luint tid  = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
            // luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
            luint nt   = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
            luint gid;
            if constexpr(dim == 1) {
                gid = ii;
            } else if constexpr(dim == 2) {
                luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
                gid  = self->xactive_grid * jj + ii;
            } else if constexpr(dim == 3) {
                luint jj   = blockIdx.y * blockDim.y + threadIdx.y;
                luint kk   = blockIdx.z * blockDim.z + threadIdx.z;
                gid  = self->yactive_grid * self->xactive_grid * kk + self->xactive_grid * jj + ii;
            }
            // reduce multiple elements per thread
            for(auto i = gid; i < nmax; i += nt) {
                min = helpers::my_min(dt_min[i], min);
            }

            min = blockReduceMin(min);
            if (tid == 0) {
                self->dt = atomicMinReal(dt_min, min);
            }
            #endif 
        };
        
        /***
         * takes a string and adds a separator character every n-th steps 
         * through the string
         * @param input input string
         * @return none
        */
        template<const unsigned num, const char separator>
        void separate(std::string & input)
        {
            for ( auto it = input.rbegin(); (num+0) <= std::distance(it, input.rend()); ++it )
            {
                std::advance(it,num);
                it = std::make_reverse_iterator(input.insert(it.base(),separator));
                std::cout << std::distance(it, input.rend()) << "\n";
            }
            std::cin.get();
        }

        template<typename T>
        int floor_or_ceil(T val) {
            constexpr T tol = 1e-16;
            if (std::abs(val) < tol) {
                return 0;
            }
            return std::floor(val);
        };
    } // namespace helpers
}

