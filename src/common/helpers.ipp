
#include "util/device_api.hpp"
#include "util/parallel_for.hpp"

namespace simbi {
    namespace helpers {
        template <typename... Args>
        std::string string_format(const std::string& format, Args... args)
        {
            size_t size = snprintf(nullptr, 0, format.c_str(), args...) +
                          1;   // Extra space for '\0'
            if (size <= 0) {
                throw std::runtime_error("Error during formatting.");
            }
            std::unique_ptr<char[]> buf(new char[size]);
            snprintf(buf.get(), size, format.c_str(), args...);
            return std::string(
                buf.get(),
                buf.get() + size - 1
            );   // We don't want the '\0' inside
        }

        template <typename T, typename U>
        typename std::enable_if<is_3D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->v2  = from->v2;
            to->v3  = from->v3;
            to->p   = from->p;
            to->chi = from->chi;
        }

        // Handle 2D primitive arrays whether SR or Newtonian
        template <typename T, typename U>
        typename std::enable_if<is_2D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->v2  = from->v2;
            to->p   = from->p;
            to->chi = from->chi;
        }

        template <typename T, typename U>
        typename std::enable_if<is_1D_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->p   = from->p;
            to->chi = from->chi;
        }

        template <typename T, typename U>
        typename std::enable_if<is_3D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->v2  = from->v2;
            to->v3  = from->v3;
            to->p   = from->p;
            to->b1  = from->b1;
            to->b2  = from->b2;
            to->b3  = from->b3;
            to->chi = from->chi;
        }

        // Handle 2D primitive arrays whether SR or Newtonian
        template <typename T, typename U>
        typename std::enable_if<is_2D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->v2  = from->v2;
            to->v3  = from->v3;
            to->p   = from->p;
            to->b1  = from->b1;
            to->b2  = from->b2;
            to->b3  = from->b3;
            to->chi = from->chi;
        }

        template <typename T, typename U>
        typename std::enable_if<is_1D_mhd_primitive<U>::value>::type
        writeToProd(T* from, PrimData* to)
        {
            to->rho = from->rho;
            to->v1  = from->v1;
            to->v2  = from->v2;
            to->v3  = from->v3;
            to->p   = from->p;
            to->b1  = from->b1;
            to->b2  = from->b2;
            to->b3  = from->b3;
            to->chi = from->chi;
        }

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_1D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.p.push_back(p[i].p);
                sprims.chi.push_back(p[i].chi);
            }

            return sprims;
        }

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_2D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
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

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_3D_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
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

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_1D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.v3.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.b1.reserve(nzones);
            sprims.b2.reserve(nzones);
            sprims.b3.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.v3.push_back(p[i].v3);
                sprims.p.push_back(p[i].p);
                sprims.b1.push_back(p[i].b1);
                sprims.b2.push_back(p[i].b2);
                sprims.b3.push_back(p[i].b3);
                sprims.chi.push_back(p[i].chi);
            }

            return sprims;
        }

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_2D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.v3.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.b1.reserve(nzones);
            sprims.b2.reserve(nzones);
            sprims.b3.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.v3.push_back(p[i].v3);
                sprims.p.push_back(p[i].p);
                sprims.b1.push_back(p[i].b1);
                sprims.b2.push_back(p[i].b2);
                sprims.b3.push_back(p[i].b3);
                sprims.chi.push_back(p[i].chi);
            }

            return sprims;
        }

        template <typename T, typename U, typename arr_type>
        typename std::enable_if<is_3D_mhd_primitive<U>::value, T>::type
        vec2struct(const arr_type& p)
        {
            T sprims;
            size_t nzones = p.size();

            sprims.rho.reserve(nzones);
            sprims.v1.reserve(nzones);
            sprims.v2.reserve(nzones);
            sprims.v3.reserve(nzones);
            sprims.p.reserve(nzones);
            sprims.b1.reserve(nzones);
            sprims.b2.reserve(nzones);
            sprims.b3.reserve(nzones);
            sprims.chi.reserve(nzones);
            for (size_t i = 0; i < nzones; i++) {
                sprims.rho.push_back(p[i].rho);
                sprims.v1.push_back(p[i].v1);
                sprims.v2.push_back(p[i].v2);
                sprims.v3.push_back(p[i].v3);
                sprims.p.push_back(p[i].p);
                sprims.b1.push_back(p[i].b1);
                sprims.b2.push_back(p[i].b2);
                sprims.b3.push_back(p[i].b3);
                sprims.chi.push_back(p[i].chi);
            }

            return sprims;
        }

        template <typename Sim_type>
        void write_to_file(Sim_type& sim_state)
        {
            sim_state.prims.copyFromGpu();
            sim_state.cons.copyFromGpu();
            sim_state.setup.x1max = sim_state.x1max;
            sim_state.setup.x1min = sim_state.x1min;

            PrimData prods;
            static auto data_directory      = sim_state.data_directory;
            static auto step                = sim_state.init_chkpt_idx;
            static auto tbefore             = sim_state.tstart;
            static lint tchunk_order_of_mag = 2;
            const auto t_interval           = [&] {
                if (sim_state.t == 0) {
                    return static_cast<real>(0.0);
                }
                else if (sim_state.dlogt != 0.0 &&
                         sim_state.init_chkpt_idx == 0) {
                    return static_cast<real>(0.0);
                }
                return sim_state.t_interval;
            }();
            const auto time_order_of_mag = std::floor(std::log10(sim_state.t));
            if (time_order_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }

            // Transform vector of primitive structs to struct of primitive
            // vectors
            auto transfer_prims = vec2struct<
                typename Sim_type::primitive_soa_t,
                typename Sim_type::primitive_t>(sim_state.prims);
            writeToProd<
                typename Sim_type::primitive_soa_t,
                typename Sim_type::primitive_t>(&transfer_prims, &prods);
            std::string tnow;
            if (sim_state.dlogt != 0) {
                const auto time_order_of_mag = std::floor(std::log10(step));
                if (time_order_of_mag > tchunk_order_of_mag) {
                    tchunk_order_of_mag += 1;
                }
                tnow = create_step_str(step, tchunk_order_of_mag);
            }
            else if (!sim_state.inFailureState) {
                tnow = create_step_str(t_interval, tchunk_order_of_mag);
            }
            else {
                if (sim_state.wasInterrupted) {
                    tnow = "interrupted";
                }
                else {
                    tnow = "crashed";
                }
            }
            const auto filename = string_format(
                "%d.chkpt." + tnow + ".h5",
                sim_state.checkpoint_zones
            );

            sim_state.setup.t         = sim_state.t;
            sim_state.setup.dt        = sim_state.t - tbefore;
            sim_state.setup.chkpt_idx = step;
            tbefore                   = sim_state.t;
            step++;
            write_hdf5(
                data_directory,
                filename,
                prods,
                sim_state.setup,
                sim_state.dimensions,
                sim_state.total_zones
            );
        }

        template <typename T, typename U>
        void config_ghosts1D(
            const ExecutionPolicy<> p,
            T* cons,
            const int grid_size,
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* outer_zones,
            const U* inflow_zones
        )
        {
            simbi::parallel_for(p, 0, 1, [=] DEV(const int gid) {
                if (first_order) {
                    switch (boundary_conditions[0]) {
                        case simbi::BoundaryCondition::INFLOW:
                            cons[0] = inflow_zones[0];
                            break;
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[0] = cons[1];
                            cons[0].momentum() *= -1;
                            break;
                        case simbi::BoundaryCondition::PERIODIC:
                            cons[0] = cons[grid_size - 2];
                        default:
                            cons[0] = cons[1];
                            break;
                    }

                    switch (boundary_conditions[1]) {
                        case simbi::BoundaryCondition::INFLOW:
                            cons[grid_size - 1] = inflow_zones[1];
                            break;
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[grid_size - 1] = cons[grid_size - 2];
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
                }
                else {

                    switch (boundary_conditions[0]) {
                        case simbi::BoundaryCondition::INFLOW:
                            cons[0] = inflow_zones[0];
                            cons[1] = inflow_zones[1];
                            break;
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[0] = cons[3];
                            cons[1] = cons[2];
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

                    switch (boundary_conditions[1]) {
                        case simbi::BoundaryCondition::INFLOW:
                            cons[grid_size - 1] = inflow_zones[0];
                            cons[grid_size - 2] = inflow_zones[0];
                            break;
                        case simbi::BoundaryCondition::REFLECTING:
                            cons[grid_size - 1] = cons[grid_size - 4];
                            cons[grid_size - 2] = cons[grid_size - 3];
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

        template <typename T, typename U>
        void config_ghosts2D(
            const ExecutionPolicy<> p,
            T* cons,
            const int x1grid_size,
            const int x2grid_size,
            const bool first_order,
            const simbi::Geometry geometry,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* outer_zones,
            const U* inflow_zones,
            const bool half_sphere
        )
        {
            const int extent = p.get_full_extent();
            const int sx     = x1grid_size;
            const int sy     = x2grid_size;
            simbi::parallel_for(p, 0, extent, [=] DEV(const int gid) {
                const int jj = axid<2, BlkAx::J>(gid, sx, sy);
                const int ii = axid<2, BlkAx::I>(gid, sx, sy);

                if (first_order) {
                    if (jj < x2grid_size - 2) {
                        const auto ing  = (jj + 1) * sx + 0;
                        const auto outg = (jj + 1) * sx + (x1grid_size - 1);
                        const auto inr  = (jj + 1) * sx + 1;
                        const auto outr = (jj + 1) * sx + (x1grid_size - 2);

                        switch (boundary_conditions[0]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[ing] = cons[inr];
                                cons[ing].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[ing] = inflow_zones[0];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[ing] = cons[outr];
                                break;
                            default:
                                cons[ing] = cons[inr];
                                break;
                        }

                        switch (boundary_conditions[1]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[outg] = cons[outr];
                                cons[outg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[outg] = inflow_zones[1];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[outg] = cons[inr];
                                break;
                            default:
                                cons[outg] = cons[outr];
                                break;
                        }

                        // if outer zones, fill them in
                        if (outer_zones) {
                            cons[outg] = outer_zones[0];
                        }
                    }
                    if (ii < x1grid_size - 2) {
                        const auto ing  = 0 * sx + (ii + 1);
                        const auto outg = (x2grid_size - 1) * sx + (ii + 1);
                        const auto inr  = 1 * sx + (ii + 1);
                        const auto outr = (x2grid_size - 2) * sx + (ii + 1);

                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]  = cons[inr];
                                cons[outg] = cons[outr];
                                if (half_sphere) {
                                    cons[outg].momentum(2) *= -1;
                                }
                                break;
                            case simbi::Geometry::CYLINDRICAL:
                                cons[ing]  = cons[outr];
                                cons[outg] = cons[inr];
                                break;
                            default:
                                switch (boundary_conditions[2]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing] = inflow_zones[2];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing] = cons[outr];
                                        break;
                                    default:
                                        cons[ing] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[3]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg] = cons[outr];
                                        cons[outg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg] = inflow_zones[3];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg] = cons[inr];
                                        break;
                                    default:
                                        cons[outg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }
                }
                else {
                    if (jj < x2grid_size - 4) {
                        const auto ing   = (jj + 2) * sx + 0;
                        const auto ingg  = (jj + 2) * sx + 1;
                        const auto outg  = (jj + 2) * sx + (x1grid_size - 1);
                        const auto outgg = (jj + 2) * sx + (x1grid_size - 2);
                        const auto inr   = (jj + 2) * sx + 2;
                        const auto inrr  = (jj + 2) * sx + 3;
                        const auto outr  = (jj + 2) * sx + (x1grid_size - 3);
                        const auto outrr = (jj + 2) * sx + (x1grid_size - 4);

                        switch (boundary_conditions[0]) {
                            case simbi::BoundaryCondition::INFLOW:
                                cons[ing]  = inflow_zones[0];
                                cons[ingg] = inflow_zones[0];
                                break;
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[ing]  = cons[inrr];
                                cons[ingg] = cons[inr];
                                cons[ing].momentum(1) *= -1;
                                cons[ingg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[ing]  = cons[outrr];
                                cons[ingg] = cons[outr];
                                break;
                            default:
                                cons[ing]  = cons[inr];
                                cons[ingg] = cons[inr];
                                break;
                        }

                        switch (boundary_conditions[1]) {
                            case simbi::BoundaryCondition::INFLOW:
                                cons[outg]  = inflow_zones[1];
                                cons[outgg] = inflow_zones[1];
                                break;
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[outg]  = cons[outr];
                                cons[outgg] = cons[outrr];
                                cons[outg].momentum(1) *= -1;
                                cons[outgg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[outg]  = cons[inrr];
                                cons[outgg] = cons[inr];
                                break;
                            default:
                                cons[outg]  = cons[outr];
                                cons[outgg] = cons[outr];
                                break;
                        }

                        // if outer zones, fill them in
                        if (outer_zones) {
                            cons[outg]  = outer_zones[0];
                            cons[outgg] = outer_zones[0];
                        }
                    }
                    if (ii < x1grid_size - 4) {
                        const auto ing   = 0 * sx + (ii + 2);
                        const auto ingg  = 1 * sx + (ii + 2);
                        const auto outg  = (x2grid_size - 1) * sx + (ii + 2);
                        const auto outgg = (x2grid_size - 2) * sx + (ii + 2);
                        const auto inr   = 2 * sx + (ii + 2);
                        const auto inrr  = 3 * sx + (ii + 2);
                        const auto outr  = (x2grid_size - 3) * sx + (ii + 2);
                        const auto outrr = (x2grid_size - 4) * sx + (ii + 2);

                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]   = cons[inrr];
                                cons[outg]  = cons[outr];
                                cons[ingg]  = cons[inr];
                                cons[outgg] = cons[outr];
                                if (half_sphere) {
                                    cons[outg].momentum(2) *= -1;
                                    cons[outgg].momentum(2) *= -1;
                                }
                                break;
                            case simbi::Geometry::CYLINDRICAL:
                                cons[ing]   = cons[outrr];
                                cons[outg]  = cons[inr];
                                cons[ingg]  = cons[outr];
                                cons[outgg] = cons[inrr];
                                break;
                            default:
                                switch (boundary_conditions[2]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(2) *= -1;
                                        cons[ingg] = cons[inrr];
                                        cons[ingg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing]  = inflow_zones[2];
                                        cons[ingg] = inflow_zones[2];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing]  = cons[outrr];
                                        cons[ingg] = cons[outr];
                                        break;
                                    default:
                                        cons[ing]  = cons[inr];
                                        cons[ingg] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[3]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outrr];
                                        cons[outg].momentum(2) *= -1;
                                        cons[outgg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg]  = inflow_zones[3];
                                        cons[outgg] = inflow_zones[3];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg]  = cons[inr];
                                        cons[outgg] = cons[inrr];
                                        break;
                                    default:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }
                }
            });
        }

        template <typename T, typename U>
        void config_ghosts3D(
            const ExecutionPolicy<> p,
            T* cons,
            const int x1grid_size,
            const int x2grid_size,
            const int x3grid_size,
            const bool first_order,
            const simbi::BoundaryCondition* boundary_conditions,
            const U* inflow_zones,
            const bool half_sphere,
            const simbi::Geometry geometry
        )
        {
            const int extent = p.get_full_extent();
            const int sx     = x1grid_size;
            const int sy     = x2grid_size;
            simbi::parallel_for(p, 0, extent, [=] DEV(const int gid) {
                const int kk = axid<3, BlkAx::K>(gid, sx, sy);
                const int jj = axid<3, BlkAx::J>(gid, sx, sy, kk);
                const int ii = axid<3, BlkAx::I>(gid, sx, sy, kk);

                if (first_order) {
                    if (jj < x2grid_size - 2 && kk < x3grid_size - 2) {
                        const auto jk  = (kk + 1) * sx * sy + (jj + 1) * sx + 0;
                        const auto ing = jk + 0;
                        const auto outg = jk + (x1grid_size - 1);
                        const auto inr  = jk + 1;
                        const auto outr = jk + (x1grid_size - 2);

                        switch (boundary_conditions[0]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[ing] = cons[inr];
                                cons[ing].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[ing] = inflow_zones[0];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[ing] = cons[outr];
                                break;
                            default:
                                cons[ing] = cons[inr];
                                break;
                        }

                        switch (boundary_conditions[1]) {
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[outg] = cons[outr];
                                cons[outg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::INFLOW:
                                cons[outg] = inflow_zones[1];
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[outg] = cons[ing];
                                break;
                            default:
                                cons[outg] = cons[outr];
                                break;
                        }
                    }
                    // Fix the ghost zones at the x2 boundaries
                    if (ii < x1grid_size - 2 && kk < x3grid_size - 2) {
                        const auto ik  = (kk + 1) * sx * sy + 0 * sx + (ii + 1);
                        const auto ing = ik;
                        const auto outg = ik + (x2grid_size - 1) * sx;
                        const auto inr  = ik + sx;
                        const auto outr = ik + (x2grid_size - 2) * sx;

                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]  = cons[inr];
                                cons[outg] = cons[outr];
                                if (half_sphere) {
                                    cons[outg].momentum(2) *= -1;
                                }
                                break;
                            case simbi::Geometry::CYLINDRICAL:
                                cons[ing]  = cons[outr];
                                cons[outg] = cons[inr];
                                break;
                            default:
                                switch (boundary_conditions[2]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing] = inflow_zones[2];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing] = cons[outr];
                                        break;
                                    default:
                                        cons[ing] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[3]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg] = cons[outr];
                                        cons[outg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg] = inflow_zones[3];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg] = cons[inr];
                                        break;
                                    default:
                                        cons[outg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }

                    // Fix the ghost zones at the x3 boundaries
                    if (jj < x2grid_size - 2 && ii < x1grid_size - 2) {
                        const auto ij  = 0 * sx * sy + (jj + 1) * sx + (ii + 1);
                        const auto ing = ij;
                        const auto outg = ij + (x3grid_size - 1) * sx * sy;
                        const auto inr  = ij + sx * sy;
                        const auto outr = ij + (x3grid_size - 2) * sx * sy;

                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]  = cons[outr];
                                cons[outg] = cons[inr];
                                break;
                            default:
                                switch (boundary_conditions[4]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(3) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing] = inflow_zones[4];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing] = cons[outr];
                                        break;
                                    default:
                                        cons[ing] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[5]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg] = cons[outr];
                                        cons[outg].momentum(3) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg] = inflow_zones[5];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg] = cons[inr];
                                        break;
                                    default:
                                        cons[outg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }
                }
                else {
                    if (jj < x2grid_size - 4 && kk < x3grid_size - 4) {
                        const auto jk  = (kk + 2) * sx * sy + (jj + 2) * sx + 0;
                        const auto ing = jk;
                        const auto ingg  = jk + 1;
                        const auto outg  = jk + (x1grid_size - 1);
                        const auto outgg = jk + (x1grid_size - 2);
                        const auto inr   = jk + 2;
                        const auto inrr  = jk + 3;
                        const auto outr  = jk + (x1grid_size - 3);
                        const auto outrr = jk + (x1grid_size - 4);

                        // fill in the two sets ghost zones for the x1
                        // boundaries at the inner edge and outer edge for
                        // second order stencil
                        switch (boundary_conditions[0]) {
                            case simbi::BoundaryCondition::INFLOW:
                                cons[ing]  = inflow_zones[0];
                                cons[ingg] = inflow_zones[0];
                                break;
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[ing]  = cons[inrr];
                                cons[ingg] = cons[inr];
                                cons[ing].momentum(1) *= -1;
                                cons[ingg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[ing]  = cons[outrr];
                                cons[ingg] = cons[outr];
                                break;
                            default:
                                cons[ing]  = cons[inr];
                                cons[ingg] = cons[inr];
                                break;
                        }

                        switch (boundary_conditions[1]) {
                            case simbi::BoundaryCondition::INFLOW:
                                cons[outg]  = inflow_zones[1];
                                cons[outgg] = inflow_zones[1];
                                break;
                            case simbi::BoundaryCondition::REFLECTING:
                                cons[outg]  = cons[outr];
                                cons[outgg] = cons[outrr];
                                cons[outg].momentum(1) *= -1;
                                cons[outgg].momentum(1) *= -1;
                                break;
                            case simbi::BoundaryCondition::PERIODIC:
                                cons[outg]  = cons[inrr];
                                cons[outgg] = cons[inr];
                                break;
                            default:
                                cons[outg]  = cons[outr];
                                cons[outgg] = cons[outr];
                                break;
                        }
                    }
                    // Fix the ghost zones at the x2 boundaries
                    if (ii < x1grid_size - 4 && kk < x3grid_size - 4) {
                        const auto ik  = (kk + 2) * sx * sy + 0 * sx + (ii + 2);
                        const auto ing = ik;
                        const auto ingg  = ik + 1 * sx;
                        const auto outg  = ik + (x2grid_size - 1) * sx;
                        const auto outgg = ik + (x2grid_size - 2) * sx;
                        const auto inr   = ik + 2 * sx;
                        const auto inrr  = ik + 3 * sx;
                        const auto outr  = ik + (x2grid_size - 3) * sx;
                        const auto outrr = ik + (x2grid_size - 4) * sx;

                        // fill in the two sets ghost zones for the x2
                        // boundaries at the inner edge and outer edge for
                        // second order stencil
                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]   = cons[inrr];
                                cons[outg]  = cons[outr];
                                cons[ingg]  = cons[inr];
                                cons[outgg] = cons[outr];
                                if (half_sphere) {
                                    cons[outg].momentum(2) *= -1;
                                    cons[outgg].momentum(2) *= -1;
                                }
                                break;
                            case simbi::Geometry::CYLINDRICAL:
                                cons[ing]   = cons[outrr];
                                cons[outg]  = cons[inr];
                                cons[ingg]  = cons[outr];
                                cons[outgg] = cons[inrr];
                                break;
                            default:
                                switch (boundary_conditions[2]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(2) *= -1;
                                        cons[ingg] = cons[inrr];
                                        cons[ingg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing]  = inflow_zones[2];
                                        cons[ingg] = inflow_zones[2];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing]  = cons[outrr];
                                        cons[ingg] = cons[outr];
                                        break;
                                    default:
                                        cons[ing]  = cons[inr];
                                        cons[ingg] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[3]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outrr];
                                        cons[outg].momentum(2) *= -1;
                                        cons[outgg].momentum(2) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg]  = inflow_zones[3];
                                        cons[outgg] = inflow_zones[3];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg]  = cons[inr];
                                        cons[outgg] = cons[inrr];
                                        break;
                                    default:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }

                    // Fix the ghost zones at the x3 boundaries
                    if (jj < x2grid_size - 4 && ii < x1grid_size - 4) {
                        const auto ij  = 0 * sx * sy + (jj + 2) * sx + (ii + 2);
                        const auto ing = ij;
                        const auto ingg  = ij + 1 * sx * sy;
                        const auto outg  = ij + (x3grid_size - 1) * sx * sy;
                        const auto outgg = ij + (x3grid_size - 2) * sx * sy;
                        const auto inr   = ij + 2 * sx * sy;
                        const auto inrr  = ij + 3 * sx * sy;
                        const auto outr  = ij + (x3grid_size - 3) * sx * sy;
                        const auto outrr = ij + (x3grid_size - 4) * sx * sy;

                        // fill in the two sets ghost zones for the x3
                        // boundaries at the inner edge and outer edge for
                        // second order stencil
                        switch (geometry) {
                            case simbi::Geometry::SPHERICAL:
                                cons[ing]   = cons[outrr];
                                cons[outg]  = cons[inr];
                                cons[ingg]  = cons[outr];
                                cons[outgg] = cons[inrr];
                                break;
                            default:
                                switch (boundary_conditions[4]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[ing]  = cons[inr];
                                        cons[ingg] = cons[inrr];
                                        cons[ing].momentum(3) *= -1;
                                        cons[ingg].momentum(3) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[ing]  = inflow_zones[4];
                                        cons[ingg] = inflow_zones[4];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[ing]  = cons[outrr];
                                        cons[ingg] = cons[outr];
                                        break;
                                    default:
                                        cons[ing]  = cons[inr];
                                        cons[ingg] = cons[inr];
                                        break;
                                }

                                switch (boundary_conditions[5]) {
                                    case simbi::BoundaryCondition::REFLECTING:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outrr];
                                        cons[outg].momentum(3) *= -1;
                                        cons[outgg].momentum(3) *= -1;
                                        break;
                                    case simbi::BoundaryCondition::INFLOW:
                                        cons[outg]  = inflow_zones[5];
                                        cons[outgg] = inflow_zones[5];
                                        break;
                                    case simbi::BoundaryCondition::PERIODIC:
                                        cons[outg]  = cons[inr];
                                        cons[outgg] = cons[inrr];
                                        break;
                                    default:
                                        cons[outg]  = cons[outr];
                                        cons[outgg] = cons[outr];
                                        break;
                                }
                                break;
                        }
                    }
                }
            });
        };

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_1D_primitive<T>::value>::type
        compute_dt(U* self, const V* prim_buffer, real* dt_min)
        {
#if GPU_CODE
            real vPlus, vMinus;
            int ii = blockDim.x * blockIdx.x + threadIdx.x;
            if (ii < self->total_zones) {
                const auto ireal =
                    helpers::get_real_idx(ii, self->radius, self->xag);
                if constexpr (is_relativistic<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        const real rho = prim_buffer[ii].rho;
                        const real p   = prim_buffer[ii].p;
                        const real v   = prim_buffer[ii].get_v();
                        real h =
                            1.0 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs = std::sqrt(self->gamma * p / (rho * h));
                        vPlus   = (v + cs) / (1.0 + v * cs);
                        vMinus  = (v - cs) / (1.0 - v * cs);
                    }
                    else {
                        vPlus  = 1.0;
                        vMinus = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[ii].rho;
                    const real p   = prim_buffer[ii].p;
                    const real v   = prim_buffer[ii].get_v();
                    const real cs  = std::sqrt(self->gamma * p / rho);
                    vPlus          = std::abs(v + cs);
                    vMinus         = std::abs(v - cs);
                }
                const real x1l = self->get_x1face(ireal, 0);
                const real x1r = self->get_x1face(ireal, 1);
                const real dx1 = x1r - x1l;
                const real vfaceL =
                    (self->geometry == simbi::Geometry::CARTESIAN)
                        ? self->hubble_param
                        : x1l * self->hubble_param;
                const real vfaceR =
                    (self->geometry == simbi::Geometry::CARTESIAN)
                        ? self->hubble_param
                        : x1r * self->hubble_param;
                const real cfl_dt = dx1 / (helpers::my_max(
                                              std::abs(vPlus + vfaceR),
                                              std::abs(vMinus + vfaceL)
                                          ));
                dt_min[ii]        = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_2D_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        )
        {
#if GPU_CODE
            real cfl_dt, v1p, v1m, v2p, v2m;
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint gid = idx2(ii, jj, self->nx, self->ny);
            if ((ii < self->nx) && (jj < self->ny)) {
                real plus_v1, plus_v2, minus_v1, minus_v2;
                if constexpr (is_relativistic<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        const real rho = prim_buffer[gid].rho;
                        const real p   = prim_buffer[gid].p;
                        const real v1  = prim_buffer[gid].get_v1();
                        const real v2  = prim_buffer[gid].get_v2();
                        real h =
                            1.0 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs  = std::sqrt(self->gamma * p / (rho * h));
                        plus_v1  = (v1 + cs) / (1.0 + v1 * cs);
                        plus_v2  = (v2 + cs) / (1.0 + v2 * cs);
                        minus_v1 = (v1 - cs) / (1.0 - v1 * cs);
                        minus_v2 = (v2 - cs) / (1.0 - v2 * cs);
                    }
                    else {
                        plus_v1  = 1.0;
                        plus_v2  = 1.0;
                        minus_v1 = 1.0;
                        minus_v2 = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[gid].rho;
                    const real p   = prim_buffer[gid].p;
                    const real v1  = prim_buffer[gid].get_v1();
                    const real v2  = prim_buffer[gid].get_v2();
                    real cs        = std::sqrt(self->gamma * p / rho);
                    plus_v1        = (v1 + cs);
                    plus_v2        = (v2 + cs);
                    minus_v1       = (v1 - cs);
                    minus_v2       = (v2 - cs);
                }

                v1p = std::abs(plus_v1);
                v1m = std::abs(minus_v1);
                v2p = std::abs(plus_v2);
                v2m = std::abs(minus_v2);
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = helpers::my_min(
                            self->dx1 / (helpers::my_max(v1p, v1m)),
                            self->dx2 / (helpers::my_max(v2m, v2m))
                        );
                        break;

                    case simbi::Geometry::SPHERICAL:
                        {
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real tl = self->get_x2face(jreal, 0);
                            const real tr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            const real rmean =
                                0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) /
                                (rr * rr * rr - rl * rl * rl);
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                rmean * (tr - tl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        {
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real tl = self->get_x2face(jreal, 0);
                            const real tr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            const real rmean = (2.0 / 3.0) *
                                               (rr * rr * rr - rl * rl * rl) /
                                               (rr * rr - rl * rl);
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                rmean * (tr - tl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                    case simbi::Geometry::AXIS_CYLINDRICAL:
                        {
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real zl = self->get_x2face(jreal, 0);
                            const real zr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                (zr - zl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                        // TODO: Implement
                }   // end switch
                dt_min[gid] = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_3D_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        )
        {
#if GPU_CODE
            real cfl_dt;
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint kk  = blockDim.z * blockIdx.z + threadIdx.z;
            const luint gid = idx3(ii, jj, kk, self->nx, self->ny, self->nz);
            if ((ii < self->nx) && (jj < self->ny) && (kk < self->nz)) {
                real plus_v1, plus_v2, minus_v1, minus_v2, plus_v3, minus_v3;

                if constexpr (is_relativistic<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        const real rho = prim_buffer[gid].rho;
                        const real p   = prim_buffer[gid].p;
                        const real v1  = prim_buffer[gid].get_v1();
                        const real v2  = prim_buffer[gid].get_v2();
                        const real v3  = prim_buffer[gid].get_v3();

                        real h =
                            1.0 + self->gamma * p / (rho * (self->gamma - 1));
                        real cs  = std::sqrt(self->gamma * p / (rho * h));
                        plus_v1  = (v1 + cs) / (1.0 + v1 * cs);
                        plus_v2  = (v2 + cs) / (1.0 + v2 * cs);
                        plus_v3  = (v3 + cs) / (1.0 + v3 * cs);
                        minus_v1 = (v1 - cs) / (1.0 - v1 * cs);
                        minus_v2 = (v2 - cs) / (1.0 - v2 * cs);
                        minus_v3 = (v3 - cs) / (1.0 - v3 * cs);
                    }
                    else {
                        plus_v1  = 1.0;
                        plus_v2  = 1.0;
                        plus_v3  = 1.0;
                        minus_v1 = 1.0;
                        minus_v2 = 1.0;
                        minus_v3 = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[gid].rho;
                    const real p   = prim_buffer[gid].p;
                    const real v1  = prim_buffer[gid].get_v1();
                    const real v2  = prim_buffer[gid].get_v2();
                    const real v3  = prim_buffer[gid].get_v3();

                    real cs  = std::sqrt(self->gamma * p / rho);
                    plus_v1  = (v1 + cs);
                    plus_v2  = (v2 + cs);
                    plus_v3  = (v3 + cs);
                    minus_v1 = (v1 - cs);
                    minus_v2 = (v2 - cs);
                    minus_v3 = (v3 - cs);
                }

                const auto ireal =
                    helpers::get_real_idx(ii, self->radius, self->xag);
                const auto jreal =
                    helpers::get_real_idx(jj, self->radius, self->yag);
                const auto kreal =
                    helpers::get_real_idx(kk, self->radius, self->zag);
                const auto x1l = self->get_x1face(ireal, 0);
                const auto x1r = self->get_x1face(ireal, 1);
                const auto dx1 = x1r - x1l;
                const auto x2l = self->get_x2face(jreal, 0);
                const auto x2r = self->get_x2face(jreal, 1);
                const auto dx2 = x2r - x2l;
                const auto x3l = self->get_x3face(kreal, 0);
                const auto x3r = self->get_x3face(kreal, 1);
                const auto dx3 = x3r - x3l;
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        {

                            cfl_dt = helpers::my_min3(
                                dx1 / (helpers::my_max(
                                          std::abs(plus_v1),
                                          std::abs(minus_v1)
                                      )),
                                dx2 / (helpers::my_max(
                                          std::abs(plus_v2),
                                          std::abs(minus_v2)
                                      )),
                                dx3 / (helpers::my_max(
                                          std::abs(plus_v3),
                                          std::abs(minus_v3)
                                      ))
                            );
                            break;
                        }
                    case simbi::Geometry::SPHERICAL:
                        {
                            const real rmean =
                                0.75 *
                                (x1r * x1r * x1r * x1r - x1l * x1l * x1l * x1l
                                ) /
                                (x1r * x1r * x1r - x1l * x1l * x1l);
                            const real th = 0.5 * (x2l + x2r);
                            cfl_dt        = helpers::my_min3(
                                dx1 / (helpers::my_max(
                                          std::abs(plus_v1),
                                          std::abs(minus_v1)
                                      )),
                                rmean * dx2 /
                                    (helpers::my_max(
                                        std::abs(plus_v2),
                                        std::abs(minus_v2)
                                    )),
                                rmean * std::sin(th) * dx3 /
                                    (helpers::my_max(
                                        std::abs(plus_v3),
                                        std::abs(minus_v3)
                                    ))
                            );
                            break;
                        }
                    case simbi::Geometry::CYLINDRICAL:
                        {
                            const real rmean =
                                (2.0 / 3.0) *
                                (x1r * x1r * x1r - x1l * x1l * x1l) /
                                (x1r * x1r - x1l * x1l);
                            const real th = 0.5 * (x2l + x2r);
                            cfl_dt        = helpers::my_min3(
                                dx1 / (helpers::my_max(
                                          std::abs(plus_v1),
                                          std::abs(minus_v1)
                                      )),
                                rmean * dx2 /
                                    (helpers::my_max(
                                        std::abs(plus_v2),
                                        std::abs(minus_v2)
                                    )),
                                dx3 / (helpers::my_max(
                                          std::abs(plus_v3),
                                          std::abs(minus_v3)
                                      ))
                            );
                            break;
                        }
                }   // end switch

                dt_min[gid] = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_1D_mhd_primitive<T>::value>::type
        compute_dt(U* self, const V* prim_buffer, real* dt_min)
        {
#if GPU_CODE
            real vPlus, vMinus;
            int ii  = blockDim.x * blockIdx.x + threadIdx.x;
            int gid = ii;
            if (ii < self->total_zones) {
                if constexpr (is_relativistic_mhd<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        real cs, speeds[4];
                        self->calc_max_wave_speeds(
                            prim_buffer[gid],
                            1,
                            speeds,
                            cs
                        );
                        vPlus  = std::abs(speeds[3]);
                        vMinus = std::abs(speeds[0]);
                    }
                    else {
                        vPlus  = 1.0;
                        vMinus = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[gid].rho;
                    const real p   = prim_buffer[gid].p;
                    const real v   = prim_buffer[gid].get_v1();
                    const real cs  = std::sqrt(self->gamma * p / rho);
                    vPlus          = (v + cs);
                    vMinus         = (v - cs);
                }
                const auto ireal =
                    helpers::get_real_idx(ii, self->radius, self->xag);
                const real x1l = self->get_x1face(ireal, 0);
                const real x1r = self->get_x1face(ireal, 1);
                const real dx1 = x1r - x1l;
                const real vfaceL =
                    (self->geometry == simbi::Geometry::CARTESIAN)
                        ? self->hubble_param
                        : x1l * self->hubble_param;
                const real vfaceR =
                    (self->geometry == simbi::Geometry::CARTESIAN)
                        ? self->hubble_param
                        : x1r * self->hubble_param;
                const real cfl_dt = dx1 / (helpers::my_max(
                                              std::abs(vPlus + vfaceR),
                                              std::abs(vMinus + vfaceL)
                                          ));
                dt_min[ii]        = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_2D_mhd_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        )
        {
#if GPU_CODE
            real cfl_dt, v1p, v1m, v2p, v2m;
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint gid = idx2(ii, jj, self->nx, self->ny);
            if ((ii < self->nx) && (jj < self->ny)) {
                real plus_v1, plus_v2, minus_v1, minus_v2;
                if constexpr (is_relativistic_mhd<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        real cs, speeds[4];
                        self->calc_max_wave_speeds(
                            prim_buffer[gid],
                            1,
                            speeds,
                            cs
                        );
                        plus_v1  = std::abs(speeds[3]);
                        minus_v1 = std::abs(speeds[0]);
                        self->calc_max_wave_speeds(
                            prim_buffer[gid],
                            2,
                            speeds,
                            cs
                        );
                        plus_v2  = std::abs(speeds[3]);
                        minus_v2 = std::abs(speeds[0]);
                    }
                    else {
                        plus_v1  = 1.0;
                        plus_v2  = 1.0;
                        minus_v1 = 1.0;
                        minus_v2 = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[gid].rho;
                    const real p   = prim_buffer[gid].p;
                    const real v1  = prim_buffer[gid].get_v1();
                    const real v2  = prim_buffer[gid].get_v2();
                    real cs        = std::sqrt(self->gamma * p / rho);
                    plus_v1        = (v1 + cs);
                    plus_v2        = (v2 + cs);
                    minus_v1       = (v1 - cs);
                    minus_v2       = (v2 - cs);
                }

                v1p = std::abs(plus_v1);
                v1m = std::abs(minus_v1);
                v2p = std::abs(plus_v2);
                v2m = std::abs(minus_v2);
                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = helpers::my_min(
                            self->dx1 / (helpers::my_max(v1p, v1m)),
                            self->dx2 / (helpers::my_max(v2m, v2m))
                        );
                        break;

                    case simbi::Geometry::SPHERICAL:
                        {
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real tl = self->get_x2face(jreal, 0);
                            const real tr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            const real rmean =
                                0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) /
                                (rr * rr * rr - rl * rl * rl);
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                rmean * (tr - tl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                    case simbi::Geometry::PLANAR_CYLINDRICAL:
                        {
                            // Compute avg spherical distance 3/4 *(rf^4 -
                            // ri^4)/(rf^3 - ri^3)
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real tl = self->get_x2face(jreal, 0);
                            const real tr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            const real rmean = (2.0 / 3.0) *
                                               (rr * rr * rr - rl * rl * rl) /
                                               (rr * rr - rl * rl);
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                rmean * (tr - tl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                    case simbi::Geometry::AXIS_CYLINDRICAL:
                        {
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->xag
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->yag
                            );
                            const real rl = self->get_x1face(ireal, 0);
                            const real rr = self->get_x1face(ireal, 1);
                            const real zl = self->get_x2face(jreal, 0);
                            const real zr = self->get_x2face(jreal, 1);
                            if (self->mesh_motion) {
                                const real vfaceL = rl * self->hubble_param;
                                const real vfaceR = rr * self->hubble_param;
                                v1p               = std::abs(plus_v1 - vfaceR);
                                v1m               = std::abs(minus_v1 - vfaceL);
                            }
                            cfl_dt = helpers::my_min(
                                (rr - rl) / (helpers::my_max(v1p, v1m)),
                                (zr - zl) / (helpers::my_max(v2p, v2m))
                            );
                            break;
                        }
                        // TODO: Implement
                }   // end switch
                dt_min[gid] = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_3D_mhd_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const simbi::Geometry geometry
        )
        {
#if GPU_CODE
            const luint ii  = blockDim.x * blockIdx.x + threadIdx.x;
            const luint jj  = blockDim.y * blockIdx.y + threadIdx.y;
            const luint kk  = blockDim.z * blockIdx.z + threadIdx.z;
            const luint ia  = ii + self->radius;
            const luint ja  = jj + self->radius;
            const luint ka  = kk + self->radius;
            const luint gid = idx3(ii, jj, kk, self->nx, self->ny, self->nz);
            const luint aid = idx3(ia, ja, ka, self->nx, self->ny, self->nz);

            if ((ii < self->nx) && (jj < self->ny) && (kk < self->nz)) {
                real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt, cs;
                real speeds[4];
                const auto prims = prim_buffer;
                // Left/Right wave speeds
                if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                    self->calc_max_wave_speeds(prims[aid], 1, speeds, cs);
                    v1p = std::abs(speeds[3]);
                    v1m = std::abs(speeds[0]);
                    self->calc_max_wave_speeds(prims[aid], 2, speeds, cs);
                    v2p = std::abs(speeds[3]);
                    v2m = std::abs(speeds[0]);
                    self->calc_max_wave_speeds(prims[aid], 3, speeds, cs);
                    v3p = std::abs(speeds[3]);
                    v3m = std::abs(speeds[0]);
                }
                else {
                    v1p = 1.0;
                    v1m = 1.0;
                    v2p = 1.0;
                    v2m = 1.0;
                    v3p = 1.0;
                    v3m = 1.0;
                }

                switch (geometry) {
                    case simbi::Geometry::CARTESIAN:
                        cfl_dt = std ::min(
                            {self->dx1 / (my_max(v1p, v1m)),
                             self->dx2 / (my_max(v2p, v2m)),
                             self->dx3 / (my_max(v3p, v3m))}
                        );

                        break;
                    case simbi::Geometry::SPHERICAL:
                        {
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->nxv
                            );
                            const auto jreal = helpers::get_real_idx(
                                jj,
                                self->radius,
                                self->nyv
                            );

                            const real x1l = self->get_x1face(ireal, 0);
                            const real x1r = self->get_x1face(ireal, 1);
                            const real dx1 = x1r - x1l;

                            const real x2l   = self->get_x2face(jreal, 0);
                            const real x2r   = self->get_x2face(jreal, 1);
                            const real rmean = get_cell_centroid(
                                x1r,
                                x1l,
                                simbi::Geometry::SPHERICAL
                            );
                            const real th    = 0.5 * (x2r + x2l);
                            const real rproj = rmean * std::sin(th);
                            cfl_dt           = std::min(
                                {dx1 / (my_max(v1p, v1m)),
                                           rmean * self->dx2 / (my_max(v2p, v2m)),
                                           rproj * self->dx3 / (my_max(v3p, v3m))}
                            );
                            break;
                        }
                    default:
                        {
                            const auto ireal = helpers::get_real_idx(
                                ii,
                                self->radius,
                                self->nxv
                            );
                            const real x1l = self->get_x1face(ireal, 0);
                            const real x1r = self->get_x1face(ireal, 1);
                            const real dx1 = x1r - x1l;

                            const real rmean = get_cell_centroid(
                                x1r,
                                x1l,
                                simbi::Geometry::CYLINDRICAL
                            );
                            cfl_dt = std::min(
                                {dx1 / (my_max(v1p, v1m)),
                                 rmean * self->dx2 / (my_max(v2p, v2m)),
                                 self->dx3 / (my_max(v3p, v3m))}
                            );
                            break;
                        }
                }

                dt_min[gid] = self->cfl * cfl_dt;
            }
#endif
        }

        template <int dim, typename T>
        KERNEL void deviceReduceKernel(T* self, real* dt_min, lint nmax)
        {
#if GPU_CODE
            real min  = INFINITY;
            luint ii  = blockIdx.x * blockDim.x + threadIdx.x;
            luint jj  = blockIdx.y * blockDim.y + threadIdx.y;
            luint kk  = blockIdx.z * blockDim.z + threadIdx.z;
            luint tid = threadIdx.z * blockDim.x * blockDim.y +
                        threadIdx.y * blockDim.x + threadIdx.x;
            luint bid = blockIdx.z * gridDim.x * gridDim.y +
                        blockIdx.y * gridDim.x + blockIdx.x;
            luint nt = blockDim.x * blockDim.y * blockDim.z * gridDim.x *
                       gridDim.y * gridDim.z;
            luint gid;
            if constexpr (dim == 1) {
                gid = ii;
            }
            else if constexpr (dim == 2) {
                gid = self->nx * jj + ii;
            }
            else if constexpr (dim == 3) {
                gid = self->ny * self->nx * kk + self->nx * jj + ii;
            }
            // reduce multiple elements per thread
            for (luint i = gid; i < nmax; i += nt) {
                min = helpers::my_min(dt_min[i], min);
            }
            min = blockReduceMin(min);
            if (tid == 0) {
                dt_min[bid] = min;
                self->dt    = dt_min[0];
            }
#endif
        };

        template <int dim, typename T>
        KERNEL void
        deviceReduceWarpAtomicKernel(T* self, real* dt_min, lint nmax)
        {
#if GPU_CODE
            real min        = INFINITY;
            const luint ii  = blockIdx.x * blockDim.x + threadIdx.x;
            const luint tid = threadIdx.z * blockDim.x * blockDim.y +
                              threadIdx.y * blockDim.x + threadIdx.x;
            // luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y *
            // gridDim.x + blockIdx.x;
            const luint nt = blockDim.x * blockDim.y * blockDim.z * gridDim.x *
                             gridDim.y * gridDim.z;
            const luint gid = [&] {
                if constexpr (dim == 1) {
                    return ii;
                }
                else if constexpr (dim == 2) {
                    luint jj = blockIdx.y * blockDim.y + threadIdx.y;
                    return self->nx * jj + ii;
                }
                else if constexpr (dim == 3) {
                    luint jj = blockIdx.y * blockDim.y + threadIdx.y;
                    luint kk = blockIdx.z * blockDim.z + threadIdx.z;
                    return self->ny * self->nx * kk + self->nx * jj + ii;
                }
            }();
            // reduce multiple elements per thread
            for (auto i = gid; i < nmax; i += nt) {
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
        template <const unsigned num, const char separator>
        void separate(std::string& input)
        {
            for (auto it = input.rbegin() + 1;
                 (num + 0) <= std::distance(it, input.rend());
                 ++it) {
                std::advance(it, num - 1);
                it = std::make_reverse_iterator(
                    input.insert(it.base(), separator)
                );
            }
        }

        template <typename T>
        DUAL T cubic(T b, T c, T d)
        {
            T p = c - b * b / 3.0;
            T q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;

            if (p == 0.0) {
                return std::pow(q, 1.0 / 3.0);
            }
            if (q == 0.0) {
                return 0.0;
            }

            T t = std::sqrt(std::abs(p) / 3.0);
            T g = 1.5 * q / (p * t);
            if (p > 0.0) {
                return -2.0 * t * std::sinh(std::asinh(g) / 3.0) - b / 3.0;
            }

            if (4.0 * p * p * p + 27.0 * q * q < 0.0) {
                return 2.0 * t * std::cos(std::acos(g) / 3.0) - b / 3.0;
            }

            if (q > 0.0) {
                return -2.0 * t * std::cosh(std::acosh(-g) / 3.0) - b / 3.0;
            }

            return 2.0 * t * std::cosh(std::acosh(g) / 3.0) - b / 3.0;
        }

        /*--------------------------------------------
            quartic solver adapted from:
            https://stackoverflow.com/a/50747781/13874039
        --------------------------------------------*/
        template <typename T>
        DUAL int quartic(T b, T c, T d, T e, T res[4])
        {
            T p = c - 0.375 * b * b;
            T q = 0.125 * b * b * b - 0.5 * b * c + d;
            T m = cubic<real>(
                p,
                0.25 * p * p + 0.01171875 * b * b * b * b - e + 0.25 * b * d -
                    0.0625 * b * b * c,
                -0.125 * q * q
            );
            if (q == 0.0) {
                if (m < 0.0) {
                    return 0;
                };

                int nroots = 0;
                T sqrt_2m  = std::sqrt(2.0 * m);
                if (-m - p > 0.0) {
                    T delta       = std::sqrt(2.0 * (-m - p));
                    res[nroots++] = -0.25 * b + 0.5 * (sqrt_2m - delta);
                    res[nroots++] = -0.25 * b - 0.5 * (sqrt_2m - delta);
                    res[nroots++] = -0.25 * b + 0.5 * (sqrt_2m + delta);
                    res[nroots++] = -0.25 * b - 0.5 * (sqrt_2m + delta);
                }

                if (-m - p == 0.0) {
                    res[nroots++] = -0.25 * b - 0.5 * sqrt_2m;
                    res[nroots++] = -0.25 * b + 0.5 * sqrt_2m;
                }

                return nroots;
            }

            if (m < 0.0) {
                return 0;
            };
            T sqrt_2m  = std::sqrt(2.0 * m);
            int nroots = 0;
            if (-m - p + q / sqrt_2m >= 0.0) {
                T delta       = std::sqrt(2.0 * (-m - p + q / sqrt_2m));
                res[nroots++] = 0.5 * (-sqrt_2m + delta) - 0.25 * b;
                res[nroots++] = 0.5 * (-sqrt_2m - delta) - 0.25 * b;
            }

            if (-m - p - q / sqrt_2m >= 0.0) {
                T delta       = std::sqrt(2.0 * (-m - p - q / sqrt_2m));
                res[nroots++] = 0.5 * (sqrt_2m + delta) - 0.25 * b;
                res[nroots++] = 0.5 * (sqrt_2m - delta) - 0.25 * b;
            }

            // printf(
            //     "roots are: r1: %.2f, r2: %.2f, r3: %.2f, r4: %.2f\n",
            //     res[0],
            //     res[1],
            //     res[2],
            //     res[3]
            // );
            if constexpr (global::BuildPlatform == global::Platform::GPU) {
                iterativeQuickSort(res, 0, 3);
            }
            else {
                recursiveQuickSort(res, 0, nroots - 1);
            }
            return nroots;
        }

        // Function to swap two elements
        template <typename T>
        DUAL void myswap(T& a, T& b)
        {
            T temp = a;
            a      = b;
            b      = temp;
        }

        // Partition the array and return the pivot index
        template <typename T, typename index_type>
        DUAL index_type partition(T arr[], index_type low, index_type high)
        {
            T pivot = arr[high];   // Choose the rightmost element as the pivot
            index_type i = low - 1;   // Index of the smaller element

            for (index_type j = low; j <= high - 1; j++) {
                if (arr[j] <= pivot) {
                    i++;
                    myswap(arr[i], arr[j]);
                }
            }
            myswap(arr[i + 1], arr[high]);
            return i + 1;   // Return the pivot index
        }

        // Quick sort implementation
        template <typename T, typename index_type>
        DUAL void recursiveQuickSort(T arr[], index_type low, index_type high)
        {
            if (low < high) {
                index_type pivotIndex = partition(arr, low, high);

                // Recursively sort the left and right subarrays
                recursiveQuickSort(arr, low, pivotIndex - 1);
                recursiveQuickSort(arr, pivotIndex + 1, high);
            }
        }

        template <typename T, typename index_type>
        DUAL void iterativeQuickSort(T arr[], index_type low, index_type high)
        {
            // Create an auxiliary stack
            T stack[4];

            // initialize top of stack
            index_type top = -1;

            // push initial values of l and h to stack
            stack[++top] = low;
            stack[++top] = high;

            // Keep popping from stack while is not empty
            while (top >= 0) {
                // Pop h and l
                high = stack[top--];
                low  = stack[top--];

                // Set pivot element at its correct position
                // in sorted array
                index_type pivotIndex = partition(arr, low, high);

                // If there are elements on left side of pivot,
                // then push left side to stack
                if (pivotIndex - 1 > low) {
                    stack[++top] = low;
                    stack[++top] = pivotIndex - 1;
                }

                // If there are elements on right side of pivot,
                // then push right side to stack
                if (pivotIndex + 1 < high) {
                    stack[++top] = pivotIndex + 1;
                    stack[++top] = high;
                }
            }
        }

        template <typename T, typename U>
        SHARED T* sm_proxy(const U object)
        {
#if GPU_CODE
            if constexpr (global::on_sm) {
                // do we need an __align__() here? I don't think so...
                EXTERN unsigned char memory[];
                return reinterpret_cast<T*>(memory);
            }
            else {
                return object;
            }
#else
            return object;
#endif
        }

        template <typename T>
        SHARED auto sm_or_identity(const T* object)
        {
#if GPU_CODE
            if constexpr (global::on_sm) {
                EXTERN unsigned char memory[];
                return reinterpret_cast<T*>(memory);
            }
            else {
                return object;
            }
#else
            return object;
#endif
        }

        template <int dim, BlkAx axis, typename T>
        DUAL T axid(T idx, T ni, T nj, T kk)
        {
            if constexpr (dim == 1) {
                if constexpr (axis != BlkAx::I) {
                    return 0;
                }
                return idx;
            }
            else if constexpr (dim == 2) {
                if constexpr (axis == BlkAx::I) {
                    if constexpr (global::on_gpu) {
                        return blockDim.x * blockIdx.x + threadIdx.x;
                    }
                    return idx % ni;
                }
                else if constexpr (axis == BlkAx::J) {
                    if constexpr (global::on_gpu) {
                        return blockDim.y * blockIdx.y + threadIdx.y;
                    }
                    return idx / ni;
                }
                else {
                    return 0;
                }
            }
            else {
                if constexpr (axis == BlkAx::I) {
                    if constexpr (global::on_gpu) {
                        return blockDim.x * blockIdx.x + threadIdx.x;
                    }
                    return get_column(idx, ni, nj, kk);
                }
                else if constexpr (axis == BlkAx::J) {
                    if constexpr (global::on_gpu) {
                        return blockDim.y * blockIdx.y + threadIdx.y;
                    }
                    return get_row(idx, ni, nj, kk);
                }
                else {
                    if constexpr (global::on_gpu) {
                        return blockDim.z * blockIdx.z + threadIdx.z;
                    }
                    return get_height(idx, ni, nj);
                }
            }
        }

        template <typename T, typename U>
        inline real getFlops(
            const luint dim,
            const luint radius,
            const luint total_zones,
            const luint real_zones,
            const float delta_t
        )
        {
            // the advance step does one write plus 1.0 + dim * 2 * radius reads
            const float advance_contr =
                real_zones * sizeof(T) * (1.0 + (1.0 + dim * 2 * radius));
            const float cons2prim_contr = total_zones * sizeof(U);
            const float ghost_conf_contr =
                (total_zones - real_zones) * sizeof(T);
            return (advance_contr + cons2prim_contr + ghost_conf_contr) /
                   (delta_t * 1e9);
        }

        template <int dim, typename T, typename U, typename V>
        DEV void load_shared_buffer(
            const ExecutionPolicy<>& p,
            T& buffer,
            const U& data,
            const V ni,
            const V nj,
            const V nk,
            const V sx,
            const V sy,
            const V tx,
            const V ty,
            const V tz,
            const V txa,
            const V tya,
            const V tza,
            const V ia,
            const V ja,
            const V ka,
            const V radius
        )
        {

            const V aid = idx3(ia, ja, ka, ni, nj, nk);
            if constexpr (dim == 1) {
                V txl = p.blockSize.x;
                // Check if the active index exceeds the active zones
                // if it does, then this thread buffer will take on the
                // ghost index at the very end and return
                buffer[txa] = data[ia];
                if (tx < radius) {
                    if (blockIdx.x == p.gridSize.x - 1 &&
                        (ia + p.blockSize.x > ni - radius + tx)) {
                        txl = ni - radius - ia + tx;
                    }
                    buffer[txa - radius] = data[ia - radius];
                    buffer[txa + txl]    = data[ia + txl];
                }
                gpu::api::synchronize();
            }
            else if constexpr (dim == 2) {
                V txl = p.blockSize.x;
                V tyl = p.blockSize.y;
                // Load Shared memory into buffer for active zones plus
                // ghosts
                buffer[idx2(txa, tya, sx, sy)] = data[aid];
                if (ty < radius) {
                    if (blockIdx.y == p.gridSize.y - 1 &&
                        (ja + p.blockSize.y > nj - radius + ty)) {
                        tyl = nj - radius - ja + ty;
                    }
                    buffer[idx2(txa, tya - radius, sx, sy)] =
                        data[idx2(ia, ja - radius, ni, nj)];
                    buffer[idx2(txa, tya + txl, sx, sy)] =
                        data[idx2(ia, ja + tyl, ni, nj)];
                }
                if (tx < radius) {
                    if (blockIdx.x == p.gridSize.x - 1 &&
                        (ia + p.blockSize.x > ni - radius + tx)) {
                        txl = ni - radius - ia + tx;
                    }
                    buffer[idx2(txa - radius, tya, sx, sy)] =
                        data[idx2(ia - radius, ja, ni, nj)];
                    buffer[idx2(txa + txl, tya, sx, sy)] =
                        data[idx2(ia + txl, ja, ni, nj)];
                }
                gpu::api::synchronize();
            }
            else {
                luint txl = p.blockSize.x;
                luint tyl = p.blockSize.y;
                luint tzl = p.blockSize.z;
                // Load Shared memory into buffer
                buffer[idx3(txa, tya, tza, sx, sy, 0)] = data[aid];
                if (tz == 0) {
                    if ((blockIdx.z == p.gridSize.z - 1) &&
                        (ka + p.blockSize.z > nk - radius)) {
                        tzl = nk - radius - ka;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const luint re = tzl + q - 1;
                        buffer[idx3(txa, tya, tza - q, sx, sy, 0)] =
                            data[idx3(ia, ja, ka - q, ni, nj, nk)];
                        buffer[idx3(txa, tya, tza + re, sx, sy, 0)] =
                            data[idx3(ia, ja, ka + re, ni, nj, nk)];

                        for (int bdr = 1; bdr < radius + 1; bdr++) {
                            // x1 zones
                            buffer[idx3(txa + bdr, tya, tza - q, sx, sy, 0)] =
                                data[idx3(ia + bdr, ja, ka - q, ni, nj, nk)];
                            buffer[idx3(txa + bdr, tya, tza + re, sx, sy, 0)] =
                                data[idx3(ia + bdr, ja, ka + re, ni, nj, nk)];

                            buffer[idx3(txa - bdr, tya, tza - q, sx, sy, 0)] =
                                data[idx3(ia - bdr, ja, ka - q, ni, nj, nk)];
                            buffer[idx3(txa - bdr, tya, tza + re, sx, sy, 0)] =
                                data[idx3(ia - bdr, ja, ka + re, ni, nj, nk)];

                            // x2 zones
                            buffer[idx3(txa, tya + bdr, tza - q, sx, sy, 0)] =
                                data[idx3(ia, ja + bdr, ka - q, ni, nj, nk)];
                            buffer[idx3(txa, tya + bdr, tza + re, sx, sy, 0)] =
                                data[idx3(ia, ja + bdr, ka + re, ni, nj, nk)];

                            buffer[idx3(txa, tya - bdr, tza - q, sx, sy, 0)] =
                                data[idx3(ia, ja - bdr, ka - q, ni, nj, nk)];
                            buffer[idx3(txa, tya - bdr, tza + re, sx, sy, 0)] =
                                data[idx3(ia, ja - bdr, ka + re, ni, nj, nk)];
                        }
                    }
                }
                if (ty == 0) {
                    if ((blockIdx.y == p.gridSize.y - 1) &&
                        (ja + p.blockSize.y > nj - radius)) {
                        tyl = nj - radius - ja;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const luint re = tyl + q - 1;
                        buffer[idx3(txa, tya - q, tza, sx, sy, 0)] =
                            data[idx3(ia, ja - q, ka, ni, nj, nk)];
                        buffer[idx3(txa, tya + re, tza, sx, sy, 0)] =
                            data[idx3(ia, ja + re, ka, ni, nj, nk)];

                        for (int bdr = 1; bdr < radius + 1; bdr++) {
                            // x1 zones
                            buffer[idx3(txa + bdr, tya - q, tza, sx, sy, 0)] =
                                data[idx3(ia + bdr, ja - q, ka, ni, nj, nk)];
                            buffer[idx3(txa + bdr, tya + re, tza, sx, sy, 0)] =
                                data[idx3(ia + bdr, ja + re, ka, ni, nj, nk)];

                            buffer[idx3(txa - bdr, tya - q, tza, sx, sy, 0)] =
                                data[idx3(ia - bdr, ja - q, ka, ni, nj, nk)];
                            buffer[idx3(txa - bdr, tya + re, tza, sx, sy, 0)] =
                                data[idx3(ia - bdr, ja + re, ka, ni, nj, nk)];

                            // x3 zones
                            buffer[idx3(txa, tya - q, tza + bdr, sx, sy, 0)] =
                                data[idx3(ia, ja - q, ka + bdr, ni, nj, nk)];
                            buffer[idx3(txa, tya + re, tza + bdr, sx, sy, 0)] =
                                data[idx3(ia, ja + re, ka + bdr, ni, nj, nk)];

                            buffer[idx3(txa, tya - q, tza - bdr, sx, sy, 0)] =
                                data[idx3(ia, ja - q, ka - bdr, ni, nj, nk)];
                            buffer[idx3(txa, tya + re, tza - bdr, sx, sy, 0)] =
                                data[idx3(ia, ja + re, ka - bdr, ni, nj, nk)];
                        }
                    }
                }
                if (tx == 0) {
                    if ((blockIdx.x == p.gridSize.x - 1) &&
                        (ia + p.blockSize.x > ni - radius + tx)) {
                        txl = ni - radius - ia;
                    }
                    for (int q = 1; q < radius + 1; q++) {
                        const luint re = txl + q - 1;
                        buffer[idx3(txa - q, tya, tza, sx, sy, 0)] =
                            data[idx3(ia - q, ja, ka, ni, nj, nk)];
                        buffer[idx3(txa + re, tya, tza, sx, sy, 0)] =
                            data[idx3(ia + re, ja, ka, ni, nj, nk)];

                        for (int bdr = 1; bdr < radius + 1; bdr++) {
                            // x2 zones
                            buffer[idx3(txa - q, tya - bdr, tza, sx, sy, 0)] =
                                data[idx3(ia - q, ja - bdr, ka, ni, nj, nk)];
                            buffer[idx3(txa + re, tya - bdr, tza, sx, sy, 0)] =
                                data[idx3(ia + re, ja - bdr, ka, ni, nj, nk)];

                            buffer[idx3(txa - q, tya + bdr, tza, sx, sy, 0)] =
                                data[idx3(ia - q, ja + bdr, ka, ni, nj, nk)];
                            buffer[idx3(txa + re, tya + bdr, tza, sx, sy, 0)] =
                                data[idx3(ia + re, ja + bdr, ka, ni, nj, nk)];

                            // x3 zones
                            buffer[idx3(txa - q, tya, tza + bdr, sx, sy, 0)] =
                                data[idx3(ia - q, ja, ka + bdr, ni, nj, nk)];
                            buffer[idx3(txa + re, tya, tza + bdr, sx, sy, 0)] =
                                data[idx3(ia + re, ja, ka + bdr, ni, nj, nk)];

                            buffer[idx3(txa - q, tya, tza - bdr, sx, sy, 0)] =
                                data[idx3(ia - q, ja, ka - bdr, ni, nj, nk)];
                            buffer[idx3(txa + re, tya, tza - bdr, sx, sy, 0)] =
                                data[idx3(ia + re, ja, ka - bdr, ni, nj, nk)];
                        }
                    }
                }
                gpu::api::synchronize();
            }
        }

        template <class IndexType>
        DUAL bool in_range(IndexType val, IndexType lower, IndexType upper)
        {
            return (luint) (val - lower) <= (luint) (upper - lower);
        }

        template <int dim, typename T, typename idx>
        DUAL void ib_modify(T& lhs, const T& rhs, const bool ib, const idx side)
        {
            if (ib) {
                lhs.rho = rhs.rho;
                lhs.v1  = (1 - 2 * (side == 1)) * rhs.v1;
                if constexpr (dim > 1) {
                    lhs.v2 = (1 - 2 * (side == 2)) * rhs.v2;
                }
                if constexpr (dim > 2) {
                    lhs.v3 = (1 - 2 * (side == 3)) * rhs.v3;
                }
                lhs.p   = rhs.p;
                lhs.chi = rhs.chi;
            }
        }

        template <int dim, typename T, typename idx>
        DUAL bool ib_check(
            T& arr,
            const idx ii,
            const idx jj,
            const idx kk,
            const idx ni,
            const idx nj,
            const int side
        )
        {
            if constexpr (dim == 1) {
                return false;
            }
            else if constexpr (dim == 2) {
                if (side == 3) {
                    return false;
                }
                return arr[kk * nj * ni + jj * ni + ii];
            }
            else {
                return arr[kk * nj * ni + jj * ni + ii];
            }
        }

        template <Plane P, Corner C, Dir s>
        DUAL lint cidx(lint ii, lint jj, lint kk, luint ni, luint nj, luint nk)
        {
            constexpr lint half     = 1;
            constexpr lint offset   = 1;
            const auto [ip, jp, kp] = [=]() -> std::tuple<lint, lint, lint> {
                if constexpr (P == Plane::IJ) {
                    if constexpr (C == Corner::NE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + half,
                                  jj + offset + (s == Dir::N),
                                  kk + offset
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset + (s == Dir::E),
                                  jj + half,
                                  kk + offset
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii + 0, jj + (s == Dir::NW), kk};
                            default:
                                return {ii + 1, jj + (s == Dir::NE), kk};
                        }
                    }
                    else if constexpr (C == Corner::NW) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii,
                                  jj + offset + (s == Dir::N),
                                  kk + offset
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset - (s == Dir::W),
                                  jj + half,
                                  kk + offset
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii - 1, jj + (s == Dir::NW), kk};
                            default:
                                return {ii + 0, jj + (s == Dir::NE), kk};
                        }
                    }
                    else if constexpr (C == Corner::SE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + half,
                                  jj + offset - (s == Dir::S),
                                  kk + offset
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset + (s == Dir::E),
                                  jj,
                                  kk + offset
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii + 0, jj - (s == Dir::SW), kk};
                            default:
                                return {ii + 1, jj - (s == Dir::SE), kk};
                        }
                    }
                    else {
                        // SW
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii,
                                  jj + offset - (s == Dir::S),
                                  kk + offset
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset - (s == Dir::W),
                                  jj,
                                  kk + offset
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii - 1, jj - (s == Dir::SW), kk};
                            default:
                                return {ii + 0, jj - (s == Dir::SE), kk};
                        }
                    }
                }
                else if constexpr (P == Plane::IK) {
                    if constexpr (C == Corner::NE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + half,
                                  jj + offset,
                                  kk + offset + (s == Dir::N)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset + (s == Dir::E),
                                  jj + offset,
                                  kk + half
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii + 0, jj, kk + (s == Dir::NW)};
                            default:
                                return {ii + 1, jj, kk + (s == Dir::NE)};
                        }
                    }
                    else if constexpr (C == Corner::NW) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii,
                                  jj + offset,
                                  kk + offset + (s == Dir::N)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset - (s == Dir::W),
                                  jj + offset,
                                  kk + half
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii - 1, jj, kk + (s == Dir::NW)};
                            default:
                                return {ii + 0, jj, kk + (s == Dir::NE)};
                        }
                    }
                    else if constexpr (C == Corner::SE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + half,
                                  jj + offset,
                                  kk + offset - (s == Dir::S)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset + (s == Dir::E),
                                  jj + offset,
                                  kk
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii + 0, jj, kk - (s == Dir::SW)};
                            default:
                                return {ii + 1, jj, kk - (s == Dir::SE)};
                        }
                    }
                    else {
                        // SW
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii,
                                  jj + offset,
                                  kk + offset - (s == Dir::S)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset - (s == Dir::W),
                                  jj + offset,
                                  kk
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii - 1, jj, kk - (s == Dir::SW)};
                            default:
                                return {ii + 0, jj, kk - (s == Dir::SE)};
                        }
                    }
                }
                else {   // JK plane
                    if constexpr (C == Corner::NE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + offset,
                                  jj + half,
                                  kk + offset + (s == Dir::N)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset,
                                  jj + offset + (s == Dir::E),
                                  kk + half
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii, jj + 0, kk + (s == Dir::NW)};
                            default:
                                return {ii, jj + 1, kk + (s == Dir::NE)};
                        }
                    }
                    else if constexpr (C == Corner::NW) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + offset,
                                  jj,
                                  kk + offset + (s == Dir::N)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset,
                                  jj + offset - (s == Dir::W),
                                  kk + half
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii, jj - 1, kk + (s == Dir::NW)};
                            default:
                                return {ii, jj + 0, kk + (s == Dir::NE)};
                        }
                    }
                    else if constexpr (C == Corner::SE) {
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + offset,
                                  jj + half,
                                  kk + offset - (s == Dir::S)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset,
                                  jj + offset + (s == Dir::E),
                                  kk
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii, jj + 0, kk - (s == Dir::SW)};
                            default:
                                return {ii, jj + 1, kk - (s == Dir::SE)};
                        }
                    }
                    else {
                        // SW
                        switch (s) {
                            case Dir::N:
                            case Dir::S:
                                return {
                                  ii + offset,
                                  jj,
                                  kk + offset - (s == Dir::S)
                                };
                            case Dir::E:
                            case Dir::W:
                                return {
                                  ii + offset,
                                  jj + offset - (s == Dir::W),
                                  kk
                                };
                            case Dir::NW:
                            case Dir::SW:
                                return {ii, jj - 1, kk - (s == Dir::SW)};
                            default:
                                return {ii, jj + 0, kk - (s == Dir::SE)};
                        }
                    }
                }
            }();

            return idx3(ip, jp, kp, ni, nj, nk);
        }
    }   // namespace helpers
}   // namespace simbi
