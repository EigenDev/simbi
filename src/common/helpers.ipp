
#include "H5Cpp.h"
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

        template <typename Sim_type>
        void write_to_file(Sim_type& sim_state)
        {
            sim_state.prims.copyFromGpu();
            sim_state.cons.copyFromGpu();
            static auto data_directory      = sim_state.data_directory;
            static auto step                = sim_state.init_chkpt_idx;
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

            std::string tnow;
            if (sim_state.dlogt != 0) {
                const auto time_order_of_mag = std::floor(std::log10(step));
                if (time_order_of_mag > tchunk_order_of_mag) {
                    tchunk_order_of_mag += 1;
                }
                tnow = format_real(step);
            }
            else if (!sim_state.inFailureState) {
                tnow = format_real(t_interval);
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
            sim_state.chkpt_idx = step;
            step++;
            write_hdf5(data_directory, filename, sim_state);
        }

        template <typename sim_state_t>
        void config_ghosts1D(sim_state_t* sim_state)
        {
            const auto grid_size   = sim_state->nx;
            const auto first_order = sim_state->use_pcm;
            const auto x1max       = sim_state->x1max;
            const auto x1min       = sim_state->x1min;
            const auto nvars       = sim_state->nvars;
            const auto mesh_motion = sim_state->mesh_motion;
            auto* cons             = sim_state->cons.data();
            parallel_for(sim_state->activeP, 0, 1, [=] DEV(const luint gid) {
                const auto cell_end =
                    sim_state->cell_factors(sim_state->xag - 1);
                const auto cell_beg = sim_state->cell_factors(0);
                const auto es       = mesh_motion ? cell_end.dV : 1.0;
                const auto bs       = mesh_motion ? cell_beg.dV : 1.0;
                if (first_order) {
                    switch (sim_state->bcs[0]) {
                        case BoundaryCondition::DYNAMIC:
                            for (auto qq = 0; qq < nvars; qq++) {
                                cons[0][qq] = sim_state->bsources[qq](
                                                  x1min,
                                                  sim_state->t
                                              ) *
                                              bs;
                            }
                            break;
                        case BoundaryCondition::REFLECTING:
                            cons[0] = cons[1];
                            cons[0].momentum() *= -1;
                            break;
                        case BoundaryCondition::PERIODIC:
                            cons[0] = cons[grid_size - 2];
                            break;
                        default:
                            cons[0] = cons[1];
                            break;
                    }

                    switch (sim_state->bcs[1]) {
                        case BoundaryCondition::DYNAMIC:
                            for (auto qq = 0; qq < nvars; qq++) {
                                cons[grid_size - 1][qq] =
                                    sim_state->bsources[qq](
                                        x1max,
                                        sim_state->t
                                    ) *
                                    es;
                            }
                            break;
                        case BoundaryCondition::REFLECTING:
                            cons[grid_size - 1] = cons[grid_size - 2];
                            cons[grid_size - 1].momentum() *= -1;
                            break;
                        case BoundaryCondition::PERIODIC:
                            cons[grid_size - 1] = cons[1];
                            break;
                        default:
                            cons[grid_size - 1] = cons[grid_size - 2];
                            break;
                    }
                }
                else {
                    switch (sim_state->bcs[0]) {
                        case BoundaryCondition::DYNAMIC:
                            for (auto qq = 0; qq < nvars; qq++) {
                                cons[0][qq] = sim_state->bsources[qq](
                                                  x1min,
                                                  sim_state->t
                                              ) *
                                              bs;
                                cons[1][qq] = sim_state->bsources[qq](
                                                  x1min,
                                                  sim_state->t
                                              ) *
                                              bs;
                            }
                            break;
                        case BoundaryCondition::REFLECTING:
                            cons[0] = cons[3];
                            cons[1] = cons[2];
                            cons[0].momentum() *= -1;
                            cons[1].momentum() *= -1;
                            break;
                        case BoundaryCondition::PERIODIC:
                            cons[0] = cons[grid_size - 4];
                            cons[1] = cons[grid_size - 3];
                            break;
                        default:
                            cons[0] = cons[2];
                            cons[1] = cons[2];
                            break;
                    }

                    switch (sim_state->bcs[1]) {
                        case BoundaryCondition::DYNAMIC:
                            for (auto qq = 0; qq < nvars; qq++) {
                                cons[grid_size - 1][qq] =
                                    sim_state->bsources[qq](
                                        x1max,
                                        sim_state->t
                                    ) *
                                    es;
                                cons[grid_size - 2][qq] =
                                    sim_state->bsources[qq](
                                        x1max,
                                        sim_state->t
                                    ) *
                                    es;
                            }
                            break;
                        case BoundaryCondition::REFLECTING:
                            cons[grid_size - 1] = cons[grid_size - 4];
                            cons[grid_size - 2] = cons[grid_size - 3];
                            cons[grid_size - 1].momentum() *= -1;
                            cons[grid_size - 2].momentum() *= -1;
                            break;
                        case BoundaryCondition::PERIODIC:
                            cons[grid_size - 1] = cons[3];
                            cons[grid_size - 2] = cons[2];
                            break;
                        default:
                            cons[grid_size - 1] = cons[grid_size - 3];
                            cons[grid_size - 2] = cons[grid_size - 3];
                            break;
                    }
                }
            });
        }

        template <typename sim_state_t>
        void config_ghosts2D(sim_state_t* sim_state)
        {
            const auto nvars       = sim_state->nvars;
            const auto xag         = sim_state->xag;
            const auto yag         = sim_state->yag;
            const auto nx          = sim_state->nx;
            const auto ny          = sim_state->ny;
            const auto geometry    = sim_state->geometry;
            const auto half_sphere = sim_state->half_sphere;
            const auto hr          = sim_state->radius;   // halo radius
            auto* cons             = sim_state->cons.data();
            parallel_for(
                sim_state->activeP,
                sim_state->activeP.nzones,
                [=] DEV(const luint gid) {
                    const luint jj = axid<2, BlkAx::J>(gid, xag, yag);
                    const luint ii = axid<2, BlkAx::I>(gid, xag, yag);

                    const auto ir = ii + hr;
                    const auto jr = jj + hr;
                    for (luint rr = 0; rr < hr; rr++) {
                        const auto rs = rr + 1;
                        // Fill ghost zones at x1 boundaries
                        if (jj < ny - 2 * hr) {
                            auto ing  = idx2(rr, jr, nx, ny);
                            auto outg = idx2(nx - rs, jr, nx, ny);

                            switch (sim_state->bcs[0]) {
                                case BoundaryCondition::REFLECTING: {
                                    const auto inr =
                                        idx2(2 * hr - rs, jr, nx, ny);
                                    cons[ing] = cons[inr];
                                    cons[ing].momentum(1) *= -1;
                                    break;
                                }
                                case BoundaryCondition::DYNAMIC:
                                    for (int qq = 0; qq < nvars; qq++) {
                                        cons[ing][qq] = sim_state->bsources[qq](
                                            sim_state->x1min,
                                            sim_state->x2[jj],
                                            sim_state->t
                                        );
                                    }
                                    break;
                                case BoundaryCondition::PERIODIC: {
                                    const auto outr =
                                        idx2(nx - 2 * hr + rr, jr, nx, ny);
                                    cons[ing] = cons[outr];
                                    break;
                                }
                                default: {
                                    const auto inr = idx2(hr, jr, nx, ny);
                                    cons[ing]      = cons[inr];
                                    break;
                                }
                            }

                            switch (sim_state->bcs[1]) {
                                case BoundaryCondition::REFLECTING: {
                                    const auto outr =
                                        idx2(nx - 2 * hr + rr, jr, nx, ny);
                                    cons[outg] = cons[outr];
                                    cons[outg].momentum(1) *= -1;
                                    break;
                                }
                                case BoundaryCondition::DYNAMIC:
                                    for (int qq = 0; qq < nvars; qq++) {
                                        cons[outg][qq] =
                                            sim_state->bsources[qq](
                                                sim_state->x1max,
                                                sim_state->x2[jj],
                                                sim_state->t
                                            );
                                    }
                                    break;
                                case BoundaryCondition::PERIODIC: {
                                    const auto inr =
                                        idx2(2 * hr - rs, jr, nx, ny);
                                    cons[outg] = cons[inr];
                                    break;
                                }
                                default: {
                                    const auto outr =
                                        idx2(nx - (hr + 1), jr, nx, ny);
                                    cons[outg] = cons[outr];
                                    break;
                                }
                            }
                        }

                        // Fill ghost zones at x2 boundaries
                        if (ii < nx - 2 * hr) {
                            auto ing  = idx2(ir, rr, nx, ny);
                            auto outg = idx2(ir, ny - rs, nx, ny);

                            switch (geometry) {
                                case Geometry::SPHERICAL: {
                                    const auto inr =
                                        idx2(ir, 2 * hr - rs, nx, ny);
                                    const auto outr =
                                        idx2(ir, ny - 2 * hr + rr, nx, ny);
                                    cons[ing]  = cons[inr];
                                    cons[outg] = cons[outr];
                                    if (half_sphere) {
                                        cons[outg].momentum(2) *= -1;
                                    }
                                    break;
                                }
                                default:
                                    switch (sim_state->bcs[2]) {
                                        case BoundaryCondition::REFLECTING: {
                                            const auto inr =
                                                idx2(ir, 2 * hr - rs, nx, ny);
                                            cons[ing] = cons[inr];
                                            cons[ing].momentum(2) *= -1;
                                            break;
                                        }
                                        case BoundaryCondition::DYNAMIC:
                                            for (int qq = 0; qq < nvars; qq++) {
                                                cons[ing][qq] =
                                                    sim_state->bsources[qq](
                                                        sim_state->x1[ii],
                                                        sim_state->x2min,
                                                        sim_state->t
                                                    );
                                            }
                                            break;
                                        case BoundaryCondition::PERIODIC: {
                                            const auto outr = idx2(
                                                ir,
                                                ny - 2 * hr + rr,
                                                nx,
                                                ny
                                            );
                                            cons[ing] = cons[outr];
                                            break;
                                        }
                                        default: {
                                            const auto inr =
                                                idx2(ir, hr, nx, ny);
                                            cons[ing] = cons[inr];
                                            break;
                                        }
                                    }

                                    switch (sim_state->bcs[3]) {
                                        case BoundaryCondition::REFLECTING: {
                                            const auto outr = idx2(
                                                ir,
                                                ny - 2 * hr + rr,
                                                nx,
                                                ny
                                            );
                                            cons[outg] = cons[outr];
                                            cons[outg].momentum(2) *= -1;
                                            break;
                                        }
                                        case BoundaryCondition::DYNAMIC:
                                            for (int qq = 0; qq < nvars; qq++) {
                                                cons[outg][qq] =
                                                    sim_state->bsources[qq](
                                                        sim_state->x1[ii],
                                                        sim_state->x2max,
                                                        sim_state->t
                                                    );
                                            }
                                            break;
                                        case BoundaryCondition::PERIODIC: {
                                            const auto inr =
                                                idx2(ir, 2 * hr - rs, nx, ny);
                                            cons[outg] = cons[inr];
                                            break;
                                        }
                                        default: {
                                            const auto outr =
                                                idx2(ir, ny - (hr + 1), nx, ny);
                                            cons[outg] = cons[outr];
                                            break;
                                        }
                                    }
                                    break;
                            }
                        }
                    }
                }
            );
        }

        template <typename sim_state_t>
        void config_ghosts3D(sim_state_t* sim_state)
        {
            const auto nvars       = sim_state->nvars;
            const auto xag         = sim_state->xag;
            const auto yag         = sim_state->yag;
            const auto nx          = sim_state->nx;
            const auto ny          = sim_state->ny;
            const auto nz          = sim_state->nz;
            const auto geometry    = sim_state->geometry;
            const auto half_sphere = sim_state->half_sphere;
            const auto hr          = sim_state->radius;   // halo radius
            auto* cons             = sim_state->cons.data();
            parallel_for(
                sim_state->activeP,
                sim_state->activeP.nzones,
                [=] DEV(const luint gid) {
                    const luint kk = axid<3, BlkAx::K>(gid, xag, yag);
                    const luint jj = axid<3, BlkAx::J>(gid, xag, yag, kk);
                    const luint ii = axid<3, BlkAx::I>(gid, xag, yag, kk);

                    const auto ir = ii + hr;
                    const auto jr = jj + hr;
                    const auto kr = kk + hr;
                    for (luint rr = 0; rr < hr; rr++) {
                        const auto rs = rr + 1;
                        if (jj < ny - 2 * hr) {
                            // Fill ghost zones at i-k corners
                            auto iknw = idx3(rr, jr, rr, nx, ny, nz);
                            auto ikne = idx3(nx - rs, jr, rr, nx, ny, nz);
                            auto ikse = idx3(nx - rs, jr, nz - rs, nx, ny, nz);
                            auto iksw = idx3(rr, jr, nz - rs, nx, ny, nz);

                            // the corner ghosts are set equal to the real zones
                            // nearest those corners
                            cons[iknw] = cons[idx3(hr, jr, hr, nx, ny, nz)];
                            cons[ikne] =
                                cons[idx3(nx - (hr + 1), jr, hr, nx, ny, nz)];
                            cons[ikse] = cons[idx3(
                                nx - (hr + 1),
                                jr,
                                nz - (hr + 1),
                                nx,
                                ny,
                                nz
                            )];
                            cons[iksw] =
                                cons[idx3(hr, jr, nz - (hr + 1), nx, ny, nz)];

                            //================================================================
                            // Fill ghosts zones at x1 boundaries
                            if (kk < nz - 2 * hr) {
                                // Fill ghost zones at i-j corners
                                auto ijnw = idx3(rr, rr, kr, nx, ny, nz);
                                auto ijne = idx3(nx - rs, rr, kr, nx, ny, nz);
                                auto ijse =
                                    idx3(nx - rs, ny - rs, kr, nx, ny, nz);
                                auto ijsw = idx3(rr, ny - rs, kr, nx, ny, nz);
                                // the corner ghosts are set equal to the
                                // real zones nearest those corners
                                cons[ijnw] = cons[idx3(hr, hr, kr, nx, ny, nz)];
                                cons[ijne] = cons
                                    [idx3(nx - (hr + 1), hr, kr, nx, ny, nz)];
                                cons[ijse] = cons[idx3(
                                    nx - (hr + 1),
                                    ny - (hr + 1),
                                    kr,
                                    nx,
                                    ny,
                                    nz
                                )];
                                cons[ijsw] = cons
                                    [idx3(hr, ny - (hr + 1), kr, nx, ny, nz)];

                                //================================================================
                                auto ing  = idx3(rr, jr, kr, nx, ny, nz);
                                auto outg = idx3(nx - rs, jr, kr, nx, ny, nz);

                                switch (sim_state->bcs[0]) {
                                    case BoundaryCondition::REFLECTING: {
                                        const auto inr = idx3(
                                            2 * hr - rs,
                                            jr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[ing] = cons[inr];
                                        cons[ing].momentum(1) *= -1;
                                        break;
                                    }
                                    case BoundaryCondition::DYNAMIC:
                                        for (int qq = 0; qq < nvars; qq++) {
                                            cons[ing][qq] =
                                                sim_state->bsources[qq](
                                                    sim_state->x1min,
                                                    sim_state->x2[jj],
                                                    sim_state->x3[kk],
                                                    sim_state->t
                                                );
                                        }
                                        break;
                                    case BoundaryCondition::PERIODIC: {
                                        const auto outr = idx3(
                                            nx - 2 * hr + rr,
                                            jr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[ing] = cons[outr];
                                        break;
                                    }
                                    default: {
                                        const auto inr =
                                            idx3(hr, jr, kr, nx, ny, nz);
                                        cons[ing] = cons[inr];
                                        break;
                                    }
                                }

                                switch (sim_state->bcs[1]) {
                                    case BoundaryCondition::REFLECTING: {
                                        const auto outr = idx3(
                                            nx - 2 * hr + rr,
                                            jr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[outg] = cons[outr];
                                        cons[outg].momentum(1) *= -1;
                                        break;
                                    }
                                    case BoundaryCondition::DYNAMIC:
                                        for (int qq = 0; qq < nvars; qq++) {
                                            cons[outg][qq] =
                                                sim_state->bsources[qq](
                                                    sim_state->x1max,
                                                    sim_state->x2[jj],
                                                    sim_state->x3[kk],
                                                    sim_state->t
                                                );
                                        }
                                        break;
                                    case BoundaryCondition::PERIODIC: {
                                        const auto inr = idx3(
                                            2 * hr - rs,
                                            jr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[outg] = cons[inr];
                                        break;
                                    }
                                    default: {
                                        const auto outr = idx3(
                                            nx - (hr + 1),
                                            jr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[outg] = cons[outr];
                                        break;
                                    }
                                }
                            }

                            // Fill ghost zones at x3 boundaries
                            if (ii < nx - 2 * hr) {
                                // Fill ghost zones at j-k corners
                                auto jknw = idx3(ir, rr, rr, nx, ny, nz);
                                auto jkne = idx3(ir, ny - rs, rr, nx, ny, nz);
                                auto jkse =
                                    idx3(ir, ny - rs, nz - rs, nx, ny, nz);
                                auto jksw = idx3(ir, rr, nz - rs, nx, ny, nz);
                                // the corner ghosts are set equal to the
                                // real zones nearest those corners
                                cons[jknw] = cons[idx3(ir, hr, kr, nx, ny, nz)];
                                cons[jkne] = cons
                                    [idx3(ir, ny - (hr + 1), kr, nx, ny, nz)];
                                cons[jkse] = cons[idx3(
                                    ir,
                                    ny - (hr + 1),
                                    nz - (hr + 1),
                                    nx,
                                    ny,
                                    nz
                                )];
                                cons[jksw] = cons
                                    [idx3(ir, hr, nz - (hr + 1), nx, ny, nz)];

                                //================================================================
                                auto ing  = idx3(ir, jr, rr, nx, ny, nz);
                                auto outg = idx3(ir, jr, nz - rs, nx, ny, nz);

                                switch (geometry) {
                                    case Geometry::SPHERICAL: {
                                        // the x3 direction is periodic in phi
                                        const auto inr = idx3(
                                            ir,
                                            jr,
                                            2 * hr - rs,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        const auto outr = idx3(
                                            ir,
                                            jr,
                                            nz - 2 * hr + rr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[ing]  = cons[outr];
                                        cons[outg] = cons[inr];
                                        break;
                                    }
                                    default:
                                        switch (sim_state->bcs[4]) {
                                            case BoundaryCondition::
                                                REFLECTING: {
                                                const auto inr = idx3(
                                                    ir,
                                                    jr,
                                                    2 * hr - rs,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[inr];
                                                cons[ing].momentum(3) *= -1;
                                                break;
                                            }
                                            case BoundaryCondition::DYNAMIC:
                                                for (auto qq = 0;
                                                     qq < sim_state->nvars;
                                                     qq++) {
                                                    cons[ing][qq] =
                                                        sim_state->bsources[qq](
                                                            sim_state->x1[ii],
                                                            sim_state->x2[jj],
                                                            sim_state->x3min,
                                                            sim_state->t
                                                        );
                                                }
                                                break;
                                            case BoundaryCondition::PERIODIC: {
                                                const auto outr = idx3(
                                                    ir,
                                                    jr,
                                                    nz - 2 * hr + rr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[outr];
                                                break;
                                            }
                                            default: {
                                                const auto inr = idx3(
                                                    ir,
                                                    jr,
                                                    hr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[inr];
                                                break;
                                            }
                                        }

                                        switch (sim_state->bcs[5]) {
                                            case BoundaryCondition::
                                                REFLECTING: {
                                                const auto outr = idx3(
                                                    ir,
                                                    jr,
                                                    nz - 2 * hr + rr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[outr];
                                                cons[outg].momentum(3) *= -1;
                                                break;
                                            }
                                            case BoundaryCondition::DYNAMIC:
                                                for (auto qq = 0;
                                                     qq < sim_state->nvars;
                                                     qq++) {
                                                    cons[outg][qq] =
                                                        sim_state->bsources[qq](
                                                            sim_state->x1[ii],
                                                            sim_state->x2[jj],
                                                            sim_state->x3max,
                                                            sim_state->t
                                                        );
                                                }
                                                break;
                                            case BoundaryCondition::PERIODIC: {
                                                const auto inr = idx3(
                                                    ir,
                                                    jr,
                                                    2 * hr - rs,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[inr];
                                                break;
                                            }
                                            default: {
                                                const auto outr = idx3(
                                                    ir,
                                                    jr,
                                                    nz - (hr + 1),
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[outr];
                                                break;
                                            }
                                        }
                                        break;
                                }
                            }
                        }

                        if (ii < nx - 2 * hr) {
                            // Fill the ghost zones at the x2 boundaries
                            if (kk < nz - 2 * hr) {
                                auto ing  = idx3(ir, rr, kr, nx, ny, nz);
                                auto outg = idx3(ir, ny - rs, kr, nx, ny, nz);

                                switch (geometry) {
                                    case Geometry::SPHERICAL: {
                                        // theta boundaries are reflecting
                                        const auto inr = idx3(
                                            ir,
                                            2 * hr - rs,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        const auto outr = idx3(
                                            ir,
                                            nz - 2 * hr + rr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[ing]  = cons[inr];
                                        cons[outg] = cons[outr];
                                        if (half_sphere) {
                                            cons[outg].momentum(2) *= -1;
                                        }
                                        break;
                                    }

                                    case Geometry::CYLINDRICAL: {
                                        // phi boundaries are periodic
                                        const auto inr = idx3(
                                            ir,
                                            2 * hr - rs,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        const auto outr = idx3(
                                            ir,
                                            ny - 2 * hr + rr,
                                            kr,
                                            nx,
                                            ny,
                                            nz
                                        );
                                        cons[ing]  = cons[outr];
                                        cons[outg] = cons[inr];
                                        break;
                                    }
                                    default:
                                        switch (sim_state->bcs[2]) {
                                            case BoundaryCondition::
                                                REFLECTING: {
                                                auto inr = idx3(
                                                    ir,
                                                    2 * hr - rs,
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[inr];
                                                cons[ing].momentum(2) *= -1;
                                                break;
                                            }
                                            case BoundaryCondition::DYNAMIC:
                                                for (auto qq = 0;
                                                     qq < sim_state->nvars;
                                                     qq++) {
                                                    cons[ing][qq] =
                                                        sim_state->bsources[qq](
                                                            sim_state->x1[ii],
                                                            sim_state->x2min,
                                                            sim_state->x3[kk],
                                                            sim_state->t
                                                        );
                                                }
                                                break;
                                            case BoundaryCondition::PERIODIC: {
                                                auto outr = idx3(
                                                    ir,
                                                    ny - 2 * hr + rr,
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[outr];
                                                break;
                                            }
                                            default: {
                                                auto inr = idx3(
                                                    ir,
                                                    hr,
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[ing] = cons[inr];
                                                break;
                                            }
                                        }

                                        switch (sim_state->bcs[3]) {
                                            case BoundaryCondition::
                                                REFLECTING: {
                                                auto outr = idx3(
                                                    ir,
                                                    ny - 2 * hr + rr,
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[outr];
                                                cons[outg].momentum(2) *= -1;
                                                break;
                                            }
                                            case BoundaryCondition::DYNAMIC:
                                                for (auto qq = 0;
                                                     qq < sim_state->nvars;
                                                     qq++) {
                                                    cons[outg][qq] =
                                                        sim_state->bsources[qq](
                                                            sim_state->x1[ii],
                                                            sim_state->x2max,
                                                            sim_state->x3[kk],
                                                            sim_state->t
                                                        );
                                                }
                                                break;
                                            case BoundaryCondition::PERIODIC: {
                                                auto inr = idx3(
                                                    ir,
                                                    2 * hr - rs,
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[inr];
                                                break;
                                            }
                                            default: {
                                                auto outr = idx3(
                                                    ir,
                                                    ny - (hr + 1),
                                                    kr,
                                                    nx,
                                                    ny,
                                                    nz
                                                );
                                                cons[outg] = cons[outr];
                                                break;
                                            }
                                        }
                                        break;
                                }
                            }
                        }
                    }
                }
            );
        };

        template <typename T>
        void config_ghosts(T* sim_state)
        {
            if constexpr (T::dimensions == 1) {
                config_ghosts1D(sim_state);
            }
            else if constexpr (T::dimensions == 2) {
                config_ghosts2D(sim_state);
            }
            else if constexpr (T::dimensions == 3) {
                config_ghosts3D(sim_state);
            }
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_1D_primitive<T>::value>::type
        compute_dt(U* self, const V* prim_buffer, real* dt_min)
        {
#if GPU_CODE
            real vPlus, vMinus;
            int ii = blockDim.x * blockIdx.x + threadIdx.x;
            if (ii < self->total_zones) {
                const auto ireal = get_real_idx(ii, self->radius, self->xag);
                if constexpr (is_relativistic<T>::value) {
                    if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                        const real rho = prim_buffer[ii].rho();
                        const real p   = prim_buffer[ii].p();
                        const real v   = prim_buffer[ii].get_v1();
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
                    const real rho = prim_buffer[ii].rho();
                    const real p   = prim_buffer[ii].p();
                    const real v   = prim_buffer[ii].get_v1();
                    const real cs  = std::sqrt(self->gamma * p / rho);
                    vPlus          = std::abs(v + cs);
                    vMinus         = std::abs(v - cs);
                }
                const auto cell   = self->cell_factors(ireal);
                const real x1l    = cell.x1L();
                const real x1r    = cell.x1R();
                const real dx1    = x1r - x1l;
                const real vfaceL = (self->geometry == Geometry::CARTESIAN)
                                        ? self->hubble_param
                                        : x1l * self->hubble_param;
                const real vfaceR = (self->geometry == Geometry::CARTESIAN)
                                        ? self->hubble_param
                                        : x1r * self->hubble_param;
                const real cfl_dt =
                    dx1 /
                    (my_max(std::abs(vPlus + vfaceR), std::abs(vMinus + vfaceL))
                    );
                dt_min[ii] = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_2D_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const Geometry geometry
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
                        const real rho = prim_buffer[gid].rho();
                        const real p   = prim_buffer[gid].p();
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
                    const real rho = prim_buffer[gid].rho();
                    const real p   = prim_buffer[gid].p();
                    const real v1  = prim_buffer[gid].get_v1();
                    const real v2  = prim_buffer[gid].get_v2();
                    real cs        = std::sqrt(self->gamma * p / rho);
                    plus_v1        = (v1 + cs);
                    plus_v2        = (v2 + cs);
                    minus_v1       = (v1 - cs);
                    minus_v2       = (v2 - cs);
                }
                const auto cell = self->cell_factors(ii, jj);
                v1p             = std::abs(plus_v1);
                v1m             = std::abs(minus_v1);
                v2p             = std::abs(plus_v2);
                v2m             = std::abs(minus_v2);
                switch (geometry) {
                    case Geometry::CARTESIAN:
                        cfl_dt = my_min(
                            self->dx1 / (my_max(v1p, v1m)),
                            self->dx2 / (my_max(v2m, v2m))
                        );
                        break;

                    case Geometry::SPHERICAL: {
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real tl = cell.x2L();
                        const real tr = cell.x1R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            rmean * (tr - tl) / (my_max(v2p, v2m))
                        );
                        break;
                    }
                    case Geometry::PLANAR_CYLINDRICAL: {
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real tl = cell.x2L();
                        const real tr = cell.x2R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            rmean * (tr - tl) / (my_max(v2p, v2m))
                        );
                        break;
                    }
                    case Geometry::AXIS_CYLINDRICAL: {
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real zl = cell.x2L();
                        const real zr = cell.x2R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        cfl_dt = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            (zr - zl) / (my_max(v2p, v2m))
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
            const Geometry geometry
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
                        const real rho = prim_buffer[gid].rho();
                        const real p   = prim_buffer[gid].p();
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
                    const real rho = prim_buffer[gid].rho();
                    const real p   = prim_buffer[gid].p();
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

                const auto ireal = get_real_idx(ii, self->radius, self->xag);
                const auto jreal = get_real_idx(jj, self->radius, self->yag);
                const auto kreal = get_real_idx(kk, self->radius, self->zag);
                const auto cell  = self->cell_factors(ireal, jreal, kreal);
                const auto x1l   = cell.x1L();
                const auto x1r   = cell.x1R();
                const auto dx1   = x1r - x1l;
                const auto x2l   = cell.x2L();
                const auto x2r   = cell.x2R();
                const auto dx2   = x2r - x2l;
                const auto x3l   = cell.x3L();
                const auto x3r   = cell.x3R();
                const auto dx3   = x3r - x3l;
                switch (geometry) {
                    case Geometry::CARTESIAN: {
                        cfl_dt = my_min3<real>(
                            dx1 /
                                (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                            dx2 /
                                (my_max(std::abs(plus_v2), std::abs(minus_v2))),
                            dx3 /
                                (my_max(std::abs(plus_v3), std::abs(minus_v3)))
                        );
                        break;
                    }
                    case Geometry::SPHERICAL: {
                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min3<real>(
                            dx1 /
                                (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                            rmean * dx2 /
                                (my_max(std::abs(plus_v2), std::abs(minus_v2))),
                            rmean * std::sin(cell.x2mean) * dx3 /
                                (my_max(std::abs(plus_v3), std::abs(minus_v3)))
                        );
                        break;
                    }
                    case Geometry::CYLINDRICAL: {
                        const real rmean = cell.x1mean;
                        const real th    = 0.5 * (x2l + x2r);
                        cfl_dt           = my_min3<real>(
                            dx1 /
                                (my_max(std::abs(plus_v1), std::abs(minus_v1))),
                            rmean * dx2 /
                                (my_max(std::abs(plus_v2), std::abs(minus_v2))),
                            dx3 /
                                (my_max(std::abs(plus_v3), std::abs(minus_v3)))
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
                        real speeds[4];
                        self->calc_max_wave_speeds(prim_buffer[gid], 1, speeds);
                        vPlus  = std::abs(speeds[3]);
                        vMinus = std::abs(speeds[0]);
                    }
                    else {
                        vPlus  = 1.0;
                        vMinus = 1.0;
                    }
                }
                else {
                    const real rho = prim_buffer[gid].rho();
                    const real p   = prim_buffer[gid].p();
                    const real v   = prim_buffer[gid].get_v1();
                    const real cs  = std::sqrt(self->gamma * p / rho);
                    vPlus          = (v + cs);
                    vMinus         = (v - cs);
                }
                const auto ireal = get_real_idx(ii, self->radius, self->xag);
                const auto cell  = self->cell_factors(ii);

                const real x1l    = cell.x1L();
                const real x1r    = cell.x1R();
                const real dx1    = x1r - x1l;
                const real vfaceL = (self->geometry == Geometry::CARTESIAN)
                                        ? self->hubble_param
                                        : x1l * self->hubble_param;
                const real vfaceR = (self->geometry == Geometry::CARTESIAN)
                                        ? self->hubble_param
                                        : x1r * self->hubble_param;
                const real cfl_dt =
                    dx1 /
                    (my_max(std::abs(vPlus + vfaceR), std::abs(vMinus + vfaceL))
                    );
                dt_min[ii] = self->cfl * cfl_dt;
            }
#endif
        }

        template <typename T, TIMESTEP_TYPE dt_type, typename U, typename V>
        KERNEL typename std::enable_if<is_2D_mhd_primitive<T>::value>::type
        compute_dt(
            U* self,
            const V* prim_buffer,
            real* dt_min,
            const Geometry geometry
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
                        real speeds[4];
                        self->calc_max_wave_speeds(prim_buffer[gid], 1, speeds);
                        plus_v1  = std::abs(speeds[3]);
                        minus_v1 = std::abs(speeds[0]);
                        self->calc_max_wave_speeds(prim_buffer[gid], 2, speeds);
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
                    const real rho = prim_buffer[gid].rho();
                    const real p   = prim_buffer[gid].p();
                    const real v1  = prim_buffer[gid].get_v1();
                    const real v2  = prim_buffer[gid].get_v2();
                    real cs        = std::sqrt(self->gamma * p / rho);
                    plus_v1        = (v1 + cs);
                    plus_v2        = (v2 + cs);
                    minus_v1       = (v1 - cs);
                    minus_v2       = (v2 - cs);
                }

                const auto cell = self->cell_factors(ii, jj);
                v1p             = std::abs(plus_v1);
                v1m             = std::abs(minus_v1);
                v2p             = std::abs(plus_v2);
                v2m             = std::abs(minus_v2);
                switch (geometry) {
                    case Geometry::CARTESIAN:
                        cfl_dt = my_min(
                            self->dx1 / (my_max(v1p, v1m)),
                            self->dx2 / (my_max(v2m, v2m))
                        );
                        break;

                    case Geometry::SPHERICAL: {
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real tl = cell.x2L();
                        const real tr = cell.x2R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            rmean * (tr - tl) / (my_max(v2p, v2m))
                        );
                        break;
                    }
                    case Geometry::PLANAR_CYLINDRICAL: {
                        // Compute avg spherical distance 3/4 *(rf^4 -
                        // ri^4)/(rf^3 - ri^3)
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real tl = cell.x2L();
                        const real tr = cell.x2R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            rmean * (tr - tl) / (my_max(v2p, v2m))
                        );
                        break;
                    }
                    case Geometry::AXIS_CYLINDRICAL: {
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->xag);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->yag);
                        const real rl = cell.x1L();
                        const real rr = cell.x1R();
                        const real zl = cell.x2L();
                        const real zr = cell.x2R();
                        if (self->mesh_motion) {
                            const real vfaceL = rl * self->hubble_param;
                            const real vfaceR = rr * self->hubble_param;
                            v1p               = std::abs(plus_v1 - vfaceR);
                            v1m               = std::abs(minus_v1 - vfaceL);
                        }
                        cfl_dt = my_min(
                            (rr - rl) / (my_max(v1p, v1m)),
                            (zr - zl) / (my_max(v2p, v2m))
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
            const Geometry geometry
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
                real v1p, v1m, v2p, v2m, v3p, v3m, cfl_dt;
                real speeds[4];
                const auto prims = prim_buffer;
                // Left/Right wave speeds
                if constexpr (dt_type == TIMESTEP_TYPE::ADAPTIVE) {
                    self->calc_max_wave_speeds(prims[aid], 1, speeds);
                    v1p = std::abs(speeds[3]);
                    v1m = std::abs(speeds[0]);
                    self->calc_max_wave_speeds(prims[aid], 2, speeds);
                    v2p = std::abs(speeds[3]);
                    v2m = std::abs(speeds[0]);
                    self->calc_max_wave_speeds(prims[aid], 3, speeds);
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

                const auto cell = self->cell_factors(ii, jj, kk);

                switch (geometry) {
                    case Geometry::CARTESIAN:
                        cfl_dt = my_min3<real>(
                            self->dx1 / (my_max(v1p, v1m)),
                            self->dx2 / (my_max(v2p, v2m)),
                            self->dx3 / (my_max(v3p, v3m))
                        );

                        break;
                    case Geometry::SPHERICAL: {
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->nxv);
                        const auto jreal =
                            get_real_idx(jj, self->radius, self->nyv);

                        const real x1l = cell.x1L();
                        const real x1r = cell.x1R();
                        const real dx1 = x1r - x1l;

                        const real x2l   = cell.x2L();
                        const real x2r   = cell.x2R();
                        const real rmean = cell.x1mean;
                        const real th    = 0.5 * (x2r + x2l);
                        const real rproj = rmean * std::sin(th);
                        cfl_dt           = my_min3<real>(
                            dx1 / (my_max(v1p, v1m)),
                            rmean * self->dx2 / (my_max(v2p, v2m)),
                            rproj * self->dx3 / (my_max(v3p, v3m))
                        );
                        break;
                    }
                    default: {
                        const auto ireal =
                            get_real_idx(ii, self->radius, self->nxv);
                        const real x1l = cell.x1L();
                        const real x1r = cell.x1R();
                        const real dx1 = x1r - x1l;

                        const real rmean = cell.x1mean;
                        cfl_dt           = my_min3<real>(
                            dx1 / (my_max(v1p, v1m)),
                            rmean * self->dx2 / (my_max(v2p, v2m)),
                            self->dx3 / (my_max(v3p, v3m))
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
                min = my_min(dt_min[i], min);
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
            // luint bid  = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y
            // * gridDim.x + blockIdx.x;
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
                min = my_min(dt_min[i], min);
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

        // solve the cubic equation (Pluto)
        // template <typename T>
        // DUAL T cubicPluto(T b, T c, T d, T res[3])
        // {
        //     double b2;
        //     double Q, R;
        //     double sQ, arg, theta, cs, sn, p;
        //     const double one_3  = 1.0 / 3.0;
        //     const double one_27 = 1.0 / 27.0;
        //     const double sqrt3  = std::sqrt(3.0);

        //     b2 = b * b;

        //     /*  ----------------------------------------------
        //          the expression for f should be
        //          Q = c - b*b/3.0; however, to avoid negative
        //          round-off making h > 0.0 or g^2/4 - h < 0.0
        //          we let c --> c(1- 1.1e-16)
        //         ---------------------------------------------- */

        //     Q = b2 * one_3 -
        //         c * (1.0 - 1.e-16); /* = 3*Q, with Q given by Eq.
        //         [5.6.10] */
        //     R = b * (2.0 * b2 - 9.0 * c) * one_27 +
        //         d; /* = 2*R, with R given by Eq. [5.6.10] */

        //     Q = my_max(Q, 0.0);
        //     /*
        //     if (fabs(Q) < 1.e-18){
        //       print ("CubicSolve() very small Q = %12.6e\n",Q);
        //       QUIT_PLUTO(1);
        //     }
        //     if (Q < 0.0){
        //       print ("! CubicSolve(): Q = %8.3 < 0 \n",Q);
        //       QUIT_PLUTO(1);
        //     }
        //     */

        //     /* -------------------------------------------------------
        //         We assume the cubic *always* has 3 real root for
        //         which R^2 > Q^3.
        //         It follows that Q is always > 0
        //        ------------------------------------------------------- */

        //     sQ  = std::sqrt(Q) / sqrt3;
        //     arg = -1.5 * R / (Q * sQ);

        //     /*  this is to prevent unpleseant situation
        //         where both g and i are close to zero       */

        //     arg = my_max(-1.0, arg);
        //     arg = my_min(1.0, arg);

        //     theta = std::acos(arg) *
        //             one_3; /* Eq. [5.6.11], note that  pi/3 < theta < 0
        //             */

        //     cs = std::cos(theta);         /*   > 0   */
        //     sn = sqrt3 * std::sin(theta); /*   > 0   */
        //     p  = -b * one_3;

        //     res[0] = -sQ * (cs + sn) + p;
        //     res[1] = -sQ * (cs - sn) + p;
        //     res[2] = 2.0 * sQ * cs + p;

        //     /* -- Debug
        //       if(debug_print) {
        //         int l;
        //         double x, f;
        //         print
        //       ("===========================================================\n");
        //         print ("> Resolvent cubic:\n");
        //         print ("  g(x)  = %18.12e + x*(%18.12e + x*(%18.12e +
        //         x))\n", d,
        //       c, b); print ("  Q     = %8.3e\n",Q); print ("  arg-1 =
        //       %8.3e\n", -1.5*R/(Q*sQ)-1.0);

        //         print ("> Cubic roots = %8.3e  %8.3e
        //         %8.3e\n",z[0],z[1],z[2]); for (l = 0; l < 3; l++){  //
        //         check accuracy of solution

        //           x = z[l];
        //           f = d + x*(c + x*(b + x));
        //           print ("  verify: g(x[%d]) = %8.3e\n",l,f);
        //         }

        //         print
        //       ("===========================================================\n");
        //       }
        //     */
        //     return (0);
        // }

        // template <typename T>
        // DUAL int quarticPluto(T b, T c, T d, T e, T res[4])
        // {
        //     int n, j, ifail;
        //     double b2, sq;
        //     double a2, a1, a0, u[4];
        //     double p, q, r, f;
        //     const double three_256 = 3.0 / 256.0;
        //     const double one_64    = 1.0 / 64.0;
        //     double sz1, sz2, sz3, sz4;

        //     b2 = b * b;

        //     /*
        //     --------------------------------------------------------------
        //        1) Compute cubic coefficients using the method outlined in
        //           http://eqworld.ipmnet.ru/En/solutions/ae/ae0108.pdf
        //        --------------------------------------------------------------
        //        */

        //     p = c - b2 * 0.375;
        //     q = d + b2 * b * 0.125 - b * c * 0.5;
        //     r = e - 3.0 * b2 * b2 / 256.0 + b2 * c / 16.0 - 0.25 * b * d;

        //     a2 = 2.0 * p;
        //     a1 = p * p - 4.0 * r;
        //     a0 = -q * q;

        //     ifail = cubicPluto(a2, a1, a0, u);
        //     if (ifail != 0) {
        //         return 1;
        //     }

        //     u[0] = my_max(u[0], 0.0);
        //     u[1] = my_max(u[1], 0.0);
        //     u[2] = my_max(u[2], 0.0);

        //     if (u[0] != u[0] || u[1] != u[1] || u[2] != u[2]) {
        //         return 1;
        //     }

        //     sq  = -0.5 * (q >= 0.0 ? 1.0 : -1.0);
        //     sz1 = sq * std::sqrt(u[0]);
        //     sz2 = sq * std::sqrt(u[1]);
        //     sz3 = sq * std::sqrt(u[2]);

        //     res[0] = -0.25 * b + sz1 + sz2 + sz3;
        //     res[1] = -0.25 * b + sz1 - sz2 - sz3;
        //     res[2] = -0.25 * b - sz1 + sz2 - sz3;
        //     res[3] = -0.25 * b - sz1 - sz2 + sz3;
        //     if constexpr (global::BuildPlatform == global::Platform::GPU)
        //     {
        //         iterativeQuickSort(res, 0, 3);
        //     }
        //     else {
        //         recursiveQuickSort(res, 0, 3);
        //     }
        //     /*
        //       if (debug_print){
        //         print ("Quartic roots = %f  %f  %f  %f; q =
        //       %8.3e\n",z[0],z[1],z[2],z[3],q); CheckSolutions(b,c,d,e,z);
        //       }
        //     */
        //     return 0;
        // }

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
            // the advance step does one write plus 1.0 + dim * 2 * hr reads
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
                lhs.rho() = rhs.rho();
                lhs.v1()  = (1 - 2 * (side == 1)) * rhs.v1();
                if constexpr (dim > 1) {
                    lhs.v2() = (1 - 2 * (side == 2)) * rhs.v2();
                }
                if constexpr (dim > 2) {
                    lhs.v3() = (1 - 2 * (side == 3)) * rhs.v3();
                }
                lhs.p()   = rhs.p();
                lhs.chi() = rhs.chi();
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

        template <typename T>
        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const T& state
        )
        {
            const auto full_filename = data_directory + filename;
            display_message(full_filename);

            // Create a new file using the default property list.
            H5::H5File file(full_filename, H5F_ACC_TRUNC);

            // Create the data space for the dataset.
            hsize_t dims[1]   = {state.nx * state.ny * state.nz};
            hsize_t dimxv[1]  = {state.nxv * state.yag * state.zag};
            hsize_t dimyv[1]  = {state.xag * state.nyv * state.zag};
            hsize_t dimzv[1]  = {state.xag * state.yag * state.nzv};
            hsize_t dimx[1]   = {state.x1.size()};
            hsize_t dimy[1]   = {state.x2.size()};
            hsize_t dimz[1]   = {state.x3.size()};
            hsize_t dim_bc[1] = {state.boundary_conditions.size()};

            H5::DataSpace hdataspace(1, dims);
            H5::DataSpace hdataspace_x1(1, dimx);
            H5::DataSpace hdataspace_x2(1, dimy);
            H5::DataSpace hdataspace_x3(1, dimz);
            H5::DataSpace hdataspace_bc(1, dim_bc);
            H5::DataSpace b1dataspace(1, dimxv);
            H5::DataSpace b2dataspace(1, dimyv);
            H5::DataSpace b3dataspace(1, dimzv);

            // Create empty dataspace for attributes
            H5::DataSpace attr_dataspace(H5S_NULL);
            H5::DataType attr_type(H5::PredType::NATIVE_INT);
            // create attribute data space that houses scalar type
            H5::DataSpace scalar_dataspace(H5S_SCALAR);

            //==================================================================
            // DATA TYPES
            //==================================================================
            // Define the real-type for the data in the file.
            H5::DataType real_type = H5::PredType::NATIVE_DOUBLE;

            // int-type
            H5::DataType int_type = H5::PredType::NATIVE_INT;

            // bool-type
            H5::DataType bool_type = H5::PredType::NATIVE_HBOOL;

            // scalar-type
            H5::DataType scalar_type = H5::PredType::NATIVE_DOUBLE;

            // Define the string type for variable string length
            H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

            //==================================================================
            //  BOUNDARY CONDITIONS
            //==================================================================
            // convert the string to a char array
            std::vector<const char*> arr_c_str;
            for (hsize_t ii = 0; ii < dim_bc[0]; ++ii) {
                arr_c_str.push_back(state.boundary_conditions[ii].c_str());
            }

            // Write the boundary conditions to the file
            H5::DataSet bc_dataset = file.createDataSet(
                "boundary_conditions",
                str_type,
                hdataspace_bc
            );
            bc_dataset.write(arr_c_str.data(), str_type);
            bc_dataset.close();

            //==================================================================
            //  X1, X2, X3 DATA
            //==================================================================
            H5::DataSet dataset;
            // helper lambda for writing to any dataset
            auto write_dataset = [&](const std::string& name,
                                     const auto& data,
                                     const auto& dataspace) {
                dataset = file.createDataSet(name, real_type, dataspace);
                dataset.write(data.host_data(), real_type);
                dataset.close();
            };

            // Create datasets for the x1, x2, and x3 data
            write_dataset("x1", state.x1, hdataspace_x1);
            write_dataset("x2", state.x2, hdataspace_x2);
            write_dataset("x3", state.x3, hdataspace_x3);

            //==================================================================
            //  PRIMITIVE DATA
            //==================================================================
            // the regime is a  constexprstring view, so convert to
            // std::string
            const auto regime = std::string(state.regime);
            // helper lambda for writing the prim data using a for 1D loop
            // and hyperslab selection
            auto write_prims = [&](const std::string& name,
                                   const auto& dataspace,
                                   const auto member) {
                // Write the data using a for loop
                dataset = file.createDataSet(name, real_type, dataspace);
                for (hsize_t i = 0; i < state.prims.size(); ++i) {
                    hsize_t offset[1] = {i};
                    hsize_t count[1]  = {1};
                    H5::DataSpace memspace(1, count);
                    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
                    dataset.write(
                        &state.prims[i][member],
                        real_type,
                        memspace,
                        dataspace
                    );
                }
                dataset.close();
            };

            auto write_fields = [&](const std::string& name,
                                    const auto& dataspace,
                                    const auto member) {
                if constexpr (T::regime == "srmhd") {
                    // Write the data using a for loop
                    dataset = file.createDataSet(name, real_type, dataspace);
                    if (member == 1) {
                        for (hsize_t i = 0; i < state.bstag1.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);
                            dataset.write(
                                &const_cast<T&>(state).bstag1[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    else if (member == 2) {
                        for (hsize_t i = 0; i < state.bstag2.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);

                            dataset.write(
                                &const_cast<T&>(state).bstag2[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    else {
                        for (hsize_t i = 0; i < state.bstag3.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);

                            dataset.write(
                                &const_cast<T&>(state).bstag3[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    dataset.close();
                }
            };
            write_prims("rho", hdataspace, 0);
            write_prims("v1", hdataspace, 1);
            if (state.dimensions > 1) {
                write_prims("v2", hdataspace, 2);
            }
            if (state.dimensions > 2) {
                write_prims("v3", hdataspace, 3);
            }
            write_prims("p", hdataspace, state.dimensions + 1);
            if constexpr (T::regime == "srmhd") {
                write_fields("b1", b1dataspace, 1);
                write_fields("b2", b2dataspace, 2);
                write_fields("b3", b3dataspace, 3);
                write_prims("chi", hdataspace, state.dimensions + 5);
            }
            else {
                write_prims("chi", hdataspace, state.dimensions + 2);
            }

            //==================================================================
            //  ATTRIBUTE DATA
            //==================================================================
            // create dataset for simulation information
            H5::DataSet sim_info =
                file.createDataSet("sim_info", attr_type, attr_dataspace);

            // write simulation information in attributes and then close the
            // file
            const std::vector<std::pair<std::string, const void*>> attributes =
                {{"current_time", &state.t},
                 {"time_step", &state.dt},
                 {"spatial_order", state.spatial_order.c_str()},
                 {"time_order", state.time_order.c_str()},
                 {"using_gamma_beta", &state.using_fourvelocity},
                 {"mesh_motion", &state.mesh_motion},
                 {"x1max", &state.x1max},
                 {"x1min", &state.x1min},
                 {"x2max", &state.x2max},
                 {"x2min", &state.x2min},
                 {"x3max", &state.x3max},
                 {"x3min", &state.x3min},
                 {"adiabatic_gamma", &state.gamma},
                 {"nx", &state.nx},
                 {"ny", &state.ny},
                 {"nz", &state.nz},
                 {"chkpt_idx", &state.chkpt_idx},
                 {"xactive_zones", &state.xag},
                 {"yactive_zones", &state.yag},
                 {"zactive_zones", &state.zag},
                 {"geometry", state.coord_system.c_str()},
                 {"regime", regime.c_str()},
                 {"dimensions", &state.dimensions},
                 {"x1_cell_spacing",
                  cell2str.at(state.x1_cell_spacing).c_str()},
                 {"x2_cell_spacing",
                  cell2str.at(state.x2_cell_spacing).c_str()},
                 {"x3_cell_spacing",
                  cell2str.at(state.x3_cell_spacing).c_str()}};

            for (const auto& [name, value] : attributes) {
                H5::DataType type;
                if (name == "spatial_order" || name == "time_order" ||
                    name == "geometry" || name == "regime" ||
                    name.find("cell_spacing") != std::string::npos) {

                    type = H5::StrType(H5::PredType::C_S1, 256);
                }
                else if (name == "using_gamma_beta" || name == "mesh_motion") {
                    type = bool_type;
                }
                else if (name == "nx" || name == "ny" || name == "nz" ||
                         name == "chkpt_idx" ||
                         name.find("active_zones") != std::string::npos ||
                         name == "dimensions") {
                    type = int_type;
                }
                else {
                    type = real_type;
                }

                auto att =
                    sim_info.createAttribute(name, type, scalar_dataspace);
                att.write(type, value);
                att.close();
            }
            sim_info.close();
        }
    }   // namespace helpers
}   // namespace simbi
