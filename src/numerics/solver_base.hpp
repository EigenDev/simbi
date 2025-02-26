#ifndef SOLVER_BASE_HPP
#define SOLVER_BASE_HPP
#include "base.hpp"
#include "comomon/traits.hpp"

template <typename Derived>
class SolverBase
{
  protected:
    // Allow derived classes to access protected members
    Derived& derived() { return static_cast<Derived&>(*this); }

    const Derived& derived() const
    {
        return static_cast<const Derived&>(*this);
    }

    // Common simulation steps
    void simulate(
        std::function<real(real)> const& a,
        std::function<real(real)> const& adot
    )
    {
        auto& d = derived();

        // Common initialization
        d.hubble_param = adot(d.t) / a(d.t);
        d.mesh_motion  = (d.hubble_param != 0);
        d.homolog      = d.mesh_motion && d.geometry != Geometry::CARTESIAN;

        // Setup boundary conditions
        d.bcs.resize(d.dim * 2);
        for (int i = 0; i < 2 * d.dim; i++) {
            d.bcs[i] = boundary_cond_map.at(d.boundary_conditions[i]);
        }

        // Initialize arrays
        d.load_functions();
        d.cons.resize(d.total_zones).reshape({d.nz, d.ny, d.nx});
        d.prims.resize(d.total_zones).reshape({d.nz, d.ny, d.nx});

        // Simulation loop setup
        boundary_manager<typename Derived::conserved_t, Derived::dim> bman;
        bman.sync_boundaries(
            d.this->full_policy(),
            d.cons,
            d.cons.contract(2),
            d.bcs
        );

        // Main simulation loop
        d.cons2prim();
        d.adapt_dt();

        simbi::detail::logger::with_logger(d, d.tend, [&] {
            d.advance();
            bman.sync_boundaries(
                d.this->full_policy(),
                d.cons,
                d.cons.contract(2),
                d.bcs
            );
            d.cons2prim();
            d.adapt_dt();

            d.t += d.step * d.dt;
            d.update_mesh_motion(scale_factor, scale_factor_derivative);
        });
    }

    // Common time step adaptation
    template <typename DerivedT = Derived>
    void adapt_dt()
    {
        auto& d = derived();

        auto calc_wave_speeds = [&d](const auto& prim) {
            return d.calc_max_speeds(prim);
        };

        auto calc_local_dt = [&d](const auto& speeds, const auto& cell) {
            return d.compute_timestep(speeds, cell);
        };

        d.dt = d.prims.reduce(
                   static_cast<real>(INFINITY),
                   [calc_wave_speeds,
                    calc_local_dt,
                    &d](const auto& acc, const auto& prim, const luint gid) {
                       const auto [ii, jj, kk] = d.get_indices(gid, d.nx, d.ny);
                       const auto speeds       = calc_wave_speeds(prim);
                       const auto cell         = d.cell_geometry(ii, jj, kk);
                       const auto local_dt     = calc_local_dt(speeds, cell);
                       return std::min(acc, local_dt);
                   },
                   d.this->full_policy()
               ) *
               d.cfl;
    }

    // Common boundary handling
    template <typename DerivedT = Derived>
    void handle_boundaries()
    {
        auto& d = derived();
        boundary_manager<typename Derived::conserved_t, Derived::dim> bman;
        bman.sync_boundaries(
            d.this->full_policy(),
            d.cons,
            d.cons.contract(2),
            d.bcs
        );
    }
};
#endif