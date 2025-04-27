/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            base.hpp
 *  * @brief           base state for all hydro states to derive from
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef BASE_HPP
#define BASE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use...
#include "core/managers/boundary_manager.hpp"      // for boundary_manager
#include "core/managers/exec_policy_manager.hpp"   // for ExecutionPolicy
#include "core/managers/io_manager.hpp"            // for IOManager
#include "core/managers/solver_manager.hpp"        // for SolverManager
#include "core/managers/time_manager.hpp"          // for TimeManager
#include "core/types/containers/vector.hpp"
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "core/types/utility/managed.hpp"           // for Managed
#include "geometry/mesh/mesh.hpp"                   // for Mesh
#include "io/console/logger.hpp"                    // for logger
#include "physics/hydro/schemes/ib/systems/body_system_operations.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"
#include "physics/hydro/schemes/ib/systems/component_generator.hpp"
#include "physics/hydro/types/context.hpp"           // for HydroContext
#include "physics/hydro/types/generic_structs.hpp"   // for anyConserved, anyPrimitive
#include <limits>

namespace simbi {
    template <typename Derived, size_type Dims, Regime R>
    class HydroBase : public Managed<global::managed_memory>
    {
      public:
        // Type aliases for child classes
        using derived_t   = Derived;
        using primitive_t = anyPrimitive<Dims, R>;
        using conserved_t = anyConserved<Dims, R>;
        using eigenvals_t = Eigenvals<Dims, R>;
        using function_t  = typename helpers::real_func<Dims>::type;

        template <typename D>
        struct WaveSpeedFunctor {
            // pointers work better on gpu
            const D* derived_ptr;

            WaveSpeedFunctor(const D* d) : derived_ptr(d) {}

            template <typename T>
            DEV real
            operator()(const real acc, const T& prim, const luint gid) const
            {
                auto speeds   = derived_ptr->get_wave_speeds(prim);
                auto cell     = derived_ptr->mesh().get_cell_from_global(gid);
                auto local_dt = calc_local_dt(speeds, cell);
                return my_min(acc, local_dt);
            }
        };

        void adapt_dt()
        {
            const auto& derived = static_cast<const Derived&>(*this);
            auto functor        = WaveSpeedFunctor<Derived>(&derived);

            const auto gas_dt = prims_.reduce(
                                    std::numeric_limits<real>::infinity(),
                                    functor,
                                    this->full_policy()
                                ) *
                                cfl_;
            real system_dt = std::numeric_limits<real>::infinity();
            if (body_system_) {
                system_dt = ibsystem::functions::get_system_timestep(
                    *body_system_,
                    cfl_
                );
            }
            time_manager_.set_dt(std::min(gas_dt, system_dt));
        }

        DEV conserved_t ib_sources(
            const auto& prim,
            const auto& cell,
            std::tuple<size_type, size_type, size_type>&& coords
        )
        {
            auto [fluid_change, delta_buffer] =
                ibsystem::functions::apply_forces_to_fluid(
                    *body_system_,
                    prim,
                    cell,
                    coords,
                    context_,
                    time_step()
                );
            accumulator_->add_buffer(delta_buffer);
            return std::move(fluid_change);
        }

      private:
        // state
        std::vector<std::vector<real>> state_;
        std::atomic<bool> in_failure_state_;

        // physical / numerical parameters
        real gamma_;
        real cfl_;

        Mesh<Dims> mesh_;
        ExecutionPolicyManager<Dims> exec_policy_manager_;
        TimeManager time_manager_;
        SolverManager solver_config_;
        util::smart_ptr<IOManager<Dims>> io_manager_;
        boundary_manager<conserved_t, Dims> conserved_boundary_manager_;

        bool was_interrupted_{false};
        bool has_crashed_{false};

        void deallocate_state()
        {
            for (auto& vec : state_) {
                vec.clear();
            }
            state_.clear();
            state_.shrink_to_fit();
        }

      protected:
        // Common state members
        ndarray<Maybe<primitive_t>, Dims> prims_;
        ndarray<conserved_t, Dims> cons_;
        util::smart_ptr<ibsystem::ComponentBodySystem<real, Dims>> body_system_;
        util::smart_ptr<ibsystem::BodyDeltaCombiner<real, Dims>> accumulator_;
        HydroContext context_;

        HydroBase() = default;

        ~HydroBase() = default;

        HydroBase(
            std::vector<std::vector<real>> state,
            InitialConditions& init_conditions
        )
            : state_(std::move(state)),
              in_failure_state_(false),
              gamma_(init_conditions.gamma),
              cfl_(init_conditions.cfl),
              mesh_(init_conditions),
              exec_policy_manager_(mesh_.grid(), init_conditions),
              time_manager_(init_conditions),
              solver_config_(init_conditions),
              io_manager_(
                  util::make_unique<IOManager<Dims>>(
                      solver_config_,
                      init_conditions
                  )
              ),
              // protected references to commonly used values
              gamma(gamma_)
        {
            init_body_system(init_conditions);
        }

        DEV conserved_t hydro_sources(const auto& prims, const auto& cell) const
        {
            if (null_sources()) {
                return conserved_t{};
            }

            const auto* iof = io_manager_.get();
            auto coords     = cell.centroid();
            return iof
                ->call_hydro_source(coords, time(), prims.to_conserved(gamma_));
        }

        DEV conserved_t
        gravity_sources(const auto& prims, const auto& cell) const
        {
            if (null_gravity()) {
                return conserved_t{};
            }

            const auto c = cell.centroid();
            spatial_vector_t<real, Dims> gravity_vec;
            const auto* iof = io_manager_.get();
            if constexpr (Dims == 1) {
                iof->call_gravity_source(c[0], time(), gravity_vec.data());
            }
            else if constexpr (Dims == 2) {
                iof->call_gravity_source(
                    c[0],
                    c[1],
                    time(),
                    gravity_vec.data()
                );
            }
            else {
                iof->call_gravity_source(
                    c[0],
                    c[1],
                    c[2],
                    time(),
                    gravity_vec.data()
                );
            }

            const auto dp_dt = prims.labframe_density() * gravity_vec;
            const auto v_old = prims.velocity();
            const auto v_new = (prims.spatial_momentum(gamma_) + dp_dt) /
                               prims.labframe_density();
            const auto v_avg = 0.5 * (v_old + v_new);
            const auto dE_dt = vecops::dot(dp_dt, v_avg);

            // gravity source term is rho * g_vec for momentum and
            // rho * v.dot(g_vec) for energy
            // here, we return the force and power to later
            // be multiplied by the timestep
            return {0.0, dp_dt, dE_dt};
        }

        void apply_boundary_conditions()
        {
            constexpr auto need_corners = sim_type::MHD<R>;
            conserved_boundary_manager_.sync_boundaries(
                full_policy(),
                cons_,
                cons_.contract(halo_radius()),
                bcs(),
                mesh_,
                io_manager_.get(),
                time(),
                time_step(),
                need_corners
            );
        }

        DEV static real calc_local_dt(const auto& speeds, const auto& cell)
        {
            auto dt = std::numeric_limits<real>::infinity();
            for (size_type ii = 0; ii < Dims; ++ii) {
                auto dx    = cell.width(ii);
                auto dt_dx = dx / my_min(speeds[2 * ii], speeds[2 * ii + 1]);
                dt         = my_min<real>(dt, dt_dx);
            }
            return dt;
        };

        void init_body_system(const InitialConditions& init)
        {
            body_system_ = ibsystem::create_body_system_from_config<real, Dims>(
                mesh_,
                init
            );
            if (body_system_) {
                accumulator_ =
                    util::make_unique<ibsystem::BodyDeltaCombiner<real, Dims>>(
                    );
            }
        }

        void simulate(
            const std::function<real(real)> a,
            const std::function<real(real)> adot
        )
        {
            auto& derived = static_cast<Derived&>(*this);

            cons_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            prims_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            // Move the state array
            for (size_type ii = 0; ii < this->total_zones(); ii++) {
                for (int q = 0; q < conserved_t::nmem; q++) {
                    cons_[ii][q] = state_[q][ii];
                }
            }
            deallocate_state();

            // Initialize simulation
            derived.init_simulation();
            derived.cons2prim_impl();
            adapt_dt();

            // Main simulation loop
            detail::logger::with_logger(derived, tend(), [&] {
                // Single timestep advance
                advance_system();

                // Update time
                time_manager_.advance(step());

                // move the mesh if needed
                mesh_.geometry_state().move_grid(a, adot, time(), time_step());
            });
        }

        void advance_system()
        {
            auto& derived = static_cast<Derived&>(*this);

            // immersed body dynamics (if any bodies are present)
            if (body_system_) {
                *body_system_ = ibsystem::functions::update_body_system(
                    std::move(*body_system_),
                    time(),
                    time_step()
                );
            }

            // gas dynamics
            derived.advance_impl();
            derived.cons2prim_impl();
            adapt_dt();

            // generate new system based on the new body
            // configuration / state
            if (body_system_) {
                // update system on the host
                // but make sure gpu finishes
                // the work before we do this
                gpu::api::deviceSynch();
                *body_system_ =
                    accumulator_->apply_to(std::move(*body_system_));
            }
        }

        // protected references to commonly used values
        const real& gamma;

      public:
        void sync_to_device()
        {
            cons_.sync_to_device();
            prims_.sync_to_device();
            bcs().sync_to_device();
        }

        void sync_to_host()
        {
            cons_.sync_to_host();
            prims_.sync_to_host();
        }
        // accessors
        const auto& primitives() const { return prims_; }
        const auto& conserveds() const { return cons_; }
        const auto& state() const { return state_; }
        DUAL const auto& mesh() const { return mesh_; }
        const auto& exec_policy_manager() const { return exec_policy_manager_; }
        const auto& time_manager() const { return time_manager_; }
        auto& time_manager() { return time_manager_; }
        const auto& solver_config() const { return solver_config_; }
        const auto& io() const { return *io_manager_; }
        auto& io() { return *io_manager_; }
        const auto& conserved_boundary_manager() const
        {
            return conserved_boundary_manager_;
        }

        auto adiabatic_index() const { return gamma; }

        // accessors from solver manager class
        DUAL auto solver_type() const { return solver_config_.solver_type(); }
        DUAL auto using_pcm() const { return solver_config_.is_pcm(); }
        auto using_rk1() const { return solver_config_.is_rk1(); }
        DUAL auto quirk_smoothing() const { return solver_config_.is_quirk(); }
        DUAL auto null_gravity() const { return solver_config_.null_gravity(); }
        DUAL auto null_sources() const { return solver_config_.null_sources(); }
        DUAL auto plm_theta() const { return solver_config_.plm_theta(); }
        DUAL auto step() const { return solver_config_.step(); }
        auto& bcs() { return solver_config_.boundary_conditions(); }
        const auto& bcs() const { return solver_config_.boundary_conditions(); }
        auto spatial_order() const { return solver_config_.spatial_order(); }
        auto temporal_order() const { return solver_config_.temporal_order(); }

        // accessors from time manager class
        DUAL auto time() const { return time_manager_.time(); }
        DUAL auto dt() const { return time_manager_.dt(); }
        auto tend() const { return time_manager_.tend(); }
        auto checkpoint_interval() const
        {
            return time_manager_.checkpoint_interval();
        }
        auto dlogt() const { return time_manager_.dlogt(); }
        auto time_to_write_checkpoint() const
        {
            return time_manager_.time_to_write_checkpoint();
        }

        // accessors from io manager class
        auto data_directory() const { return (*io_manager_).data_directory(); }
        auto current_iter() const { return (*io_manager_).current_iter(); }
        auto checkpoint_zones() const
        {
            return (*io_manager_).checkpoint_zones();
        }
        auto checkpoint_index() const
        {
            return (*io_manager_).checkpoint_index();
        }

        // accessors from execution manager class
        auto full_policy() const { return exec_policy_manager_.full_policy(); }

        auto interior_policy() const
        {
            return exec_policy_manager_.interior_policy();
        }

        auto full_xvertex_policy() const
        {
            return exec_policy_manager_.full_xvertex_policy();
        }

        auto full_yvertex_policy() const
        {
            return exec_policy_manager_.full_yvertex_policy();
        }

        auto full_zvertex_policy() const
        {
            return exec_policy_manager_.full_zvertex_policy();
        }

        auto xvertex_policy() const
        {
            return exec_policy_manager_.xvertex_policy();
        }

        auto yvertex_policy() const
        {
            return exec_policy_manager_.yvertex_policy();
        }

        auto zvertex_policy() const
        {
            return exec_policy_manager_.zvertex_policy();
        }

        // some mixed accesors
        DUAL auto time_step() const { return dt() * step(); }
        // accessors from grid manager class
        auto nx() const { return mesh_.grid().total_gridsize(0); }
        auto ny() const { return mesh_.grid().total_gridsize(1); }
        auto nz() const { return mesh_.grid().total_gridsize(2); }
        auto active_nx() const { return mesh_.grid().active_gridsize(0); }
        auto active_ny() const { return mesh_.grid().active_gridsize(1); }
        auto active_nz() const { return mesh_.grid().active_gridsize(2); }
        auto total_zones() const { return mesh_.grid().total_zones(); }
        DUAL auto has_immersed_bodies() const
        {
            return body_system_ != nullptr;
        };

        auto checkpoint_identifier() const
        {
            // if log-time checkpointing is enabled
            // the checkpoint identifier is the current checkpoint index
            // otherwise it is the current time
            return time_manager_.log_time_enabled()
                       ? io().checkpoint_index()
                       : time_manager_.checkpoint_time();
        }

        void update_next_checkpoint_location()
        {
            io().increment_checkpoint_idx();
            time_manager().update_next_checkpoint_time();
        }

        constexpr collapsable<Dims> get_shape(const auto& policy) const
        {
            return {policy.grid_size.z, policy.grid_size.y, policy.grid_size.x};
        }

        // allow derived classes to check/set failure state
        bool is_in_failure_state() const { return in_failure_state_.load(); }

        void set_failure_state(bool state) { in_failure_state_.store(state); }

        void was_interrupted() { was_interrupted_ = true; }
        void has_crashed() { has_crashed_ = true; }
        void has_failed() { set_failure_state(true); }
        bool has_been_interrupted() const { return was_interrupted_; }
        bool is_in_initial_primitive_state() const
        {
            return time() == 0.0 || checkpoint_index() == 0;
        }

        auto body_system() const { return body_system_.get(); }

        // accessors from mesh class
        auto halo_radius() const { return mesh_.halo_radius(); }

        // utility method for derived classes
        bool check_and_set_failure(bool condition) const
        {
            if (condition) {
                set_failure_state(true);
                return true;
            }
            return false;
        }
    };
}   // namespace simbi
#endif
