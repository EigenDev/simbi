/**
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 * @file       base.hpp
 * @brief      base state for all hydro states to derive from
 *
 * @note
 * @history:
 *   Version   Date            Author          Modification    Email
 *   V0.8.0    Dec-03-2023     Marcus DuPont marcus.dupont@princeton.edu
 *
 * @verbatim
 * ==============================================================================
 *
 * ==============================================================================
 * @endverbatim
 * ***********************(C) COPYRIGHT 2025 Marcus DuPont**********************
 */
#ifndef BASE_HPP
#define BASE_HPP

#include "build_options.hpp"   // for real, luint, global::managed_memory, use...
#include "core/managers/boundary_manager.hpp"       // for boundary_manager
#include "core/managers/exec_policy_manager.hpp"    // for ExecutionPolicy
#include "core/managers/io_manager.hpp"             // for IOManager
#include "core/managers/solver_manager.hpp"         // for SolverManager
#include "core/managers/time_manager.hpp"           // for TimeManager
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "core/types/utility/managed.hpp"           // for Managed
#include "geometry/mesh/mesh.hpp"                   // for Mesh
#include "io/console/logger.hpp"                    // for logger
#include "physics/hydro/types/generic_structs.hpp"   // for anyConserved, anyPrimitive

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

      private:
        // state
        std::vector<std::vector<real>> state_;
        atomic_bool in_failure_state_;

        // physical / numerical parameters
        real gamma_;
        real cfl_;
        real hllc_z_;

        Mesh<Dims> mesh_;
        ExecutionPolicyManager<Dims> exec_policy_manager_;
        TimeManager time_manager_;
        SolverManager solver_config_;
        IOManager io_manager_;
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
              hllc_z_((gamma_ - 1.0) / (2.0 * gamma_)),
              mesh_(init_conditions),
              exec_policy_manager_(mesh_.grid(), init_conditions),
              time_manager_(init_conditions),
              solver_config_(init_conditions),
              io_manager_(solver_config_, init_conditions),
              // protected references to commonly used values
              gamma(gamma_),
              cfl(cfl_),
              hllc_z(hllc_z_)

        {
            // if (std::getenv("USE_OMP")) {
            //     global::use_omp = true;
            //     if (const char* omp_tnum = std::getenv("OMP_NUM_THREADS")) {
            //         omp_set_num_threads(std::stoi(omp_tnum));
            //     }
            // }
        }

        DUAL conserved_t hydro_sources(const auto& cell) const
        {
            if (null_sources()) {
                return conserved_t{};
            }

            conserved_t res;
            if constexpr (Dims == 1) {
                const auto x1c = cell.centroid()[0];
                // hydro_source(x1c, t, res);
            }
            else if constexpr (Dims == 2) {
                const auto [x1c, x2c] = cell.centroid();
                // hydro_source(x1c, x2c, t, res);
            }
            else {
                const auto [x1c, x2c, x3c] = cell.centroid();
                // hydro_source(x1c, x2c, x3c, t, res);
            }

            return res;
        }

        DUAL conserved_t
        gravity_sources(const auto& prims, const auto& cell) const
        {
            if (null_gravity()) {
                return conserved_t{};
            }
            const auto x1c = cell.centroid_coordinate(0);

            conserved_t res;
            // // gravity only changes the momentum and energy
            // if constexpr (dim > 1) {
            //     const auto x2c = cell.centroid_coordinate(1);
            //     if constexpr (dim > 2) {
            //         const auto x3c = cell.centroid_coordinate(2);
            //         gravity_source(x1c, x2c, x3c, t, res);
            //         res[dimensions + 1] = res[1] * prims[1] +
            //                               res[2] * prims[2] + res[3] *
            //                               prims[3];
            //     }
            //     else {
            //         gravity_source(x1c, x2c, t, res);
            //         res[dimensions + 1] = res[1] * prims[1] + res[2] *
            //         prims[2];
            //     }
            // }
            // else {
            //     gravity_source(x1c, t, res);
            //     res[dimensions + 1] = res[1] * prims[1];
            // }

            return res;
        }

        void adapt_dt()
        {
            auto& derived = static_cast<Derived&>(*this);

            auto calc_local_dt =
                [] DEV(const auto& speeds, const auto& cell) -> real {
                auto dt = INFINITY;
                for (size_type ii = 0; ii < Dims; ++ii) {
                    auto dx = cell.width(ii);
                    auto dt_dx =
                        dx / std::min(speeds[2 * ii], speeds[2 * ii + 1]);
                    dt = std::min<real>(dt, dt_dx);
                }
                return dt;
            };

            // Single-pass reduction that combines fold and reduce
            time_manager_.set_dt(
                prims_.reduce(
                    static_cast<real>(INFINITY),
                    [&derived,
                     calc_local_dt,
                     this](const auto& acc, const auto& prim, const luint gid) {
                        auto speeds   = derived.get_wave_speeds(prim);
                        auto cell     = this->mesh().get_cell_from_global(gid);
                        auto local_dt = calc_local_dt(speeds, cell);

                        return std::min(acc, local_dt);
                    },
                    this->full_policy()
                ) *
                cfl_
            );
        };

        void simulate(
            const std::function<real(real)> a,
            const std::function<real(real)> adot
        )
        {
            auto& derived = static_cast<Derived&>(*this);
            cons_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            prims_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            std::cout << "Simulating with " << this->total_zones()
                      << " zones\n";

            // Copy the state array into real& profile variables
            for (size_type ii = 0; ii < this->total_zones(); ii++) {
                for (int q = 0; q < conserved_t::nmem; q++) {
                    cons_[ii][q] = state_[q][ii];
                }
            }
            deallocate_state();

            // // Initialize simulation
            derived.init_simulation();
            derived.cons2prim_impl();
            adapt_dt();

            // Main simulation loop
            detail::logger::with_logger(derived, tend(), [&] {
                // Single timestep advance
                advance_system();

                // Update time
                time_manager_.advance();

                // move the mesh if needed
                mesh_.geometry_state().move_grid(a, adot, time(), time_step());
            });
        }

        DUAL void advance_system()
        {
            auto& derived = static_cast<Derived&>(*this);
            // 1. advance the regime-specific system
            derived.advance_impl();

            // 2. convert to primitives
            derived.cons2prim_impl();

            // 3. adapt timestep
            adapt_dt();
        }

        // protected references to commonly used values
        const real& gamma;
        const real& cfl;
        const real& hllc_z;

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
        DUAL const auto& primitives() const { return prims_; }
        DUAL const auto& conserveds() const { return cons_; }
        DUAL const auto& state() const { return state_; }
        DUAL const auto& mesh() const { return mesh_; }
        DUAL const auto& exec_policy_manager() const
        {
            return exec_policy_manager_;
        }
        DUAL const auto& time_manager() const { return time_manager_; }
        DUAL auto& time_manager() { return time_manager_; }
        DUAL const auto& solver_config() const { return solver_config_; }
        DUAL const auto& io() const { return io_manager_; }
        DUAL auto& io() { return io_manager_; }
        DUAL const auto& conserved_boundary_manager() const
        {
            return conserved_boundary_manager_;
        }

        DUAL auto adiabatic_gamma() const { return gamma; }

        // accessors from solver manager class
        DUAL auto solver_type() const { return solver_config_.solver_type(); }
        DUAL auto using_pcm() const { return solver_config_.is_pcm(); }
        DUAL auto using_rk1() const { return solver_config_.is_rk1(); }
        DUAL auto quirk_smoothing() const { return solver_config_.is_quirk(); }
        DUAL auto null_gravity() const { return solver_config_.null_gravity(); }
        DUAL auto null_sources() const { return solver_config_.null_sources(); }
        DUAL auto plm_theta() const { return solver_config_.plm_theta(); }
        DUAL auto step() const { return solver_config_.step(); }
        DUAL auto& bcs() { return solver_config_.boundary_conditions(); }
        DUAL const auto& bcs() const
        {
            return solver_config_.boundary_conditions();
        }
        DUAL auto spatial_order() const
        {
            return solver_config_.spatial_order();
        }
        DUAL auto time_order() const { return solver_config_.time_order(); }

        // accessors from time manager class
        DUAL auto time() const { return time_manager_.time(); }
        DUAL auto dt() const { return time_manager_.dt(); }
        DUAL auto tend() const { return time_manager_.tend(); }
        DUAL auto checkpoint_interval() const
        {
            return time_manager_.checkpoint_interval();
        }
        DUAL auto dlogt() const { return time_manager_.dlogt(); }
        DUAL auto time_to_write_checkpoint() const
        {
            return time_manager_.time_to_write_checkpoint();
        }

        // accessors from io manager class
        DUAL auto data_directory() const
        {
            return io_manager_.data_directory();
        }
        DUAL auto current_iter() const { return io_manager_.current_iter(); }
        DUAL auto checkpoint_zones() const
        {
            return io_manager_.checkpoint_zones();
        }
        DUAL auto checkpoint_idx() const
        {
            return io_manager_.checkpoint_idx();
        }

        // accessors from execution manager class
        DUAL auto full_policy() const
        {
            return exec_policy_manager_.full_policy();
        }

        DUAL auto interior_policy() const
        {
            return exec_policy_manager_.interior_policy();
        }

        DUAL auto full_xvertex_policy() const
        {
            return exec_policy_manager_.full_xvertex_policy();
        }

        DUAL auto full_yvertex_policy() const
        {
            return exec_policy_manager_.full_yvertex_policy();
        }

        DUAL auto full_zvertex_policy() const
        {
            return exec_policy_manager_.full_zvertex_policy();
        }

        DUAL auto xvertex_policy() const
        {
            return exec_policy_manager_.xvertex_policy();
        }

        DUAL auto yvertex_policy() const
        {
            return exec_policy_manager_.yvertex_policy();
        }

        DUAL auto zvertex_policy() const
        {
            return exec_policy_manager_.zvertex_policy();
        }

        // some mixed accesors
        DUAL auto time_step() const { return dt() * step(); }
        // accessors from grid manager class
        DUAL auto nx() const { return mesh_.grid().total_gridsize(0); }
        DUAL auto ny() const { return mesh_.grid().total_gridsize(1); }
        DUAL auto nz() const { return mesh_.grid().total_gridsize(2); }
        DUAL auto total_zones() const { return mesh_.grid().total_zones(); }

        DUAL auto checkpoint_identifier() const
        {
            // if log-time checkpointing is enabled
            // the checkpoint identifier is the current checkpoint index
            // otherwise it is the current time
            return time_manager_.log_time_enabled()
                       ? io().checkpoint_idx()
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
        DUAL bool is_in_failure_state() const
        {
            return in_failure_state_.load();
        }

        DUAL void set_failure_state(bool state)
        {
            in_failure_state_.store(state);
        }

        void was_interrupted() { was_interrupted_ = true; }
        void has_crashed() { has_crashed_ = true; }
        void has_failed() { set_failure_state(true); }
        bool has_been_interrupted() const { return was_interrupted_; }
        bool is_in_initial_state() const
        {
            return time() == 0.0 || checkpoint_idx() == 0;
        }

        // accessors from mesh class
        DUAL auto halo_radius() const { return mesh_.halo_radius(); }

        // utility method for derived classes
        DUAL bool check_and_set_failure(bool condition) const
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