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
#include "core/functional/fp.hpp"
#include "core/managers/boundary_manager.hpp"      // for boundary_manager
#include "core/managers/exec_policy_manager.hpp"   // for ExecutionPolicy
#include "core/managers/io_manager.hpp"            // for IOManager
#include "core/managers/solver_manager.hpp"        // for SolverManager
#include "core/managers/time_manager.hpp"          // for TimeManager
#include "core/types/containers/collapsable.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/init_conditions.hpp"   // for InitialConditions
#include "core/types/utility/managed.hpp"           // for Managed
#include "geometry/mesh/mesh.hpp"                   // for Mesh
#include "geometry/mesh/refinement/fmr/refinement_functions.hpp"   // for prolongate_value, prolongate
#include "geometry/mesh/refinement/fmr/refinement_manager.hpp"   // for refinement_manager
#include "io/console/logger.hpp"                                 // for logger
#include "physics/hydro/schemes/ib/systems/body_system_operations.hpp"
#include "physics/hydro/schemes/ib/systems/component_body_system.hpp"
#include "physics/hydro/schemes/ib/systems/component_generator.hpp"
#include "physics/hydro/schemes/viscosity/viscous.hpp"   // for visc::get_minimum_viscous_time
#include "physics/hydro/types/context.hpp"           // for HydroContext
#include "physics/hydro/types/generic_structs.hpp"   // for anyConserved, anyPrimitive
#include "util/tools/helpers.hpp"
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

            // if viscosity is present, limit the dt to the minimum viscous time
            real viscous_dt = std::numeric_limits<real>::infinity();
            if (!goes_to_zero(viscosity())) {
                viscous_dt = cfl_ * visc::get_minimum_viscous_time(
                                        full_policy(),
                                        mesh(),
                                        viscosity()
                                    );
            }

            time_manager_.set_dt(std::min({gas_dt, system_dt, viscous_dt}));
        }

        DEV conserved_t ib_sources(
            const auto& prim,
            const auto& cell,
            std::tuple<size_type, size_type, size_type>&& coords
        )
        {
            return ibsystem::functions::apply_forces_to_fluid(
                *body_system_,
                prim,
                cell,
                coords,
                context_,
                time_step(),
                *collector_
            );
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
        util::smart_ptr<ibsystem::GridBodyDeltaCollector<real, Dims>>
            collector_;
        HydroContext context_;

        // fmr-related members
        struct level_data_t {
            ndarray<Maybe<primitive_t>, Dims> prims;
            ndarray<conserved_t, Dims> cons;
        };
        std::vector<level_data_t> level_data_;
        refinement::refinement_manager<Dims> refine_mgr_;
        bool using_refinement_{false};

        // ctors and dtors
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
            context_.viscosity = viscosity();
        }

        DEV conserved_t hydro_sources(const auto& prims, const auto& cell) const
        {
            if (null_sources()) {
                return conserved_t{};
            }

            const auto* iof   = io_manager_.get();
            const auto coords = cell.centroid();
            return iof
                ->call_hydro_source(coords, time(), prims.to_conserved(gamma_));
        }

        DEV conserved_t
        gravity_sources(const auto& prims, const auto& cell) const
        {
            if (null_gravity()) {
                return conserved_t{};
            }

            const auto coords      = cell.centroid();
            const auto* iof        = io_manager_.get();
            const auto gravity_vec = iof->call_gravity_source(coords, time());

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

        DEV real local_sound_speed_squared(const auto& cell) const
        {
            const auto coords   = cell.centroid();
            const auto* iof     = io_manager_.get();
            const auto local_cs = iof->call_local_sound_speed(coords);
            return local_cs * local_cs;
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
                const auto [nax, nay, naz] = init.active_zones();
                collector_                 = util::make_unique<
                                    ibsystem::GridBodyDeltaCollector<real, Dims>>(
                    collapsable<Dims>{naz, nay, nax},
                    2
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
            for (size_type ii = 0; ii < this->total_zones(); ii++) {
                for (int q = 0; q < conserved_t::nmem; q++) {
                    cons_[ii][q] = state_[q][ii];
                }
            }
            deallocate_state();

            if (using_refinement_) {
                init_refinement_levels();
            }

            // init simulation
            derived.init_simulation();
            derived.cons2prim_impl();
            adapt_dt();

            // main simulation loop
            detail::logger::with_logger(derived, tend(), [&] {
                // single timestep advance
                advance_system();

                // update time
                time_manager_.advance(step());

                // move the mesh if needed
                mesh_.geometry_state().move_grid(a, adot, time(), time_step());
            });
        }

        void advance_system()
        {
            auto& derived = static_cast<Derived&>(*this);
            // gas dynamics (might include immersed body effects)
            derived.advance_impl();
            // ensure consistency b/w levels via restriction
            // synchronize_levels();
            derived.cons2prim_impl();
            adapt_dt();

            // generate new system based on the new body
            // configuration / state and move the bodies
            if (body_system_) {
                *body_system_ = collector_->apply_to(std::move(*body_system_));
                *body_system_ = ibsystem::functions::update_body_system(
                    std::move(*body_system_),
                    time(),
                    time_step()
                );
                body_system_->sync_to_device();
            }
        }

        //--- Level-Specific Functions ---//

        // calculate the shape of the level arrays
        auto calculate_level_shape(size_type level) const
        {
            if (level == 0) {
                // base level shape - the entire domain
                return array_t<size_type, Dims>{nz(), ny(), nx()};
            }

            array_t<size_type, Dims> shape;
            const size_type rf = refine_mgr_.refinement_factor;

            for (size_type i = 0; i < Dims; ++i) {
                shape[i] = 0;
            }

            // expand shape to cover all refinement regions at this level
            for (const auto& region : refine_mgr_.regions[level - 1]) {
                for (size_type ii = 0; ii < Dims; ++ii) {
                    // calc refined size of this region
                    size_type region_size =
                        (region.max_bounds[ii] - region.min_bounds[ii]) * rf;
                    // add ghost zones on both sides
                    region_size += 2 * halo_radius();
                    // update shape to maximum required
                    shape[ii] = std::max(shape[ii], region_size);
                }
            }

            return shape;
        }

        // create a policy for the given level
        auto create_level_policy(size_type level) const
        {
            if (level == 0) {
                return full_policy();
            }

            // for refine levels, create policies based on the level's
            // array sizes
            return ExecutionPolicy<>{
              // cons and prim have the same shape
              level_data_[level].cons.shape(),
              {128, 1, 1}   // default block size, TODO: make this configurable
            };
        }

        // update the boundaries of the level data
        void update_level_boundaries(size_type level)
        {
            if (!using_refinement_ || level >= refine_mgr_.max_level) {
                return;
            }

            // coarse level -> fine level (prolongation to fill ghost zones)
            // this prepares the fine level for computation
            for (const auto& region : refine_mgr_.regions[level]) {
                // create expanded region that includes ghost zones
                refinement::refinement_region<Dims> ghost_region = region;

                // expand region to include ghost zones
                for (size_type i = 0; i < Dims; ++i) {
                    ghost_region.min_bounds[i] =
                        (region.min_bounds[i] > halo_radius())
                            ? region.min_bounds[i] - halo_radius()
                            : 0;

                    ghost_region.max_bounds[i] = std::min(
                        region.max_bounds[i] + halo_radius(),
                        level_data_[level].cons.shape()[i]
                    );
                }

                // fill ghost zones via prolongation
                refinement::prolongate(
                    level_data_[level].cons,
                    level_data_[level + 1].cons,
                    ghost_region,
                    refine_mgr_.refinement_factor
                );
            }
        }

        // synchronize levels
        void synchronize_levels()
        {
            // work from finest to coarsest level
            for (size_type level = refine_mgr_.max_level; level > 0; --level) {
                // for each refinement region
                for (const auto& region : refine_mgr_.regions[level - 1]) {
                    // Restrict data from fine to coarse
                    refinement::restrict(
                        level_data_[level].cons,
                        level_data_[level - 1].cons,
                        region,
                        refine_mgr_.refinement_factor
                    );
                }
            }
        }

        //=== FMR ===//
        // my attempt at learning to implement fmr :P
        void init_refinement_levels()
        {
            // set up the level_data_ vector (level 0 is the base level)
            const size_type max_level = refine_mgr_.max_level;
            level_data_.resize(max_level + 1);

            // init level 0 with the base grid data
            level_data_[0].cons  = cons_;
            level_data_[0].prims = prims_;

            // for each subsequent level, init refined regions
            for (size_type level = 1; level <= max_level; ++level) {
                // determine shape for this level's arrays
                auto level_shape = calculate_level_shape(level);
                auto level_size  = fp::product(level_shape);

                // resize arrays for this level
                level_data_[level].cons.resize(level_size);
                level_data_[level].prims.resize(level_size);

                // for each region at this level, init via prolongation
                const size_type prev_level = level - 1;
                for (const auto& region : refine_mgr_.regions[prev_level]) {
                    // prolongate conserved variables from coarse to fine level
                    refinement::prolongate(
                        level_data_[prev_level].cons,
                        level_data_[level].cons,
                        region,
                        refine_mgr_.refinement_factor
                    );
                }
            }
        }

        // get conserved variables at a specific level
        const ndarray<conserved_t, Dims>& level_conserved(size_type level) const
        {
            if (level == 0 || !using_refinement_) {
                return cons_;   // backwards compat
            }
            else {
                return level_data_[level].cons;
            }
        }

        // get primitive variables at a specific level
        const ndarray<Maybe<primitive_t>, Dims>&
        level_primitives(size_type level) const
        {
            if (level == 0 || !using_refinement_) {
                return prims_;   // backwards compat
            }
            else {
                return level_data_[level].prims;
            }
        }

        conserved_t conserved_at_position(
            const array_t<size_type, Dims>& coarse_coords
        ) const
        {
            if (!using_refinement_) {
                return cons_.access(coarse_coords);
            }

            // find the finest level that contains this position
            size_type level = refine_mgr_.level_at(coarse_coords);

            if (level == 0) {
                return level_data_[0].cons.access(coarse_coords);
            }
            else {
                auto fine_coords = refine_mgr_.coarse_to_fine(coarse_coords);
                return level_data_[level].cons.access(fine_coords);
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
        DUAL auto viscosity() const { return solver_config_.viscosity(); }
        DUAL auto shakura_sunyaev_alpha() const
        {
            return solver_config_.shakura_sunyaev_alpha();
        }

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
        DUAL auto cfl_number() const { return cfl_; }
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
