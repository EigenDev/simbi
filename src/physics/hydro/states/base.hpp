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
#include "physics/hydro/schemes/ib/systems/gravitational_system.hpp"   // for GravitationalSystem
#include "physics/hydro/schemes/ib/systems/system_config.hpp"   // for system_config
#include "physics/hydro/types/context.hpp"           // for HydroContext
#include "physics/hydro/types/generic_structs.hpp"   // for anyConserved, anyPrimitive
#include <list>

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

            DUAL WaveSpeedFunctor(const D* d) : derived_ptr(d) {}

            template <typename T>
            DUAL real
            operator()(const real acc, const T& prim, const luint gid) const
            {
                auto speeds   = derived_ptr->get_wave_speeds(prim);
                auto cell     = derived_ptr->mesh().get_cell_from_global(gid);
                auto local_dt = calc_local_dt(speeds, cell);
                return std::min(acc, local_dt);
            }
        };

        void adapt_dt()
        {
            const auto& derived = static_cast<const Derived&>(*this);
            auto functor        = WaveSpeedFunctor<Derived>(&derived);

            const auto gas_dt = prims_.reduce(
                                    static_cast<real>(INFINITY),
                                    functor,
                                    this->full_policy()
                                ) *
                                cfl_;

            real orbital_dt = INFINITY;
            if (gravitational_system_) {
                orbital_dt = gravitational_system_->get_orbital_timestep(cfl_);
            }

            time_manager_.set_dt(std::min(gas_dt, orbital_dt));
        }

        DUAL conserved_t ib_sources(
            const auto& prim,
            const auto& cell,
            std::tuple<size_type, size_type, size_type>&& coords
        )
        {
            return gravitational_system_->apply_forces_to_fluid(
                prim,
                cell,
                coords,
                context_,
                time_step()
            );
        }

      private:
        // state
        std::vector<std::vector<real>> state_;
        atomic_bool in_failure_state_;

        // physical / numerical parameters
        real gamma_;
        real cfl_;

        Mesh<Dims> mesh_;
        ExecutionPolicyManager<Dims> exec_policy_manager_;
        TimeManager time_manager_;
        SolverManager solver_config_;
        std::unique_ptr<IOManager<Dims>> io_manager_;
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
        std::unique_ptr<ibsystem::GravitationalSystem<real, Dims>>
            gravitational_system_;
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
                  std::make_unique<IOManager<Dims>>(
                      solver_config_,
                      init_conditions
                  )
              ),
              // protected references to commonly used values
              gamma(gamma_)

        {
            init_gravitational_system(init_conditions);
        }

        DUAL conserved_t hydro_sources(const auto& cell) const
        {
            if (null_sources()) {
                return conserved_t{};
            }

            conserved_t res;
            const auto iof = io_manager_.get();
            if constexpr (Dims == 1) {
                const auto x1c = cell.centroid()[0];
                iof->call_hydro_source(x1c, time(), res.data());
            }
            else if constexpr (Dims == 2) {
                const auto [x1c, x2c] = cell.centroid();
                iof->call_hydro_source(x1c, x2c, time(), res.data());
            }
            else {
                const auto [x1c, x2c, x3c] = cell.centroid();
                iof->call_hydro_source(x1c, x2c, x3c, time(), res.data());
            }
            return res;
        }

        DUAL conserved_t
        gravity_sources(const auto& prims, const auto& cell) const
        {
            if (null_gravity()) {
                return conserved_t{};
            }
            const auto c = cell.centroid();

            conserved_t gravity;
            const auto iof = io_manager_.get();
            if constexpr (Dims > 1) {
                if constexpr (Dims > 2) {
                    iof->call_gravity_source(
                        c[0],
                        c[1],
                        c[2],
                        time(),
                        gravity.data()
                    );
                }
                else {
                    iof->call_gravity_source(
                        c[0],
                        c[1],
                        time(),
                        gravity.data()
                    );
                }
            }
            else {
                iof->call_gravity_source(c[0], time(), gravity.data());
            }

            // gravity source term is rho * g_vec for momentum and
            // rho * v.dot(g_vec) for energy
            return {
              0.0,
              prims.rho() * gravity.momentum(),
              prims.velocity().dot(gravity.momentum())
            };
        }

        void apply_boundary_conditions()
        {
            constexpr auto need_corners = sim_type::MHD<R>;
            conserved_boundary_manager_.sync_boundaries(
                full_policy(),
                cons_,
                cons_.contract(halo_radius()),
                bcs(),
                io_manager_ ? Maybe<const IOManager<Dims>*>(io_manager_.get())
                            : Nothing,
                mesh_,
                time(),
                need_corners
            );
        }

        static DUAL real calc_local_dt(const auto& speeds, const auto& cell)
        {
            auto dt = INFINITY;
            for (size_type ii = 0; ii < Dims; ++ii) {
                auto dx    = cell.width(ii);
                auto dt_dx = dx / std::min(speeds[2 * ii], speeds[2 * ii + 1]);
                dt         = std::min<real>(dt, dt_dx);
            }
            return dt;
        };

        void init_gravitational_system(const InitialConditions& init)
        {
            if (init.contains("body_system")) {
                const auto& sys_props = init.get_dict("body_system");

                // Extract basic gravitational config
                ibsystem::config::GravitationalConfig<real> grav_config;
                if (sys_props.contains("prescribed_motion")) {
                    grav_config.prescribed_motion =
                        sys_props.at("prescribed_motion").get<bool>();
                }
                if (sys_props.contains("reference_frame")) {
                    grav_config.reference_frame =
                        sys_props.at("reference_frame").get<std::string>();
                }

                // Check system type
                if (sys_props.contains("system_type")) {
                    const auto& system_type =
                        sys_props.at("system_type").get<std::string>();

                    if (system_type == "binary" &&
                        sys_props.contains("binary_config")) {
                        if constexpr (Dims >= 2) {
                            const auto& binary_props =
                                sys_props.at("binary_config").get<ConfigDict>();

                            // Extract binary config
                            real total_mass =
                                binary_props.contains("total_mass")
                                    ? binary_props.at("total_mass").get<real>()
                                    : 1.0;
                            real semi_major =
                                binary_props.contains("semi_major")
                                    ? binary_props.at("semi_major").get<real>()
                                    : 1.0;
                            real eccentricity =
                                binary_props.contains("eccentricity")
                                    ? binary_props.at("eccentricity")
                                          .get<real>()
                                    : 0.0;
                            real mass_ratio =
                                binary_props.contains("mass_ratio")
                                    ? binary_props.at("mass_ratio").get<real>()
                                    : 1.0;

                            auto binary_comps =
                                binary_props.at("components")
                                    .get<std::list<ConfigDict>>();
                            auto body_component = [](const ConfigDict& props) {
                                ibsystem::config::GravitationalComponent<real>
                                    comp;
                                comp.mass   = props.at("mass").get<real>();
                                comp.radius = props.at("radius").get<real>();
                                comp.softening_length =
                                    props.at("softening_length").get<real>();
                                comp.accretion_efficiency =
                                    props.at("accretion_efficiency")
                                        .get<real>();
                                comp.accretion_radius =
                                    props.at("accretion_radius").get<real>();
                                comp.two_way_coupling =
                                    props.at("two_way_coupling").get<bool>();
                                comp.is_an_accretor =
                                    props.at("is_an_accretor").get<bool>();
                                return comp;
                            };
                            // Use factory method to create binary system
                            gravitational_system_ =
                                ibsystem::GravitationalSystem<real, Dims>::
                                    create_binary_system(
                                        mesh_,
                                        init.gamma,
                                        total_mass,
                                        semi_major,
                                        {body_component(binary_comps.front()),
                                         body_component(binary_comps.back())},
                                        eccentricity,
                                        mass_ratio,
                                        grav_config.prescribed_motion
                                    );
                        }
                    }
                    else {
                        throw std::runtime_error(
                            "Invalid gravitational system type: " + system_type
                        );
                    }
                }
            }
            else if (!init.immersed_bodies.empty()) {
                gravitational_system_ =
                    std::make_unique<ibsystem::GravitationalSystem<real, Dims>>(
                        mesh_,
                        init.gamma
                    );

                for (const auto& [body_type, props] : init.immersed_bodies) {
                    // Extract common properties
                    const auto& position =
                        props.at("position").get<std::vector<real>>();
                    const auto& velocity =
                        props.at("velocity").get<std::vector<real>>();
                    const real mass   = props.at("mass").get<real>();
                    const real radius = props.at("radius").get<real>();

                    gravitational_system_->add_body(
                        body_type,
                        spatial_vector_t<real, Dims>(position),
                        spatial_vector_t<real, Dims>(velocity),
                        mass,
                        radius,
                        props
                    );
                }

                gravitational_system_->init_system();
            }
        }

        void simulate(
            const std::function<real(real)> a,
            const std::function<real(real)> adot
        )
        {
            auto& derived = static_cast<Derived&>(*this);
            // load the user-defined functions if any
            io().load_functions();

            cons_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            prims_.resize(this->total_zones()).reshape({nz(), ny(), nx()});
            // Copy the state array into real& profile variables
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

        DUAL void advance_system()
        {
            auto& derived = static_cast<Derived&>(*this);

            // orbital dynamics (if any bodies are present)
            if (gravitational_system_) {
                if constexpr (sim_type::Newtonian<R>) {
                    gravitational_system_->update_system(time(), time_step());
                }
            }

            // gas dynamics
            derived.advance_impl();
            derived.cons2prim_impl();
            adapt_dt();
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
        DUAL const auto& io() const { return *io_manager_; }
        DUAL auto& io() { return *io_manager_; }
        DUAL const auto& conserved_boundary_manager() const
        {
            return conserved_boundary_manager_;
        }

        DUAL auto adiabatic_index() const { return gamma; }

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
        DUAL auto temporal_order() const
        {
            return solver_config_.temporal_order();
        }

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
            return (*io_manager_).data_directory();
        }
        DUAL auto current_iter() const { return (*io_manager_).current_iter(); }
        DUAL auto checkpoint_zones() const
        {
            return (*io_manager_).checkpoint_zones();
        }
        DUAL auto checkpoint_index() const
        {
            return (*io_manager_).checkpoint_index();
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
        DUAL auto has_immersed_bodies() const
        {
            return gravitational_system_ != nullptr;
        };

        DUAL auto checkpoint_identifier() const
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
        bool is_in_initial_primitive_state() const
        {
            return time() == 0.0 || checkpoint_index() == 0;
        }

        auto gravitational_system() const
        {
            return gravitational_system_.get();
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
