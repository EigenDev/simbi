/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            body.hpp
 *  * @brief           Immersed Boundary Method of Peskin (2002)
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-15
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
 *  * 2025-02-15      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */

#ifndef IMMERSED_BOUNDARY_HPP
#define IMMERSED_BOUNDARY_HPP

#include "build_options.hpp"
#include "core/types/containers/ndarray.hpp"
#include "core/types/containers/vector.hpp"
#include "core/types/utility/enums.hpp"

namespace simbi {
    template <size_type Dims>
    class Cell;

    template <size_type Dims, Regime R>
    struct anyConserved;

    template <size_type Dims, Regime R>
    struct anyPrimitive;

    namespace ib {
        enum class BodyType {
            RIGID,                // Rigid body
            ELASTIC,              // Elastic body
            VISCOUS,              // Viscous body
            POROUS,               // Porous body
            SINK,                 // Fluid sink (accretes mass/momentum)
            SOURCE,               // Fluid source (injections mass/momentum)
            PASSIVE,              // Passive body
            GRAVITATIONAL,        // Gravitational body
            GRAVITATIONAL_SINK,   // Gravitational sink
        };

        // base class for all immersed boundary bodies
        template <typename T, size_type Dims, typename MeshType>
        class BaseBody
        {
          protected:
            const MeshType& mesh_;
            spatial_vector_t<T, Dims> position_;
            spatial_vector_t<T, Dims> velocity_;
            spatial_vector_t<T, Dims> force_;
            spatial_vector_t<T, Dims> fluid_velocity_;
            T mass_;
            T radius_;

            // default policy based on mesh size and whatnot
            DUAL auto get_default_policy() const
            {
                auto grid_sizes = array_t<luint, 3>{
                  mesh_.grid().active_gridsize(0),
                  mesh_.grid().active_gridsize(1),
                  mesh_.grid().active_gridsize(2)
                };

                // default block sizes based on dimensionality
                auto block_sizes = array_t<luint, 3>{256, 1, 1};
                if constexpr (Dims > 1) {
                    block_sizes = {16, 16, 1};
                }
                if constexpr (Dims > 2) {
                    block_sizes = {8, 8, 8};
                }

                return ExecutionPolicy<>(grid_sizes, block_sizes);
            }

          public:
            DUAL BaseBody() = default;
            DUAL BaseBody(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius
            )
                : mesh_(mesh),
                  position_(position),
                  velocity_(velocity),
                  mass_(mass),
                  radius_(radius)
            {
            }

            // accessors
            DUAL auto position() const { return position_; }

            DUAL auto velocity() const { return velocity_; }

            DUAL auto mass() const { return mass_; }

            DUAL auto radius() const { return radius_; }

            DUAL auto force() const { return force_; }

            DUAL auto fluid_velocity() const { return fluid_velocity_; }

            // pure virtual interace
            virtual DUAL void advance_position(const T dt) = 0;
            virtual ~BaseBody()                            = default;

            void interpolate_fluid_velocity(const auto& prim_state) {}
        };

        // Immersed boundary body class
        template <typename T, size_type Dims, typename MeshType>
        class ImmersedBody : public BaseBody<T, Dims, MeshType>
        {
          protected:
            // grid-aware cell info
            struct CellInfo {
                bool is_cut;
                T volume_fraction;
                spatial_vector_t<T, Dims> normal;
                T distance;
            };
            // Cut cell data
            ndarray<CellInfo, Dims> cell_info_;

            T drag_coeff_{0.47};     // default sphere drag coefficient
            bool is_sink_ = false;   // is this a sink particle?

            DUAL auto cut_cell_indices() const
            {
                return cell_info_.filter_indices(
                    [](const auto& cell) { return cell.is_cut; },
                    this->get_default_policy()
                );
            }

            DUAL T compute_volume_fraction(
                const T distance,
                const auto& mesh_cell
            ) const
            {
                // Simple linear approximation
                switch (this->mesh_.geometry()) {
                    case Geometry::SPHERICAL:
                        return compute_spherical_volume_fraction(
                            distance,
                            mesh_cell
                        );
                    case Geometry::CYLINDRICAL:
                        return compute_cylindrical_volume_fraction(
                            distance,
                            mesh_cell
                        );
                    default:
                        return compute_cartesian_volume_fraction(
                            distance,
                            mesh_cell
                        );
                }
            }

            DUAL T compute_spherical_volume_fraction(
                const T distance,
                const auto& mesh_cell
            ) const
            {
                // Compute volume fraction
                return 1.0 - std::abs(distance) / mesh_cell.max_cell_width();
            }

            DUAL T compute_cylindrical_volume_fraction(
                const T distance,
                const auto& mesh_cell
            ) const
            {
                // Compute volume fraction
                return 1.0 - std::abs(distance) / mesh_cell.max_cell_width();
            }

            DUAL T compute_cartesian_volume_fraction(
                const T distance,
                const auto& mesh_cell
            ) const
            {
                // Compute volume fraction
                return 1.0 - std::abs(distance) / mesh_cell.max_cell_width();
            }

            // helper for drag calculation
            DUAL spatial_vector_t<T, Dims> compute_drag_force(
                const auto& cons_state,
                const auto& cell,
                const auto& dA_normal
            ) const
            {
                const auto v_rel = this->velocity_ - this->fluid_velocity_;
                if (v_rel.norm() > 0) {
                    return -0.5 * cons_state.dens() * drag_coeff_ *
                           dA_normal.norm() * v_rel.norm() * v_rel;
                }
                return spatial_vector_t<T, Dims>();
            }

          public:
            // constructor
            DUAL ImmersedBody(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const T drag_coeff = 0.47,
                const bool is_sink = false
            )
                : BaseBody<T, Dims, MeshType>(
                      mesh,
                      position,
                      velocity,
                      mass,
                      radius
                  ),
                  cell_info_(mesh.size()),
                  drag_coeff_(drag_coeff),
                  is_sink_(is_sink)
            {
                update_cut_cells();
            }

            ~ImmersedBody() = default;

            DUAL void advance_position(const T dt) override
            {
                this->position_ += this->velocity_ * dt;
                update_cut_cells();
            }

            DUAL void advance_velocity(const T dt)
            {
                this->velocity_ += this->force_ * dt / this->mass_;
            }

            DUAL void interpolate_fluid_velocity(const auto& prim_state)
            {
                this->fluid_velocity_ = spatial_vector_t<T, Dims>();
                T total_volume        = 0.0;

                for (const auto& idx : cut_cell_indices()) {
                    const auto& cell = cell_info_[idx];
                    this->fluid_velocity_ +=
                        prim_state[idx].velocity() * cell.volume_fraction;
                    total_volume += cell.volume_fraction;
                }

                this->fluid_velocity_ /= total_volume;
            }

            // IBM specific methods
            DUAL void update_cut_cells()
            {
                cell_info_.transform_with_indices(
                    [&](auto& cell, size_type idx) {
                        const auto mesh_cell =
                            this->mesh_.get_cell_from_global(idx);
                        // Use mesh geometry for distance calc
                        auto r =
                            mesh_cell.compute_distance_vector(this->position_);
                        cell.distance = r.norm() - this->radius_;

                        // Use mesh geometry for volume fraction
                        if (std::abs(cell.distance) <=
                            mesh_cell.max_cell_width()) {
                            r.normalize();
                            cell.is_cut          = true;
                            cell.normal          = r;
                            cell.volume_fraction = compute_volume_fraction(
                                cell.distance,
                                mesh_cell
                            );
                        }
                        // std::cout << cell.distance << " "
                        //           << cell.volume_fraction << " " <<
                        //           cell.is_cut
                        //           << " ";
                        // std::cout << cell.normal;
                        // std::cin.get();
                        return cell;
                    },
                    this->get_default_policy()
                );
            }

            DUAL void spread_boundary_forces(auto& cons_state, const T dt)
            {
                using ndarray_t = std::remove_reference_t<decltype(cons_state)>;
                using conserved_t = typename ndarray_t::value_type;
                for (const auto& idx : cut_cell_indices()) {
                    const auto& cell = cell_info_[idx];
                    const auto mesh_cell =
                        this->mesh_.get_cell_from_global(idx);
                    const auto dA_normal = mesh_cell.area_normal(cell.normal);
                    const auto dV        = mesh_cell.volume();
                    auto& state          = cons_state[idx];
                    compute_surface_forces(cell, dA_normal);
                    this->force_ += compute_drag_force(state, cell, dA_normal);

                    // std::cout << "Force: " << this->force_ << std::endl;
                    // std::cin.get();
                    // add to fluid conserved state
                    state += conserved_t{
                      0.0,
                      this->force_ * cell.volume_fraction / dV,
                      this->force_.dot(this->velocity_) * cell.volume_fraction /
                          dV
                    };
                }
            }

            DUAL void update_conserved_state(auto& cons_state, const T dt)
            {
                // Update position
                this->advance_position(dt);

                // Update cut cells
                this->update_cut_cells();

                // Spread force
                this->spread_boundary_forces(cons_state, dt);

                // Apply body forces
                this->apply_body_forces(cons_state, dt);
            }

            // read-only accesors
            DUAL bool is_sink() const { return this->is_sink_; }
            DUAL auto position() const { return this->position_; }
            DUAL auto velocity() const { return this->velocity_; }
            DUAL auto force() const { return this->force_; }
            DUAL auto mass() const { return this->mass_; }
            DUAL auto radius() const { return this->radius_; }
            DUAL auto drag_coeff() const { return this->drag_coeff_; }
            DUAL auto fluid_velocity() const { return this->fluid_velocity_; }
            DUAL auto cell_info() const { return this->cell_info_; }

            // these default implementations can be overridden by derived
            // classes optionally

            // Virtual method for N-body forces
            virtual DUAL void compute_body_forces(
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies
            )
            {
                // Default implementation - no forces
                // this->force_ = spatial_vector_t<T, Dims>();
            }

            virtual DUAL void compute_surface_forces(
                const CellInfo& cell,
                const spatial_vector_t<T, Dims>& dA_normal
            ) = 0;

            virtual DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::NEWTONIAN>, Dims>&
                    cons_states,
                const ndarray<
                    Maybe<anyPrimitive<Dims, Regime::NEWTONIAN>>,
                    Dims>& prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            )
            {
                // no body forces by default
            }

            virtual DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::SRHD>, Dims>& cons_states,
                const ndarray<Maybe<anyPrimitive<Dims, Regime::SRHD>>, Dims>&
                    prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            )
            {
                // no body forces by default
            }

            virtual DUAL void apply_body_forces(
                ndarray<anyConserved<Dims, Regime::RMHD>, Dims>& cons_states,
                const ndarray<Maybe<anyPrimitive<Dims, Regime::RMHD>>, Dims>&
                    prims,
                const std::vector<
                    std::unique_ptr<ImmersedBody<T, Dims, MeshType>>>& bodies,
                const T dt
            )
            {
                // no body forces by default
            }
        };
    }   // namespace ib

}   // namespace simbi

#endif