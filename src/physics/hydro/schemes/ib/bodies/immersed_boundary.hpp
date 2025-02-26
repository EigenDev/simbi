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
    namespace ib {
        enum class BodyType {
            RIGID,     // Rigid body
            ELASTIC,   // Elastic body
            VISCOUS,   // Viscous body
            POROUS,    // Porous body
            SINK,      // Fluid sink (accretes mass/momentum)
            SOURCE,    // Fluid source (injections mass/momentum)
            PASSIVE    // Passive body
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

          public:
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
            virtual DUAL void update_position(const T dt) = 0;

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

            T drag_coeff_{0.47};     // default sphere drag coefficient
            bool is_sink_ = false;   // is this a sink particle?

            // Cut cell data
            ndarray<CellInfo, Dims> cell_info_;

            DUAL auto cut_cell_indices() const
            {
                return cell_info_.filter_indices([](const auto& cell) {
                    return cell.is_cut;
                });
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
                const MeshType& mesh_cell
            ) const
            {
                // Compute volume fraction
                return 1.0 - std::abs(distance) / mesh_cell.max_cell_width();
            }

            DUAL T compute_cylindrical_volume_fraction(
                const T distance,
                const MeshType& mesh_cell
            ) const
            {
                // Compute volume fraction
                return 1.0 - std::abs(distance) / mesh_cell.max_cell_width();
            }

            DUAL T compute_cartesian_volume_fraction(
                const T distance,
                const MeshType& mesh_cell
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
            }

            DUAL void update_position(const T dt)
            {
                this->position_ += this->velocity_ * dt;
                update_cut_cells();
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
                cell_info_.transform([&](auto& cell, size_t idx) {
                    auto mesh_cell = this->mesh_.cell_geometry(idx);

                    // Use mesh geometry for distance calc
                    const auto r =
                        mesh_cell.compute_distance_vector(this->position_);
                    cell.distance = r.norm() - this->radius_;

                    // Use mesh geometry for volume fraction
                    if (std::abs(cell.distance) <= mesh_cell.max_cell_width()) {
                        cell.is_cut = true;
                        cell.normal = r.normalize();
                        cell.volume_fraction =
                            compute_volume_fraction(cell.distance, mesh_cell);
                    }
                });
            }

            DUAL void spread_surface_forces(auto& cons_state, const T dt)
            {
                for (const auto& idx : cut_cell_indices()) {
                    const auto& cell     = cell_info_[idx];
                    const auto mesh_cell = this->mesh_.cell_geometry(idx);
                    const auto dA_normal = mesh_cell.area_normal(cell.normal);
                    const auto dV        = mesh_cell.volume()();
                    const auto& state    = cons_state[idx];
                    compute_surface_forces(state, cell, dA_normal);
                    this->force_ += compute_drag_force(state, cell, dA_normal);

                    // add to fluid conserved state
                    state.momentum() +=
                        this->force_ * dt * cell.volume_fraction / dV;
                    state.nrg() += this->force_.dot(this->velocity_) * dt *
                                   cell.volume_fraction / dV;
                }
            }

            DUAL void
            apply_body_forces(auto& cons_states, const auto& bodies, const T dt)
            {
                // force affects whole domain, not just cut cells
                const auto body_force = compute_body_forces(bodies);
                cons_states.transform([&](auto& state) {
                    state.momentum() += state.dens() * body_force * dt;
                });
            }

            DUAL void update_conserved_state(auto& cons_state, const T dt)
            {
                // Update position
                this->update_position(dt);

                // Update cut cells
                this->update_cut_cells();

                // Spread force
                this->spread_surface_forces(cons_state, dt);

                // Apply body forces
                this->apply_body_forces(cons_state, dt);
            }

            DUAL bool is_sink() const { return this->is_sink_; }

            // these default implementations can be overridden by derived
            // classes optionally
            DUAL void
            compute_surface_forces(const auto& cell, const auto& dA_normal)
            {
                // for elastic, viscous, porous, etc.
            }

            DUAL void compute_body_forces(const auto& bodies)
            {
                // for gravity, electromagnetic, etc.
            }
        };
    }   // namespace ib

}   // namespace simbi

#endif