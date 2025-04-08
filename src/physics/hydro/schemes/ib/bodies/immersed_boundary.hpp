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

#include "build_options.hpp"                   // for DUAL, DEV, real, size_type
#include "core/types/containers/ndarray.hpp"   // for ndarray
#include "core/types/containers/vector.hpp"    // for spatial_vector_t
#include "core/types/utility/enums.hpp"        // for BodyType
#include "physics/hydro/types/generic_structs.hpp"
#include "physics/hydro/schemes/ib/bodies/body_traits.hpp"
namespace simbi {
    template <size_type Dims>
    class Mesh;

    template <size_type Dims>
    class Cell;

    namespace ib {
        // base class for all immersed boundary bodies
        template <typename T, size_type Dims>
        class BaseBody
        {
          protected:
            using MeshType = Mesh<Dims>;
            const MeshType& mesh_;
            spatial_vector_t<T, Dims> position_;
            spatial_vector_t<T, Dims> velocity_;
            spatial_vector_t<T, Dims> force_;
            spatial_vector_t<T, Dims> fluid_velocity_;
            T mass_;
            T radius_;

          public:
            BaseBody()  = default;
            ~BaseBody() = default;
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
            void interpolate_fluid_velocity(const auto& prim_state) {}

            // reference to various body properties that are needed by the
            // policies
            DUAL auto& force_ref() { return force_; }
            DUAL auto& fluid_velocity_ref() { return fluid_velocity_; }
            DUAL auto& position_ref() { return position_; }
            DUAL auto& velocity_ref() { return velocity_; }
            DUAL auto& mass_ref() { return mass_; }
            DUAL auto& radius_ref() { return radius_; }

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
        };

        // Immersed boundary body class
        template <
            typename T,
            size_type Dims,
            template <typename, size_type> class ForcePolicy,
            template <typename, size_type> class MaterialPolicy,
            template <typename, size_type> class FluidInteractionPolicy,
            template <typename, size_type> class MotionPolicy>
        class ImmersedBody : public BaseBody<T, Dims>,
                             public ForcePolicy<T, Dims>,
                             public MaterialPolicy<T, Dims>,
                             public FluidInteractionPolicy<T, Dims>,
                             public MotionPolicy<T, Dims>
        {
          public:
            using MeshType  = Mesh<Dims>;
            using ForceP    = ForcePolicy<T, Dims>;
            using MaterialP = MaterialPolicy<T, Dims>;
            using FluidP    = FluidInteractionPolicy<T, Dims>;
            using MotionP   = MotionPolicy<T, Dims>;

            // grid-aware cell info. public for gpu lambdas
            struct CellInfo {
                bool is_cut;
                T volume_fraction;
                spatial_vector_t<T, Dims> normal;
                T distance;
            };

            template <template <typename> class Trait>
            DUAL static constexpr bool has_trait()
            {
                return has_trait_impl<
                    Trait,
                    ForceP,
                    MaterialP,
                    FluidP,
                    MotionP>();
            }

          private:
            template <template <typename> class Trait, typename... Policies>
            DUAL static constexpr bool has_trait_impl()
            {
                return (has_policy_trait<Trait, Policies>::value || ...);
            }

            // check if a policy uses a specific trait
            template <template <typename> class Trait, typename Policy>
            struct has_policy_trait {
                template <typename P>
                static auto test(
                    int
                ) -> decltype(std::declval<typename P::template trait_t<T>>(), std::true_type{});

                template <typename>
                static std::false_type test(...);

                static constexpr bool value = decltype(test<Policy>(0))::value;
            };

            friend ForceP;
            friend MaterialP;
            friend FluidP;
            friend MotionP;

          protected:
            // Cut cell data
            ndarray<CellInfo, Dims> cell_info_;

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

          public:

            // constructor
            DUAL ImmersedBody(
                const MeshType& mesh,
                const spatial_vector_t<T, Dims>& position,
                const spatial_vector_t<T, Dims>& velocity,
                const T mass,
                const T radius,
                const typename ForceP::Params& force_params       = {},
                const typename MaterialP::Params& material_params = {},
                const typename FluidP::Params& fluid_params       = {},
                const typename MotionP::Params& motion_params     = {}
            )
                : BaseBody<T, Dims>(mesh, position, velocity, mass, radius),
                  ForceP(force_params),
                  MaterialP(material_params),
                  FluidP(fluid_params),
                  MotionP(motion_params),
                  cell_info_(mesh.size())
            {
                update_cut_cells();
            }
            ~ImmersedBody() = default;

            DUAL ndarray<size_type, 1> cut_cell_indices() const
            {
                return cell_info_.filter_indices(
                    []DEV(const auto& cell) -> bool { return cell.is_cut; },
                    this->get_default_policy()
                );
            }

            DUAL void advance_position(const T dt)
            {
                MotionP::advance_position(*this, dt);
                update_cut_cells();
            }

            DUAL void advance_velocity(const T dt)
            {
                MotionP::advance_velocity(*this, dt);
            }

            DUAL void calculate_forces(const auto& other_bodies, const T dt)
            {
                this->force_ = spatial_vector_t<T, Dims>();
                ForceP::calculate_forces(*this, other_bodies, dt);
                FluidP::calculate_fluid_forces(*this, this->mesh_, dt);
            }

            DUAL void update_material_state(const T dt)
            {
                MaterialP::update_material_state(*this, dt);
            }

            DUAL anyConserved<Dims, Regime::NEWTONIAN> apply_forces_to_fluid(
                const auto& prim,
                const auto& mesh_cell,
                const auto& coords,
                const auto& context,
                const T dt
            )
            {
                return FluidP::apply_forces_to_fluid(
                    *this,
                    prim,
                    mesh_cell,
                    coords,
                    context,
                    dt
                );
            }

            DUAL anyConserved<Dims, Regime::NEWTONIAN> accrete_from_cell(
                const auto& prim,
                const auto& mesh_cell,
                const auto& coords,
                const auto& context,
                const T dt
            )
            {
                return FluidP::accrete_from_cell(
                    *this,
                    prim,
                    mesh_cell,
                    coords,
                    context,
                    dt
                );
            }

            // IBM specific methods
            DUAL void update_cut_cells()
            {
                cell_info_.transform_with_indices(
                    [=, this] DEV(auto& cell, size_type idx) {
                        const auto mesh_cell =
                            this->mesh_.get_cell_from_global(idx);
                        // Use mesh geometry for distance calc
                        auto r =
                            mesh_cell.compute_distance_vector(this->position_);
                        cell.distance = r.norm() - this->radius_;
                        const auto dx = mesh_cell.max_cell_width() * T(0.5);
                        const auto max_corner_dist = dx * std::sqrt(T(Dims));

                        // Use mesh geometry for volume fraction
                        if (std::abs(cell.distance) <= max_corner_dist) {
                            // find dominant direction by comparing absolute
                            // values of components
                            auto max_dir = 0;
                            auto max_val = std::abs(r[0]);

                            for (size_type ii = 1; ii < Dims; ++ii) {
                                if (std::abs(r[ii]) > max_val) {
                                    max_val = std::abs(r[ii]);
                                    max_dir = ii;
                                }
                            }

                            cell.is_cut          = true;
                            cell.normal          = spatial_vector_t<T, Dims>();
                            cell.normal[max_dir] = r[max_dir] > 0 ? 1.0 : -1.0;
                            cell.volume_fraction = compute_volume_fraction(
                                cell.distance,
                                mesh_cell
                            );
                        }
                        else if (cell.distance < 0.0) {
                            // the cell is completely inside the body
                            cell.is_cut          = true;
                            cell.volume_fraction = 1.0;
                        }
                        return cell;
                    },
                    this->get_default_policy()
                );
            }

            DUAL bool has_gravitational_capability() const {
                return has_trait<traits::Gravitational>();
            }

            DUAL bool has_elastic_capability() const {
                return has_trait<traits::Elastic>();
            }

            // DUAL bool has_accretion_capability() const {
            //     return has_trait<traits::Accretion>();
            // }

            DUAL bool has_deformable_capability() const {
                return has_trait<traits::Deformable>();
            }





            // read-only accesors
            DUAL auto position() const { return this->position_; }
            DUAL auto velocity() const { return this->velocity_; }
            DUAL auto force() const { return this->force_; }
            DUAL auto mass() const { return this->mass_; }
            DUAL auto radius() const { return this->radius_; }
            DUAL auto fluid_velocity() const { return this->fluid_velocity_; }
            DUAL auto cell_info() const { return this->cell_info_; }
            DUAL auto mesh() const { return this->mesh_; }

            // setters for position and velocity
            DUAL void set_position(const auto& position)
            {
                this->position_ = position;
                update_cut_cells();
            }
            DUAL void set_velocity(const auto& velocity)
            {
                this->velocity_ = velocity;
            }

            DUAL void set_mass(const T mass) { this->mass_ = mass; }
            DUAL void set_radius(const T radius) { this->radius_ = radius; }
        };
    }   // namespace ib

}   // namespace simbi

#endif
