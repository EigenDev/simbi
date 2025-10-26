#ifndef ECS_SIMULATION_HPP
#define ECS_SIMULATION_HPP

#include "components.hpp"   // for hydro_fields_t, mesh_geometry_t, simulation_metadata_t, sources_t
#include "entity.hpp"              // for entity_t
#include "hydro_state_types.hpp"   // for vtraits
#include "utility/enums.hpp"       // for Geometry, Regime

#include <cstdint>   // for std::uint64_t
#include <vector>    // for std::vector

namespace simbi::ecs {

    template <Regime R, std::uint64_t Dims, Geometry G, typename EoS>
    struct simulation_t {
        using conserved_t = typename vtraits<R, Dims, EoS>::conserved_type;
        using primitive_t = typename vtraits<R, Dims, EoS>::primitive_type;
        using eos_t       = EoS;
        static constexpr std::uint64_t dimensions = Dims;
        static constexpr Regime regime_t          = R;
        static constexpr bool is_mhd = (R == Regime::MHD || R == Regime::RMHD);
        static constexpr auto nvars  = (is_mhd) ? 9 : Dims + 3;

        registry_t registry;
        std::vector<entity_t> levels;
        entity_t global;

        bool in_failure_state{false};
        bool was_interrupted{false};

        // query interface
        std::uint64_t num_levels() const { return levels.size(); }

        bool has_refinement() const { return num_levels() > 1; }

        auto& level_info(std::uint64_t lvl)
        {
            return registry.get<level_info_t>(levels[lvl]);
        }
        auto& level_info(std::uint64_t lvl) const
        {
            return registry.get<level_info_t>(levels[lvl]);
        }

        auto& refinement(std::uint64_t lvl)
        {
            return registry.get<refinement_child_t<Dims>>(levels[lvl]);
        }
        auto& refinement(std::uint64_t lvl) const
        {
            return registry.get<refinement_child_t<Dims>>(levels[lvl]);
        }

        auto& metadata()
        {
            return registry.get<simulation_metadata_t<Dims>>(global);
        }

        const auto& metadata() const
        {
            return registry.get<simulation_metadata_t<Dims>>(global);
        }

        auto& sources() const { return registry.get<sources_t<Dims>>(global); }
        const auto& sources() { return registry.get<sources_t<Dims>>(global); }

        // level accessors
        auto& hydro(std::uint64_t level_id)
        {
            return registry.get<hydro_fields_t<conserved_t, primitive_t, Dims>>(
                levels[level_id]
            );
        }
        const auto& hydro(std::uint64_t level_id) const
        {
            return registry.get<hydro_fields_t<conserved_t, primitive_t, Dims>>(
                levels[level_id]
            );
        }

        auto& mesh(std::uint64_t level_id)
        {
            return registry.get<mesh_geometry_t<Dims, G>>(levels[level_id])
                .config;
        }
        const auto& mesh(std::uint64_t level_id) const
        {
            return registry.get<mesh_geometry_t<Dims, G>>(levels[level_id])
                .config;
        }

        bool has_bodies() const
        {
            return registry.has<immersed_bodies_t<Dims>>(global);
        }

        auto& bodies()
        {
            return registry.get<immersed_bodies_t<Dims>>(global).bodies;
        }
        const auto& bodies() const
        {
            return registry.get<immersed_bodies_t<Dims>>(global).bodies;
        }
        auto& diagnostics()
        {
            return registry.get<body_info_t<Dims>>(global).diagnostics;
        }
        const auto& diagnostics() const
        {
            return registry.get<body_info_t<Dims>>(global).diagnostics;
        }

        auto& hierarchy()
        {
            return registry.get<fmr_hierarchy_t<Dims>>(global).hierarchy;
        }
        const auto& hierarchy() const
        {
            return registry.get<fmr_hierarchy_t<Dims>>(global).hierarchy;
        }
    };

}   // namespace simbi::ecs

#endif
