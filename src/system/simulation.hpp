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

        // query interface
        std::uint64_t num_levels() const { return levels.size(); }

        bool has_refinement() const { return num_levels() > 1; }

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

        auto& mesh(std::uint64_t level_id)
        {
            return registry.get<mesh_geometry_t<Dims, G>>(levels[level_id])
                .config;
        }
    };

}   // namespace simbi::ecs

#endif
