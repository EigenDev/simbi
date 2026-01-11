#ifndef SINK_HPP
#define SINK_HPP

#include "body.hpp"
#include "build_config.hpp"
#include "containers/vector.hpp"
#include "decorators.hpp"
#include "ecs/geometry_visitor.hpp"
#include "geometry/block_geometry.hpp"
#include "physics/hydro/physics.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>

namespace simbi::body {
    inline real accretion_coefficient(real gamma)
    {
        // approximate lambda(gamma) from Bondi (1952) and Krumholz et al.
        // (2004)
        if (std::abs(gamma - 1.0) < 1e-5) {
            return std::exp(1.5) / 4.0; // isothermal
        }
        else if (std::abs(gamma - 5.0 / 3.0) < 1e-5) {
            return 0.25; // adiabatic monoatomic
        }
        else {
            // general case (approximate)
            return 0.25 *
                   std::pow(2.0 / (5.0 - 3.0 * gamma), (5.0 - 3.0 * gamma) / (2.0 * gamma - 2.0));
        }
    }

    template <std::uint64_t Rank>
    struct sink_properties_t
    {
        std::uint64_t body_idx;

        // Weighted averages from Krumholz prescription
        real mdot{0}; // Bondi-Hoyle accretion rate
        real r_bh{0}; // Bondi-Hoyle  radius
        real total_weight{0};
    };

    template <std::uint64_t Rank, std::uint64_t MaxBodies = 2>
    struct sink_cache_t
    {
        vector_t<sink_properties_t<Rank>, MaxBodies> properties;
        std::size_t                                  count{0};

        DUAL const auto& operator[](std::uint64_t idx) const
        {
            return properties[idx];
        }

        DUAL auto& operator[](std::uint64_t idx)
        {
            return properties[idx];
        }

        DUAL bool empty() const
        {
            return count == 0;
        }
    };

    template <std::uint64_t Rank>
    struct weighted_sums_t
    {
        real                 weighted_density{0};
        real                 weighted_cs{0};
        real                 sum_weight{0};
        real                 sum_mass{0};
        vector_t<real, Rank> weighted_v_vec{0};

        constexpr DUAL auto operator+(const weighted_sums_t& other) const
        {
            return weighted_sums_t{
                weighted_density + other.weighted_density,
                weighted_cs + other.weighted_cs,
                sum_weight + other.sum_weight,
                sum_mass + other.sum_mass,
                weighted_v_vec + other.weighted_v_vec
            };
        }
    };

    struct weight_reducer_t
    {
        template <std::uint64_t Rank>
        constexpr DUAL auto
        operator()(const weighted_sums_t<Rank>& a, const weighted_sums_t<Rank>& b) const
            -> weighted_sums_t<Rank>
        {
            return a + b;
        }
    };

    // functor for sink weight mapping (hoisted for cuda compatibility)
    template <std::uint64_t Rank, typename BlockGeo, typename PrimAccessor>
    struct sink_weight_mapper_t
    {
        BlockGeo             block_geo;
        PrimAccessor         prims;
        vector_t<real, Rank> body_position;
        real                 r_acc;
        real                 gamma;
        bool                 is_binary;

        constexpr DEV auto operator()(auto coord) const -> weighted_sums_t<Rank>
        {
            using namespace simbi::hydro;

            const auto cell_pos = block_geo.centroid(coord);
            const auto r_mag    = (cell_pos - body_position).norm();

            // gaussian weight
            const auto weight = [&]() {
                if constexpr (Rank == 2) {
                    if (is_binary) {
                        const auto r_norm = r_mag / r_acc;
                        return std::exp(-0.25 * std::pow(r_norm, 4));
                    }
                    const auto r_k    = 0.5 * r_acc;
                    const auto r_norm = r_mag / r_k;
                    return std::exp(-r_norm * r_norm);
                }
                else {
                    const auto r_k    = 0.5 * r_acc;
                    const auto r_norm = r_mag / r_k;
                    return std::exp(-r_norm * r_norm);
                }
            }();

            if (weight < 1e-10) {
                return weighted_sums_t<Rank>{};
            }

            const auto prim  = prims(coord);
            const auto rho   = labframe_density(prim);
            const auto v_vec = prim.vel;
            const auto cs    = sound_speed(prim, gamma);
            const auto mass  = block_geo.volume(coord) * rho;

            return weighted_sums_t<Rank>{
                .weighted_density = weight * rho,
                .weighted_cs      = weight * mass * cs,
                .sum_weight       = weight,
                .sum_mass         = weight * mass,
                .weighted_v_vec   = weight * mass * v_vec
            };
        }
    };

    template <typename Body, typename SimState, std::uint64_t Rank = SimState::rank>
    auto compute_sink_properties(const Body& body, const SimState& sim, std::uint64_t lvl)
        -> sink_properties_t<Rank>
    {
        using namespace simbi::hydro;
        static_assert(has_accretion_capability_c<Body>, "Body must have accretion capability");

        const auto& meta     = sim.metadata();
        const auto& bodies   = sim.bodies();
        const auto& mesh_cfg = sim.mesh(lvl);

        const auto r_acc     = accretion_radius(body);
        const auto gamma     = meta.gamma;
        const auto is_binary = (bodies.size() == 2);

        // need motion state for geometry construction
        auto motion = geometry::motion_state_t::static_mesh();

        // accumulate over all partitions at this level
        weighted_sums_t<Rank> sum{};

        for (std::uint64_t pp = 0; pp < sim.num_partitions(lvl); ++pp) {
            const auto& hydro = sim.partition_hydro(lvl, pp);
            const auto& part  = sim.partition(lvl, pp);
            auto&       exec  = sim.partition_executor(lvl, pp);

            const auto prims  = hydro.prim;
            const auto domain = part.owned_domain;

            // use geometry visitor to properly construct block geometry
            auto part_sums = ecs::with_block_geometry<SimState::coord_system>(
                mesh_cfg,
                motion,
                [&](const auto& block_geo) {
                    auto mapper = sink_weight_mapper_t{
                        .block_geo     = block_geo,
                        .prims         = prims[domain],
                        .body_position = body.position,
                        .r_acc         = r_acc,
                        .gamma         = gamma,
                        .is_binary     = is_binary
                    };

                    return exec.reduce(domain, weighted_sums_t<Rank>{}, mapper, weight_reducer_t{});
                }
            );

            sum = sum + part_sums;
        }

        sink_properties_t<Rank> props{.body_idx = body.idx};
        if (sum.sum_weight > 1e-10) {
            const auto rho_eff   = sum.weighted_density / sum.sum_weight;
            const auto cs_eff    = sum.weighted_cs / sum.sum_mass;
            props.total_weight   = sum.sum_weight;
            const auto v_gas_avg = sum.weighted_v_vec / sum.sum_mass;
            // effective relative velocity. Even if standard spherical accretion.
            // the average gas velocity should vanish, so we recover standard
            // Bondi accretion.
            const auto v_eff_mag = (body.velocity - v_gas_avg).norm();

            const auto     v_sq    = v_eff_mag * v_eff_mag;
            const auto     cs_sq   = cs_eff * cs_eff;
            const real     lambda  = accretion_coefficient(gamma);
            constexpr real four_pi = 4.0 * std::numbers::pi_v<real>;
            const auto     r_bh    = body.mass / (cs_eff * cs_eff + v_eff_mag * v_eff_mag);

            props.mdot =
                four_pi * r_bh * r_bh * rho_eff * std::sqrt(lambda * lambda * cs_sq + v_sq);
            props.r_bh = r_bh;
        }

        return props;
    }

    template <typename SimState>
    void update_sink_cache(SimState& sim)
    {
        if (!sim.has_bodies()) {
            return;
        }

        auto& bodies = sim.bodies();
        if (bodies.accretion_count() == 0) {
            return;
        }

        constexpr auto Rank      = SimState::rank;
        constexpr auto MaxBodies = 2;

        if (!bodies.sink_cache.has_value()) {
            bodies.sink_cache = sink_cache_t<Rank, MaxBodies>{};
        }

        auto finest_level = sim.num_levels() - 1;
        bodies.visit_accretion([&](const auto& body) {
            auto props                     = compute_sink_properties(body, sim, finest_level);
            (*bodies.sink_cache)[body.idx] = props;
            bodies.sink_cache->count++;
        });
    }
} // namespace simbi::body

#endif
