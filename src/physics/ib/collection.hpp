#ifndef BODY_COLLECTION_HPP
#define BODY_COLLECTION_HPP

#include "body.hpp"
#include "compat.hpp"
#include "containers/vector.hpp"
#include "execution/executor.hpp"
#include "mesh/mesh_ops.hpp"
#include "physics/hydro/physics.hpp"
#include "utility/config_dict.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace simbi::body {
    using namespace simbi::config;

    // forward declarations for common body types
    template <std::uint64_t Dims>
    using body_variant_t = std::variant<
        rigid_sphere_t<Dims>,
        gravitational_body_t<Dims>,
        black_hole_t<Dims>,
        planet_t<Dims>,
        passive_body_t<Dims>
        // add more combinations as needed (maybe...)
        // [TODO]: revisit later
        >;

    // ========================================================================
    // binary system parameters
    // ========================================================================

    struct binary_parameters_t {
        real total_mass;
        real semi_major;
        real eccentricity;
        real mass_ratio;
        real orbital_period;
        bool is_circular_orbit;
        bool prescribed_motion;

        static auto from_config(const config_dict_t& props)
            -> binary_parameters_t
        {
            auto total_mass   = try_read<real>(props, "total_mass");
            auto semi_major   = try_read<real>(props, "semi_major");
            auto eccentricity = try_read<real>(props, "eccentricity");
            auto mass_ratio   = try_read<real>(props, "mass_ratio");

            auto orbital_period =
                real{2} * std::numbers::pi_v<real> *
                std::sqrt((semi_major * semi_major * semi_major) / total_mass);

            bool is_circular = (eccentricity < real{1e-10});

            return {
              .total_mass        = total_mass,
              .semi_major        = semi_major,
              .eccentricity      = eccentricity,
              .mass_ratio        = mass_ratio,
              .orbital_period    = orbital_period,
              .is_circular_orbit = is_circular,
              .prescribed_motion = true
            };
        }
    };

    template <std::uint64_t Dims>
    struct sink_properties_t {
        std::uint64_t body_idx;

        // Weighted averages from Krumholz prescription
        real mdot{0};   // Bondi-Hoyle accretion rate
        real r_bh{0};   // Bondi-Hoyle  radius
        real total_weight{0};
    };

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    struct sink_cache_t {
        vector_t<sink_properties_t<Dims>, MaxBodies> properties;
        std::size_t count{0};

        const auto& operator[](std::uint64_t idx) const
        {
            return properties[idx];
        }

        auto& operator[](std::uint64_t idx) { return properties[idx]; }

        bool empty() const { return count == 0; }
    };

    template <std::uint64_t Dims>
    struct weighted_sums_t {
        real weighted_density{0};
        real weighted_v{0};
        real weighted_cs{0};
        real sum_weight{0};
        real sum_mass{0};

        constexpr auto operator+(const weighted_sums_t& other) const
        {
            return weighted_sums_t{
              weighted_density + other.weighted_density,
              weighted_v + other.weighted_v,
              weighted_cs + other.weighted_cs,
              sum_weight + other.sum_weight,
              sum_mass + other.sum_mass
            };
        }
    };

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    struct body_collection_t {
        static constexpr std::uint64_t dimensions = Dims;

        vector_t<body_variant_t<Dims>, MaxBodies> bodies_;
        std::optional<binary_parameters_t> binary_params_;
        std::optional<sink_cache_t<Dims>> sink_cache;
        std::size_t size_            = 0;
        std::string system_name_     = "Untitled";
        std::string reference_frame_ = "inertial";   // or "corotating"

        template <typename Body>
        constexpr auto add(Body&& body) &&
        {
            if (size_ >= MaxBodies) {
                throw std::runtime_error("Body collection is full");
            }
            auto indexed_body = body;
            // indexed_body.index = size_;
            bodies_[size_++] = std::move(indexed_body);
            return std::move(*this);
        }

        template <typename Body>
        constexpr auto add(Body&& body) const&
        {
            auto result = *this;
            return std::move(result).add(std::forward<Body>(body));
        }

        constexpr auto with_name(std::string name) &&
        {
            system_name_ = std::move(name);
            return std::move(*this);
        }

        constexpr auto with_name(const std::string& name) const&
        {
            auto result = *this;
            return std::move(result).with_name(name);
        }

        constexpr auto with_reference_frame(std::string frame) &&
        {
            if (frame != "inertial" && frame != "corotating") {
                throw std::runtime_error("Invalid reference frame: " + frame);
            }
            reference_frame_ = std::move(frame);
            return std::move(*this);
        }

        constexpr auto with_reference_frame(const std::string& frame) const&
        {
            auto result = *this;
            return std::move(result).with_reference_frame(frame);
        }

        constexpr auto with_system_config(const binary_parameters_t& params) &&
        {
            binary_params_ = params;
            return std::move(*this);
        }

        constexpr auto
        with_system_config(const binary_parameters_t& params) const&
        {
            auto result = *this;
            return std::move(result).with_system_config(params);
        }

        constexpr std::size_t size() const { return size_; }
        constexpr std::size_t capacity() const { return MaxBodies; }
        constexpr bool empty() const { return size_ == 0; }
        constexpr bool full() const { return size_ == MaxBodies; }
        constexpr const std::string& name() const { return system_name_; }
        constexpr const std::string& reference_frame() const
        {
            return reference_frame_;
        }
        constexpr auto binary_params() const
        {
            if (!binary_params_) {
                throw std::runtime_error("No binary parameters set");
            }
            return *binary_params_;
        }

        constexpr auto begin() const { return bodies_.begin(); }
        constexpr auto end() const { return bodies_.begin() + size_; }

        constexpr const auto& operator[](std::size_t idx) const
        {
            if constexpr (global::bounds_checking) {
                assert(idx < size_);
            }
            return bodies_[idx];
        }

        // visitor pattern for compile-time dispatch
        template <typename Visitor>
        constexpr void visit_all(Visitor&& visitor) const
        {
            for (std::size_t ii = 0; ii < size_; ++ii) {
                std::visit(visitor, bodies_[ii]);
            }
        }

        // capability-based filtering
        template <typename Tag, typename Visitor>
        constexpr void visit_with_capability(Visitor&& visitor) const
        {
            visit_all([&](const auto& body) {
                using body_type = std::decay_t<decltype(body)>;
                if constexpr (body_type::template has_capability_v<Tag>) {
                    visitor(body);
                }
            });
        }

        // specific capability visitors
        template <typename Visitor>
        constexpr void visit_gravitational(Visitor&& visitor) const
        {
            visit_with_capability<capabilities::gravitational_tag>(visitor);
        }

        template <typename Visitor>
        constexpr void visit_accretion(Visitor&& visitor) const
        {
            visit_with_capability<capabilities::accretion_tag>(visitor);
        }

        template <typename Visitor>
        constexpr void visit_rigid(Visitor&& visitor) const
        {
            visit_with_capability<capabilities::rigid_tag>(visitor);
        }

        // fp integration b/c I dig it a lot lately
        template <typename Op>
        constexpr auto operator|(Op&& op) const
        {
            return std::forward<Op>(op)(*this);
        }

        // utility functions
        constexpr std::size_t count_with_capability(auto tag) const
        {
            std::size_t count = 0;
            visit_all([&](const auto& body) {
                if (body.template has_capability_v<decltype(tag)>) {
                    ++count;
                }
            });
            return count;
        }

        constexpr std::size_t gravitational_count() const
        {
            return count_with_capability(capabilities::gravitational_tag{});
        }

        constexpr std::size_t accretion_count() const
        {
            return count_with_capability(capabilities::accretion_tag{});
        }

        constexpr std::size_t rigid_count() const
        {
            return count_with_capability(capabilities::rigid_tag{});
        }

        template <typename Predicate>
        constexpr auto find_if(Predicate&& pred) const
        {
            for (std::size_t i = 0; i < size_; ++i) {
                if (std::visit(pred, bodies_[i])) {
                    return i;
                }
            }
            return size_;
        }
    };

    inline auto accretion_coefficient(real gamma) -> real
    {
        // approximate lambda(gamma) from Bondi (1952) and Krumholz et al.
        // (2004)
        if (std::abs(gamma - 1.0) < 1e-5) {
            return std::exp(1.5) / 4.0;   // isothermal
        }
        else if (std::abs(gamma - 5.0 / 3.0) < 1e-5) {
            return 0.25;   // adiabatic monoatomic
        }
        else {
            // general case (approximate)
            return 0.25 * std::pow(
                              2.0 / (5.0 - 3.0 * gamma),
                              (5.0 - 3.0 * gamma) / (2.0 * gamma - 2.0)
                          );
        }
    }

    // template <
    //     typename Body,
    //     typename HydroState,
    //     typename MeshConfig,
    //     std::uint64_t Dims = MeshConfig::dimensions>
    // auto compute_sink_properties(
    //     const Body& body,
    //     const HydroState& state,
    //     const MeshConfig& mesh
    // ) -> sink_properties_t<Dims>
    // {
    //     using namespace simbi::hydro;
    //     static_assert(
    //         has_accretion_capability_c<Body>,
    //         "Body must have accretion capability"
    //     );

    //     const auto r_acc     = accretion_radius(body);
    //     const auto gamma     = state.metadata.gamma;
    //     const auto is_binary = (state.bodies->size() == 2);
    //     const auto prims     = state.prim;

    //     // Mapper: compute weighted contribution from each cell
    //     auto mapper = [=](auto coord) -> weighted_sums_t<Dims> {
    //         // get cell position and compute distance to sink
    //         const auto cell_pos = mesh::centroid(coord, mesh);
    //         const auto r_mag    = (cell_pos - body.position).norm();

    //         // Gaussian weight
    //         const auto weight = [is_binary, r_mag, r_acc]() {
    //             if constexpr (Dims == 2) {
    //                 if (is_binary) {
    //                     const auto r_norm = r_mag / r_acc;
    //                     // from Dittmann & Ryan (2021)
    //                     return std::exp(-0.25 * std::pow(r_norm, 4));
    //                 }
    //                 const auto r_k    = 0.5 * r_acc;
    //                 const auto r_norm = r_mag / r_k;
    //                 // from Krumholz et al. (2004)
    //                 return std::exp(-r_norm * r_norm);
    //             }
    //             else {
    //                 (void) is_binary;
    //                 const auto r_k    = 0.5 * r_acc;
    //                 const auto r_norm = r_mag / r_k;
    //                 // from Krumholz et al. (2004)
    //                 return std::exp(-r_norm * r_norm);
    //             }
    //         }();

    //         if (weight < 1e-10) {
    //             return weighted_sums_t<Dims>{};
    //         }
    //         const auto prim = prims[mesh.domain][coord];
    //         const auto rho  = labframe_density(prim);
    //         // most problems will start with zero velcity
    //         // field at infinity.
    //         // [TODO]: revisit later for more complex flows
    //         const auto v_mag = body.velocity.norm();
    //         const auto cs    = sound_speed(prim, gamma);
    //         const auto mass  = mesh::volume(coord, mesh) * rho;

    //         return weighted_sums_t<Dims>{
    //           .weighted_density = weight * rho,
    //           .weighted_v       = weight * mass * v_mag,
    //           .weighted_cs      = weight * mass * cs,
    //           .sum_weight       = weight,
    //           .sum_mass         = weight * mass
    //         };
    //     };

    //     auto reducer = [](const auto& a, const auto& b) { return a + b; };

    //     auto sums =
    //         exec::default_executor()
    //             .reduce(mesh.domain, weighted_sums_t<Dims>{}, mapper,
    //             reducer) .wait();

    //     sink_properties_t<Dims> props{.body_idx = body.idx};
    //     if (sums.sum_weight > 1e-10) {
    //         const auto rho_eff   = sums.weighted_density / sums.sum_weight;
    //         const auto v_eff_mag = sums.weighted_v / sums.sum_mass;
    //         const auto cs_eff    = sums.weighted_cs / sums.sum_mass;
    //         props.total_weight   = sums.sum_weight;

    //         // Compute Bondi-Hoyle accretion rate
    //         // \dot{M} = 4 \pi \lambda(\gamma) G^2 M^2 \rho / (c_s^2 +
    //         // v^2)^(3/2) w/ G = 1 in code units
    //         const auto v_sq        = v_eff_mag * v_eff_mag;
    //         const auto cs_sq       = cs_eff * cs_eff;
    //         const auto gamma       = state.metadata.gamma;
    //         const real lambda      = accretion_coefficient(gamma);
    //         constexpr real four_pi = 4.0 * std::numbers::pi_v<real>;
    //         const auto r_bh =
    //             body.mass / (cs_eff * cs_eff + v_eff_mag * v_eff_mag);

    //         props.mdot = four_pi * r_bh * r_bh * rho_eff *
    //                      std::sqrt(lambda * lambda * cs_sq + v_sq);
    //         props.r_bh = r_bh;
    //     }

    //     return props;
    // }

    template <
        typename Body,
        typename SimState,
        std::uint64_t Dims = SimState::dimensions>
    auto compute_sink_properties(
        const Body& body,
        const SimState& sim,
        std::uint64_t lvl
    ) -> sink_properties_t<Dims>
    {
        using namespace simbi::hydro;
        static_assert(
            has_accretion_capability_c<Body>,
            "Body must have accretion capability"
        );

        const auto& meta   = sim.metadata();
        const auto& hydro  = sim.hydro(lvl);
        const auto& mesh   = sim.mesh(lvl);
        const auto& bodies = sim.bodies();

        const auto r_acc     = accretion_radius(body);
        const auto gamma     = meta.gamma;
        const auto is_binary = (bodies.size() == 2);
        const auto prims     = hydro.prim;

        // Mapper: compute weighted contribution from each cell
        auto mapper = [=](auto coord) -> weighted_sums_t<Dims> {
            // get cell position and compute distance to sink
            const auto cell_pos = mesh::centroid(coord, mesh);
            const auto r_mag    = (cell_pos - body.position).norm();

            // Gaussian weight
            const auto weight = [is_binary, r_mag, r_acc]() {
                if constexpr (Dims == 2) {
                    if (is_binary) {
                        const auto r_norm = r_mag / r_acc;
                        // from Dittmann & Ryan (2021)
                        return std::exp(-0.25 * std::pow(r_norm, 4));
                    }
                    const auto r_k    = 0.5 * r_acc;
                    const auto r_norm = r_mag / r_k;
                    // from Krumholz et al. (2004)
                    return std::exp(-r_norm * r_norm);
                }
                else {
                    (void) is_binary;
                    const auto r_k    = 0.5 * r_acc;
                    const auto r_norm = r_mag / r_k;
                    // from Krumholz et al. (2004)
                    return std::exp(-r_norm * r_norm);
                }
            }();

            if (weight < 1e-10) {
                return weighted_sums_t<Dims>{};
            }
            const auto prim = prims[mesh.domain][coord];
            const auto rho  = labframe_density(prim);
            // most problems will start with zero velcity
            // field at infinity.
            // [TODO]: revisit later for more complex flows
            const auto v_mag = body.velocity.norm();
            const auto cs    = sound_speed(prim, gamma);
            const auto mass  = mesh::volume(coord, mesh) * rho;

            return weighted_sums_t<Dims>{
              .weighted_density = weight * rho,
              .weighted_v       = weight * mass * v_mag,
              .weighted_cs      = weight * mass * cs,
              .sum_weight       = weight,
              .sum_mass         = weight * mass
            };
        };

        auto reducer = [](const auto& a, const auto& b) { return a + b; };

        auto sums =
            exec::default_executor()
                .reduce(mesh.domain, weighted_sums_t<Dims>{}, mapper, reducer)
                .wait();

        sink_properties_t<Dims> props{.body_idx = body.idx};
        if (sums.sum_weight > 1e-10) {
            const auto rho_eff   = sums.weighted_density / sums.sum_weight;
            const auto v_eff_mag = sums.weighted_v / sums.sum_mass;
            const auto cs_eff    = sums.weighted_cs / sums.sum_mass;
            props.total_weight   = sums.sum_weight;

            // Compute Bondi-Hoyle accretion rate
            // \dot{M} = 4 \pi \lambda(\gamma) G^2 M^2 \rho / (c_s^2 +
            // v^2)^(3/2) w/ G = 1 in code units
            const auto v_sq        = v_eff_mag * v_eff_mag;
            const auto cs_sq       = cs_eff * cs_eff;
            const real lambda      = accretion_coefficient(gamma);
            constexpr real four_pi = 4.0 * std::numbers::pi_v<real>;
            const auto r_bh =
                body.mass / (cs_eff * cs_eff + v_eff_mag * v_eff_mag);

            props.mdot = four_pi * r_bh * r_bh * rho_eff *
                         std::sqrt(lambda * lambda * cs_sq + v_sq);
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

        constexpr auto Dims      = SimState::dimensions;
        constexpr auto MaxBodies = 2;

        if (!bodies.sink_cache.has_value()) {
            bodies.sink_cache = sink_cache_t<Dims, MaxBodies>{};
        }

        auto finest_level = sim.num_levels() - 1;
        bodies.visit_accretion([&](const auto& body) {
            auto props = compute_sink_properties(body, sim, finest_level);
            (*bodies.sink_cache)[body.idx] = props;
            bodies.sink_cache->count++;
        });
    }

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    constexpr auto make_body_collection()
    {
        return body_collection_t<Dims, MaxBodies>{};
    }

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 2>
    constexpr auto create_binary_system(
        const vector_t<real, Dims>& pos1,
        const vector_t<real, Dims>& vel1,
        const vector_t<real, Dims>& pos2,
        const vector_t<real, Dims>& vel2,
        real mass1,
        real mass2,
        real radius1,
        real radius2,
        real softening1,
        real softening2,
        real sink_rate1   = 0.0,
        real sink_rate2   = 0.0,
        real accr_radius1 = 0.0,
        real accr_radius2 = 0.0,
        real sink_delta1  = 1.0,
        real sink_delta2  = 1.0
    )
    {
        if (sink_rate1 > 0.0 && sink_rate2 > 0.0) {
            // this is a binary black hole system
            return make_body_collection<Dims, MaxBodies>()
                .add(
                    make_black_hole<Dims>(
                        pos1,
                        vel1,
                        mass1,
                        radius1,
                        softening1,
                        sink_rate1,
                        sink_delta1,
                        accr_radius1
                    )
                )
                .add(
                    make_black_hole<Dims>(
                        pos2,
                        vel2,
                        mass2,
                        radius2,
                        softening2,
                        sink_rate2,
                        sink_delta2,
                        accr_radius2
                    )
                );
        }
        else if (sink_rate1 <= 0.0 && sink_rate2 <= 0.0) {
            // this is a binary gravitational system
            return make_body_collection<Dims, MaxBodies>()
                .add(
                    make_gravitational_body<
                        Dims>(pos1, vel1, mass1, radius1, softening1)
                )
                .add(
                    make_gravitational_body<
                        Dims>(pos2, vel2, mass2, radius2, softening2)
                );
        }
        else {
            // this is a mixed system with one gravitational and one
            // accretion body
            if (sink_rate1 > 0.0) {
                return make_body_collection<Dims, MaxBodies>()
                    .add(
                        make_black_hole<Dims>(
                            pos1,
                            vel1,
                            mass1,
                            radius1,
                            softening1,
                            sink_rate1,
                            sink_delta1,
                            accr_radius1
                        )
                    )
                    .add(
                        make_gravitational_body<
                            Dims>(pos2, vel2, mass2, radius2, softening2)
                    );
            }
            else {
                return make_body_collection<Dims, MaxBodies>()
                    .add(
                        make_gravitational_body<
                            Dims>(pos1, vel1, mass1, radius1, softening1)
                    )
                    .add(
                        make_black_hole<Dims>(
                            pos2,
                            vel2,
                            mass2,
                            radius2,
                            softening2,
                            sink_rate2,
                            sink_delta2,
                            accr_radius2
                        )
                    );
            }
        }
    }

    namespace collection_ops {
        // map operation over collection
        template <typename Func>
        struct map_bodies_t {
            Func func_;

            template <typename Collection>
            constexpr auto operator()(const Collection& collection) const
            {
                // returns array of results
                vector_t<
                    std::invoke_result_t<Func, decltype(*collection.begin())>,
                    2>
                    results;
                std::size_t idx = 0;

                collection.visit_all([&](const auto& body) {
                    results[idx++] = func_(body);
                });

                return results;   // or return a view/span of first 'idx'
                                  // elements
            }
        };

        template <typename Func>
        constexpr auto map_bodies(Func&& func)
        {
            return map_bodies_t<std::decay_t<Func>>{std::forward<Func>(func)};
        }

        // filter operation
        template <typename Predicate>
        struct filter_bodies_t {
            Predicate pred_;

            template <typename Collection>
            constexpr auto operator()(const Collection& collection) const
            {
                // returns new collection with only matching bodies
                auto result = make_body_collection<Collection::dimensions>();

                collection.visit_all([&](const auto& body) {
                    if (std::visit(pred_, body)) {
                        result = std::move(result).add(body);
                    }
                });

                return result;
            }
        };

        template <typename Predicate>
        constexpr auto filter_bodies(Predicate&& pred)
        {
            return filter_bodies_t<std::decay_t<Predicate>>{
              std::forward<Predicate>(pred)
            };
        }
    }   // namespace collection_ops
}   // namespace simbi::body

#endif
