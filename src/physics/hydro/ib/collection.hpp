#ifndef SIMBI_BODY_COLLECTION_HPP
#define SIMBI_BODY_COLLECTION_HPP

#include "body.hpp"
#include "config.hpp"
#include "containers/vector.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace simbi::body {

    // forward declarations for common body types
    template <std::uint64_t Dims>
    using body_variant_t = std::variant<
        rigid_sphere_t<Dims>,
        gravitational_body_t<Dims>,
        black_hole_t<Dims>,
        planet_t<Dims>
        // add more combinations as needed
        >;

    struct binary_system_config_t {
        real semi_major;
        real mass_ratio;
        real eccentricity;
        real orbital_period;
        bool prescribed_motion;
        bool is_circular_orbit;
        std::pair<std::uint64_t, std::uint64_t> body_indices;
    };

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 8>
    class body_collection_t
    {
      private:
        vector_t<body_variant_t<Dims>, MaxBodies> bodies_;
        std::size_t size_ = 0;

      public:
        static constexpr std::uint64_t dimensions = Dims;

        // constructors
        constexpr body_collection_t() = default;

        // add body with perfect forwarding
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

        // immutable add (returns new collection)
        template <typename Body>
        constexpr auto add(Body&& body) const&
        {
            auto result = *this;
            return std::move(result).add(std::forward<Body>(body));
        }

        // size and capacity
        constexpr std::size_t size() const { return size_; }
        constexpr std::size_t capacity() const { return MaxBodies; }
        constexpr bool empty() const { return size_ == 0; }
        constexpr bool full() const { return size_ == MaxBodies; }

        // iteration support
        constexpr auto begin() const { return bodies_.begin(); }
        constexpr auto end() const { return bodies_.begin() + size_; }

        // indexed access
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

        // fo integration b/c I dig it a lot lately
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
                if constexpr (body.template has_capability_v<decltype(tag)>) {
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

        // find body by index
        template <typename Predicate>
        constexpr auto find_if(Predicate&& pred) const
        {
            for (std::size_t i = 0; i < size_; ++i) {
                if (std::visit(pred, bodies_[i])) {
                    return i;
                }
            }
            return size_;   // not found
        }
    };

    // factory functions for fluent interface
    template <std::uint64_t Dims, std::uint64_t MaxBodies = 8>
    constexpr auto make_body_collection()
    {
        return body_collection_t<Dims, MaxBodies>{};
    }

    template <std::uint64_t Dims, std::uint64_t MaxBodies = 8>
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
        real accr_efficiency1 = 0.0,
        real accr_efficiency2 = 0.0,
        real accr_radius1     = 0.0,
        real accr_radius2     = 0.0
    )
    {
        if (accr_efficiency1 > 0.0 && accr_efficiency2 > 0.0) {
            // this is a binary black hole system
            return make_body_collection<Dims, MaxBodies>()
                .add(
                    make_black_hole<Dims>(
                        pos1,
                        vel1,
                        mass1,
                        radius1,
                        softening1,
                        accr_efficiency1,
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
                        accr_efficiency2,
                        accr_radius2
                    )
                );
        }
        else if (accr_efficiency1 <= 0.0 && accr_efficiency2 <= 0.0) {
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
            if (accr_efficiency1 > 0.0) {
                return make_body_collection<Dims, MaxBodies>()
                    .add(
                        make_black_hole<Dims>(
                            pos1,
                            vel1,
                            mass1,
                            radius1,
                            softening1,
                            accr_efficiency1,
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
                            accr_efficiency2,
                            accr_radius2
                        )
                    );
            }
        }
    }

    // functional programming helpers
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
                    Collection::capacity()>
                    results;
                std::size_t idx = 0;

                collection.visit_all([&](const auto& body) {
                    results[idx++] = std::visit(func_, body);
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
                auto result = make_body_collection<
                    Collection::dimensions,
                    Collection::capacity()>();

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
