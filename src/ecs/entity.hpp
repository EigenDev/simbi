// =============================================================================
// entity.hpp
//
// a minimal, type-erased entity-component-system (ecs) framework.
// this file provides `entity_t` (a unique id) and `registry_t`, which
// manages the association of components (plain data structs) with entities.
// it supports creating entities, adding/removing components, and creating
// views to iterate over entities with specific components.
//
// usage:
//   registry_t registry;
//   entity_t entity = registry.create();
//   registry.add<position>(entity, {1.0, 2.0});
//   auto& pos = registry.get<position>(entity);
// =============================================================================
#pragma once

#include <cstdint>       // for std::uint64_t
#include <memory>        // for std::any
#include <tuple>         // for std::tuple
#include <typeindex>     // for std::type_index
#include <unordered_map> // for std::unordered_map
#include <utility>       // for std::pair, std::move
#include <vector>        // for std::vector

namespace simbi::ecs {
    using entity_t = std::uint64_t;

    class registry_t
    {
        std::uint64_t next_id_{0};

        // type_index -> (entity_id -> shared_ptr<void>)
        std::unordered_map<std::type_index, std::unordered_map<entity_t, std::shared_ptr<void>>>
            storage_;

      public:
        entity_t create()
        {
            return next_id_++;
        }

        template <typename T>
        void add(entity_t entity, T component)
        {
            auto type = std::type_index(typeid(T));

            // shared_ptr with custom deleter that knows the real type
            storage_[type][entity] =
                std::shared_ptr<void>(new T(std::move(component)), [](void* ptr) {
                    delete static_cast<T*>(ptr);
                });
        }

        template <typename T>
        T& get(entity_t entity)
        {
            auto  type = std::type_index(typeid(T));
            void* ptr  = storage_[type].at(entity).get();
            return *static_cast<T*>(ptr);
        }

        template <typename T>
        const T& get(entity_t entity) const
        {
            auto  type = std::type_index(typeid(T));
            void* ptr  = storage_.at(type).at(entity).get();
            return *static_cast<const T*>(ptr);
        }

        template <typename T>
        bool has(entity_t entity) const
        {
            auto type = std::type_index(typeid(T));
            auto it   = storage_.find(type);
            if (it == storage_.end()) {
                return false;
            }
            return it->second.contains(entity);
        }

        template <typename T>
        void remove(entity_t entity)
        {
            auto type = std::type_index(typeid(T));
            storage_[type].erase(entity);
        }

        template <typename T>
        auto view()
        {
            auto                                 type = std::type_index(typeid(T));
            std::vector<std::pair<entity_t, T*>> result;

            if (storage_.contains(type)) {
                for (auto& [entity, ptr] : storage_[type]) {
                    result.emplace_back(entity, static_cast<T*>(ptr.get()));
                }
            }

            return result;
        }

        template <typename T, typename U>
        auto view()
        {
            std::vector<std::tuple<entity_t, T*, U*>> result;

            for (auto [entity, t_ptr] : view<T>()) {
                if (has<U>(entity)) {
                    result.emplace_back(entity, t_ptr, &get<U>(entity));
                }
            }

            return result;
        }
    };

} // namespace simbi::ecs
